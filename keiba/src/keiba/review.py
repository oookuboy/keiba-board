"""結果照合と回顧。

SKILL.md の「レース結果が出たら必ず振り返りを行い、反省点を反映する」を
自動化する。人が読む教訓ログ（LESSONS.md）と、機械が使う成績（reviews）の
両方を残す。重み自体は backtest.py が調整するので、ここは事実の記録に徹する。
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import date
from pathlib import Path

from keiba.backtest import _actual_top3, _settle
from keiba.betting import Ticket
from keiba.store import Store

log = logging.getLogger(__name__)

# 予想の根拠テキストから、どの教訓が効いた／外したかを引き当てる索引。
# features/engine が出す文言と対応させてある。
#
# 展開（教訓2）は全頭に注記が付くので「想定」で拾うと必ず発火してしまい、
# 効き目の指標にならない。恩恵を受ける側に振れた馬（展開評価0.8以上）だけを
# 数えるようにしてある。
LESSON_MARKERS: dict[str, tuple[str, ...]] = {
    "教訓1 血統で馬場替わりを拾う": ("替わり。父",),
    "教訓2 展開読み": ("展開評価0.8", "展開評価0.9", "展開評価1.0"),
    "教訓8 確逃げ馬を切らない": ("教訓8",),
    "教訓11 実績馬を休み明けで切らない": ("教訓11",),
    "教訓12 降級・格上帰りを上位に": ("格上",),
}


def review_day(store: Store, payload: dict) -> dict:
    """予想 payload に実結果を埋め、成績を集計して返す。

    payload は predict.predict_day が作ったものをそのまま受け取り、
    各レースの "result" を埋めて返す（破壊的に書き換える）。
    """
    spent = returned = hits = 0
    worked: Counter[str] = Counter()
    failed: Counter[str] = Counter()

    for race in payload["races"]:
        card = store.load_card(race["race_id"])
        if card is None or not card.results:
            continue
        top3 = _actual_top3(card)
        if top3 is None:
            continue

        tickets = [
            Ticket(b["type"], b["combination"], b["amount"], b["why"])
            for b in race["bets"]
        ]
        race_spent, race_returned, hit_types, best = _settle(tickets, card, top3)
        hit = bool(hit_types)

        names = {e.umaban: e.horse_name for e in card.entries}
        race["result"] = {
            "top3": top3,
            "top3_names": [names.get(u, "") for u in top3],
            "hit": hit,
            "hit_types": hit_types,
            "spent": race_spent,
            "returned": race_returned,
            "best_payout": best,
        }

        spent += race_spent
        returned += race_returned
        hits += int(hit)

        # 印を打った馬が実際に絡んだか。教訓の効き目はここで判定する
        placed = set(top3)
        for horse in race["horses"]:
            if not horse["mark"]:
                continue
            bucket = worked if horse["umaban"] in placed else failed
            for lesson, markers in LESSON_MARKERS.items():
                if any(m in r for r in horse["reasons"] for m in markers):
                    bucket[lesson] += 1

        store.conn.execute(
            "INSERT OR REPLACE INTO reviews"
            " (race_id, hit, spent, returned, lessons_worked, lessons_failed, note)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                race["race_id"],
                int(hit),
                race_spent,
                race_returned,
                json.dumps(dict(worked), ensure_ascii=False),
                json.dumps(dict(failed), ensure_ascii=False),
                race["confidence_reason"],
            ),
        )

    store.conn.commit()

    payload["summary"]["returned"] = returned
    payload["summary"]["hits"] = hits
    payload["summary"]["roi"] = round(returned / spent * 100, 1) if spent else None
    payload["summary"]["lessons_worked"] = dict(worked)
    payload["summary"]["lessons_failed"] = dict(failed)
    return payload


def append_lessons(payload: dict, path: Path) -> None:
    """人が読む回顧ログを追記する。

    SKILL.md 本体は手で育てるものなので上書きしない。機械が観測した事実だけを
    別ファイルに積み、人が読んで SKILL.md に反映できるようにする。
    """
    summary = payload["summary"]
    day = payload["date"]

    hit_races = [r for r in payload["races"] if (r.get("result") or {}).get("hit")]
    lines = [
        f"\n## {day}",
        "",
        f"- 対象 {summary['races']}R / 買い {summary['bet']}R / 見送り {summary['skipped']}R",
        f"- 投資 {summary['spend']:,}円 → 払戻 {summary.get('returned', 0):,}円 "
        f"（回収率 {summary.get('roi') or 0:.1f}%・的中 {summary.get('hits', 0)}R）",
    ]

    if hit_races:
        lines.append("- 的中:")
        for r in hit_races:
            res = r["result"]
            lines.append(
                f"  - {r['venue']}{r['race_no']}R {r['name']}（{r['confidence']}）"
                f" {'-'.join(map(str, res['top3']))} "
                f"最高配当{res['best_payout']:,}円 → 払戻{res['returned']:,}円"
            )
    else:
        lines.append("- 的中なし")

    if summary.get("lessons_worked"):
        lines.append("- 効いた教訓: " + ", ".join(
            f"{k}({v})" for k, v in sorted(
                summary["lessons_worked"].items(), key=lambda kv: -kv[1]
            )
        ))
    if summary.get("lessons_failed"):
        lines.append("- 外した教訓: " + ", ".join(
            f"{k}({v})" for k, v in sorted(
                summary["lessons_failed"].items(), key=lambda kv: -kv[1]
            )
        ))

    header = (
        "# 実戦ログ\n\n"
        "> keiba-review ワークフローが開催日ごとに追記する。機械が観測した事実の\n"
        "> 記録であり、判断は含まない。ここを読んで SKILL.md の「学習済み教訓」を\n"
        "> 育てるのは人の仕事。\n"
    )
    if not path.exists():
        path.write_text(header, encoding="utf-8")
    # 同じ日を二重に書かない
    existing = path.read_text(encoding="utf-8")
    if f"\n## {day}\n" in existing:
        log.info("%s は記録済み", day)
        return
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")


def totals(store: Store) -> dict:
    """通算成績。ボードのヘッダに出す。"""
    row = store.conn.execute(
        "SELECT COUNT(*) n, SUM(hit) hits, SUM(spent) spent, SUM(returned) returned"
        " FROM reviews"
    ).fetchone()
    spent = row["spent"] or 0
    returned = row["returned"] or 0
    return {
        "races": row["n"] or 0,
        "hits": row["hits"] or 0,
        "spent": spent,
        "returned": returned,
        "roi": round(returned / spent * 100, 1) if spent else None,
    }
