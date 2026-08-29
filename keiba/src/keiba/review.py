"""結果照合と回顧。

SKILL.md の「レース結果が出たら必ず振り返りを行い、反省点を反映する」を
自動化する。人が読む教訓ログ（LESSONS.md）と、機械が使う成績（reviews）の
両方を残す。重み自体は backtest.py が調整するので、ここは事実の記録に徹する。
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path

from keiba import betting
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


PLACE = "複勝"

# 毎回そろえて測る「もし別の買い方をしていたら」。買い方を変えるかどうかは、
# バックテストだけでなく本番の積み上げでも見たい。バックテストと本番がずれる
# のがこのプロジェクトで一番よく起きた壊れ方なので、同じ土俵で並べておく。
ALTERNATIVES = {
    "box_all": "印ボックス（全レース）",
    "box_bet": "印ボックス（買ったRだけ）",
    "place3": "印上位3頭の複勝",
}


def _payout_of(card, bet_type: str, combination: str) -> int:
    for p in card.payouts:
        if p.bet_type == bet_type and p.combination == combination:
            return p.payout
    return 0


def _marked_by_score(race: dict) -> list[int]:
    """印を打った馬の馬番を、評価の高い順に返す。"""
    marked = [h for h in race["horses"] if h.get("mark") and h.get("umaban")]
    marked.sort(key=lambda h: -h.get("score", 0))
    return [h["umaban"] for h in marked]


def _alternatives(race: dict, card, top3: list[int], unit: int = 100) -> dict:
    """その日の別の買い方だと、このレースがいくらになっていたかを返す。

    予想 payload に残っている印だけで計算できる形にしてある。買い目を組み直す
    のではないので、見送ったレース（買い目0点）についても測れる。そこが肝心で、
    「見送りは正しかったのか」は見送ったレースを測らないと永遠に分からない。
    """
    numbers = _marked_by_score(race)
    if len(numbers) < 3:
        return {}

    # ① 全レースで印ボックスを買った場合。ボックスは印の3頭組を全部買うので、
    #    的中する条件は「1〜3着が全員印」と同値。組み直さずに判定できる。
    points = math.comb(len(numbers), 3)
    trio_hit = set(top3) <= set(numbers)
    trio_payout = (
        _payout_of(card, betting.TRIO, "-".join(str(n) for n in sorted(top3)))
        if trio_hit
        else 0
    )

    # ② 印上位3頭の複勝。モデルが予測しているのは「3着以内に入るか」なので、
    #    賭式としてはこちらが素直（控除率も 20% と三連複の 27.5% より低い）。
    place = {
        int(p.combination): p.payout
        for p in card.payouts
        if p.bet_type == PLACE and p.combination.isdigit()
    }

    box = (points * unit, trio_payout * unit // 100)
    out = {
        "box_all": box,
        "place3": (3 * unit, sum(place.get(n, 0) for n in numbers[:3]) * unit // 100),
    }
    # 見送らなかったレースだけのボックス。box_all との差が「見送りの損得」、
    # 実際の成績との差が「予算の削りと三連単の損得」になる。分けて測らないと
    # どちらを直せばいいのか分からない。
    if race.get("bets"):
        out["box_bet"] = box
    return out


def review_day(store: Store, payload: dict) -> dict:
    """予想 payload に実結果を埋め、成績を集計して返す。

    payload は predict.predict_day が作ったものをそのまま受け取り、
    各レースの "result" を埋めて返す（破壊的に書き換える）。
    """
    spent = returned = hits = 0
    graded = 0
    worked: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    # 教訓の効き目は率でしか読めない。母数が要る。
    marks_total = marks_placed = 0
    # [投資, 払戻, 単一レースの最高払戻]。3つめは「1本の高配当で数字が
    # 立っているだけ」を見抜くために要る。回収率だけ見て買い方を変えると、
    # 偶然の1本に合わせて設定をいじることになる。
    alternatives: dict[str, list[int]] = {k: [0, 0, 0] for k in ALTERNATIVES}
    by_type: dict[str, list[int]] = {}
    best_race = 0

    for race in payload["races"]:
        card = store.load_card(race["race_id"])
        if card is None or not card.results:
            continue
        graded += 1
        top3 = _actual_top3(card)
        if top3 is None:
            continue

        tickets = [
            Ticket(b["type"], b["combination"], b["amount"], b["why"])
            for b in race["bets"]
        ]
        race_spent, race_returned, hit_types, best = _settle(tickets, card, top3)
        hit = bool(hit_types)

        # 券種ごとの内訳。三連単は当たれば大きいが点数あたりの確率が桁違いに
        # 低いので、全体に混ぜたままだと三連複の成績が読めない。
        trio_key = "-".join(str(n) for n in sorted(top3))
        trifecta_key = "-".join(str(n) for n in top3)
        for t in tickets:
            row = by_type.setdefault(t.bet_type, [0, 0, 0])
            row[0] += t.amount
            key = trifecta_key if t.bet_type == betting.TRIFECTA else trio_key
            if t.combination == key:
                won = _payout_of(card, t.bet_type, key) * t.amount // 100
                row[1] += won
                row[2] = max(row[2], won)

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
        best_race = max(best_race, race_returned)
        hits += int(hit)

        for key, (alt_spent, alt_returned) in _alternatives(race, card, top3).items():
            alternatives[key][0] += alt_spent
            alternatives[key][1] += alt_returned
            alternatives[key][2] = max(alternatives[key][2], alt_returned)

        # 印を打った馬が実際に絡んだか。教訓の効き目はここで判定する。
        # レース単位の集計を別に持つ。以前は日単位の Counter をそのまま
        # reviews へ書いており、各行に「その日のそこまでの累計」が入っていた。
        placed = set(top3)
        race_worked: Counter[str] = Counter()
        race_failed: Counter[str] = Counter()
        for horse in race["horses"]:
            if not horse["mark"]:
                continue
            marks_total += 1
            in_top3 = horse["umaban"] in placed
            marks_placed += int(in_top3)
            bucket = race_worked if in_top3 else race_failed
            for lesson, markers in LESSON_MARKERS.items():
                if any(m in r for r in horse["reasons"] for m in markers):
                    bucket[lesson] += 1
        worked += race_worked
        failed += race_failed

        store.conn.execute(
            "INSERT OR REPLACE INTO reviews"
            " (race_id, hit, spent, returned, lessons_worked, lessons_failed, note)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                race["race_id"],
                int(hit),
                race_spent,
                race_returned,
                json.dumps(dict(race_worked), ensure_ascii=False),
                json.dumps(dict(race_failed), ensure_ascii=False),
                race["confidence_reason"],
            ),
        )

    store.conn.commit()

    # 何レース照合できたかを必ず残す。これが無いと「結果を取れていない」と
    # 「全部外した」がどちらも 払戻0円・的中0R になって見分けられない。
    # 2026-08-08 に実際そうなり、全敗したように見えていた。
    payload["summary"]["graded"] = graded
    payload["summary"]["returned"] = returned
    payload["summary"]["hits"] = hits
    payload["summary"]["roi"] = round(returned / spent * 100, 1) if spent else None
    payload["summary"]["lessons_worked"] = dict(worked)
    payload["summary"]["lessons_failed"] = dict(failed)
    payload["summary"]["marks"] = {"total": marks_total, "placed": marks_placed}
    payload["summary"]["best_race"] = best_race
    payload["summary"]["by_type"] = {
        k: {"spend": v[0], "returned": v[1], "best": v[2]} for k, v in by_type.items()
    }
    payload["summary"]["alternatives"] = {
        k: {"spend": v[0], "returned": v[1], "best": v[2]}
        for k, v in alternatives.items()
        if v[0]
    }
    return payload


def append_lessons(payload: dict, path: Path) -> None:
    """人が読む回顧ログを追記する。

    SKILL.md 本体は手で育てるものなので上書きしない。機械が観測した事実だけを
    別ファイルに積み、人が読んで SKILL.md に反映できるようにする。
    """
    summary = payload["summary"]
    day = payload["date"]

    hit_races = [r for r in payload["races"] if (r.get("result") or {}).get("hit")]
    graded = summary.get("graded", 0)
    lines = [
        f"\n## {day}",
        "",
        f"- 対象 {summary['races']}R / 買い {summary['bet']}R / 見送り {summary['skipped']}R",
    ]

    # 結果が1件も取れていないのに「回収率0%・的中0R」と書くと、全部外したのと
    # 区別がつかない。db.netkeiba は結果の反映に時間がかかるので、開催当日の
    # 夜に走らせるとこの状態になる。
    if graded == 0:
        lines += [
            "- **結果未照合**。着順を1件も取得できていない（成績ではない）",
            "  db.netkeiba への反映待ちの可能性が高い。翌日に再実行すること",
        ]
        return _write(path, lines, day, graded)

    if graded < summary["races"]:
        lines.append(f"- 照合できたのは {graded}/{summary['races']}R のみ（残りは結果待ち）")

    lines.append(
        f"- 投資 {summary['spend']:,}円 → 払戻 {summary.get('returned', 0):,}円 "
        f"（回収率 {summary.get('roi') or 0:.1f}%・的中 {summary.get('hits', 0)}R）"
    )

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

    lines += _alternative_lines(summary)
    lines += _lesson_lines(summary)

    _write(path, lines, day, graded)


def _rate(returned: int, spend: int) -> str:
    return f"{returned / spend * 100:5.1f}%" if spend else "    -"


def _alternative_lines(summary: dict) -> list[str]:
    """別の買い方だといくらだったかを並べる。

    実際の成績だけを見ていると「見送りが正しかったのか」「三連複でよいのか」に
    永久に答えが出ない。同じレースを別の買い方で毎回測って積んでおく。
    """
    alternatives = summary.get("alternatives") or {}
    if not alternatives:
        return []
    lines = ["- 対案（同じ印で買い方だけ変えた場合）:"]
    for key, label in ALTERNATIVES.items():
        alt = alternatives.get(key)
        if not alt:
            continue
        lines.append(
            f"  - {label} {alt['spend']:,}円 → {alt['returned']:,}円"
            f" （{_rate(alt['returned'], alt['spend'])}）"
        )
    return lines


def _lesson_lines(summary: dict) -> list[str]:
    """教訓ごとの3着内率を、印全体の率と並べて出す。

    以前は「効いた回数／外した回数」を出していたが、これは効き目ではなく
    発火回数を測っていた。よく発火する教訓が両方の一覧で1位になるだけで、
    どれを伸ばすべきかが何も分からない。率にして初めて比較できる。
    """
    worked = summary.get("lessons_worked") or {}
    failed = summary.get("lessons_failed") or {}
    marks = summary.get("marks") or {}
    total, placed = marks.get("total", 0), marks.get("placed", 0)
    if not (worked or failed) or not total:
        return []

    base = placed / total
    lines = [f"- 教訓の効き（印全体の3着内率 {base:.1%}・n={total}）:"]
    rows = []
    for lesson in sorted(set(worked) | set(failed)):
        hit, miss = worked.get(lesson, 0), failed.get(lesson, 0)
        n = hit + miss
        if not n:
            continue
        rows.append((hit / n - base, lesson, hit, n))
    for lift, lesson, hit, n in sorted(rows, reverse=True):
        lines.append(
            f"  - {lesson} {hit / n:.1%} ({hit}/{n}・{lift * 100:+.1f}pt)"
        )
    return lines


HEADER = (
    "# 実戦ログ\n\n"
    "> keiba-weekend の review が開催日ごとに追記する。機械が観測した事実の\n"
    "> 記録であり、判断は含まない。ここを読んで SKILL.md の「学習済み教訓」を\n"
    "> 育てるのは人の仕事。\n"
    ">\n"
    "> 先頭の「通算」は毎回積み直す。買い方を変えるかどうかは、1日の成績では\n"
    "> なくここが十分な日数で動いてから決める。\n"
)

LEDGER_START = "<!-- ledger:start -->"
LEDGER_END = "<!-- ledger:end -->"


def build_ledger(data_dir: Path) -> list[str]:
    """回顧の済んだ日を全部足して、買い方ごとの通算を出す。

    1日ぶんの数字は分散が大きすぎて何も言えない。回を重ねるほど意味が出る形で
    積んでおき、対案が実際の買い方を上回り続けるなら乗り換えを検討する材料に
    する。1日や2日の上下で動かさない。
    """
    days: list[tuple[str, dict]] = []
    for path in sorted(data_dir.glob("[0-9]*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data.get("summary") or {}
        if not summary.get("graded") or not summary.get("spend"):
            continue
        days.append((data["date"], summary))
    if not days:
        return []

    spend = sum(s["spend"] for _, s in days)
    returned = sum(s.get("returned", 0) for _, s in days)
    over = [d for d, s in days if s.get("returned", 0) >= s["spend"]]

    def totals_of(get) -> tuple[int, int, int]:
        """(投資, 払戻, 単一レースの最高払戻) を全日ぶん足す。"""
        parts = [get(s) for _, s in days]
        return (
            sum(p.get("spend", 0) for p in parts),
            sum(p.get("returned", 0) for p in parts),
            max((p.get("best", 0) for p in parts), default=0),
        )

    rows = [("実際の買い方", spend, returned, max(s.get("best_race", 0) for _, s in days))]
    for bet_type in (betting.TRIO, betting.TRIFECTA):
        s, r, b = totals_of(lambda x, t=bet_type: (x.get("by_type") or {}).get(t, {}))
        if s:
            rows.append((f"  うち{bet_type}", s, r, b))
    for key, label in ALTERNATIVES.items():
        s, r, b = totals_of(lambda x, k=key: (x.get("alternatives") or {}).get(k, {}))
        if s:
            rows.append((label, s, r, b))

    width = max(len(label) for label, *_ in rows)
    lines = [
        LEDGER_START,
        "",
        "## 通算",
        "",
        f"回顧済み {len(days)}日（{days[0][0]} 〜 {days[-1][0]}）。"
        f"100%超えの日 {len(over)}日" + (f"（{', '.join(over)}）" if over else ""),
        "",
        "```",
        f"{'買い方':<{width}}      投資          払戻     回収率  最高配当を除くと",
    ]
    for label, s, r, b in rows:
        lines.append(
            f"{label:<{width}}  {s:>9,}円  {r:>9,}円  {_rate(r, s)}"
            f"    {_rate(r - b, s)}"
        )
    lines += [
        "```",
        "",
        "> 対案は同じ印のまま買い方だけ変えた場合。見送ったレースも含めて測って",
        "> いるので、「見送りが正しかったか」もここに出る。",
        ">",
        "> 右端は、通算で一番大きかった1レースの払戻を引いた回収率。三連複は",
        "> 当たりが少なく配当の幅が大きいので、数十日ぶんでも1本で全体が立つ。",
        "> **左右で結論が変わる行は、まだ何も言えていない**とみなす。1日や2日の",
        "> 上下で設定を動かさない。",
        "",
        LEDGER_END,
    ]
    return lines


def write_ledger(data_dir: Path, path: Path) -> None:
    """実戦ログの先頭にある通算ブロックを差し替える。"""
    lines = build_ledger(data_dir)
    if not lines:
        return
    block = "\n".join(lines) + "\n"
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    text = path.read_text(encoding="utf-8")

    if LEDGER_START in text and LEDGER_END in text:
        head, _, rest = text.partition(LEDGER_START)
        _, _, tail = rest.partition(LEDGER_END)
        text = head + block + tail.lstrip("\n")
    else:
        head, sep, tail = text.partition(HEADER)
        text = (head + sep if sep else text + "\n") + "\n" + block
        if sep:
            text += tail.lstrip("\n")
    path.write_text(text, encoding="utf-8")

_UNGRADED = "結果未照合"
# 何レース照合できた記録なのかを機械可読で残す。文面から判断しようとすると、
# 書式を変えた瞬間に判定が壊れる（実際そうなった）。
_GRADED_RE = re.compile(r"<!-- graded:(\d+) -->")


def _write(path: Path, lines: list[str], day: str | None = None, graded: int = 0) -> None:
    """実戦ログに1日ぶんを書く。

    同じ日を二重に書かない。ただし**前回より多く照合できていれば置き換える**。

    当日夜は結果が出ておらず翌日に取り直す運用なので、未照合の記録が残り
    続けると成績が二度と記録されない。逆に、一度ちゃんと記録できた日を
    後から薄い内容で上書きするのも困る。照合数の大小で決める。
    """
    day = day or next((line[4:].strip() for line in lines if line.startswith("\n## ")), "")
    lines = [*lines, f"<!-- graded:{graded} -->"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")

    existing = path.read_text(encoding="utf-8")
    marker = f"\n## {day}\n"
    if marker in existing:
        head, _, rest = existing.partition(marker)
        block, sep, tail = rest.partition("\n## ")
        m = _GRADED_RE.search(block)
        # 印の無い記録は修正前のコードが書いたもの。0件扱いにして置き換える
        previous = int(m.group(1)) if m else 0
        if previous >= graded:
            log.info("%s は記録済み（照合 %d件）", day, previous)
            return
        log.info("%s を置き換える（照合 %d件 → %d件）", day, previous, graded)
        existing = head + (sep + tail if sep else "")

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
