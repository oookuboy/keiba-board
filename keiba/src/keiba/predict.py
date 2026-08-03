"""1開催日ぶんの予想を作り、ボードが読む JSON に落とす。

「全てのレースを記録する」という要求どおり、× で見送ったレースも payload に
残す。買わない判断も記録の一部であり、あとから「なぜ買わなかったか」を
辿れないと回顧ができない。
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from keiba import betting, confidence, engine
from keiba.features import build_features
from keiba.store import Store

log = logging.getLogger(__name__)

JST = timezone.utc  # 表示は日付単位なのでUTCのまま持ち、描画側で扱う


def predict_race(race_id: str, store: Store, weights: dict, sire_table: dict) -> dict | None:
    """1レースを予想して、そのまま JSON に落とせる dict を返す。"""
    card = store.load_card(race_id)
    if card is None:
        return None

    race = card.race
    entries = card.live_entries()
    if len(entries) < 6:
        return None

    features = build_features(
        race, entries, store, sire_table, weights, as_of=race.race_date
    )
    horses = engine.run(features, {e.umaban: e for e in entries}, weights)
    grade = confidence.grade(horses, weights)
    tickets = betting.build(horses, grade, weights)

    # 展開は全頭共通の判断なので、1頭ぶんの理由から取り出して見出しにする
    pace_note = next(
        (r for f in features for r in f.reasons if "想定" in r and "ペース" in r), ""
    )

    return {
        "race_id": race.race_id,
        "venue": race.venue,
        "race_no": race.race_no,
        "name": race.name,
        "grade": race.grade,
        "race_class": race.race_class,
        "surface": race.surface,
        "distance": race.distance,
        "direction": race.direction,
        "going": race.going,
        "weather": race.weather,
        "post_time": race.post_time,
        "field_size": race.field_size,
        "confidence": grade.grade,
        "confidence_reason": grade.reason,
        "popularity_sum": grade.popularity_sum,
        "separation": grade.separation,
        "expected_odds": grade.expected_odds,
        "pace": pace_note.split("・")[0].replace("想定", "") if pace_note else None,
        "horses": [
            {
                "umaban": h.umaban,
                "name": h.horse_name,
                "mark": h.mark,
                "score": h.score,
                "style": h.style,
                "popularity": h.market_popularity,
                "odds": h.market_odds,
                "lone_front_runner": h.is_lone_front_runner,
                "reasons": h.reasons,
            }
            for h in horses
        ],
        "bets": [
            {
                "type": t.bet_type,
                "combination": t.combination,
                "amount": t.amount,
                "why": t.rationale,
            }
            for t in tickets
        ],
        "spend": sum(t.amount for t in tickets),
        # 結果は review が後から埋める
        "result": None,
    }


# 予算を削る順序。◎ は最後まで守る。穴で勝負するレースを削ったら
# 「固い予想はいらない」という方針そのものが崩れるため。
TRIM_ORDER = ("△", "○", "◎")


def apply_daily_cap(races: list[dict], cfg: dict) -> list[str]:
    """1日の総額を上限に収める。

    開催日は36レースあり、1レース10点前後を素直に積むと簡単に数万円になる。
    削り方には順序があり、まず自信度の低いレースを最小単位へ落とし、それでも
    収まらなければ低いほうから買い目ごと落とす。◎ は最後まで守る。

    戻り値は何をしたかの記録（ボードとログに出す）。
    """
    cap = cfg.get("daily_cap")
    unit = cfg["unit"]
    if not cap:
        return []

    def total() -> int:
        return sum(r["spend"] for r in races)

    actions: list[str] = []
    if total() <= cap:
        return actions

    # 第1段: 自信度の低い順に、全点を最小単位へ落とす
    for grade in TRIM_ORDER:
        for race in races:
            if race["confidence"] != grade or not race["bets"]:
                continue
            if all(b["amount"] <= unit for b in race["bets"]):
                continue
            for bet in race["bets"]:
                bet["amount"] = unit
            race["spend"] = sum(b["amount"] for b in race["bets"])
        if total() <= cap:
            actions.append(f"{grade} までを最小単位に減額して上限に収めた")
            return actions
    actions.append("全レースを最小単位まで減額した")

    # 第2段: それでも超えるなら、低い自信度のレースから買い目ごと落とす
    for grade in TRIM_ORDER[:-1]:  # ◎ は落とさない
        dropped = 0
        for race in sorted(
            (r for r in races if r["confidence"] == grade and r["bets"]),
            key=lambda r: r["popularity_sum"],  # 妙味の薄いレースから捨てる
        ):
            if total() <= cap:
                break
            race["bets"] = []
            race["spend"] = 0
            race["budget_skipped"] = True
            dropped += 1
        if dropped:
            actions.append(f"{grade} を {dropped}R 見送り（予算上限）")
        if total() <= cap:
            return actions

    actions.append(f"◎ だけで上限を超えている（{total():,}円 > {cap:,}円）")
    return actions


def predict_day(store: Store, weights: dict, sire_table: dict, day: date) -> dict:
    """その日の中央全レースを予想する。"""
    race_ids = [
        r[0]
        for r in store.conn.execute(
            "SELECT race_id FROM races WHERE race_date = ? ORDER BY race_id",
            (day.isoformat(),),
        )
    ]
    log.info("%s: %d レース", day, len(race_ids))

    races: list[dict] = []
    for race_id in race_ids:
        try:
            payload = predict_race(race_id, store, weights, sire_table)
        except (AssertionError, KeyError, ValueError) as exc:
            log.warning("%s の予想に失敗: %s", race_id, exc)
            continue
        if payload:
            races.append(payload)

    budget_actions = apply_daily_cap(races, weights["betting"])
    for action in budget_actions:
        log.info("予算調整: %s", action)

    bet_races = [r for r in races if r["bets"]]
    return {
        "date": day.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_version": weights["engine_version"],
        "summary": {
            "races": len(races),
            "bet": len(bet_races),
            "skipped": len(races) - len(bet_races),
            "spend": sum(r["spend"] for r in races),
            "budget_actions": budget_actions,
            "by_confidence": {
                g: sum(1 for r in races if r["confidence"] == g)
                for g in ("◎", "○", "△", "×")
            },
        },
        "races": races,
    }


def save_predictions(store: Store, payload: dict) -> None:
    """予想と買い目を DB にも残す。回顧のときに突き合わせる。"""
    for race in payload["races"]:
        store.conn.execute(
            "INSERT OR REPLACE INTO predictions"
            " (race_id, generated_at, engine_version, confidence, payload)"
            " VALUES (?,?,?,?,?)",
            (
                race["race_id"],
                payload["generated_at"],
                payload["engine_version"],
                race["confidence"],
                json.dumps(race, ensure_ascii=False),
            ),
        )
        for bet in race["bets"]:
            store.conn.execute(
                "INSERT OR REPLACE INTO bets"
                " (race_id, bet_type, combination, amount, rationale) VALUES (?,?,?,?,?)",
                (race["race_id"], bet["type"], bet["combination"], bet["amount"], bet["why"]),
            )
    store.conn.commit()


def write_day(payload: dict, data_dir: Path) -> Path:
    """1日ぶんの JSON を書き、日付インデックスを更新する。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"{payload['date']}.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    rebuild_index(data_dir)
    return out


def rebuild_index(data_dir: Path) -> Path:
    """ボードが最初に読む日付一覧。新しい順。"""
    days = []
    for path in sorted(data_dir.glob("[0-9]*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        days.append(
            {
                "date": data["date"],
                "races": data["summary"]["races"],
                "bet": data["summary"]["bet"],
                "spend": data["summary"]["spend"],
                "returned": data["summary"].get("returned"),
                "hits": data["summary"].get("hits"),
            }
        )

    index = data_dir / "index.json"
    index.write_text(
        json.dumps({"days": days}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return index
