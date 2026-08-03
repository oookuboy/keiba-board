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
