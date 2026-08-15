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
from dataclasses import replace

from keiba.models import RaceCard
from keiba.store import Store, read_jsonl

log = logging.getLogger(__name__)

JST = timezone.utc  # 表示は日付単位なのでUTCのまま持ち、描画側で扱う


def predict_race(race_id: str, store: Store, weights: dict, sire_table: dict) -> dict | None:
    """1レースを予想して、そのまま JSON に落とせる dict を返す。"""
    card = store.load_card(race_id)
    if card is None:
        return None
    return predict_card(card, store, weights, sire_table)


def model_scores(store: Store, day: date, config_dir: Path) -> dict[str, dict[int, float]]:
    """その日の全出走馬に、学習モデルの3着以内確率を付ける。

    ## なぜ要るのか

    バックテストは `--no-model` を付けない限り学習モデルで採点している。
    一方、実際にボードへ出していた予想は `weights.yml` の手置きスコアだけで
    動いていた。**測っているものと出しているものが別だった**。

    タイム指数を足して AUC が上がったのも、調教を測っていたのも、すべて
    モデル側の話で、ボードには一度も届いていない。ここを繋いで初めて、
    集めたデータが予想に効く。

    モデルが無い・今日のレースが表に出ない場合は空を返す。呼び出し側は
    手置きスコアにそのまま落ちる。
    """
    from keiba import dataset, ml

    booster = ml.load(config_dir / "model.txt")
    if booster is None:
        log.warning(
            "%s が無い。手置きの重みで採点する（順位付けの精度は落ちる）",
            config_dir / "model.txt",
        )
        return {}

    # まだ結果の無いレースを含めて読む。内側 JOIN のままだと、これから走る
    # レースは results が無いという理由だけで1行も出てこない。
    df = dataset.prepare(store, include_unfinished=True)
    today = df[df["race_date"] == str(day)]
    if today.empty:
        log.warning("%s の行が特徴量の表に無い。手置きの重みで採点する", day)
        return {}

    today = today.copy()
    today["p"] = ml.predict(booster, today)
    scores = {
        race_id: dict(zip(group["umaban"], group["p"]))
        for race_id, group in today.groupby("race_id", observed=True)
    }
    log.info("学習モデルで採点: %d レース / %d頭", len(scores), len(today))
    return scores


def predict_card(
    card: RaceCard,
    store: Store,
    weights: dict,
    sire_table: dict,
    ml_scores: dict[int, float] | None = None,
) -> dict | None:
    """出馬表そのものから予想する。

    木曜の JRA 出馬表は枠順確定前で、馬番が空のまま返る。以前はそこで予想ごと
    捨てていたが、馬名も血統も騎手も揃っているので能力評価はできる。
    捨てていたのは「買い目を偽の馬番で組まないため」であって、印まで出せない
    理由は無い。

    枠順が未確定のときは、内部の突き合わせにだけ仮の通し番号を使い、
    **出力には馬番を載せず、買い目も組まない**。仮番号が券に化けないよう、
    ここで断ち切る。
    """
    race = card.race
    entries = card.live_entries()
    if len(entries) < 6:
        return None

    unconfirmed = any(e.umaban is None for e in entries)
    if unconfirmed:
        # 仮番号は engine と features の突き合わせにしか使わない。
        # 出馬表に並んでいる順（＝JRA の表示順）をそのまま振る。
        entries = [
            replace(e, umaban=i) if e.umaban is None else e
            for i, e in enumerate(entries, 1)
        ]
        log.info("%s: 枠順未確定。印だけ出して買い目は組まない", race.race_id)
        # 仮番号はこの関数の中だけの通し番号で、モデル側の馬番とは無関係。
        # 突き合わせると別の馬のスコアを付けることになるので捨てる。
        ml_scores = None

    features = build_features(
        race, entries, store, sire_table, weights, as_of=race.race_date
    )
    horses = engine.run(
        features, {e.umaban: e for e in entries}, weights, ml_scores
    )
    grade = confidence.grade(horses, weights)
    tickets = [] if unconfirmed else betting.build(horses, grade, weights)

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
        "post_positions_confirmed": not unconfirmed,
        "confidence": grade.grade,
        "confidence_reason": grade.reason,
        "popularity_sum": grade.popularity_sum,
        "separation": grade.separation,
        "expected_odds": grade.expected_odds,
        "pace": pace_note.split("・")[0].replace("想定", "") if pace_note else None,
        "horses": [
            {
                # 仮番号を本物として出さない。枠順が決まるまで馬番は無い。
                "umaban": None if unconfirmed else h.umaban,
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


# 予算を削る順序。◎ は最後まで守る。実測で最も回収率の高い帯なので、
# ここを削ったら期待値の一番良いところから捨てることになる。
TRIM_ORDER = ("△", "○", "◎")
LONGSHOT_GRADE = "△"


def apply_longshot_cap(races: list[dict], cfg: dict) -> list[str]:
    """穴枠（△）の1日上限を、本線とは別の財布として適用する。

    穴の帯は実測で回収率が本線より明確に低い（9-13が89.9%に対し14-19が63.3%、
    20-27が18.2%）。当たれば大きいので残すが、本線の予算を食わないよう
    財布を分ける。ここが混ざると、期待値の良い本線が穴に押し出される。
    """
    cap = cfg.get("longshot_daily_cap")
    if not cap:
        return []

    longshots = [r for r in races if r["confidence"] == LONGSHOT_GRADE and r["bets"]]
    spent = sum(r["spend"] for r in longshots)
    if spent <= cap:
        return []

    # 妙味の薄い（人気和の小さい）穴レースから落とす
    dropped = 0
    for race in sorted(longshots, key=lambda r: r["popularity_sum"]):
        if spent <= cap:
            break
        spent -= race["spend"]
        race["bets"] = []
        race["spend"] = 0
        race["budget_skipped"] = True
        dropped += 1
    return [f"穴枠を {dropped}R 見送り（穴枠上限 {cap:,}円）"] if dropped else []


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


def predict_day(
    store: Store,
    weights: dict,
    sire_table: dict,
    day: date,
    *,
    provisional: bool = False,
    raw_dir: Path | None = None,
    config_dir: Path | None = None,
) -> dict:
    """その日の中央全レースを予想する。

    provisional=True は木曜など発走前の暫定運用。能力評価と印はそのまま出すが、
    買い目は組まない。オッズが確定していない段階の妙味判定はあてにならず、
    たまたま一部レースだけオッズが出ていると「一部だけ買い目が付く」という
    中途半端な出方をするため、金額に関わる部分は当日の本予想に一本化する。
    """
    # 枠順確定前の馬は DB に入れられない（馬番が entries の主キーなので）。
    # 暫定運用ではその馬こそ見たいので、生データから直に読む。ここを DB 経由に
    # したままだと、木曜は毎回「0レース」で終わる（実際そうなっていた）。
    cards: list[RaceCard] = []
    if provisional and raw_dir is not None:
        path = Path(raw_dir) / str(day.year) / f"{day.isoformat()}.jsonl.gz"
        if path.exists():
            cards = [c for c in read_jsonl(path) if c.race.race_date == day]
            log.info("%s: 生データから %d レース", day, len(cards))

    if not cards:
        cards = [
            card
            for card in (
                store.load_card(r[0])
                for r in store.conn.execute(
                    "SELECT race_id FROM races WHERE race_date = ? ORDER BY race_id",
                    (day.isoformat(),),
                )
            )
            if card is not None
        ]
        log.info("%s: %d レース", day, len(cards))

    # 学習モデルで能力を採点する。ここが無いと、バックテストで測っている
    # ものとボードに出るものが別になる。
    scores: dict[str, dict[int, float]] = {}
    if config_dir is not None and cards:
        try:
            scores = model_scores(store, day, config_dir)
        except (KeyError, ValueError) as exc:
            # 採点できなくても手置きの重みで予想自体は出せる。ただし黙って
            # 落ちると「モデルを繋いだつもりで繋がっていない」に戻るので残す。
            log.warning("モデルでの採点に失敗: %s（手置きの重みで続ける）", exc)

    races: list[dict] = []
    for card in cards:
        race_id = card.race.race_id
        try:
            payload = predict_card(
                card, store, weights, sire_table, scores.get(race_id)
            )
        except (AssertionError, KeyError, ValueError) as exc:
            log.warning("%s の予想に失敗: %s", race_id, exc)
            continue
        if payload:
            races.append(payload)

    if provisional:
        # 暫定段階では金額に関わる判断をしない。印と能力評価だけを残す。
        for race in races:
            race["bets"] = []
            race["spend"] = 0
        budget_actions = []
    else:
        # 穴枠を先に財布ごと締めてから、全体の上限を当てる
        budget_actions = apply_longshot_cap(races, weights["betting"])
        budget_actions += apply_daily_cap(races, weights["betting"])
        for action in budget_actions:
            log.info("予算調整: %s", action)

    bet_races = [r for r in races if r["bets"]]
    return {
        "date": day.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_version": weights["engine_version"],
        "provisional": provisional,
        # 何で採点したか。手置きと学習モデルでは順位付けの精度が違うので、
        # 後から回顧するときにこれが分からないと成績を比べられない。
        "scored_by": "model" if scores else "weights",
        "summary": {
            "races": len(races),
            "bet": len(bet_races),
            "skipped": len(races) - len(bet_races),
            "spend": sum(r["spend"] for r in races),
            "spend_main": sum(
                r["spend"] for r in races if r["confidence"] != LONGSHOT_GRADE
            ),
            "spend_longshot": sum(
                r["spend"] for r in races if r["confidence"] == LONGSHOT_GRADE
            ),
            "budget_actions": budget_actions,
            "by_confidence": {
                g: sum(1 for r in races if r["confidence"] == g)
                for g in ("◎", "○", "△", "×", "?")
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
                # 暫定（木曜の下見）か本予想かをボードで区別できるようにする
                "provisional": data.get("provisional", False),
            }
        )

    index = data_dir / "index.json"
    index.write_text(
        json.dumps({"days": days}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return index
