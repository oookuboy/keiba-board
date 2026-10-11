"""本番のモデルが直近の検証期間も学んでいること。

以前は検証用に取っておいた直近3か月を学習に戻さず、そのまま本番に置いて
いた。2026-10 の予想は 6/28 までのレースしか学んでいないモデルで出ていた。
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from keiba import ml, predict


def _toy(n_days: int = 120, per_day: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("2026-01-01")
    for d in range(n_days):
        for i in range(per_day):
            row = {c: rng.normal() for c in ml.FEATURE_COLUMNS}
            for c in ml.CATEGORICAL:
                row[c] = int(rng.integers(0, 3))
            row["race_date"] = start + pd.Timedelta(days=d)
            row["market_popularity"] = i % 10 + 1
            row["market_odds"] = float(i % 10 + 2)
            row["finish_pos"] = i % 10 + 1
            row["race_id"] = f"r{d:03d}{i // 10}"
            row[ml.TARGET] = int(row[ml.FEATURE_COLUMNS[0]] + rng.normal() > 0.8)
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in ml.CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def test_検証期間も学び直して最後の日まで学ぶ(tmp_path) -> None:
    df = _toy()
    result = ml.train(df, valid_from=date(2026, 3, 1), rounds=50)
    assert result.train_rows < len(df)

    full = ml.refit(df, result)
    assert full.train_rows == len(df)
    assert full.trained_through == date(2026, 4, 30)
    # AUC は検証で測った数字のまま（学んだ行で測り直さない）
    assert full.auc == result.auc

    ml.save(full, tmp_path / "model.txt", tmp_path / "model.json")
    meta = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert meta["trained_through"] == "2026-04-30"
    assert predict._model_trained_through(tmp_path) == "2026-04-30"


def test_古い形式のモデルは検証開始の前日までと読む(tmp_path) -> None:
    (tmp_path / "model.json").write_text(json.dumps({"valid_from": "2026-06-29"}))
    assert predict._model_trained_through(tmp_path) == "2026-06-28"


def test_再学習は本番に置く前に学び直す() -> None:
    import inspect

    from keiba import cli

    source = inspect.getsource(cli.cmd_train)
    assert source.index("ml.refit(") < source.index("ml.save("), (
        "検証期間を学ばないモデルを本番に置いている"
    )


def test_予想の文章の最後に調教評価を載せる() -> None:
    from keiba.models import Entry, Race, RaceCard

    race = Race(race_id="202605040407", race_date=date(2026, 10, 11), venue="東京",
                venue_code="05", kai=4, nichi=4, race_no=7, name="1勝クラス",
                surface="芝", distance=1600)
    card = RaceCard(race=race, entries=[
        Entry(race_id=race.race_id, umaban=1, horse_name="ヒラバイチ", horse_id="2022104123"),
        Entry(race_id=race.race_id, umaban=2, horse_name="ヒラバニ", horse_id="2022100002"),
    ], results=[], payouts=[])
    payload = {"horses": [
        {"name": "ヒラバイチ", "reasons": ["芝マイルは1-0-0-0（適性＋）"]},
        {"name": "ヒラバニ", "reasons": []},
    ]}
    predict._append_workout_rank(payload, card, {"2022104123": "A"})
    assert payload["horses"][0]["reasons"][-1] == "調教評価 A"
    assert payload["horses"][0]["workout_rank"] == "A"
    assert "workout_rank" not in payload["horses"][1]
