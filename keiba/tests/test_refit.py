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


def test_評価を6項目に分けてレース内の順位にする() -> None:
    from keiba import factors

    assert factors.category("w_rank_score") == "調教"
    assert factors.category("j_place_rate_z") == "騎手"
    assert factors.category("sp_r3") == "タイム"
    assert factors.category("h_course_place_rate") == "適性"
    assert factors.category("h_finish_ratio_r3_z") == "最近の着順"
    assert factors.category("body_weight") == "その他"

    ranked = factors.ranks({
        1: {"騎手": 0.3, "調教": -0.1},
        2: {"騎手": -0.2, "調教": 0.4},
    })
    assert ranked[1] == {"騎手": 1, "調教": 2}
    assert ranked[2] == {"騎手": 2, "調教": 1}


def test_寄与を足すとモデルの出力になる() -> None:
    import numpy as np

    from keiba import factors

    df = _toy(n_days=60)
    result = ml.train(df, valid_from=date(2026, 2, 15), rounds=30)
    parts = factors.breakdown(result.booster, df.head(50))
    assert list(parts.columns) == list(factors.CATEGORIES)
    raw = result.booster.predict(df.head(50)[result.booster.feature_name()], raw_score=True)
    bias = result.booster.predict(
        df.head(50)[result.booster.feature_name()], pred_contrib=True
    )[:, -1]
    assert np.allclose(parts.sum(axis=1).to_numpy() + bias, raw)


def test_重賞の格をオープンと区別する() -> None:
    import pandas as pd

    from keiba import dataset, features

    assert features.class_rank("オープン", "GI") == 7
    assert features.class_rank("オープン", "GII") == 6
    assert features.class_rank("オープン", "GIII") == 5
    assert features.class_rank("オープン", "L") == 4
    assert features.class_rank("オープン", None) == 4
    got = dataset._class_rank(
        pd.Series(["オープン", "オープン", "3勝クラス"]), pd.Series(["GI", None, None])
    )
    assert got.tolist() == [7, 4, 3]
