"""学習データの先読み検証。

過去成績から作る特徴量に「その出走の結果」が1つでも混ざると、バックテストの
回収率が嘘をつく。しかも静かに起きて、見た目は良い数字になる。ここで止める。
"""

from __future__ import annotations

import pandas as pd
import pytest

from keiba import dataset


def frame() -> pd.DataFrame:
    """同じ馬が3走する最小の表。1走目→3走目で成績が変わる。"""
    rows = []
    for i, (finish, size) in enumerate([(1, 10), (10, 10), (1, 10)]):
        rows.append(
            {
                "race_id": f"R{i}", "race_date": pd.Timestamp(2026, 1, 1 + i * 7),
                "venue": "東京", "surface": "芝", "distance": 1600, "direction": "左",
                "going": "良", "grade": None, "race_class": "1勝クラス",
                "field_size": size, "race_no": 11, "umaban": 1, "waku": 1,
                "horse_id": "H1", "horse_name": "テスト", "sex": "牡", "age": 4,
                "weight_carried": 57.0, "jockey_id": "J1", "trainer_id": "T1",
                "body_weight": 480, "body_weight_diff": 0,
                "sire": "S1", "damsire": "D1",
                "market_popularity": 1, "market_odds": 2.0,
                "finish_pos": finish, "time_sec": 95.0, "last3f": 34.0,
                "corner_ratio": 0.3, "finish_ratio": finish / size,
                "top3": int(finish <= 3), "won": int(finish == 1),
                "placed": int(finish <= 3), "band": "mile", "class_rank": 1,
            }
        )
    return pd.DataFrame(rows)


def test_first_run_has_no_history() -> None:
    """初出走の行に過去成績が入っていないこと。"""
    out = dataset.build_features(frame())
    first = out.iloc[0]
    assert first["h_runs"] == 0
    assert pd.isna(first["h_place_rate"])
    assert pd.isna(first["h_prev_finish_ratio"])
    assert pd.isna(first["h_days_since"])


def test_prior_stats_exclude_own_result() -> None:
    """自分自身の結果が特徴量に入らないこと。

    1走目=1着(top3), 2走目=10着, 3走目=1着。3走目の h_place_rate は
    「1走目と2走目」だけの平均 0.5 でなければならない。自分（3走目=好走）が
    混ざれば 2/3 になる。
    """
    out = dataset.build_features(frame())
    assert out.iloc[1]["h_place_rate"] == 1.0     # 1走目のみ（好走）
    assert out.iloc[2]["h_place_rate"] == 0.5     # 1走目と2走目のみ
    assert out.iloc[2]["h_prev_finish_ratio"] == 1.0  # 2走目の10/10


def test_days_since_uses_previous_race() -> None:
    out = dataset.build_features(frame())
    assert out.iloc[1]["h_days_since"] == 7
    assert out.iloc[2]["h_days_since"] == 7


def test_feature_columns_contain_no_market_data() -> None:
    """オッズ・人気が特徴量に紛れていないこと。

    ここが漏れると「人気馬は勝つ」を学習してしまい、市場を上回る余地が消える。
    """
    dataset.assert_no_market_leakage(dataset.FEATURE_COLUMNS)
    for column in dataset.FEATURE_COLUMNS:
        assert "odds" not in column
        assert "popularity" not in column


def test_leakage_guard_actually_fires() -> None:
    """検証そのものが機能していること。"""
    with pytest.raises(AssertionError, match="オッズ由来"):
        dataset.assert_no_market_leakage(["h_place_rate", "market_odds"])
    with pytest.raises(AssertionError):
        dataset.assert_no_market_leakage(["market_popularity"])


def test_all_feature_columns_exist_after_build() -> None:
    """宣言した特徴量が実際に作られていること。"""
    out = dataset.build_features(frame())
    missing = [c for c in dataset.FEATURE_COLUMNS if c not in out.columns]
    assert not missing, f"作られていない特徴量: {missing}"
