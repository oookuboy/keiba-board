"""タイム指数の検証。

一番危ないのは先読み。指数は結果（走破タイム）から作るので、うっかり
そのレース自身の指数を特徴量に入れると、答えを見て予想することになる。
バックテストの数字だけが良くなって本番では効かない、という最悪の壊れ方をする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba import speed


def _frame(times: list[float], dates: list[str], horse: str = "H1") -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {
            "race_id": [f"R{i}" for i in range(n)],
            "race_date": pd.to_datetime(dates),
            "horse_id": [horse] * n,
            "venue": ["東京"] * n,
            "surface": ["芝"] * n,
            "distance": [1600] * n,
            "class_rank": [1] * n,
            "band": ["mile"] * n,
            "time_sec": times,
        }
    )


def test_faster_times_give_higher_figures() -> None:
    """速い時計ほど指数が高いこと。符号の向きの確認。"""
    df = _frame([100.0] * 40 + [95.0], [f"2025-01-{i % 28 + 1:02d}" for i in range(41)])
    # 同じ日に固まると馬場差で全部打ち消されるので、日付をばらしてある
    out = speed.attach_figures(df)
    figures = out["speed_figure"].dropna()
    assert figures.iloc[-1] > figures.iloc[0], "速い時計の指数が高くなっていない"


def test_the_horses_own_race_is_never_used(monkeypatch) -> None:
    """そのレース自身の指数を特徴量にしないこと。

    ここが漏れると、答えを見て予想することになる。
    """
    df = _frame(
        [100.0, 100.0, 60.0],           # 3走目だけ極端に速い
        ["2025-01-05", "2025-02-05", "2025-03-05"],
    )
    df["speed_figure"] = [0.0, 0.0, 99.0]
    out = speed.build_features(df)

    assert out["sp_prev"].iloc[2] == 0.0, "前走の指数に今走が混ざっている"
    assert out["sp_best"].iloc[2] == 0.0, "自己ベストに今走が混ざっている"
    assert pd.isna(out["sp_prev"].iloc[0]), "初出走に前走があってはいけない"


def test_track_variant_cancels_a_fast_day() -> None:
    """その日そのコース全体が速ければ、指数は上がらないこと。

    馬場が速いだけの日に走った馬を「強い」と誤認しないための補正。
    ここが無いと、高速馬場の開催に出ただけで指数が跳ね上がる。
    """
    # 基準を作るための普通の日を並べ、最後に「全馬2秒速い日」を足す
    normal = _frame([100.0] * 40, [f"2025-0{i//28+1}-{i % 28 + 1:02d}" for i in range(40)])
    fast = _frame([98.0] * 6, ["2025-06-01"] * 6)
    fast["race_id"] = [f"F{i}" for i in range(6)]
    out = speed.attach_figures(pd.concat([normal, fast], ignore_index=True))

    fast_day = out[out["race_date"] == pd.Timestamp("2025-06-01")]["speed_figure"]
    assert abs(fast_day.median()) < 0.5, (
        "馬場が速いだけの日の指数が中立になっていない"
        f"（中央値 {fast_day.median():.2f}）"
    )


def test_baselines_can_be_restricted_to_the_past() -> None:
    """検証期間より前のデータだけで基準を作れること。

    バックテストで未来の時計を混ぜると、そのコースの基準を未来から知って
    いることになる。種牡馬適性表と同じ扱いにする。

    確かめ方: 過去に一度も無かった条件（新しい競馬場）を未来に置く。
    基準を過去だけから作るなら、その条件の基準は存在せず指数は付かない。
    未来を混ぜていれば指数が付いてしまう。
    """
    rng = np.random.default_rng(0)
    past = _frame(
        list(100.0 + rng.normal(0, 1.0, 40)),
        [f"2025-01-{i % 28 + 1:02d}" for i in range(40)],
    )
    future = _frame(list(100.0 + rng.normal(0, 1.0, 40)), ["2026-01-05"] * 40)
    future["race_id"] = [f"X{i}" for i in range(40)]
    future["venue"] = "新設競馬場"          # 過去に存在しない条件
    both = pd.concat([past, future], ignore_index=True)

    restricted = speed.attach_figures(both, before=pd.Timestamp("2026-01-01"))
    assert restricted[restricted["venue"] == "新設競馬場"]["speed_figure"].isna().all(), (
        "過去に無い条件に指数が付いた。未来のデータで基準を作っている"
    )

    # 制限しなければ基準が作れるので、指数が付くこと（テスト自体の妥当性確認）
    unrestricted = speed.attach_figures(both)
    assert unrestricted[unrestricted["venue"] == "新設競馬場"]["speed_figure"].notna().any()


def test_jump_races_are_excluded() -> None:
    """障害戦を指数の対象にしないこと。走り方も時計の意味も別物。"""
    df = _frame([100.0] * 40, [f"2025-01-{i % 28 + 1:02d}" for i in range(40)])
    df.loc[:5, "surface"] = "障"
    out = speed.attach_figures(df)
    assert out[out["surface"] == "障"]["speed_figure"].isna().all()


def test_features_do_not_reference_odds() -> None:
    """指数の特徴量にオッズ由来のものを混ぜないこと。"""
    from keiba.dataset import assert_no_market_leakage

    assert_no_market_leakage(speed.SPEED_FEATURES)
