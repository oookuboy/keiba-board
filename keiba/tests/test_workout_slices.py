"""切り口ごとの比較の検証。

一番大事なのは「小さい標本で差が出たように見えるものを、差だと言わない」
こと。20個も切れば、効果がゼロでも1つは偶然良く見える。そこを拾って本番に
入れると次の週に消える。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba import workout_slices


def _frame(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "placed": rng.integers(0, 2, rows),
            "h_days_since": rng.integers(1, 200, rows),
            "h_runs": rng.integers(0, 20, rows),
            "c_surface_switch": rng.integers(0, 2, rows),
            "c_class_delta": rng.integers(-2, 3, rows),
            "market_popularity": rng.integers(1, 18, rows),
        }
    )


def test_slices_all_have_a_stated_reason() -> None:
    """理由の書けない切り口を置かないこと。

    「なんとなく効きそう」で足していくと、偶然の当たりを拾う確率が上がる。
    """
    assert workout_slices.SLICES
    for name, reason, _ in workout_slices.SLICES:
        assert name and reason, f"{name} に理由が無い"
        assert len(reason) > 10, f"{name} の理由が短すぎる"


def test_small_slices_are_flagged_as_untrustworthy() -> None:
    """行数の少ない切り口を信用しないこと。"""
    df = _frame(50)
    rows = workout_slices.compare(
        df, "placed", np.zeros(50), np.arange(50, dtype=float)
    )
    assert rows
    assert all(not r.get("trustworthy", False) for r in rows if r.get("rows")), (
        "50行しかないのに信用できる扱いになっている"
    )


def test_identical_scores_show_no_difference() -> None:
    """同じスコアを渡したら差が0になること。計算そのものの確認。"""
    df = _frame(5000)
    scores = np.random.default_rng(1).random(5000)
    for row in workout_slices.compare(df, "placed", scores, scores):
        if row.get("delta") is not None:
            assert abs(row["delta"]) < 1e-9


def test_auc_is_one_for_a_perfect_ranking() -> None:
    labels = np.array([0, 0, 1, 1])
    assert workout_slices.auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert workout_slices.auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auc_is_none_when_one_class_is_missing() -> None:
    """片方のクラスしか無い切り口で、でたらめな数字を出さないこと。"""
    assert workout_slices.auc(np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0])) is None


def test_report_always_shows_row_counts() -> None:
    """行数を必ず出すこと。差だけ見せると小さい標本を信じてしまう。"""
    df = _frame(3000)
    rng = np.random.default_rng(2)
    text = workout_slices.format_slices(
        workout_slices.compare(df, "placed", rng.random(3000), rng.random(3000))
    )
    assert "行数" in text
    assert "別の期間で再現するまで採用しない" in text
