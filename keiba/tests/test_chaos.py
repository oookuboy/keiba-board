"""荒れやすさの測定の検証。

配当は極端に歪んだ分布なので、平均で語ると1本の大穴に引きずられる。
中央値と超過確率で見ること、区分ごとのレース数を必ず出すことを固定する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba import chaos


def _races(n: int, payouts: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": [f"R{i}" for i in range(n)],
            "trio": payouts,
            "field_size": np.full(n, 16),
            "race_class": ["3歳未勝利"] * n,
            "fav_odds": np.full(n, 3.0),
            "top3_share": np.full(n, 0.5),
        }
    )


def test_median_is_used_instead_of_the_mean() -> None:
    """1本の大穴で区分の評価が変わらないこと。

    平均で見ると、100万円が1本入っただけでその区分が「おいしい」ように
    見えてしまう。狙えるかどうかは「普通いくらか」で判断する。
    """
    payouts = np.concatenate([np.full(99, 1000.0), [1_000_000.0]])
    table = chaos.summarise(_races(100, payouts), pd.Series(["A"] * 100), "区分")
    assert table.loc["A", "配当中央値"] == 1000, "中央値が大穴に引きずられている"


def test_exceedance_probabilities_are_reported() -> None:
    """「大きいのが出る確率」を出すこと。中央値だけでは狙えるか分からない。"""
    payouts = np.array([1000.0] * 50 + [50_000.0] * 50)
    table = chaos.summarise(_races(100, payouts), pd.Series(["A"] * 100), "区分")
    assert table.loc["A", "10千超"] == 0.5
    assert table.loc["A", "30千超"] == 0.5


def test_table_always_shows_race_counts() -> None:
    """レース数を必ず出すこと。少ない区分の数字を信じないため。"""
    table = chaos.summarise(_races(20, np.full(20, 5000.0)),
                            pd.Series(["A"] * 20), "区分")
    assert "R数" in table.columns
    assert "R数" in chaos.format_table(table)


def test_big_and_huge_lines_are_meaningful_for_trifecta() -> None:
    """狙う配当の線が、三連複として意味のある水準にあること。"""
    assert chaos.BIG == 10_000
    assert chaos.HUGE == 30_000
    assert chaos.HUGE > chaos.BIG
