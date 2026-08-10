"""素直な買い方の測定の検証。

ここで確かめたいのは「人気で足切りしていないこと」。それをやったせいで
回収率を22ポイント捨てていた。同じ間違いを二度としないよう、選び方が
モデルの評価順だけで決まることを固定する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba import strategy


def _race(popularities: list[int]) -> pd.DataFrame:
    n = len(popularities)
    return pd.DataFrame(
        {
            "race_id": ["R1"] * n,
            "umaban": list(range(1, n + 1)),
            "market_popularity": popularities,
        }
    )


def test_selection_ignores_popularity() -> None:
    """人気を見ずに、モデルの評価順だけで選ぶこと。

    16番人気でもモデルが1位に置いたなら買う。「穴を狙う」のではなく
    「良いと思った馬を買ったら、たまたま穴だった」を成立させるため。
    """
    valid = _race([1, 2, 16])
    scores = np.array([0.1, 0.2, 0.9])      # 16番人気が最高評価
    picks = strategy.top_n_by_race(valid, scores, 1)

    assert list(picks["umaban"]) == [3], "人気薄が最高評価なのに選ばれていない"
    assert list(picks["market_popularity"]) == [16]


def test_no_odds_filter_removes_cheap_horses() -> None:
    """安い配当というだけで馬を落とさないこと。

    以前は「想定配当50倍以上の組み合わせしか買わない」というフィルタが
    あり、勝ちそうな馬を切り捨てていた。
    """
    valid = _race([1, 2, 3])
    picks = strategy.top_n_by_race(valid, np.array([0.9, 0.5, 0.1]), 3)
    assert len(picks) == 3, "全頭が候補に残るはず"
    assert 1 in list(picks["market_popularity"]), "1番人気を除外している"


def test_returns_use_actual_payouts() -> None:
    """払戻は実データから引くこと。当たっていても払戻が無ければ0円。

    複勝は7頭立て以下だと3着に付かない。自前で場合分けすると、実際には
    受け取れない金額を数えてしまう。
    """
    picks = _race([1, 2]).assign(score=[1.0, 0.5], rank=[1.0, 2.0])
    got = strategy.evaluate(picks, {("R1", 1): 250}, "テスト", "複勝")
    assert got["払戻"] == 250
    assert got["的中"] == 1
    assert got["回収率"] == 1.25


def test_break_even_line_matches_the_bet_type() -> None:
    """賭式ごとの基準線が正しいこと。

    単複は控除率20%で約80%、三連複は27.5%で約72.5%。ここを取り違えると
    「基準を超えた」の判断ごと狂う。
    """
    assert strategy.RANDOM_ROI["複勝"] == 0.80
    assert strategy.RANDOM_ROI["単勝"] == 0.80
    assert strategy.RANDOM_ROI["三連複"] == 0.725


def test_report_shows_the_break_even_line() -> None:
    """表に基準線を出すこと。回収率だけ見せると読み手が誤解する。"""
    picks = _race([1]).assign(score=[1.0], rank=[1.0])
    text = strategy.format_results(
        [strategy.evaluate(picks, {}, "テスト", "複勝")]
    )
    assert "基準" in text
    assert "100%を超えて初めて勝ち" in text
