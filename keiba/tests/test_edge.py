"""市場との乖離の測定の検証。

ここで一番危ないのは2つ。

  1. 市場カーブに検証期間を混ぜること。市場が実際より賢く見え、
     モデルの乖離が過小評価される（＝妙味を見逃す）
  2. 回収率を実払戻ではなく自前の場合分けで作ること。複勝は少頭数だと
     3着に払戻が付かないなど例外が多く、自前で書けば必ず間違える
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba import edge


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "race_id": [f"2026{i // 10:08d}" for i in range(n)],
            "umaban": rng.integers(1, 17, n),
            "market_odds": rng.uniform(1.2, 200.0, n),
            "market_popularity": rng.integers(1, 18, n),
            "placed": rng.integers(0, 2, n),
        }
    )


def test_market_curve_is_built_from_training_data_only() -> None:
    """カーブが渡された表だけから作られること。

    呼び出し側が学習期間を渡す設計なので、ここでは「渡した行数しか
    使っていない」ことを、値が変わることで確かめる。
    """
    early = _frame(4000, seed=1)
    early["placed"] = 1                      # 学習期間は全部3着以内
    late = _frame(4000, seed=2)
    late["placed"] = 0                       # 検証期間は全部圏外

    curve = edge.market_curve(early, "placed")
    assert (curve == 1.0).all(), "学習期間だけを見れば全部1になるはず"

    mixed = edge.market_curve(pd.concat([early, late]), "placed")
    assert (mixed < 1.0).any(), "検証期間を混ぜれば値が下がるはず（＝混ぜてはいけない）"


def test_odds_bins_separate_favourites() -> None:
    """人気馬側が1つの箱に潰れないこと。

    等間隔で切ると1倍台〜5倍台が全部同じ箱に入り、市場の織り込みが
    のっぺりする。対数寄りの刻みにしてある。
    """
    edges = [b for b in edge.ODDS_BINS if b <= 10]
    assert len(edges) >= 8, "10倍以下の刻みが粗すぎる"


def test_returns_come_from_actual_payouts() -> None:
    """払戻は実データから引くこと。当たっていても払戻が無ければ0円。

    複勝は7頭立て以下だと3着に付かない。自前で「3着以内なら払戻」と
    書くと、こういうレースで実際には受け取れない金額を数えてしまう。
    """
    frame = pd.DataFrame({"race_id": ["A", "A", "B"], "umaban": [1, 2, 3]})
    payouts = {("A", 1): 250}
    got = edge.returns_for(frame, payouts)
    assert list(got) == [250.0, 0.0, 0.0]


def test_buckets_report_row_counts_and_a_break_even_line() -> None:
    """回収率だけでなく行数と基準線を出すこと。

    80%（控除率20%）を超えているかどうかが判断のすべてなので、
    表にその基準が書かれていないと読み手が誤解する。
    """
    frame = _frame(5000)
    rng = np.random.default_rng(3)
    table = edge.by_edge(
        frame, "placed", rng.normal(size=5000),
        rng.integers(0, 400, 5000).astype(float),
        rng.integers(0, 900, 5000).astype(float),
    )
    text = edge.format_table(table, "テスト")
    assert "行数" in text
    assert "80%" in text, "でたらめに買った場合の基準が書かれていない"
    assert len(table) >= 5


def test_edge_of_zero_puts_everything_in_the_middle() -> None:
    """乖離が全部同じなら、区分が割れないこと（計算そのものの確認）。"""
    frame = _frame(2000)
    table = edge.by_edge(
        frame, "placed", np.zeros(2000), np.zeros(2000), np.zeros(2000)
    )
    assert len(table) == 1
