"""自信度判定の検証。

「オッズが無い」と「堅いから買わない」は意味がまったく違う。前日夜の予想は
オッズ未確定なので必ず前者になるが、これを × にしてしまうと開催前の予想が
すべて「見送り」に見えてしまう。
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from keiba.confidence import PENDING, SKIP, estimate_trio_odds, grade
from keiba.engine import ScoredHorse

WEIGHTS = yaml.safe_load(
    (pathlib.Path(__file__).parents[1] / "config/weights.yml").read_text()
)


def horses(pops: list[int | None], scores: list[float] | None = None) -> list[ScoredHorse]:
    scores = scores or [90 - i * 10 for i in range(len(pops))]
    return [
        ScoredHorse(
            umaban=i + 1, horse_id=f"h{i}", horse_name=f"m{i}",
            score=scores[i], style="先行",
            market_popularity=p, market_odds=(p * 3.0 if p else None),
        )
        for i, p in enumerate(pops)
    ]


def test_no_odds_is_pending_not_skip() -> None:
    """オッズ未確定は ? であって × ではない。"""
    c = grade(horses([None] * 8), WEIGHTS)
    assert c.grade == PENDING
    assert c.is_pending
    assert not c.should_bet          # まだ買わない
    assert "オッズ未確定" in c.reason


def test_chalk_race_still_gets_a_buy_list() -> None:
    """上位人気で収まりそうな形でも、買い目は出すこと。

    以前はここを × にして1点も買わなかった。**自分が高く評価した3頭が
    たまたま人気だった**という理由でレースごと捨てる形で、「人気上位だから
    3着以内に入るは違う」という方針の逆をやっていた。

    買うかどうかは人が決める。自信度は判断材料として添えるだけにする。
    """
    c = grade(horses([1, 2, 3, 4, 5, 6, 7, 8]), WEIGHTS)
    assert c.grade != SKIP, "上位人気というだけで見送っている"
    assert c.should_bet
    assert not c.is_pending
    # 形自体は判断材料として残す
    assert "上位人気" in c.reason


def test_skipping_can_be_switched_back_on() -> None:
    """見送りの仕組みは残してある（confidence.skip: true で戻る）。

    外したのは既定値であって、機能ではない。戻したくなったら設定1行。
    """
    cfg = {**WEIGHTS, "confidence": {**WEIGHTS["confidence"], "skip": True}}
    c = grade(horses([1, 2, 3, 4, 5, 6, 7, 8]), cfg)
    assert c.grade == SKIP
    assert not c.should_bet


def test_main_band_is_backed_as_anchor() -> None:
    """本線の帯（人気和 9-13）で能力が分離していれば ◎。"""
    c = grade(horses([5, 4, 3, 1, 2, 6, 7, 8]), WEIGHTS)
    assert c.popularity_sum == 12
    assert c.grade == "◎"
    assert c.should_bet
    assert not c.is_longshot


def test_longshot_band_is_separate_bucket() -> None:
    """穴の帯（人気和14以上）は △＝穴枠。買うが本線とは別扱い。

    実測でこの帯の回収率は本線より明確に低い（9-13が89.9%、14-19が63.3%、
    20-27が18.2%）。切り捨てはしないが、厚く買う対象にはしない。
    """
    c = grade(horses([12, 10, 11, 1, 2, 3, 4, 5]), WEIGHTS)
    assert c.popularity_sum == 33
    assert c.grade == "△"
    assert c.should_bet
    assert c.is_longshot
    assert "穴枠" in c.reason


def test_flat_scores_demote_from_anchor() -> None:
    """本線の帯でも能力が横並びなら ◎ にしない。軸を立てられないため。"""
    flat = [70.0] * 8
    c = grade(horses([5, 4, 3, 1, 2, 6, 7, 8], flat), WEIGHTS)
    assert c.grade == "○"
    assert "スコア差" in c.reason


def test_thin_expected_payout_is_noted_not_skipped() -> None:
    """想定配当が薄くても買い目は出し、薄いことだけ書き添えること。

    「配当が大きい組を探す」形は、勝つ馬を買うという目的と順序が逆になる。
    strategy.py に自分でそう書いておきながら、本番の設定には残っていた。
    """
    cfg = {**WEIGHTS, "confidence": {**WEIGHTS["confidence"], "min_expected_odds": 1e9}}
    c = grade(horses([5, 4, 3, 1, 2, 6, 7, 8]), cfg)
    assert c.grade != SKIP
    assert c.should_bet
    assert "配当は小さめ" in c.reason

    # 戻したいときは skip: true
    on = {**cfg, "confidence": {**cfg["confidence"], "skip": True}}
    assert grade(horses([5, 4, 3, 1, 2, 6, 7, 8]), on).grade == SKIP


@pytest.mark.parametrize(
    ("odds", "expect_cheap"),
    [([1.5, 2.0, 3.0], True), ([30.0, 25.0, 40.0], False)],
)
def test_estimate_trio_odds_orders_correctly(odds, expect_cheap) -> None:
    """人気サイドほど安く、人気薄サイドほど高い見積りになること。"""
    hs = [
        ScoredHorse(umaban=i, horse_id="", horse_name="", score=1.0, style="先行",
                    market_odds=o)
        for i, o in enumerate(odds)
    ]
    est = estimate_trio_odds(hs)
    assert est is not None
    assert (est < 100) is expect_cheap


def test_estimate_trio_odds_none_without_odds() -> None:
    hs = [ScoredHorse(umaban=i, horse_id="", horse_name="", score=1.0, style="先行")
          for i in range(3)]
    assert estimate_trio_odds(hs) is None
