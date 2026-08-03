"""バックテストの決済ロジック検証。

ここが壊れると回収率が永久に 0% を返し、しかも一見それらしく見える。
的中を検出できること・払戻額が正しいことを固定しておく。
"""

from __future__ import annotations

import glob
import pathlib
from datetime import date

import pytest

from keiba import betting
from keiba.backtest import BacktestResult, RaceOutcome, _actual_top3, _settle
from keiba.models import Payout, Race, RaceCard, Result
from keiba.sources.netkeiba import parse_race_page

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture(pattern: str) -> str:
    matches = glob.glob(str(FIXTURES / pattern))
    if not matches:
        pytest.skip(f"フィクスチャが無い: {pattern}")
    return pathlib.Path(matches[0]).read_text(encoding="utf-8")


def make_card(top3: list[int], payouts: list[Payout]) -> RaceCard:
    race = Race(
        race_id="R1", race_date=date(2026, 8, 1), venue="東京", venue_code="05",
        kai=1, nichi=1, race_no=11, name="T", surface="芝", distance=1600,
    )
    results = [
        Result(race_id="R1", umaban=u, finish_pos=i + 1) for i, u in enumerate(top3)
    ]
    return RaceCard(race=race, results=results, payouts=payouts)


def test_actual_top3_uses_finish_order() -> None:
    card = make_card([7, 2, 11], [])
    assert _actual_top3(card) == [7, 2, 11]


def test_actual_top3_none_when_incomplete() -> None:
    """中止・除外で3頭揃わないレースは集計対象にしない。"""
    race = Race(
        race_id="R1", race_date=date(2026, 8, 1), venue="東京", venue_code="05",
        kai=1, nichi=1, race_no=11, name="T", surface="芝", distance=1600,
    )
    card = RaceCard(
        race=race,
        results=[Result(race_id="R1", umaban=1, finish_pos=1),
                 Result(race_id="R1", umaban=2, finish_pos=2),
                 Result(race_id="R1", umaban=3, finish_pos=None)],
    )
    assert _actual_top3(card) is None


def test_settle_detects_trio_hit_and_scales_payout() -> None:
    """3連複的中。払戻は100円あたりなので 300円買いなら3倍で戻る。"""
    card = make_card([7, 2, 11], [Payout(race_id="R1", bet_type="三連複",
                                         combination="2-7-11", payout=5500)])
    tickets = [
        betting.Ticket("三連複", "2-7-11", 300, "本線"),
        betting.Ticket("三連複", "1-2-3", 100, "外れ"),
    ]
    spent, returned, hit_types, best = _settle(tickets, card, [7, 2, 11])
    assert spent == 400
    assert returned == 16500  # 5,500円 × (300円 ÷ 100円)
    assert hit_types == ["三連複"]
    assert best == 5500


def test_settle_trio_is_order_independent() -> None:
    """3連複は着順に関係なく、組み合わせが一致すれば的中。"""
    card = make_card([11, 7, 2], [Payout(race_id="R1", bet_type="三連複",
                                         combination="2-7-11", payout=1230)])
    spent, returned, hit_types, _ = _settle(
        [betting.Ticket("三連複", "2-7-11", 100, "")], card, [11, 7, 2]
    )
    assert returned == 1230 and hit_types == ["三連複"]


def test_settle_trifecta_requires_exact_order() -> None:
    """3連単は着順どおりでなければ的中しない。"""
    payouts = [Payout(race_id="R1", bet_type="三連単", combination="7-2-11", payout=73530)]
    card = make_card([7, 2, 11], payouts)

    hit = _settle([betting.Ticket("三連単", "7-2-11", 100, "")], card, [7, 2, 11])
    assert hit[1] == 73530

    miss = _settle([betting.Ticket("三連単", "2-7-11", 100, "")], card, [7, 2, 11])
    assert miss[1] == 0 and miss[2] == []


def test_settle_missing_payout_does_not_silently_score_zero(caplog) -> None:
    """的中しているのに払戻データが無いときは警告を出す。

    黙って0円にすると、データ欠損が「予想が外れた」ように見えてしまう。
    """
    card = make_card([7, 2, 11], [])  # 払戻テーブルが空
    with caplog.at_level("WARNING"):
        spent, returned, hit_types, _ = _settle(
            [betting.Ticket("三連複", "2-7-11", 100, "")], card, [7, 2, 11]
        )
    assert returned == 0 and hit_types == []
    assert any("払戻データ欠損" in r.message for r in caplog.records)


def test_settle_against_real_payout_data() -> None:
    """実フィクスチャの払戻で決済できること。

    中京12R の3連複 2-5-7 は 5,500円、3連単 2→5→7 は 73,530円。
    """
    card = parse_race_page(fixture("db_race_main__*"), "202607020212")
    top3 = _actual_top3(card)
    assert top3 == [2, 5, 7]

    tickets = [
        betting.Ticket("三連複", "2-5-7", 300, "本線"),
        betting.Ticket("三連単", "2-5-7", 100, "3連単"),
    ]
    spent, returned, hit_types, best = _settle(tickets, card, top3)
    assert spent == 400
    assert returned == (5500 * 3) + 73530
    assert sorted(hit_types) == ["三連単", "三連複"]


def test_roi_reporting() -> None:
    result = BacktestResult()
    result.outcomes = [
        RaceOutcome("A", date(2026, 8, 1), "◎", spent=1000, returned=3690, hit=True,
                    payout=1230, top3_popularity=21),
        RaceOutcome("B", date(2026, 8, 1), "○", spent=1000, returned=0, hit=False),
    ]
    assert result.spent == 2000
    assert result.returned == 3690
    assert result.hits == 1
    assert result.hit_rate == 50.0
    assert round(result.roi, 1) == 184.5
    assert "回収率" in result.report()
