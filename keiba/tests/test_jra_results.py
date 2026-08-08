"""JRA公式のレース結果・払戻金パーサの検証。

db.netkeiba は結果データベースだが反映が遅く、2026-08-08 は開催当日の夜に
なっても race_id を1件も出していなかった（実測）。当日中に回顧を回すには
JRA公式から取るしかない。ここが落ちると成績が記録できない。

フィクスチャは 2026-08-08 新潟（2回5日）の実物。
"""

from __future__ import annotations

import pathlib

import pytest

from keiba.sources.jra import (
    RESULT_KAISAI_RE,
    parse_payouts_page,
    parse_results_page,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "pinned"


def fixture(name: str) -> str:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"フィクスチャが無い: {name}")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def results():
    return parse_results_page(fixture("jra_results.html"))


@pytest.fixture(scope="module")
def payouts():
    return parse_payouts_page(fixture("jra_payouts.html"))


def test_all_twelve_races_are_parsed(results) -> None:
    """1開催＝12レース。取りこぼすとそのレースの成績が記録されない。"""
    assert len(results) == 12
    assert sorted(results) == [f"20260402050{i}" if i < 10 else f"2026040205{i}"
                               for i in range(1, 13)]


def test_finish_order_and_times(results) -> None:
    """着順・タイム・通過順・上り・馬体重が取れること。"""
    rows = sorted(results["202604020501"], key=lambda r: r.finish_pos or 99)
    first = rows[0]
    assert (first.finish_pos, first.umaban) == (1, 13)
    assert first.time_sec == pytest.approx(81.7)   # 1:21.7
    assert first.corners == [3, 3]
    assert first.last3f == pytest.approx(35.2)
    assert (first.body_weight, first.body_weight_diff) == (424, 4)


def test_time_is_seconds_not_text(results) -> None:
    """タイムは秒に直す。文字列のままだと着差の計算に使えない。"""
    for rows in results.values():
        for r in rows:
            if r.time_sec is not None:
                assert 50 < r.time_sec < 300, f"秒として異常: {r.time_sec}"


def test_payouts_cover_the_bets_we_place(payouts) -> None:
    """三連複・三連単の配当が取れること。買っているのはこの2つ。"""
    got = {(p.bet_type, p.combination): p.payout for p in payouts["202604020501"]}
    assert got[("三連複", "2-8-13")] == 4520
    assert got[("三連単", "13-8-2")] == 12520


def test_payout_combination_matches_finish_order(results, payouts) -> None:
    """払戻の組み合わせが、実際の1〜3着と一致すること。

    ここがずれると、当たっているのに不的中と判定される（あるいはその逆）。
    """
    for race_id, rows in results.items():
        top3 = [r.umaban for r in sorted(rows, key=lambda r: r.finish_pos or 99)[:3]]
        trio = next(
            (p for p in payouts.get(race_id, []) if p.bet_type == "三連複"), None
        )
        if trio is None:
            continue
        assert set(trio.combination.split("-")) == {str(u) for u in top3}, race_id


def test_result_kaisai_cname_is_decoded() -> None:
    """結果の開催リンクは pw01srl。直近は pw01srl0…、過去は pw01srl1… と
    1桁目が変わるので、そこを固定しない。"""
    assert RESULT_KAISAI_RE.match("pw01srl00042026020520260808/21").groups() == (
        "04", "2026", "02", "05", "20260808"
    )
    assert RESULT_KAISAI_RE.match("pw01srl10042026020420260802/B3").groups() == (
        "04", "2026", "02", "04", "20260802"
    )


def test_rejects_a_page_that_is_not_results() -> None:
    with pytest.raises(ValueError):
        parse_results_page("<html><h1>払戻金ランキング</h1></html>")
