"""実HTMLフィクスチャに対するパーサ検証。

フィクスチャは keiba-probe ワークフローが本物のサイトから採取したもの。
netkeiba の HTML が変わればここが落ちる。それが目的で、静かに壊れて
おかしなデータを溜め込むより落ちたほうがよい。
"""

from __future__ import annotations

import glob
import pathlib

import pytest

from keiba.sources.netkeiba import (
    parse_body_weight,
    parse_corners,
    parse_finish_pos,
    parse_pedigree,
    parse_race_list,
    parse_race_page,
    time_to_sec,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture(pattern: str) -> str:
    matches = glob.glob(str(FIXTURES / pattern))
    if not matches:
        pytest.skip(f"フィクスチャが無い: {pattern}（keiba-probe を実行すること）")
    return pathlib.Path(matches[0]).read_text(encoding="utf-8")


# ------------------------------------------------------- 単体の変換関数

@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1:07.8", 67.8), ("59.6", 59.6), ("2:24.3", 144.3), ("", None), ("--", None)],
)
def test_time_to_sec(raw: str, expected: float | None) -> None:
    assert time_to_sec(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", 1), ("15", 15), ("中止", None), ("除外", None), ("失格", None), ("", None)],
)
def test_parse_finish_pos(raw: str, expected: int | None) -> None:
    assert parse_finish_pos(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("466(-2)", (466, -2)), ("438(+2)", (438, 2)), ("480(0)", (480, 0)),
     ("計不", (None, None)), ("", (None, None))],
)
def test_parse_body_weight(raw: str, expected: tuple) -> None:
    assert parse_body_weight(raw) == expected


def test_parse_corners() -> None:
    assert parse_corners("8-7") == [8, 7]
    assert parse_corners("15-14-12-10") == [15, 14, 12, 10]
    assert parse_corners("") == []


# ------------------------------------------------------------- レース

def test_parse_race_list_filters_to_jra() -> None:
    """地方を弾いて中央だけ返す。中央開催日は12R×3場で36件になる。"""
    ids = parse_race_list(fixture("db_race_list_20260726__*"))
    assert len(ids) == 36
    assert all(i[4:6] in {f"{n:02d}" for n in range(1, 11)} for i in ids)

    # 絞らなければ地方が混ざる（プローブで盛岡・船橋・高知・帯広を確認済み）
    everything = parse_race_list(fixture("db_race_list_20260802__*"), jra_only=False)
    assert len(everything) > len(parse_race_list(fixture("db_race_list_20260802__*")))


def test_parse_race_page_header() -> None:
    card = parse_race_page(fixture("db_race_main__*"), "202607020212")
    race = card.race
    assert (race.venue, race.race_no, race.name) == ("中京", 12, "3歳未勝利")
    assert (race.surface, race.distance, race.direction) == ("芝", 1200, "左")
    assert (race.going, race.weather, race.post_time) == ("良", "晴", "18:20")
    assert race.race_date.isoformat() == "2026-07-26"
    assert (race.kai, race.nichi) == (2, 2)
    # 条件記号 [指](馬齢) を落として素の条件名にする（降級判定に使うため）
    assert race.race_class == "3歳未勝利"
    assert race.field_size == 15


def test_parse_race_page_entries_and_results() -> None:
    card = parse_race_page(fixture("db_race_main__*"), "202607020212")
    assert len(card.entries) == 15
    # 全頭に着順が付いている（この日のレースは中止馬なし）
    assert all(r.finish_pos is not None for r in card.results)

    winner = next(e for e in card.entries if e.umaban == 2)
    assert winner.horse_name == "ベルセクレト"
    assert winner.horse_id == "2023109072"
    assert (winner.sex, winner.age) == ("牝", 3)
    assert winner.weight_carried == 54.0
    assert (winner.jockey, winner.jockey_id) == ("田山旺佑", "01220")
    assert (winner.trainer, winner.affiliation) == ("藤岡健一", "栗東")
    assert (winner.body_weight, winner.body_weight_diff) == (466, -2)

    result = next(r for r in card.results if r.umaban == 2)
    assert (result.finish_pos, result.time_sec) == (1, 67.8)
    assert result.corners == [8, 7]
    assert result.last3f == 33.4


def test_odds_are_recorded_but_never_lost() -> None:
    """オッズと人気は記録する。エンジンが使わないだけで、収集はする。

    穴かどうかの判定（confidence）に人気が要るため、欠けていると
    ◎○△× が付けられなくなる。
    """
    card = parse_race_page(fixture("db_race_main__*"), "202607020212")
    winner = next(e for e in card.entries if e.umaban == 2)
    assert winner.market_odds == 18.8
    assert winner.market_popularity == 4
    assert all(e.market_popularity is not None for e in card.entries)


def test_parse_payouts() -> None:
    card = parse_race_page(fixture("db_race_main__*"), "202607020212")
    payouts = {(p.bet_type, p.combination): p.payout for p in card.payouts}
    assert payouts[("三連複", "2-5-7")] == 5500
    assert payouts[("三連単", "2-5-7")] == 73530
    assert payouts[("単勝", "2")] == 1880
    # 複勝・ワイドは1セルに複数行入る。<br> で割れていること
    assert payouts[("複勝", "2")] == 240
    assert payouts[("複勝", "5")] == 340
    assert payouts[("複勝", "7")] == 110


def test_parse_race_page_dirt_course() -> None:
    """ダート・右回り・別会場でもヘッダを読めること。"""
    card = parse_race_page(fixture("db_race_sub__*"), "202601010201")
    race = card.race
    assert (race.venue, race.surface, race.distance, race.direction) == (
        "札幌", "ダ", 1000, "右",
    )
    assert race.race_class == "2歳未勝利"


# --------------------------------------------------------------- 血統

def test_parse_pedigree() -> None:
    ped = parse_pedigree(fixture("ped_2023100490__*"))
    assert ped["sire"] == "アルアイン"
    assert ped["dam"] == "フレイムコード"
    assert ped["damsire"] == "タヤスツヨシ"
    assert ped["sire_line"] == "Halo系"


def test_parse_pedigree_foreign_dam() -> None:
    """外国産馬は 'カナ名 English(産地)' の形。カナ名だけ取れること。"""
    ped = parse_pedigree(fixture("ped_2023100918__*"))
    assert ped["sire"] == "コントレイル"
    assert ped["dam"] == "コンクエストハーラネイト"
    assert ped["damsire"] == "Harlan's Holiday"
