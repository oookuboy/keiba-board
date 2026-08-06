"""JRA公式の出馬表パーサの検証。

db.netkeiba.com には発走前の race_id が存在しない（2026-08-06 実測）ため、
ここが**唯一の発走前データ経路**になる。落ちたら週末の予想が丸ごと出ない。

フィクスチャは 2026-08-08 新潟（2回5日）の出走馬一覧。木曜に採取したので
枠順確定前で、枠・馬番・馬体重・オッズが空という状態も一緒に固定してある。
"""

from __future__ import annotations

import pathlib
from datetime import date

import pytest

from keiba.sources.jra import (
    find_kaisai_links,
    parse_racecard_page,
    post_positions_confirmed,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "pinned"


def fixture(name: str) -> str:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"フィクスチャが無い: {name}")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cards():
    return parse_racecard_page(fixture("jra_racecard.html"))


def test_all_twelve_races_are_parsed(cards) -> None:
    """1開催＝12レースが1ページに載っている。取りこぼすと予想から丸ごと消える。"""
    assert len(cards) == 12
    assert [c.race.race_no for c in cards] == list(range(1, 13))


def test_race_id_matches_netkeiba_format(cards) -> None:
    """race_id は netkeiba 形式（年+場+回+日+R）で組む。

    3年分の過去走が netkeiba 形式で入っているので、ここがずれると
    同じレースを二重に持つことになる。
    """
    race = cards[0].race
    assert race.race_id == "202604020501"
    assert (race.venue, race.venue_code) == ("新潟", "04")
    assert (race.kai, race.nichi) == (2, 5)
    assert race.race_date == date(2026, 8, 8)


def test_course_is_parsed_including_straight_course(cards) -> None:
    """新潟1000mは「（芝・直）」で、「直線」とは書かれない。"""
    by_no = {c.race.race_no: c.race for c in cards}
    assert (by_no[1].surface, by_no[1].distance, by_no[1].direction) == ("芝", 1400, "左")
    assert (by_no[2].surface, by_no[2].distance) == ("ダ", 1200)
    assert (by_no[11].distance, by_no[11].direction) == (1000, "直線")


def test_entries_carry_ids_that_join_with_history(cards) -> None:
    """馬IDが netkeiba と同一体系であること。

    これが崩れると過去走・血統と結合できず、能力評価が新馬同然になる。
    """
    entry = cards[0].entries[0]
    assert entry.horse_name == "アミノショコラ"
    assert entry.horse_id == "2024100830"
    assert (entry.sex, entry.age) == ("牡", 2)
    assert entry.weight_carried == 55.0
    assert entry.jockey == "木幡 巧也"
    assert entry.trainer == "杉浦 宏昭"


def test_apprentice_mark_is_stripped_from_jockey_name(cards) -> None:
    """「△ 石神 深道」の減量記号を騎手名に混ぜない。

    混ざると騎手成績の突き合わせが名前一致で外れる。
    """
    names = [e.jockey for c in cards for e in c.entries if e.jockey]
    assert names, "騎手名が1つも取れていない"
    assert not any(n[0] in "▲△☆★◇" for n in names)


def test_thursday_card_has_no_post_positions(cards) -> None:
    """木曜の出馬表は枠順確定前。馬番が空であることを None として持つ。

    ここを 0 などで埋めると、偽の馬番で買い目が組まれる。
    """
    assert all(e.umaban is None for e in cards[0].entries)
    assert not post_positions_confirmed(cards[0])


def test_field_size_matches_entry_count(cards) -> None:
    for card in cards:
        assert card.race.field_size == len(card.entries)
        assert len(card.entries) >= 9


def test_kaisai_links_cover_every_venue_and_day() -> None:
    """開催選択ページから3場×2日を拾えること。"""
    links = find_kaisai_links(fixture("jra_kaisai.html"))
    assert len(links) == 6
    assert {x["venue"] for x in links} == {"新潟", "中京", "札幌"}
    assert {x["race_date"] for x in links} == {date(2026, 8, 8), date(2026, 8, 9)}
    # cname は組み立てず拾う。末尾2桁が何なのか分かっていないため
    assert all(x["cname"].startswith("pw01drl") for x in links)


def test_rejects_a_page_that_is_not_a_racecard() -> None:
    """別のページを食わせたら黙って空を返さず落ちること。"""
    with pytest.raises(ValueError):
        parse_racecard_page("<html><h1>払戻金一覧</h1></html>")
