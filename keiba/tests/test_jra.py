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


def test_gelding_is_parsed(cards) -> None:
    """JRA公式は騸馬を「せん」（ひらがな）と書く。

    「セン」だけを見ていたため性齢が丸ごと None になり、実データ957頭のうち
    42頭が該当した。性齢は能力モデルの特徴量なので、黙って欠けると効く。
    """
    by_name = {e.horse_name: e for c in cards for e in c.entries}
    gelding = by_name["ジーティーホクサイ"]
    assert (gelding.sex, gelding.age) == ("セ", 3)


def test_every_entry_has_sex_and_age(cards) -> None:
    """性齢が取れない馬を出さない。表記ゆれはここで気づけるようにする。"""
    missing = [
        e.horse_name
        for c in cards
        for e in c.entries
        if e.sex is None or e.age is None
    ]
    assert not missing, f"性齢を取れない馬がいる: {missing[:5]}"


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


def test_race_list_page_is_one_hop_short_of_the_racecard() -> None:
    """開催リンクの飛び先（レース選択）には出走表が無いこと。

    ここを出馬表だと思って直接パースしていたため、実機で0レースになった。
    出走馬一覧へは「全てのレースを表示」（pw01des01…）をもう1回辿る。
    """
    from keiba.sources.jra import ALL_RACES_RE, DOACTION_RE

    html = fixture("jra_racelist.html")
    # レース選択ページ自体からは1レースも取れない
    assert parse_racecard_page(html) == []

    hops = [c for _, c in DOACTION_RE.findall(html) if ALL_RACES_RE.fullmatch(c)]
    assert hops, "「全てのレースを表示」への遷移が見つからない"
    assert hops[0].startswith("pw01des01")


def test_scratched_horse_is_marked_not_silently_dropped() -> None:
    """枠順確定後に馬番が空なのは取消。印を付けて記録に残す。

    付けないと「馬番が無い」という理由だけで DB から静かに落ち、記録上は
    最初から居なかったことになる。2026-08-08 に実際そうなった。
    """
    from keiba.models import Entry
    from keiba.sources.jra import _mark_scratches

    entries = [
        Entry(race_id="R", umaban=n, horse_name=f"h{i}", horse_id="")
        for i, n in enumerate([1, 2, None, 4, 5])
    ]
    _mark_scratches(entries)
    assert [e.scratched for e in entries] == [False, False, True, False, False]


def test_pre_draw_card_is_not_treated_as_all_scratched(cards) -> None:
    """枠順確定前は全頭の馬番が空。これを取消と読んではいけない。"""
    assert all(not e.scratched for e in cards[0].entries)


def test_popularity_is_derived_from_odds() -> None:
    """JRA公式の出馬表には「人気」欄が無いので、単勝オッズの昇順から導出する。

    これが無いと confidence が人気を見られず、オッズが揃っていても全レースが
    「オッズ未確定」に落ちて1点も買えない。実際に当日そうなった。
    """
    from keiba.models import Entry
    from keiba.sources.jra import _assign_popularity

    entries = [
        Entry(race_id="R", umaban=i, horse_name=f"h{i}", horse_id="", market_odds=o)
        for i, o in enumerate([12.3, 2.1, 5.0, 2.1, 88.0], start=1)
    ]
    _assign_popularity(entries)
    assert [e.market_popularity for e in entries] == [4, 1, 3, 1, 5]


def test_popularity_stays_unset_before_odds_open() -> None:
    """オッズが無い段階で人気をでっち上げない。"""
    from keiba.models import Entry
    from keiba.sources.jra import _assign_popularity

    entries = [
        Entry(race_id="R", umaban=i, horse_name=f"h{i}", horse_id="") for i in range(1, 6)
    ]
    _assign_popularity(entries)
    assert all(e.market_popularity is None for e in entries)


def test_rejects_a_page_that_is_not_a_racecard() -> None:
    """別のページを食わせたら黙って空を返さず落ちること。"""
    with pytest.raises(ValueError):
        parse_racecard_page("<html><h1>払戻金一覧</h1></html>")
