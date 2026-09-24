"""JRA公式経由のデータで欠けていた欄を押さえる。

2026-08-08 に発走前の出馬表を JRA公式から取るよう切り替えてから、
次の欄が**黙って**欠けていた（2026-09-24 に発覚）。

- 枠番: 画像（alt="枠1白"）で書かれていて、文字として読めず全頭空
- 馬場状態: 発走前の出馬表には無く、結果ページからも拾っていなかった
- 騎手・調教師ID: 先頭の0が落ちた4桁で、netkeiba の5桁と別人扱い。
  騎手の成績は切り替え後の1か月ぶんしか数えられず、全馬が「乗り替わり」
- 血統: 一度まとめて引いたきりで、以降に出てきた馬は父も母父も空

どれも予想は普通に出るので、ログにも成果物にも異常が現れない。
"""

from __future__ import annotations

import pathlib
from datetime import date

import pytest

from keiba import backfill, collect
from keiba.features import _person_key
from keiba.models import Entry, Race, RaceCard, Result
from keiba.sources import jra
from keiba.store import Store

FIX = pathlib.Path(__file__).parent / "fixtures"


def _read(rel: str) -> str:
    path = FIX / rel
    if not path.exists():
        pytest.skip(f"フィクスチャが無い: {rel}")
    return path.read_text(encoding="utf-8")


def test_waku_is_read_from_the_image() -> None:
    """枠順確定後の出馬表で、枠番が画像から読めること。"""
    cards = jra.parse_racecard_page(
        _read("probe/card_L3_race0__www.jra.go.jp_JRADB_accessD.html.html")
    )
    entries = [e for c in cards for e in c.entries if e.umaban is not None]
    assert entries, "枠順確定後のフィクスチャのはず"
    assert all(e.waku is not None for e in entries), "枠が空の馬がいる"
    assert all(1 <= e.waku <= 8 for e in entries)
    first = cards[0].entries
    assert [(e.umaban, e.waku) for e in first[:5]] == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 4)]


def test_ids_are_padded_to_netkeiba_width() -> None:
    """騎手・調教師IDが netkeiba と同じ5桁で入ること。"""
    cards = jra.parse_racecard_page(_read("pinned/jra_racecard.html"))
    ids = [e.jockey_id for c in cards for e in c.entries if e.jockey_id]
    ids += [e.trainer_id for c in cards for e in c.entries if e.trainer_id]
    assert ids and all(len(i) == 5 for i in ids)
    assert jra.normalize_id("5339") == "05339"
    assert jra.normalize_id("05339") == "05339"
    assert jra.normalize_id(None) is None


def test_store_pads_ids_already_on_disk(tmp_path) -> None:
    """直す前に溜まった raw（4桁ID）も、DB へ入れるときにそろうこと。"""
    race = Race(
        race_id="202606040611", race_date=date(2026, 9, 20), venue="中山",
        venue_code="06", kai=4, nichi=6, race_no=11, name="t", surface="芝",
        distance=1200,
    )
    entries = [
        Entry(race_id=race.race_id, umaban=i, horse_name=f"馬{i}", horse_id=f"20200000{i:02d}",
              jockey_id="5339", trainer_id="1061")
        for i in range(1, 7)
    ]
    with Store(tmp_path / "t.db") as s:
        s.save_cards([RaceCard(race=race, entries=entries)])
        rows = s.conn.execute("SELECT DISTINCT jockey_id, trainer_id FROM entries").fetchall()
    assert [tuple(r) for r in rows] == [("05339", "01061")]


def test_results_page_gives_going_weather_and_waku() -> None:
    """結果ページから、レースごとの天候・馬場状態・枠番が取れること。"""
    conds = jra.parse_results_conditions(_read("pinned/jra_results.html"))
    assert len(conds) == 12
    first = conds["202604020501"]
    assert first["weather"] == "晴"
    assert first["going"] == {"芝": "良"}
    assert first["waku"][13] == 8
    assert conds["202604020502"]["going"] == {"ダ": "良"}
    assert all(c["waku"] for c in conds.values())


def test_conditions_fill_the_card() -> None:
    """結果の取り込みで、出馬表の馬場状態と枠が埋まること。"""
    race = Race(
        race_id="r", race_date=date(2026, 9, 20), venue="中山", venue_code="06",
        kai=4, nichi=6, race_no=1, name="t", surface="ダ", distance=1200,
    )
    card = RaceCard(race=race, entries=[
        Entry(race_id="r", umaban=1, horse_name="a", horse_id="1"),
        Entry(race_id="r", umaban=2, horse_name="b", horse_id="2", waku=5),
    ])
    collect.apply_conditions(
        card, {"weather": "曇", "going": {"芝": "良", "ダ": "重"}, "waku": {1: 1, 2: 2}}
    )
    assert card.race.going == "重", "ダートのレースに芝の馬場を入れている"
    assert card.race.weather == "曇"
    assert [e.waku for e in card.entries] == [1, 5], "既にある枠を書き換えている"


def test_jockey_spelling_is_not_a_jockey_change() -> None:
    """「武豊」と「武 豊」、「ルメール」と「C.ルメール」は同じ騎手。"""
    assert _person_key("武 豊") == _person_key("武豊")
    assert _person_key("C.ルメール") == _person_key("ルメール")
    assert _person_key("横山 武史") != _person_key("横山 和生")


def _card(going=None, waku=None, results=True) -> RaceCard:
    race = Race(
        race_id="r", race_date=date(2026, 9, 20), venue="中山", venue_code="06",
        kai=4, nichi=6, race_no=1, name="t", surface="芝", distance=1200, going=going,
    )
    entries = [
        Entry(race_id="r", umaban=1, horse_name="a", horse_id="h1", waku=waku),
        Entry(race_id="r", umaban=2, horse_name="b", horse_id="h2", waku=waku, body_weight=480),
    ]
    res = [Result(race_id="r", umaban=1, finish_pos=1, last3f=None),
           Result(race_id="r", umaban=2, finish_pos=2, last3f=34.0)] if results else []
    return RaceCard(race=race, entries=entries, results=res)


def test_refill_fills_only_blanks() -> None:
    """db.netkeiba の値で空欄だけを埋め、既にある値は書き換えないこと。"""
    card = _card()
    assert backfill.needs_refill(card)

    ref = _card(going="稍重", waku=3)
    ref.entries[1].body_weight = 999          # 既にある値は上書きしない
    ref.results[0].last3f = 33.5
    ref.results[1].last3f = 99.9
    n = backfill.fill_blanks(card, ref)

    assert n > 0
    assert card.race.going == "稍重"
    assert [e.waku for e in card.entries] == [3, 3]
    assert card.entries[1].body_weight == 480
    assert card.results[0].last3f == 33.5
    assert card.results[1].last3f == 34.0
    assert not backfill.needs_refill(card)


def test_refill_matches_horses_by_id_not_number() -> None:
    """馬番が食い違う馬には埋めない（別の馬の枠や馬体重を付けないため）。"""
    card = _card()
    ref = _card(going="良", waku=3)
    ref.entries[0].umaban = 9
    backfill.fill_blanks(card, ref)
    assert card.entries[0].waku is None
    assert card.entries[1].waku == 3


def test_refill_takes_missing_results() -> None:
    """着順ごと欠けたレースには、db.netkeiba の着順を入れること（9/21 の半分）。"""
    card = _card(going="良", waku=1, results=False)
    assert backfill.needs_refill(card)
    backfill.fill_blanks(card, _card(going="良", waku=1))
    assert len(card.results) == 2


def test_current_going_is_read_from_the_info_page() -> None:
    """発走前の馬場状態が「開催お知らせ」から取れること。

    出馬表には馬場状態が無い。これが無いと、学習では馬場を見ているのに
    本番だけ馬場が常に空になる。
    """
    day, going = jra.parse_current_going(
        _read("probe/jra_info_L1__www.jra.go.jp_JRADB_accessI.html.html"), date(2026, 8, 8)
    )
    assert day == date(2026, 8, 9)
    assert set(going) == {"新潟", "中京", "札幌"}
    assert going["新潟"] == {"weather": "晴", "going": {"芝": "良", "ダ": "良"}}


def test_recollecting_the_card_keeps_going_and_waku() -> None:
    """出馬表を取り直しても、結果ページで埋めた馬場状態と枠を消さないこと。"""
    old = _card(going="重", waku=4)
    new = _card(going=None, waku=None)
    kept = collect._keep_extras(old, new)
    assert kept.race.going == "重"
    assert [e.waku for e in kept.entries] == [4, 4]
