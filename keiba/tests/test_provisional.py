"""枠順確定前の暫定予想の検証。

木曜のJRA出馬表は枠順確定前で馬番が空。以前はそこで予想ごと捨てており、
「情報が出たら集めるだけ集めて軽く予想する」という運用ができていなかった
（実際 2026-08-13 の暫定予想は 0レースで出た）。

馬名・血統・騎手は揃っているので能力評価はできる。捨てる理由は「買い目を
偽の馬番で組まないため」だけなので、印は出して買い目だけ止める。
"""

from __future__ import annotations

import datetime

import pytest

from keiba.models import Entry, Race, RaceCard


def _card(with_umaban: bool) -> RaceCard:
    race = Race(
        race_id="202604020611", race_date=datetime.date(2026, 8, 15),
        venue="新潟", venue_code="04", kai=2, nichi=6, race_no=11,
        name="3歳未勝利", surface="芝", distance=1600, field_size=8,
    )
    entries = [
        Entry(
            race_id=race.race_id,
            umaban=i if with_umaban else None,
            horse_name=f"馬{i}", horse_id=f"20231000{i:02d}",
            sex="牡", age=3, weight_carried=56.0,
            jockey=f"騎手{i}", jockey_id=f"j{i}",
        )
        for i in range(1, 9)
    ]
    return RaceCard(race=race, entries=entries)


def test_marks_come_out_without_post_positions(tmp_path) -> None:
    """馬番が無くても印と能力評価が出ること。

    これができないと、木曜の暫定予想が毎回0レースになる。
    """
    from keiba.predict import predict_card
    from keiba.store import Store

    with Store(tmp_path / "t.db") as store:
        got = predict_card(_card(with_umaban=False), store, _weights(), {})

    assert got is not None, "枠順未確定で予想ごと落ちている"
    assert len(got["horses"]) == 8
    assert any(h["mark"] for h in got["horses"]), "印が1つも付いていない"


def test_provisional_output_carries_no_horse_numbers(tmp_path) -> None:
    """仮の通し番号を、本物の馬番として出力しないこと。

    ここが漏れると、確定前の並びで買い目を組んだり、利用者が
    その番号で馬券を買ったりする。
    """
    from keiba.predict import predict_card
    from keiba.store import Store

    with Store(tmp_path / "t.db") as store:
        got = predict_card(_card(with_umaban=False), store, _weights(), {})

    assert all(h["umaban"] is None for h in got["horses"]), "仮番号が出力に出ている"
    assert got["post_positions_confirmed"] is False


def test_no_tickets_before_post_positions(tmp_path) -> None:
    """枠順が決まるまで買い目を組まないこと。"""
    from keiba.predict import predict_card
    from keiba.store import Store

    with Store(tmp_path / "t.db") as store:
        got = predict_card(_card(with_umaban=False), store, _weights(), {})

    assert got["bets"] == []
    assert got["spend"] == 0


def test_confirmed_cards_still_get_numbers_and_tickets(tmp_path) -> None:
    """枠順が確定していれば、これまでどおり馬番も買い目も出ること。

    暫定対応を入れたせいで本番が壊れていないことの確認。
    """
    from keiba.predict import predict_card
    from keiba.store import Store

    with Store(tmp_path / "t.db") as store:
        got = predict_card(_card(with_umaban=True), store, _weights(), {})

    assert got["post_positions_confirmed"] is True
    assert all(h["umaban"] is not None for h in got["horses"])


def _weights() -> dict:
    import pathlib

    import yaml

    path = pathlib.Path(__file__).parents[1] / "config" / "weights.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
