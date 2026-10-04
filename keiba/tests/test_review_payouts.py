"""払戻が未反映のレースを、外れと区別して記録すること。

2026-10-03 は東京11R・12Rの払戻が入る前に回顧を走らせ、的中を外れと
数えて 112.2% を 94.3% と報告した。
"""

from __future__ import annotations

from datetime import date

from keiba import review
from keiba.models import Entry, Payout, Race, RaceCard, Result
from keiba.store import Store


def _card(rid, with_payout):
    race = Race(race_id=rid, race_date=date(2026, 10, 3), venue="東京", venue_code="05",
                kai=4, nichi=1, race_no=int(rid[-2:]), name="t", surface="芝", distance=1600)
    entries = [Entry(race_id=rid, umaban=i, horse_name=f"馬{i}", horse_id=f"h{rid}{i}") for i in range(1, 7)]
    results = [Result(race_id=rid, umaban=i, finish_pos=i) for i in range(1, 7)]
    payouts = [Payout(race_id=rid, bet_type="三連複", combination="1-2-3", payout=1320, popularity=1)] if with_payout else []
    return RaceCard(race=race, entries=entries, results=results, payouts=payouts)


def test_払戻未反映のレースを数える(tmp_path) -> None:
    with Store(tmp_path / "t.db") as s:
        s.save_cards([_card("202605040111", True), _card("202605040112", False)])
        bet = [{"type": "三連複", "combination": "1-2-3", "amount": 100, "why": ""}]
        payload = {"date": "2026-10-03", "summary": {"spend": 200},
                   "races": [{"race_id": "202605040111", "bets": bet, "horses": [], "confidence": "○", "confidence_reason": ""},
                             {"race_id": "202605040112", "bets": bet, "horses": [], "confidence": "○", "confidence_reason": ""}]}
        review.review_day(s, payload)
    assert payload["summary"]["payout_missing"] == ["東京12R"]
    assert payload["races"][0]["result"]["hit"] is True
