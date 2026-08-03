"""1日の予算上限の検証。

開催日は36レースあり、1レース10点前後を素直に積むと簡単に数万円になる。
削り方には順序があり、穴で勝負する ◎ は最後まで守らないと
「固い予想はいらない」という方針そのものが崩れる。
"""

from __future__ import annotations

import pathlib

import yaml

from keiba.predict import apply_daily_cap

WEIGHTS = yaml.safe_load(
    (pathlib.Path(__file__).parents[1] / "config/weights.yml").read_text()
)
CFG = WEIGHTS["betting"]


def race(grade: str, points: int, per_point: int, pop_sum: int = 20) -> dict:
    bets = [
        {"type": "三連複", "combination": f"1-2-{i + 3}", "amount": per_point, "why": ""}
        for i in range(points)
    ]
    return {
        "confidence": grade,
        "popularity_sum": pop_sum,
        "bets": bets,
        "spend": points * per_point,
    }


def total(races: list[dict]) -> int:
    return sum(r["spend"] for r in races)


def test_under_cap_is_untouched() -> None:
    races = [race("◎", 10, 300), race("○", 10, 200)]
    before = total(races)
    actions = apply_daily_cap(races, CFG)
    assert actions == []
    assert total(races) == before


def test_reduces_stakes_before_dropping_races() -> None:
    """まず減額で収める。点数を削るのは最後の手段。"""
    races = [race("◎", 10, 300) for _ in range(6)]  # 18,000円
    races += [race("△", 10, 100) for _ in range(10)]  # 10,000円 → 計28,000円
    apply_daily_cap(races, CFG)

    assert total(races) <= CFG["daily_cap"]
    # 減額で収まった時点では、まだどのレースも買い目を持っている
    assert all(r["bets"] for r in races if r["confidence"] == "◎")


def test_drops_lowest_confidence_first() -> None:
    """落とすなら △ から。◎ は守る。"""
    races = [race("◎", 12, 300) for _ in range(8)]   # 28,800円
    races += [race("○", 12, 200) for _ in range(8)]  # 19,200円
    races += [race("△", 12, 100) for _ in range(20)] # 24,000円
    apply_daily_cap(races, CFG)

    assert total(races) <= CFG["daily_cap"]
    kept = {g: sum(1 for r in races if r["confidence"] == g and r["bets"])
            for g in ("◎", "○", "△")}
    assert kept["◎"] == 8, "◎ のレースを落としている"
    assert kept["△"] < 20, "△ を落とさずに収まっているのはおかしい"


def test_drops_least_interesting_race_first() -> None:
    """同じ自信度なら、妙味（人気和）の薄いレースから捨てる。"""
    races = [race("◎", 20, 500) for _ in range(3)]  # 30,000円で既に上限超
    races += [race("△", 10, 100, pop_sum=9) for _ in range(5)]   # 妙味薄
    races += [race("△", 10, 100, pop_sum=30) for _ in range(5)]  # 妙味濃

    apply_daily_cap(races, CFG)
    thin = [r for r in races if r["confidence"] == "△" and r["popularity_sum"] == 9]
    rich = [r for r in races if r["confidence"] == "△" and r["popularity_sum"] == 30]
    assert sum(1 for r in thin if r["bets"]) <= sum(1 for r in rich if r["bets"])


def test_reports_when_anchor_races_alone_exceed_cap() -> None:
    """◎ だけで上限を超えるなら、黙って削らず事実を報告する。

    最小単位まで落としても収まらない規模でないと再現しない。
    上限20,000円・1点100円なので、200点を超える必要がある。
    """
    races = [race("◎", 20, 500) for _ in range(12)]  # 最小単位でも 24,000円
    actions = apply_daily_cap(races, CFG)
    assert any("◎ だけで上限を超えている" in a for a in actions), actions
    assert all(r["bets"] for r in races), "◎ を落としている"
    # 減額だけはされている
    assert all(b["amount"] == CFG["unit"] for r in races for b in r["bets"])
