"""自信度 ◎○△× の判定。

「固い予想はいらない、狙いは穴・大穴」という要求を機械的に実装する層。

ここが、オッズを使ってよい唯一の場所のひとつ（もう一方は betting.py）。
能力評価は features/engine が人気を一切見ずに済ませており、この段階では
スコアはもう確定している。人気は「その並びが世間とどれだけズレているか」を
測るためだけに使う。

  能力上位3頭の人気順位の和が大きい  → 世間とズレている → 穴 → 買う
  人気和が小さい                     → 上位人気で堅く収まる → × → 見送る
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from keiba.engine import ScoredHorse

log = logging.getLogger(__name__)

SKIP = "×"


@dataclass
class Confidence:
    grade: str              # ◎ ○ △ ×
    popularity_sum: int     # 能力上位3頭の人気順位の和。大きいほど穴
    separation: float       # スコア1位と4位の差。能力序列の明確さ
    expected_odds: float | None
    reason: str

    @property
    def should_bet(self) -> bool:
        return self.grade != SKIP


def estimate_trio_odds(top3: list[ScoredHorse]) -> float | None:
    """上位3頭で決まった場合の3連複配当のおおまかな見積り。

    単勝オッズから各馬の勝率を出し、その積の逆数を配当の目安にする。
    実際の3連複オッズとは乖離するが、「堅いか荒れるか」の閾値判定には足りる。
    控除率ぶんを 0.75 で寄せている。
    """
    odds = [h.market_odds for h in top3 if h.market_odds]
    if len(odds) < 3:
        return None
    probability = 1.0
    for o in odds:
        probability *= min(0.75 / o, 0.95)
    # 3頭の順不同なので組み合わせ数 3! を掛けて確率を膨らませる
    probability = min(probability * 6, 0.99)
    return round(1.0 / probability, 1) if probability > 0 else None


def grade(horses: list[ScoredHorse], weights: dict) -> Confidence:
    """レース単位の自信度を決める。"""
    cfg = weights["confidence"]
    ranked = sorted(horses, key=lambda h: -h.score)
    top3 = ranked[:3]

    pops = [h.market_popularity for h in top3 if h.market_popularity]
    # 人気が取れていないと穴かどうかを判定できない。買わない側に倒す。
    if len(pops) < 3:
        return Confidence(SKIP, 0, 0.0, None, "人気が取得できず妙味を判定できない")

    pop_sum = sum(pops)
    separation = round(ranked[0].score - ranked[3].score, 2) if len(ranked) > 3 else 0.0
    expected = estimate_trio_odds(top3)

    thresholds = cfg["popularity_sum"]
    if pop_sum >= thresholds["◎"]:
        result = "◎"
    elif pop_sum >= thresholds["○"]:
        result = "○"
    elif pop_sum >= thresholds["△"]:
        result = "△"
    else:
        return Confidence(
            SKIP, pop_sum, separation, expected,
            f"能力上位3頭の人気和{pop_sum}。上位人気で堅く収まる形のため見送る",
        )

    reason = f"能力上位3頭の人気和{pop_sum}"

    # ◎ には能力の分離も要る。上位が横並びなら穴でも勝負にならない
    if result == "◎" and separation < cfg["min_score_separation"]:
        result = "○"
        reason += f"（ただし1位と4位のスコア差{separation}が小さく○へ降格）"

    # 想定配当が薄いなら、人気和が大きくても買う意味がない
    if expected is not None and expected < cfg["min_expected_odds"]:
        return Confidence(
            SKIP, pop_sum, separation, expected,
            f"{reason}だが想定3連複配当{expected}倍は妙味不足のため見送る",
        )

    if expected is not None:
        reason += f"・想定3連複{expected}倍"
    return Confidence(result, pop_sum, separation, expected, reason)
