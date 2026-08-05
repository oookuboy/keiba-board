"""自信度 ◎○△× の判定。

ここが、オッズを使ってよい場所のひとつ（もう一方は betting.py）。能力評価は
features/engine が人気を一切見ずに済ませており、この段階でスコアはもう確定して
いる。人気は「その並びが世間とどれだけズレているか」を測るためだけに使う。

閾値は手置きではなく実測に基づく。2026年の1,782レースを、能力上位3頭の
人気順位の和で層別して回収率を測ったところ、次のようになった。

    〜8（堅い）  76.9% ／ 9-13  89.9% ／ 14-19  63.3%
    20-27  18.2% ／ 28+（大穴）  0.0%

穴に寄るほど回収率が単調に下がり、100%を超える区分は存在しない（3連複の
控除率は27.5%なので、でたらめに買っても約72.5%）。したがって:

  ◎ ○   本線。最も回収率の高い 9-13 の帯。厚く買う
  △     穴枠。14以上。当たれば大きいが期待値は本線に劣ると測定済みなので、
        別予算・最小単位で薄く買う
  ×     見送り。8以下（堅すぎる）か、想定配当が薄いレース
  ?     オッズ未確定。判断できていない状態であり、× とは意味が違う
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from keiba.engine import ScoredHorse

log = logging.getLogger(__name__)

SKIP = "×"
LONGSHOT = "△"
# オッズがまだ無くて妙味を判定できない状態。× とは意味が違う。
# × は「堅いから買わない」という判断だが、こちらは判断がまだできていない。
# 前日夜の予想は必ずここに落ちる（単勝オッズは当日にならないと出ない）。
PENDING = "?"


@dataclass
class Confidence:
    grade: str              # ◎ ○ △ × ?
    popularity_sum: int     # 能力上位3頭の人気順位の和。大きいほど穴
    separation: float       # スコア1位と4位の差。能力序列の明確さ
    expected_odds: float | None
    reason: str

    @property
    def should_bet(self) -> bool:
        return self.grade not in (SKIP, PENDING)

    @property
    def is_pending(self) -> bool:
        """能力評価は出ているが、オッズ待ちで自信度を決められない。"""
        return self.grade == PENDING

    @property
    def is_longshot(self) -> bool:
        """穴枠。本線とは別予算で薄く買う。"""
        return self.grade == LONGSHOT


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
    # 人気が無いと妙味を判定できない。ただしこれは「堅いから見送る」のとは
    # 別物なので × にはしない。能力評価と印はそのまま使えるので、
    # オッズが出てから再評価する前提で PENDING を返す。
    if len(pops) < 3:
        return Confidence(
            PENDING, 0, 0.0, None,
            "オッズ未確定のため妙味を判定できない（当日の再評価で確定する）",
        )

    pop_sum = sum(pops)
    separation = round(ranked[0].score - ranked[3].score, 2) if len(ranked) > 3 else 0.0
    expected = estimate_trio_odds(top3)
    low, high = cfg["main_band"]

    # --- 堅すぎる。実測でも本線の帯に劣る ---------------------------------
    if pop_sum < low:
        return Confidence(
            SKIP, pop_sum, separation, expected,
            f"能力上位3頭の人気和{pop_sum}。上位人気で堅く収まる形のため見送る",
        )

    # --- 穴枠。当たれば大きいが期待値は本線に劣ると測定済み ---------------
    if pop_sum >= cfg["longshot_min"]:
        return Confidence(
            LONGSHOT, pop_sum, separation, expected,
            f"能力上位3頭の人気和{pop_sum}。穴枠として別予算で薄く買う"
            f"（この帯の実測回収率は本線より低い）",
        )

    # --- 本線。実測で最も回収率の高い帯 -----------------------------------
    reason = f"能力上位3頭の人気和{pop_sum}（本線の帯 {low}-{high}）"

    if expected is not None and expected < cfg["min_expected_odds"]:
        return Confidence(
            SKIP, pop_sum, separation, expected,
            f"{reason}だが想定3連複配当{expected}倍は妙味不足のため見送る",
        )
    if expected is not None:
        reason += f"・想定3連複{expected}倍"

    # ◎ には能力の分離も要る。上位が横並びなら軸を立てられない
    if separation < cfg["min_score_separation"]:
        return Confidence(
            "○", pop_sum, separation, expected,
            f"{reason}（1位と4位のスコア差{separation}が小さく○）",
        )
    return Confidence("◎", pop_sum, separation, expected, reason)
