"""スコアリングと印付け。

features.py が返した 0〜1 の各要素に weights.yml の重みを掛けて合算し、
0〜100 のスコアにする。そのうえで SKILL.md の「床」ルールを適用する。

床は2つある。どちらも「評価が低くても切らない」という教訓の実装で、
順位を上げるのではなく最低ラインを保証するだけ。

  教訓8  確逃げ馬は絶対に切らない → lone_front_runner_floor
  教訓11 実績の証明がある馬は休み明けでも3着候補に残す → proven_ability_floor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from keiba.features import HorseFeatures

log = logging.getLogger(__name__)

# 印。SKILL.md の定義に合わせる
MARKS = ("◎", "○", "▲", "☆")


@dataclass
class ScoredHorse:
    umaban: int
    horse_id: str
    horse_name: str
    score: float
    style: str
    mark: str | None = None
    reasons: list[str] = field(default_factory=list)
    is_lone_front_runner: bool = False
    has_proven_ability: bool = False
    # 記録専用。confidence / betting でのみ使い、スコアには一切影響させない
    market_popularity: int | None = None
    market_odds: float | None = None


def score_horse(f: HorseFeatures, weights: dict) -> float:
    """各要素の加重和。条件一変だけは加点型なので 0〜1 に丸めてから掛ける。"""
    w = weights["weights"]
    raw = (
        f.pedigree * w["pedigree"]
        + min(f.condition_change, 1.0) * w["condition_change"]
        + f.pace * w["pace"]
        + min(f.form, 1.0) * w["form"]
        + f.jockey * w["jockey"]
        + f.condition * w["condition"]
    )
    return round(raw, 2)


def apply_floors(horses: list[ScoredHorse], weights: dict) -> None:
    """切ってはいけない馬にスコアの床を与える（ルールベース採点のときだけ）。

    ルールベースのスコアは中央値が 50 前後になるよう組んであるので、床 50〜55 は
    「下位から中位へ引き上げる」程度の効果しか持たない。

    一方、学習モデルのスコアは3着以内確率をそのまま 0〜100 に写したもので、
    中央値は 20 前後（全体の3着内率 21.6%）。同じ床を当てると確逃げ馬が
    ほぼ全レースで1位に躍り出てしまい、較正された確率が壊れる。そのため
    ML スコアのときは床を当てない（run() 側で制御）。

    教訓8・11 の本来の趣旨は「買い目から外さない」ことであり、それは
    betting.build が3着欄への組み込みとして明示的に保証している。順位を
    いじらなくても教訓は守られる。
    """
    front_floor = weights["pace"]["lone_front_runner_floor"]
    proven_floor = weights["form"]["proven_ability_floor"]

    for h in horses:
        if h.is_lone_front_runner and h.score < front_floor:
            h.score = front_floor
            h.reasons.append(f"確逃げのためスコア下限{front_floor:.0f}を適用（教訓8）")
        if h.has_proven_ability and h.score < proven_floor:
            h.score = proven_floor
            h.reasons.append(f"実績馬のためスコア下限{proven_floor:.0f}を適用（教訓11）")


def assign_marks(horses: list[ScoredHorse], weights: dict) -> None:
    """上位から ◎○▲☆ を打つ。

    ◎ は必ず1頭に絞る（SKILL.md）。☆（穴）はスコア上位の中から
    最も人気の無い馬に回す。ここだけは人気を見るが、スコアの計算には
    影響していない — 既に確定したスコア順の中で、どれを穴印にするかの
    振り分けにしか使っていない。
    """
    ranked = sorted(horses, key=lambda h: -h.score)
    limit = min(weights["betting"]["max_marks"], len(ranked))
    if not limit:
        return

    ranked[0].mark = "◎"
    pool = ranked[1:limit]

    # ☆ は上位評価の中で最も人気薄の馬。人気が取れないときは最下位スコア馬
    longshot = None
    with_pop = [h for h in pool if h.market_popularity]
    if with_pop:
        longshot = max(with_pop, key=lambda h: h.market_popularity or 0)
    elif pool:
        longshot = pool[-1]

    for h in pool:
        if h is longshot:
            h.mark = "☆"
    rest = [h for h in pool if h.mark is None]
    for i, h in enumerate(rest):
        h.mark = "○" if i == 0 else "▲"


def run(
    features: list[HorseFeatures],
    entries_by_umaban: dict,
    weights: dict,
    ml_scores: dict[int, float] | None = None,
) -> list[ScoredHorse]:
    """特徴量から採点済みの全頭リストを作る。スコアの降順で返す。

    ml_scores（馬番 → 3着以内確率）を渡すと、能力スコアをそちらに差し替える。
    手置きの重みでは能力順位が市場の半分しか当たらなかったため、順位付けは
    学習モデルに任せ、SKILL.md の教訓は「床」と買い目の組み方として残す。
    確率 0〜1 を 0〜100 のスコアに写すので、床の値はそのまま使える。
    """
    horses = []
    for f in features:
        entry = entries_by_umaban.get(f.umaban)
        if ml_scores is not None and f.umaban in ml_scores:
            score = round(ml_scores[f.umaban] * 100, 2)
        else:
            score = score_horse(f, weights)
        horses.append(
            ScoredHorse(
                umaban=f.umaban,
                horse_id=f.horse_id,
                horse_name=f.horse_name,
                score=score,
                style=f.style,
                reasons=list(f.reasons),
                is_lone_front_runner=f.is_lone_front_runner,
                has_proven_ability=f.has_proven_ability,
                market_popularity=entry.market_popularity if entry else None,
                market_odds=entry.market_odds if entry else None,
            )
        )

    # 床はルールベースのスケール（中央値50前後）を前提にしている。
    # 学習モデルの確率スケール（中央値20前後）に当てると順位が壊れる。
    if ml_scores is None:
        apply_floors(horses, weights)
    horses.sort(key=lambda h: -h.score)
    assign_marks(horses, weights)
    return horses
