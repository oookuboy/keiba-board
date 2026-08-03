"""特徴量の抽出。

SKILL.md の各観点を、DB から計算できる 0.0〜1.0 の数値に落とす。engine.py は
ここが返した値に重みを掛けて足すだけにしてある。

**オッズと人気はこのモジュールに入れない。** 能力評価は純粋なデータと状態だけで
行うという SKILL.md の絶対ルールを守るため、Entry の market_ フィールドは
ここから一切参照しない。人気を使うのは confidence.py（穴かどうかの判定）だけ。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from keiba.models import Entry, PastRun, Race
from keiba.store import Store, band_of

log = logging.getLogger(__name__)

# クラスの序列。降級・格上帰りの判定に使う（教訓12）。
CLASS_RANK = {
    "新馬": 0, "未勝利": 0,
    "1勝クラス": 1, "500万下": 1,
    "2勝クラス": 2, "1000万下": 2,
    "3勝クラス": 3, "1600万下": 3,
    "オープン": 4, "OP": 4, "L": 4,
    "G3": 5, "G2": 6, "G1": 7,
}


def class_rank(race_class: str | None, grade: str | None) -> int:
    """レース格を数値化する。分からなければ条件戦の下位とみなす。"""
    if grade and grade in CLASS_RANK:
        return CLASS_RANK[grade]
    if not race_class:
        return 0
    for key, rank in sorted(CLASS_RANK.items(), key=lambda kv: -len(kv[0])):
        if key in race_class:
            return rank
    return 0


@dataclass
class HorseFeatures:
    """1頭ぶんの評価材料。engine が読む唯一の入力。"""

    umaban: int
    horse_id: str
    horse_name: str

    style: str = "先行"
    position_ratio: float = 0.5      # 過去走の平均通過順位率。小さいほど前
    is_lone_front_runner: bool = False
    has_proven_ability: bool = False

    pedigree: float = 0.5
    condition_change: float = 0.0
    pace: float = 0.5
    form: float = 0.0
    jockey: float = 0.5
    condition: float = 0.5

    reasons: list[str] = field(default_factory=list)
    past_runs: list[PastRun] = field(default_factory=list)

    def note(self, text: str) -> None:
        self.reasons.append(text)


# --------------------------------------------------------------- 脚質

def running_style(
    past_runs: list[PastRun], thresholds: dict[str, float]
) -> tuple[str, float]:
    """過去走のコーナー通過順÷頭数の平均から脚質を決める。

    通過順が取れない馬（新馬・データ欠損）は中団扱いにして、展開の加減点を
    実質ゼロにする。分からないものを分かったふりで評価しないため。
    """
    ratios: list[float] = []
    for run in past_runs[:5]:
        if not run.corners or not run.field_size:
            continue
        ratios.append(min(run.corners[0] / run.field_size, 1.0))
    if not ratios:
        return "先行", 0.5

    ratio = sum(ratios) / len(ratios)
    if ratio < thresholds["逃げ"]:
        return "逃げ", ratio
    if ratio < thresholds["先行"]:
        return "先行", ratio
    if ratio < thresholds["差し"]:
        return "差し", ratio
    return "追込", ratio


def project_pace(features: list[HorseFeatures], field_size: int, cfg: dict) -> str:
    """想定ペース。

    教訓7: 少頭数（10頭以下）では逃げ馬が複数いても隊列がすんなり決まるので、
    「複数逃げ馬＝ハイペース」を機械的に適用しない。大井4Rでこれを誤って
    適用し、確逃げ馬を切って3連複を落としている。
    """
    front = [f for f in features if f.style == "逃げ"]

    if field_size <= cfg["small_field_max"]:
        # 少頭数は先行有利が基本シナリオ。差し台頭は次点に格下げする
        return "slow"
    if len(front) >= cfg["multi_front_runners"]:
        return "fast"
    return "slow" if len(front) == 1 else "mid"


def mark_lone_front_runner(features: list[HorseFeatures]) -> None:
    """単騎逃げが確定的な馬に印を付ける（教訓8）。

    逃げ馬が1頭だけで、かつ2番手候補と前へ行く意思に差があるとき、
    その馬は隊列がすんなり決まれば残る。能力評価が低くても切らない。
    """
    front = [f for f in features if f.style == "逃げ"]
    if len(front) != 1:
        return
    lone = front[0]
    contenders = sorted(
        (f for f in features if f is not lone), key=lambda f: f.position_ratio
    )
    # 2番手候補が明確に後ろなら単騎確定とみなす
    if not contenders or contenders[0].position_ratio - lone.position_ratio > 0.10:
        lone.is_lone_front_runner = True
        lone.note("単騎逃げ濃厚。隊列が決まれば残る（教訓8）")


# --------------------------------------------------------------- 血統

def pedigree_fit(
    sire: str | None,
    damsire: str | None,
    race: Race,
    table: dict[str, dict[str, dict]],
    cfg: dict,
) -> tuple[float, str | None]:
    """今走の馬場・距離に対する血統適性を 0〜1 で返す。

    父のサンプルが足りなければ母父に落とし、それも無ければ中立値。
    複勝率をそのまま使うと 0.2 前後に潰れるので、全体の水準（0.3）を
    基準に 0〜1 へ引き伸ばす。
    """
    key = f"{race.surface}:{band_of(race.distance)}"
    neutral = cfg["neutral"]

    for name, label in ((sire, "父"), (damsire if cfg["fallback_to_damsire"] else None, "母父")):
        if not name or name not in table:
            continue
        bucket = table[name].get(key)
        if not bucket or bucket["runs"] < cfg["min_sire_runs"]:
            continue
        # 複勝率 0.30 を中立、0.50 以上を上限として 0〜1 に写す
        score = min(max(bucket["place_rate"] / 0.50, 0.0), 1.0)
        return score, f"{label}{name}の{key}複勝率{bucket['place_rate']:.0%}"

    return neutral, None


def surface_switch_bonus(
    sire: str | None,
    race: Race,
    past_runs: list[PastRun],
    table: dict[str, dict[str, dict]],
    cfg: dict,
) -> tuple[float, str | None]:
    """ダート→芝（およびその逆）替わりを血統から拾う（教訓1）。

    Palace Pier や Noble Mission を名指しでハードコードする代わりに、
    「今走の馬場では父の成績が良いのに、前走までその馬場を使えていなかった」
    という形をデータから検出する。同じ性質の父を全部拾える。

    前走の大敗は「能力不足」ではなく「コース不適性」として扱い、減点しない。
    """
    if not cfg["enabled"] or not past_runs or not sire or sire not in table:
        return 0.0, None

    previous = past_runs[0]
    if previous.surface == race.surface:
        return 0.0, None

    key = f"{race.surface}:{band_of(race.distance)}"
    bucket = table[sire].get(key)
    if not bucket or bucket["runs"] < cfg.get("min_sire_runs", 30):
        return 0.0, None
    if bucket["place_rate"] < cfg["min_place_rate"]:
        return 0.0, None

    strength = min(bucket["place_rate"] / 0.50, 1.0) * cfg["max_bonus"]
    note = (
        f"{previous.surface}→{race.surface}替わり。"
        f"父{sire}は{race.surface}複勝率{bucket['place_rate']:.0%}"
    )
    if previous.finish_pos and previous.finish_pos >= cfg["forgive_finish_worse_than"]:
        note += f"（前走{previous.finish_pos}着は馬場不適と判断）"
    return strength, note


# ------------------------------------------------------- 条件一変7パターン

def condition_changes(
    entry: Entry,
    race: Race,
    past_runs: list[PastRun],
    sire_table: dict,
    store: Store,
    cfg: dict,
    ped_cfg: dict,
    jockey_cfg: dict,
    as_of: date | None = None,
) -> tuple[float, list[str]]:
    """SKILL.md の条件一変7パターンを検出して加点する。"""
    score = 0.0
    notes: list[str] = []

    if not past_runs:
        return 0.0, notes
    previous = past_runs[0]

    # 1. 馬場替わり（血統裏付きのみ）
    bonus, note = surface_switch_bonus(entry.sire, race, past_runs, sire_table, ped_cfg["surface_switch"] | {"min_sire_runs": ped_cfg["min_sire_runs"]})
    if bonus:
        score += bonus * cfg["surface_switch"]
        notes.append(note or "")

    # 2-3. 距離延長／短縮
    delta = race.distance - previous.distance
    if abs(delta) >= cfg["distance_delta"]:
        key = f"{race.surface}:{band_of(race.distance)}"
        fit = sire_table.get(entry.sire or "", {}).get(key)
        if fit and fit["place_rate"] >= ped_cfg["surface_switch"]["min_place_rate"]:
            weight = cfg["distance_stretch"] if delta > 0 else cfg["distance_shorten"]
            score += weight
            notes.append(
                f"{'距離延長' if delta > 0 else '距離短縮'}{abs(delta)}m。"
                f"父{entry.sire}は{key}複勝率{fit['place_rate']:.0%}"
            )

    # 4. コース替わりで当該場の好走歴あり
    same_venue = [r for r in past_runs if r.venue == race.venue]
    if previous.venue != race.venue and same_venue:
        placed = sum(1 for r in same_venue if r.finish_pos and r.finish_pos <= 3)
        if placed:
            score += cfg["course_switch"]
            notes.append(f"{race.venue}替わり。同場で{placed}回の好走歴")

    # 5. 休養明け2走目（叩き良化型）
    if len(past_runs) >= 2:
        gap_prev = (race.race_date - past_runs[0].run_date).days
        gap_before = (past_runs[0].run_date - past_runs[1].run_date).days
        if gap_before >= cfg["layoff_days"] and gap_prev < cfg["recent_days"]:
            score += cfg["second_off_layoff"]
            notes.append(f"休養{gap_before}日明けの2走目。叩き良化を見込む")

    # 6. 騎手強化
    if entry.jockey_id and previous.jockey and entry.jockey != previous.jockey:
        since = (
            as_of - timedelta(days=jockey_cfg["lookback_days"]) if as_of else None
        )
        now_n, now_win, _ = store.jockey_record(
            entry.jockey_id, since=since, before=as_of
        )
        if now_n >= 50 and now_win >= cfg["jockey_winrate_gap"]:
            score += cfg["jockey_upgrade"]
            notes.append(f"{previous.jockey}→{entry.jockey}へ乗り替わり（勝率{now_win:.0%}）")

    # 7. 調教師コメントに前向きな語
    comment = store.conn.execute(
        "SELECT body FROM comments WHERE race_id = ? AND umaban = ?",
        (race.race_id, entry.umaban),
    ).fetchone()
    if comment and comment[0]:
        hits = [k for k in cfg["positive_keywords"] if k in comment[0]]
        if hits:
            score += cfg["trainer_comment"]
            notes.append(f"厩舎コメントに前向きな語（{'・'.join(hits[:2])}）")

    return score, [n for n in notes if n]


# --------------------------------------------------------------- 実績

def form_score(
    entry: Entry, race: Race, past_runs: list[PastRun], cfg: dict
) -> tuple[float, bool, list[str]]:
    """同条件好走歴と格を評価する（教訓9・11・12）。

    戻り値の bool は「能力の証明がある馬」か。長期休み明けでも3着候補から
    外さない判断に使う（教訓11）。
    """
    notes: list[str] = []
    if not past_runs:
        return 0.0, False, notes

    recent = past_runs[: cfg["lookback_runs"]]
    band = band_of(race.distance)
    here = class_rank(race.race_class, race.grade)

    # 同条件（馬場・距離帯・場）での好走
    same = [
        r
        for r in recent
        if r.surface == race.surface
        and band_of(r.distance) == band
        and r.finish_pos
        and r.finish_pos <= cfg["good_finish"]
    ]
    score = min(len(same) / 2.0, 1.0) * cfg["same_condition_place"]
    if same:
        notes.append(f"同条件（{race.surface}{band}）で{len(same)}回の好走歴")

    # 教訓12: 格上で揉まれた降級馬を、下級条件の連続好走より上に置く
    top_class = max((class_rank(None, r.grade) for r in recent), default=0)
    if top_class > here:
        score += cfg["class_drop_bonus"]
        notes.append(f"格上（{top_class}）からの降級・格上帰り")

    # 教訓11: 実績該当馬は休み明けでも切らない
    proven = bool(same) or top_class > here or any(
        r.finish_pos == 1 and class_rank(None, r.grade) >= here for r in recent
    )

    layoff = (race.race_date - past_runs[0].run_date).days
    if layoff >= 180 and not proven:
        score -= cfg["layoff_penalty"]
        notes.append(f"{layoff}日の長期休み明けで実績の裏付けなし")
    elif layoff >= 180:
        notes.append(f"{layoff}日の休み明けだが実績馬（3着候補から外さない）")

    return max(score, 0.0), proven, notes


# --------------------------------------------------------------- 騎手

def jockey_score(
    entry: Entry, race: Race, store: Store, cfg: dict, as_of: date | None = None
) -> tuple[float, str | None]:
    """騎手の当該場成績と、馬とのコンビ成績を合わせる。

    as_of を渡すとその日より前の騎乗だけを見る。バックテストで予想日以降の
    成績を混ぜないため。
    """
    if not entry.jockey_id:
        return 0.5, None

    # 騎手の調子は年単位で動く。全期間の通算ではなく直近だけを見る。
    since = (as_of - timedelta(days=cfg["lookback_days"])) if as_of else None
    _, _, venue_place = store.jockey_record(
        entry.jockey_id, venue=race.venue, since=since, before=as_of
    )
    combo_n, combo_place = store.horse_jockey_record(
        entry.horse_id, entry.jockey_id, before=as_of
    )

    venue_part = min(venue_place / 0.40, 1.0)
    if combo_n >= cfg["min_combo_runs"]:
        combo_part = min(combo_place / 0.50, 1.0)
        score = venue_part * cfg["venue_weight"] + combo_part * cfg["combo_weight"]
        note = f"{entry.jockey}は{race.venue}複勝率{venue_place:.0%}／当馬とは{combo_n}戦"
    else:
        # サンプルが薄いコンビ成績は中立に寄せる
        score = venue_part * cfg["venue_weight"] + 0.5 * cfg["combo_weight"]
        note = f"{entry.jockey}は{race.venue}複勝率{venue_place:.0%}"
    return min(score, 1.0), note


# ----------------------------------------------------------- 馬体重・調教

def condition_score(
    entry: Entry, past_runs: list[PastRun], race: Race, cfg: dict
) -> tuple[float, str | None]:
    """馬体重の増減から状態を見る。

    馬体重は発走約50分前の確定なので、予想時点では前走からの推定しか
    持てないことがある。取れていなければ中立値を返す。
    """
    if entry.body_weight_diff is None:
        return cfg["neutral"], None

    swing = abs(entry.body_weight_diff)
    if swing > cfg["weight_swing_penalty_kg"]:
        return 0.25, f"馬体重{entry.body_weight_diff:+d}kgは変動が大きい"

    layoff = (race.race_date - past_runs[0].run_date).days if past_runs else 0
    if layoff >= 90 and entry.body_weight_diff < 0:
        return min(cfg["neutral"] + cfg["layoff_trim_bonus"], 1.0), (
            f"休み明けで{entry.body_weight_diff:+d}kg。絞れている"
        )
    return cfg["neutral"] + 0.2, f"馬体重{entry.body_weight_diff:+d}kg"


# ------------------------------------------------------------ まとめ

def build_features(
    race: Race,
    entries: list[Entry],
    store: Store,
    sire_table: dict,
    weights: dict,
    as_of: date | None = None,
) -> list[HorseFeatures]:
    """全出走馬ぶんの特徴量を組み立てる。

    全頭を同じ手順で通すのが要点。人気上位だけ丁寧に見るということをしない
    （SKILL.md「全頭個別調査」）。
    """
    as_of = as_of or race.race_date
    features: list[HorseFeatures] = []

    for entry in entries:
        past = store.past_runs_for(entry.horse_id, as_of) if entry.horse_id else []
        f = HorseFeatures(
            umaban=entry.umaban,
            horse_id=entry.horse_id,
            horse_name=entry.horse_name,
            past_runs=past,
        )
        f.style, f.position_ratio = running_style(past, weights["pace"]["style_thresholds"])

        f.pedigree, note = pedigree_fit(
            entry.sire, entry.damsire, race, sire_table, weights["pedigree"]
        )
        if note:
            f.note(note)

        f.condition_change, notes = condition_changes(
            entry, race, past, sire_table, store,
            weights["condition_change"], weights["pedigree"], weights["jockey"], as_of,
        )
        for n in notes:
            f.note(n)

        f.form, f.has_proven_ability, notes = form_score(entry, race, past, weights["form"])
        for n in notes:
            f.note(n)

        f.jockey, note = jockey_score(entry, race, store, weights["jockey"], as_of)
        if note:
            f.note(note)

        f.condition, note = condition_score(entry, past, race, weights["condition"])
        if note:
            f.note(note)

        features.append(f)

    # 展開は他馬との関係で決まるので、全頭そろってから評価する
    mark_lone_front_runner(features)
    pace = project_pace(features, race.field_size or len(entries), weights["pace"])
    bonus = weights["pace"]["bonus"][pace]
    label = {"slow": "スロー", "mid": "ミドル", "fast": "ハイ"}[pace]
    for f in features:
        f.pace = bonus.get(f.style, 0.5)
        f.note(f"想定{label}ペース・{f.style}（展開評価{f.pace:.1f}）")

    return features
