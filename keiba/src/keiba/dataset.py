"""学習用データセットの構築。

1行 = 1頭の1出走。目的変数は「3着以内に入ったか」。

**先読みの排除がこのモジュールの全て。** 過去成績から作る特徴量は、必ず
「その出走より前の走りだけ」で計算する。実装は日付順に並べたうえで
groupby().shift(1) を挟んでから累積集計する形に統一してある。shift を
忘れた瞬間に「その日の結果」が特徴量に混ざり、バックテストの回収率が
嘘をつく。

**オッズと人気は入れない。** 能力評価は純粋なデータと状態だけで行うという
SKILL.md の絶対ルールをここでも守る。人気を使うのは confidence.py だけ。
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from keiba.features import CLASS_RANK
from keiba.store import DISTANCE_BANDS, Store

log = logging.getLogger(__name__)

# 目的変数。3連複を狙うので「3着以内」を当てにいく。
TARGET = "top3"

CATEGORICAL = ["surface", "venue", "going", "sex", "direction"]

# オッズ由来の列がうっかり紛れ込んでいないかを確かめるための番人
FORBIDDEN = ("odds", "popularity", "market", "ninki")


def _band(distance: pd.Series) -> pd.Series:
    conditions = [(distance >= lo) & (distance <= hi) for _, lo, hi in DISTANCE_BANDS]
    return pd.Series(
        np.select(conditions, [name for name, _, _ in DISTANCE_BANDS], default="long"),
        index=distance.index,
    )


def _class_rank(race_class: pd.Series, grade: pd.Series) -> pd.Series:
    """features.class_rank をベクトル化したもの。序列の定義は共有する。"""
    rank = pd.Series(0, index=race_class.index, dtype="int64")
    for key, value in sorted(CLASS_RANK.items(), key=lambda kv: len(kv[0])):
        rank = rank.mask(race_class.fillna("").str.contains(key, regex=False), value)
    for key, value in CLASS_RANK.items():
        rank = rank.mask(grade.fillna("") == key, value)
    return rank


def load_frame(store: Store, include_unfinished: bool = False) -> pd.DataFrame:
    """races × entries × results を1枚の表にする。

    既定では結果のある出走だけを返す。学習はこれでよい。

    include_unfinished を立てると、まだ走っていないレースも含める（結果列は
    欠損になる）。**これから予想するレースを特徴量に載せるために要る**。
    内側 JOIN のままだと、出馬表は取れているのに results がまだ無いという
    理由だけで今週のレースが1行も出てこない。

    先読みにはならない。特徴量はすべて groupby().shift(1) で「その行より前」
    だけから作るので、結果の無い行は自分の欠損値を誰にも渡さない。
    """
    join = "LEFT JOIN" if include_unfinished else "JOIN"
    sql = f"""
    SELECT r.race_id, r.race_date, r.venue, r.surface, r.distance, r.direction,
           r.going, r.grade, r.race_class, r.field_size, r.race_no,
           e.umaban, e.waku, e.horse_id, e.horse_name, e.sex, e.age,
           e.weight_carried, e.jockey_id, e.trainer_id, e.body_weight,
           e.body_weight_diff, e.sire, e.damsire,
           e.market_popularity, e.market_odds,
           res.finish_pos, res.time_sec, res.corners, res.last3f
    FROM entries e
    JOIN races r     ON r.race_id = e.race_id
    {join} results res ON res.race_id = e.race_id AND res.umaban = e.umaban
    WHERE e.scratched = 0
    """
    df = pd.read_sql_query(sql, store.conn)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "race_id", "umaban"]).reset_index(drop=True)

    # 通過順は JSON 文字列。1コーナーの位置取りだけ使う。
    # まだ走っていない行では NaN で来る。NaN は真なので `s or "[]"` を
    # すり抜けて json.loads に渡ってしまう（実際そこで落ちた）。
    def _first_corner(raw) -> float:
        if not isinstance(raw, str) or not raw:
            return np.nan
        try:
            values = json.loads(raw)
        except ValueError:
            return np.nan
        return values[0] if values else np.nan

    first_corner = df["corners"].apply(_first_corner)
    df["corner_ratio"] = first_corner / df["field_size"]
    df["finish_ratio"] = df["finish_pos"] / df["field_size"]
    # 走り終わったか。TARGET は欠損を False に潰すので、これが無いと
    # 「まだ走っていない」と「3着以内に入らなかった」が同じ 0 になる。
    # 学習側はこの列で落とす。
    df["finished"] = df["finish_pos"].notna()
    df[TARGET] = (df["finish_pos"] <= 3).astype("int8")
    df["won"] = (df["finish_pos"] == 1).astype("int8")
    df["placed"] = df[TARGET]
    df["band"] = _band(df["distance"])
    df["class_rank"] = _class_rank(df["race_class"], df["grade"])
    return df


def _prior_mean(df: pd.DataFrame, keys: list[str], column: str) -> pd.Series:
    """キーごとの「その行より前」の平均。

    shift(1) を挟んでから expanding().mean() を取るので、自分自身の結果は
    絶対に入らない。ここが先読み防止の要。
    """
    grouped = df.groupby(keys, observed=True)[column]
    return grouped.transform(lambda s: s.shift(1).expanding().mean())


def _prior_count(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    return df.groupby(keys, observed=True).cumcount()


def _prior_roll(df: pd.DataFrame, keys: list[str], column: str, window: int) -> pd.Series:
    """直近 window 走の平均（自分より前のみ）。"""
    grouped = df.groupby(keys, observed=True)[column]
    return grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )


def build_features(
    df: pd.DataFrame, speed_before=None, workouts: pd.DataFrame | None = None
) -> pd.DataFrame:
    """先読みなしの特徴量を組み立てる。

    FEATURE_COLUMNS に載っているものは、この関数を通せば全部そろう。
    タイム指数もここで作る（別の場所で作ると、呼び出し側によって列が
    あったり無かったりする）。

    workouts を渡さなければ調教の列は欠損のまま作る。調教は netkeiba の
    有料データで、手元に無い環境（テストなど）でも同じ列がそろっていないと
    「呼び出し側によって列が違う」に逆戻りする。
    """
    from keiba import speed, workout_features

    out = df.copy()

    # --- 馬の履歴 ------------------------------------------------------
    horse = ["horse_id"]
    out["h_runs"] = _prior_count(out, horse)
    out["h_place_rate"] = _prior_mean(out, horse, "placed")
    out["h_win_rate"] = _prior_mean(out, horse, "won")
    out["h_finish_ratio_avg"] = _prior_mean(out, horse, "finish_ratio")
    out["h_finish_ratio_r3"] = _prior_roll(out, horse, "finish_ratio", 3)
    out["h_corner_ratio_r5"] = _prior_roll(out, horse, "corner_ratio", 5)
    out["h_last3f_r3"] = _prior_roll(out, horse, "last3f", 3)
    out["h_best_class"] = out.groupby(horse, observed=True)["class_rank"].transform(
        lambda s: s.shift(1).expanding().max()
    )
    out["h_prev_finish_ratio"] = out.groupby(horse, observed=True)[
        "finish_ratio"
    ].shift(1)
    out["h_prev_distance"] = out.groupby(horse, observed=True)["distance"].shift(1)
    out["h_prev_surface"] = out.groupby(horse, observed=True)["surface"].shift(1)
    out["h_prev_class"] = out.groupby(horse, observed=True)["class_rank"].shift(1)
    prev_date = out.groupby(horse, observed=True)["race_date"].shift(1)
    out["h_days_since"] = (out["race_date"] - prev_date).dt.days

    # --- 条件一変（SKILL.md の7パターンを数値にしたもの） ---------------
    out["c_distance_delta"] = out["distance"] - out["h_prev_distance"]
    out["c_surface_switch"] = (
        out["h_prev_surface"].notna() & (out["h_prev_surface"] != out["surface"])
    ).astype("int8")
    out["c_class_delta"] = out["class_rank"] - out["h_prev_class"]
    # 教訓12: 格上帰り・降級
    out["c_class_drop"] = (out["h_best_class"] - out["class_rank"]).clip(lower=0)
    # 教訓: 休養明け2走目（叩き良化）
    prev_gap = out.groupby(horse, observed=True)["h_days_since"].shift(1)
    out["c_second_off_layoff"] = (
        (prev_gap >= 90) & (out["h_days_since"] < 60)
    ).astype("int8")

    # --- 同条件（馬場×距離帯）の実績 -----------------------------------
    out["hs_place_rate"] = _prior_mean(out, ["horse_id", "surface"], "placed")
    out["hsb_place_rate"] = _prior_mean(out, ["horse_id", "surface", "band"], "placed")
    out["hv_place_rate"] = _prior_mean(out, ["horse_id", "venue"], "placed")

    # --- 騎手 ----------------------------------------------------------
    out["j_runs"] = _prior_count(out, ["jockey_id"])
    out["j_place_rate"] = _prior_mean(out, ["jockey_id"], "placed")
    out["j_win_rate"] = _prior_mean(out, ["jockey_id"], "won")
    out["jv_place_rate"] = _prior_mean(out, ["jockey_id", "venue"], "placed")
    out["hj_place_rate"] = _prior_mean(out, ["horse_id", "jockey_id"], "placed")
    out["hj_runs"] = _prior_count(out, ["horse_id", "jockey_id"])
    prev_jockey = out.groupby(horse, observed=True)["jockey_id"].shift(1)
    out["c_jockey_change"] = (
        prev_jockey.notna() & (prev_jockey != out["jockey_id"])
    ).astype("int8")

    # --- 調教師 --------------------------------------------------------
    out["t_place_rate"] = _prior_mean(out, ["trainer_id"], "placed")

    # --- 血統（教訓1の土台） --------------------------------------------
    # 父×馬場×距離帯の複勝率。全部「その行より前」だけで作る
    out["s_place_rate"] = _prior_mean(out, ["sire"], "placed")
    out["ssb_place_rate"] = _prior_mean(out, ["sire", "surface", "band"], "placed")
    out["ss_place_rate"] = _prior_mean(out, ["sire", "surface"], "placed")
    out["ds_place_rate"] = _prior_mean(out, ["damsire", "surface"], "placed")
    # 教訓1: 馬場替わり × 父が今走の馬場で走る、を掛け合わせた項
    out["c_switch_x_sire"] = out["c_surface_switch"] * out["ss_place_rate"].fillna(0)

    # --- 当日の状態 ----------------------------------------------------
    out["body_weight_abs_diff"] = out["body_weight_diff"].abs()
    out["draw_ratio"] = out["umaban"] / out["field_size"]

    # --- レース内での相対化 ---------------------------------------------
    # 同じレースの他馬と比べてどうか。絶対値より効く
    for col in ["h_place_rate", "h_finish_ratio_r3", "j_place_rate", "ssb_place_rate"]:
        grouped = out.groupby("race_id", observed=True)[col]
        out[f"{col}_rank"] = grouped.rank(pct=True)
        out[f"{col}_z"] = (out[col] - grouped.transform("mean")) / grouped.transform(
            "std"
        ).replace(0, np.nan)

    # --- 走破時計 ------------------------------------------------------
    out = speed.attach_figures(out, before=speed_before)
    out = speed.build_features(out)

    # --- 調教（netkeiba 有料） -------------------------------------------
    # attach は先に全列を欠損で作ってから埋めるので、渡すものが空でも
    # 列はそろう。先読みは merge_asof(allow_exact_matches=False) で断つ。
    out = workout_features.attach(
        out, pd.DataFrame() if workouts is None else workouts
    )

    return out


# 走破時計と調教の特徴量。定義はそれぞれのモジュール側に置いてある
# （作り方の説明が長いため）。
from keiba.speed import SPEED_FEATURES  # noqa: E402
from keiba.workout_features import WORKOUT_FEATURES  # noqa: E402

FEATURE_COLUMNS = [
    # 馬の履歴
    "h_runs", "h_place_rate", "h_win_rate", "h_finish_ratio_avg",
    "h_finish_ratio_r3", "h_corner_ratio_r5", "h_last3f_r3", "h_best_class",
    "h_prev_finish_ratio", "h_days_since",
    # 条件一変
    "c_distance_delta", "c_surface_switch", "c_class_delta", "c_class_drop",
    "c_second_off_layoff", "c_jockey_change", "c_switch_x_sire",
    # 同条件実績
    "hs_place_rate", "hsb_place_rate", "hv_place_rate",
    # 騎手・調教師
    "j_runs", "j_place_rate", "j_win_rate", "jv_place_rate",
    "hj_place_rate", "hj_runs", "t_place_rate",
    # 血統
    "s_place_rate", "ssb_place_rate", "ss_place_rate", "ds_place_rate",
    # 当日
    "age", "weight_carried", "body_weight", "body_weight_diff",
    "body_weight_abs_diff", "waku", "umaban", "draw_ratio",
    # レース条件
    "distance", "field_size", "class_rank", "race_no",
    # レース内相対
    "h_place_rate_rank", "h_place_rate_z",
    "h_finish_ratio_r3_rank", "h_finish_ratio_r3_z",
    "j_place_rate_rank", "j_place_rate_z",
    "ssb_place_rate_rank", "ssb_place_rate_z",
    # 走破時計（馬場差で補正した指数）。着順だけでは「強い相手に速い時計で
    # 勝った」と「低調な組をゆっくり勝った」を区別できない。
    *SPEED_FEATURES,
    # 調教（netkeiba 有料）。指数を入れたうえで測って AUC 0.75422 → 0.75717。
    # 効き幅は小さいが、狙っている人気薄（6番人気以下・5,658行）では
    # +0.0083 と、決めておいた閾値（2,000行・0.005）を両方超えた。
    *WORKOUT_FEATURES,
    # カテゴリ
    *CATEGORICAL,
]


def assert_no_market_leakage(columns: list[str]) -> None:
    """オッズ由来の列が特徴量に紛れていないことを確かめる。

    ここが漏れると「人気が高い馬は勝つ」を学習してしまい、市場を上回る
    余地がなくなる。静かに起きるので機械で止める。
    """
    leaked = [
        c for c in columns if any(bad in c.lower() for bad in FORBIDDEN)
    ]
    if leaked:
        raise AssertionError(f"オッズ由来の特徴量が混入している: {leaked}")


def prepare(
    store: Store, speed_before=None, include_unfinished: bool = False
) -> pd.DataFrame:
    """DB から学習可能な表を作る。

    speed_before を渡すと、タイム指数の基準（基準タイム・馬場差）をその日より
    前のデータだけで作る。バックテストで未来の時計を混ぜないため。本番の
    学習・予想では未来のデータが存在しないので None でよい。

    include_unfinished は、これから走るレースにもモデルのスコアを付けるため
    のもの。学習には使わない（ml.train が finished で落とす）。
    """
    log.info("DB を読み込み中…")
    df = load_frame(store, include_unfinished=include_unfinished)
    log.info("  %d 行 / %d レース", len(df), df["race_id"].nunique())
    if include_unfinished:
        pending = int((~df["finished"]).sum())
        log.info("  うち未走 %d 行（予想対象。学習には使わない）", pending)

    from keiba import workout_features

    workouts = workout_features.load_workouts(store)
    if workouts.empty:
        # 有料データが手元に無いだけで学習も予想も動く（列は欠損で作られる）。
        # ただし黙って欠損のまま動くと「集めたのに使われていない」に戻るので、
        # 分かる形で残す。
        log.warning(
            "調教が1件も入っていない。調教の列は欠損のまま進む"
            "（artifact から raw/workouts.jsonl.gz を戻すこと）"
        )
    else:
        log.info("調教 %d本を読み込み", len(workouts))

    log.info("特徴量を構築中（先読みなし）…")
    df = build_features(df, speed_before=speed_before, workouts=workouts)

    assert_no_market_leakage(FEATURE_COLUMNS)
    for column in CATEGORICAL:
        df[column] = df[column].astype("category")
    return df
