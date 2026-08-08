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


def load_frame(store: Store) -> pd.DataFrame:
    """races × entries × results を1枚の表にする。"""
    sql = """
    SELECT r.race_id, r.race_date, r.venue, r.surface, r.distance, r.direction,
           r.going, r.grade, r.race_class, r.field_size, r.race_no,
           e.umaban, e.waku, e.horse_id, e.horse_name, e.sex, e.age,
           e.weight_carried, e.jockey_id, e.trainer_id, e.body_weight,
           e.body_weight_diff, e.sire, e.damsire,
           e.market_popularity, e.market_odds,
           res.finish_pos, res.time_sec, res.corners, res.last3f
    FROM entries e
    JOIN races r     ON r.race_id = e.race_id
    JOIN results res ON res.race_id = e.race_id AND res.umaban = e.umaban
    WHERE e.scratched = 0
    """
    df = pd.read_sql_query(sql, store.conn)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "race_id", "umaban"]).reset_index(drop=True)

    # 通過順は JSON 文字列。1コーナーの位置取りだけ使う
    first_corner = df["corners"].apply(
        lambda s: (lambda v: v[0] if v else np.nan)(__import__("json").loads(s or "[]"))
    )
    df["corner_ratio"] = first_corner / df["field_size"]
    df["finish_ratio"] = df["finish_pos"] / df["field_size"]
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """先読みなしの特徴量を組み立てる。"""
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

    # --- 乗り替わりの「質」 ---------------------------------------------
    # c_jockey_change は替わったかどうかの 0/1 でしかない。上手い騎手へ
    # 乗り替わったのか下手な方へ替わったのかを区別できていなかった。
    # 前走騎手の複勝率との差を持たせる（教訓の「騎手強化」の数値版）。
    prev_jockey_skill = out.groupby(horse, observed=True)["j_place_rate"].shift(1)
    out["c_jockey_upgrade"] = out["j_place_rate"] - prev_jockey_skill

    # --- 斤量の増減 ------------------------------------------------------
    # weight_carried の絶対値しか見ておらず、前走から背負い増したのか
    # 減ったのかを拾えていなかった。ハンデ戦や昇級で効く。
    out["c_weight_delta"] = out["weight_carried"] - out.groupby(
        horse, observed=True
    )["weight_carried"].shift(1)

    # --- 間隔の非線形性 --------------------------------------------------
    # h_days_since は生の日数なので、「詰めすぎ」「叩き頃」「長期休養」の
    # 差が線形に見えてしまう。区分として持たせて木に判断させる。
    days = out["h_days_since"]
    out["c_short_rest"] = (days < 15).astype("int8")       # 連闘・中1週
    out["c_fresh"] = days.between(21, 63).astype("int8")   # 中3週〜2か月
    out["c_layoff"] = (days >= 180).astype("int8")         # 半年以上

    # --- コース形状との相性 ----------------------------------------------
    # hv_place_rate（場）はあるが、回り・距離帯まで込みの実績が無かった。
    # 「右回りの中距離だけ走る」型を拾うため。
    out["hd_place_rate"] = _prior_mean(
        out, ["horse_id", "direction", "band"], "placed"
    )
    out["hvd_place_rate"] = _prior_mean(
        out, ["horse_id", "venue", "distance"], "placed"
    )

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

    return out


FEATURE_COLUMNS = [
    # 馬の履歴
    "h_runs", "h_place_rate", "h_win_rate", "h_finish_ratio_avg",
    "h_finish_ratio_r3", "h_corner_ratio_r5", "h_last3f_r3", "h_best_class",
    "h_prev_finish_ratio", "h_days_since",
    # 条件一変
    "c_distance_delta", "c_surface_switch", "c_class_delta", "c_class_drop",
    "c_second_off_layoff", "c_jockey_change", "c_switch_x_sire",
    # 乗り替わりの質・斤量の増減・間隔の区分
    "c_jockey_upgrade", "c_weight_delta",
    "c_short_rest", "c_fresh", "c_layoff",
    # 同条件実績（場・回り・距離まで込み）
    "hs_place_rate", "hsb_place_rate", "hv_place_rate",
    "hd_place_rate", "hvd_place_rate",
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


def prepare(store: Store) -> pd.DataFrame:
    """DB から学習可能な表を作る。"""
    log.info("DB を読み込み中…")
    df = load_frame(store)
    log.info("  %d 行 / %d レース", len(df), df["race_id"].nunique())

    log.info("特徴量を構築中（先読みなし）…")
    df = build_features(df)

    assert_no_market_leakage(FEATURE_COLUMNS)
    for column in CATEGORICAL:
        df[column] = df[column].astype("category")
    return df
