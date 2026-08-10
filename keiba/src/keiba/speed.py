"""走破時計から能力指数を作る。

## なぜ要るのか

いまのモデルは着順（何着だったか）しか見ていない。走破タイムは
`results.time_sec` に 99.2% 入っているのに、特徴量として1つも使っていない。

着順は「その日その相手の中で何番目か」でしかない。同じ勝ち方でも、強い相手を
相手に速い時計で勝ったのか、低調な組を相手にゆっくり勝ったのかを区別できない。
**時計はそこを区別する**。競馬の能力指標として最も標準的なのがこれで、市販の
予想ソフトが売っている「指数」も本質的には同じものを作っている。

## 生のタイムは使えない

1600mの1分34秒と1200mの1分08秒は比べられない。同じ距離でも、東京と中山、
芝とダート、良馬場と不良では基準が違う。だから3段階で揃える。

1. **基準タイム** … (競馬場 × 芝ダ × 距離 × クラス) ごとの中央値。
   クラスまで入れるのは、未勝利とオープンで時計の水準が違うため。
2. **馬場差** … その日そのコースが基準よりどれだけ速かったか。同じ開催日の
   全レースが揃って速ければ、それは馬のせいではなく馬場のせい。
3. **指数** … 基準からの差を、その条件のばらつきで割る。こうすると距離や
   コースをまたいでも比較できる数字になる。**正なら基準より速い**。

## 先読みについて

指数は「過去のレースの結果」から作り、**そのレース自身の指数は特徴量に
しない**（それは答えそのもの）。馬ごとの集計は過去走と同じく shift して
から取る。

基準タイムと馬場差も、バックテストでは検証期間より前のデータだけで作れる
ように `before` を受ける。種牡馬適性表と同じ扱い。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# 基準を作るのに最低これだけのサンプルが要る。足りない条件は粗いキーに落とす。
MIN_SAMPLES = 30

# 細かいキーと、サンプルが足りないときに落とす先。
FINE_KEY = ["venue", "surface", "distance", "class_rank"]
COARSE_KEY = ["venue", "surface", "distance"]

# 馬場差を出す単位。同じ日・同じ競馬場・同じ馬場が1つの「コンディション」。
VARIANT_KEY = ["race_date", "venue", "surface"]

# 障害戦は走り方も時計の意味も別物なので、指数の対象から外す。
EXCLUDED_SURFACES = ("障",)


def _baselines(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = df.groupby(keys, observed=True)["time_sec"]
    out = grouped.agg(["median", "std", "size"])
    return out[out["size"] >= MIN_SAMPLES]


def attach_figures(df: pd.DataFrame, before: pd.Timestamp | None = None) -> pd.DataFrame:
    """各行に speed_figure を付ける。

    before を渡すと、基準タイムと馬場差をその日より前のデータだけで作る。
    バックテストで未来を混ぜないため（種牡馬適性表と同じ扱い）。
    """
    out = df.copy()
    out["speed_figure"] = np.nan

    usable = out["time_sec"].notna() & (~out["surface"].isin(EXCLUDED_SURFACES))
    if not usable.any():
        log.warning("走破タイムのある行が無い。指数は作れない")
        return out

    source = out[usable]
    if before is not None:
        source = source[source["race_date"] < before]
        if source.empty:
            log.warning("%s より前に走破タイムが無い。指数は作れない", before.date())
            return out
    log.info("基準タイムを %d 行から作る", len(source))

    fine = _baselines(source, FINE_KEY)
    coarse = _baselines(source, COARSE_KEY)
    log.info("  条件別の基準 %d 件（粗い基準 %d 件）", len(fine), len(coarse))

    work = out[usable].copy()
    joined = work.join(fine, on=FINE_KEY, rsuffix="_fine")
    fallback = work.join(coarse, on=COARSE_KEY, rsuffix="_coarse")
    # 細かいキーでサンプルが足りなければ粗いキーに落とす。両方無ければ諦める。
    median = joined["median"].fillna(fallback["median"])
    std = joined["std"].fillna(fallback["std"])

    raw_gap = median - work["time_sec"]          # 正なら基準より速い

    # 馬場差。その日そのコースが全体として速かったぶんを差し引く。
    # 中央値を使うのは、大敗した馬や出遅れに引きずられないため。
    variant = (
        pd.DataFrame({"gap": raw_gap, **{k: work[k] for k in VARIANT_KEY}})
        .groupby(VARIANT_KEY, observed=True)["gap"]
        .transform("median")
    )

    figure = (raw_gap - variant) / std.replace(0, np.nan)
    out.loc[usable, "speed_figure"] = figure.to_numpy()

    filled = out["speed_figure"].notna().mean()
    log.info("指数を付けた行: %.1f%%", filled * 100)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """指数から、先読みなしの特徴量を組み立てる。

    そのレース自身の指数は使わない（答えそのもの）。すべて shift してから取る。
    """
    out = df.copy()
    horse = out.groupby("horse_id", observed=True)["speed_figure"]

    # 前走の指数。いま何ができる馬かを一番素直に表す
    out["sp_prev"] = horse.shift(1)
    # 直近3走の平均。1走の凡走に振り回されないための平滑
    out["sp_r3"] = horse.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    # これまでの最高。能力の天井
    out["sp_best"] = horse.transform(lambda s: s.shift(1).expanding().max())
    # 今回と同じ距離帯での最高。距離適性込みの天井
    out["sp_band_best"] = out.groupby(["horse_id", "band"], observed=True)[
        "speed_figure"
    ].transform(lambda s: s.shift(1).expanding().max())
    # 天井からどれだけ落ちているか。仕上がり途上・下降を捉える
    out["sp_gap_from_best"] = out["sp_prev"] - out["sp_best"]

    # レース内での相対化。絶対値より「この相手より速いか」が効く
    for column in ("sp_r3", "sp_best"):
        grouped = out.groupby("race_id", observed=True)[column]
        out[f"{column}_rank"] = grouped.rank(pct=True)
        out[f"{column}_z"] = (out[column] - grouped.transform("mean")) / grouped.transform(
            "std"
        ).replace(0, np.nan)
    return out


SPEED_FEATURES = [
    "sp_prev", "sp_r3", "sp_best", "sp_band_best", "sp_gap_from_best",
    "sp_r3_rank", "sp_r3_z", "sp_best_rank", "sp_best_z",
]


def coverage(df: pd.DataFrame) -> float:
    """指数の付いた行の割合。学習の前に必ず見る。"""
    if df.empty:
        return 0.0
    return float(df["sp_r3"].notna().mean())
