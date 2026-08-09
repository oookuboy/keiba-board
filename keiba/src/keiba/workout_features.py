"""調教タイムから特徴量を作る。

netkeiba の有料プラン限定データ。horse_workouts に馬・日付単位で入っている
ものを、レース単位の表へ「そのレースより前の調教だけ」で畳み込む。

## なぜ絶対タイムをそのまま使わないか

調教タイムはコースで意味が変わる。坂路（800m）は4Fしか計らないし、
ウッドチップと芝でも基準が違う。他馬と絶対値で比べると、速い馬ではなく
「速いコースで追われた馬」を選んでしまう。

効くのは**その馬自身の過去との差**。同じ馬が同じ坂路でいつもより1秒速い、
というのが「状態が上がっている」の中身で、これは人気に織り込まれにくい。
穴を狙うという要求とも噛み合う。

## 先読みについて

workout_date < race_date で切る。当日の追い切りは存在しない（最終追いは
水木）ので等号は含めない。自分の過去平均も同様に shift して作る。
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from keiba.store import Store

log = logging.getLogger(__name__)

# 調教タイムの5区間のうち、どれを使うか。
# 坂路は先頭2つ（6F・5F）が常に空なので、全馬に入っている 3F と 1F を使う。
IDX_3F = 3
IDX_1F = 4

# 「直近の調教」として認める日数。これより前しか無い馬は状態が読めないので
# 欠損のままにする。無理に埋めると「調教が無い」と「普通だった」が混ざる。
RECENT_DAYS = 21

WORKOUT_FEATURES = [
    "w_days_since",
    "w_count_21d",
    "w_last_3f",
    "w_last_1f",
    "w_3f_vs_self",
    "w_1f_vs_self",
    "w_rank_score",
    "w_hard",
    "w_is_slope",
]

# 追い切り評価。netkeiba が S/A/B/C で付けている。順序があるので数値にする。
RANK_SCORE = {"S": 4.0, "A": 3.0, "B": 2.0, "C": 1.0}

# 脚色。強く追ったかどうか。「馬也（馬なり）で好時計」が本来いちばん良い形で、
# タイムと組み合わせて初めて意味を持つので、タイムとは別の列にしておく。
HARD_LEGS = ("一杯", "強め")


def load_workouts(store: Store) -> pd.DataFrame:
    """horse_workouts を、特徴量にしやすい形へ展開する。"""
    df = pd.read_sql_query(
        "SELECT horse_id, workout_date, course, times, rank, leg FROM horse_workouts",
        store.conn,
    )
    if df.empty:
        return df.assign(
            workout_date=pd.to_datetime(pd.Series([], dtype="object")),
            t3f=np.nan, t1f=np.nan, rank_score=np.nan, hard=np.nan, is_slope=np.nan,
        )

    df["workout_date"] = pd.to_datetime(df["workout_date"])

    def slot(raw: str, index: int) -> float:
        try:
            values = json.loads(raw or "[]")
        except (TypeError, ValueError):
            return np.nan
        if len(values) <= index or values[index] is None:
            return np.nan
        return float(values[index])

    df["t3f"] = df["times"].apply(lambda s: slot(s, IDX_3F))
    df["t1f"] = df["times"].apply(lambda s: slot(s, IDX_1F))
    df["rank_score"] = df["rank"].map(RANK_SCORE)
    df["hard"] = df["leg"].fillna("").apply(
        lambda s: 1.0 if any(w in s for w in HARD_LEGS) else 0.0
    )
    df["is_slope"] = df["course"].fillna("").str.contains("坂").astype(float)

    df = df.sort_values(["horse_id", "workout_date"]).reset_index(drop=True)

    # その馬の「それまでの」平均。自分自身を含めないので shift(1) を挟む。
    # コースごとに基準が違うため、馬とコースの組で取る。
    by_course = ["horse_id", "course"]
    for column in ("t3f", "t1f"):
        df[f"{column}_self"] = df.groupby(by_course, observed=True)[column].transform(
            lambda s: s.shift(1).expanding().mean()
        )
    return df


def attach(entries: pd.DataFrame, workouts: pd.DataFrame) -> pd.DataFrame:
    """レースの表に、そのレース直前の調教を貼り付ける。

    entries は race_date と horse_id を持つ表。merge_asof で「レース日より
    前の最後の調教」を引く。allow_exact_matches=False にしてあるので、
    同日のものは入らない。
    """
    out = entries.copy()
    for column in WORKOUT_FEATURES:
        out[column] = np.nan
    if workouts.empty:
        return out

    left = out[["race_date", "horse_id"]].copy()
    left["_row"] = np.arange(len(left))
    left = left.sort_values("race_date")

    right = workouts.sort_values("workout_date")
    merged = pd.merge_asof(
        left,
        right[
            [
                "workout_date", "horse_id", "t3f", "t1f",
                "t3f_self", "t1f_self", "rank_score", "hard", "is_slope",
            ]
        ],
        left_on="race_date",
        right_on="workout_date",
        by="horse_id",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("_row").sort_index()

    days = (merged["race_date"] - merged["workout_date"]).dt.days
    # 古すぎる調教は「直前の状態」ではない。埋めずに欠損のままにする。
    stale = days > RECENT_DAYS

    out["w_days_since"] = days.where(~stale)
    out["w_last_3f"] = merged["t3f"].where(~stale)
    out["w_last_1f"] = merged["t1f"].where(~stale)
    # 負なら自己平均より速い。これが狙っている信号。
    out["w_3f_vs_self"] = (merged["t3f"] - merged["t3f_self"]).where(~stale)
    out["w_1f_vs_self"] = (merged["t1f"] - merged["t1f_self"]).where(~stale)
    out["w_rank_score"] = merged["rank_score"].where(~stale)
    out["w_hard"] = merged["hard"].where(~stale)
    out["w_is_slope"] = merged["is_slope"].where(~stale)
    out["w_count_21d"] = _count_recent(out, workouts)
    return out


def _count_recent(entries: pd.DataFrame, workouts: pd.DataFrame) -> pd.Series:
    """レース前 RECENT_DAYS 日の調教本数。

    間隔が詰まっているほど強い調整。本数そのものより「いつもの本数と違うか」
    が効くはずだが、まずは素の本数から入れて、効かなければ捨てる。
    """
    counts = pd.Series(np.nan, index=entries.index, dtype=float)
    if workouts.empty:
        return counts

    grouped = {
        horse_id: group["workout_date"].to_numpy()
        for horse_id, group in workouts.groupby("horse_id", observed=True)
    }
    race_dates = entries["race_date"].to_numpy()
    for position, (horse_id, race_date) in enumerate(
        zip(entries["horse_id"], race_dates, strict=True)
    ):
        dates = grouped.get(horse_id)
        if dates is None:
            continue
        window = np.timedelta64(RECENT_DAYS, "D")
        counts.iloc[position] = int(
            ((dates < race_date) & (dates >= race_date - window)).sum()
        )
    return counts


def coverage(frame: pd.DataFrame) -> float:
    """調教が付いた行の割合。

    「収集は成功・中身は空」を何度も踏んでいるので、学習の前に必ず見る。
    ここが低いまま学習すると、効かない理由がデータ不足なのか調教が無意味
    なのか分からなくなる。
    """
    if frame.empty:
        return 0.0
    return float(frame["w_last_3f"].notna().mean())
