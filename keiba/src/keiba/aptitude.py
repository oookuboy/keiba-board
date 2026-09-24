"""その馬が、その条件で何をしてきたかを持たせる。

## 複勝率だけに潰していた

条件ごとの実績は `hv_place_rate`（その競馬場の複勝率）など、**複勝率の形で
しか持っていなかった**。複勝率は「3着以内に入ったか」の平均なので、
**4着と最下位が同じ 0 になる。**

2026-09-21 阪神10R の3着馬チルカーノがこの形だった。

    阪神 芝2000  5着  上がり33.9
    阪神 芝2200  4着  上がり33.9
    札幌 芝2000  5着  上がり35.9
    札幌 芝2000 12着  上がり38.5

阪神での複勝率は 0.0。しかし**阪神では上がり33.9秒を2度使えており、札幌では
35.9・38.5秒しか使えていない**。適性はタイムに出ているのに、着順の平均だけを
見ていたので 0 としか表現できていなかった。

着順に潰さず、その条件で出した**上がり3F・走破指数・最高着順・出走数**を
そのまま持つ。gain は実測で hv_place_rate 0.28% / hs_place_rate 0.40% /
hsb_place_rate 0.56% と、ほぼ効いていない。

## 洋芝と野芝を分ける

札幌・函館は洋芝で、中央の他場とは別の馬場。いまは venue を1つのカテゴリと
して持っているだけで、「洋芝が合う／合わない」という軸が無かった。

同じ 2026-09-21 阪神10R の1番人気アルマデオロは、函館・札幌で 1着1着2着の
実績があり、**今回が初の阪神で11着**。洋芝専用の馬を1番人気に支持し、
こちらも ◎ を打っている。

## 薄いサンプルも切り捨てない

2走しかない馬の「芝2200の成績」もそのまま持つ。出走数を別の列として渡し、
**どれだけ信用するかはモデルに決めさせる**。こちらで足切りしない。

## 先読みはしない

すべて groupby().shift(1) を通してから集計する。自分自身の結果は入らない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 洋芝の開催場。中央では札幌と函館だけ。
YOSHIBA_VENUES = {"札幌", "函館"}

APTITUDE_FEATURES = [
    "turf_type",            # 0=ダート 1=野芝 2=洋芝（カテゴリではなく順序なしの数値）
    "h_yoshiba_runs",       # 洋芝の出走数
    "h_yoshiba_place_rate", # 洋芝の複勝率
    "h_noshiba_place_rate", # 野芝の複勝率
    "h_turf_type_runs",     # 今走の芝種別での出走数
    "h_turf_type_place",    # 今走の芝種別での複勝率
    # 条件ごとの「走りの中身」。着順に潰さない
    "h_cond_runs",          # 馬場×距離帯での出走数
    "h_cond_best_last3f",   # 同条件で出した最速の上がり3F
    "h_cond_avg_last3f",    # 同条件の平均上がり
    "h_cond_best_finish",   # 同条件での最高着順（小さいほど良い）
    "h_venue_runs",         # その競馬場での出走数
    "h_venue_best_last3f",  # その競馬場で出した最速の上がり
    "h_venue_best_finish",  # その競馬場での最高着順
    # 今回の条件と、その馬が得意な条件の差
    "h_last3f_gap_here",    # その条件の最速上がり − 全体の最速上がり
    # --- 経験の無い馬を血統で埋める --------------------------------------
    # その競馬場で走った経験がある馬は半分（hv_place_rate の非欠損 49.1%）。
    # 残り半分には自身の履歴が何も無いので、父・母父の同条件成績を当てる。
    # 洋芝（札幌・函館）の適性は血統に強く出るところ。
    "s_turf_type_place",    # 父の、今走の芝種別での複勝率
    "ds_turf_type_place",   # 母父の、今走の芝種別での複勝率
    "s_cond_last3f",        # 父産駒がその条件で出す上がりの平均
    "cond_place_filled",    # 自身の同条件複勝率。無ければ父の値
    "turf_place_filled",    # 自身の芝種別複勝率。無ければ父の値
    "cond_last3f_filled",   # 自身の同条件最速上がり。無ければ父産駒の平均
    "filled_from_pedigree", # 血統で埋めた数（0〜3）。どれだけ推測かを渡す
    # --- そのコースそのもの・回りの向き ------------------------------------
    # 競馬場だけ・距離帯だけでは「中山芝1200」が見えない。スプリンターズSで
    # ジューンブレア（中山芝1200 4戦4連対、他場の大敗2つ）を 8.9% とした。
    # レッドモンレーヴ（右回り 0-0-0-4 / 左回り 80%）も区別できていなかった。
    # 着順そのものではなく「普段よりここで走るか」を差で渡す。
    "h_course_runs",        # 同じ競馬場・芝ダ・距離での出走数
    "h_course_place_rate",  # そのコースの複勝率
    "h_course_vs_all",      # そのコースの着順比 − 全体の着順比（負ほどここが得意）
    "h_dir_runs",           # 同じ芝ダ・回りの向きでの出走数
    "h_dir_place_rate",     # その向きの複勝率
    "h_dir_vs_all",         # その向きの着順比 − 全体の着順比
    "s_course_place",       # 父産駒のそのコースの複勝率
    "course_place_filled",  # 自身のコース複勝率。無ければ父産駒の値
    "course_place_filled_z",  # 上をレース内で比べた値
    "h_course_vs_all_z",    # 「ここが得意か」をレース内で比べた値
]


def _prior_min(df: pd.DataFrame, keys: list[str], column: str) -> pd.Series:
    """キーごとの「その行より前」の最小値。上がりは小さいほど速い。"""
    return df.groupby(keys, observed=True)[column].transform(
        lambda s: s.shift(1).expanding().min()
    )


def _prior_mean(df: pd.DataFrame, keys: list[str], column: str) -> pd.Series:
    return df.groupby(keys, observed=True)[column].transform(
        lambda s: s.shift(1).expanding().mean()
    )


def _prior_count(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    return df.groupby(keys, observed=True).cumcount()


def turf_type(venue: pd.Series, surface: pd.Series) -> pd.Series:
    """0=ダート・障害 1=野芝 2=洋芝。

    数値にしてあるのは、カテゴリ列を増やすと本番で未知の値が来たときに
    落ちるため。順序に意味は無いが、木は等値で分岐できる。
    """
    is_turf = surface.astype(str).str.startswith("芝")
    return np.where(~is_turf, 0, np.where(venue.isin(YOSHIBA_VENUES), 2, 1))


def attach(frame: pd.DataFrame) -> pd.DataFrame:
    """出走の表に適性の列を足す。

    渡す表は race_date・race_id で並んでいること（dataset.build_features が
    そうしている）。並びが崩れると shift が別の走りを指す。
    """
    out = frame.copy()
    for name in APTITUDE_FEATURES:
        if name not in out.columns:
            out[name] = np.nan

    if out.empty:
        return out

    out["turf_type"] = turf_type(out["venue"], out["surface"])

    horse = ["horse_id"]
    # --- 洋芝 / 野芝 ----------------------------------------------------
    out["is_yoshiba"] = (out["turf_type"] == 2).astype("int8")
    out["is_noshiba"] = (out["turf_type"] == 1).astype("int8")
    # 「洋芝を走ったうちの複勝率」を作るため、洋芝以外を欠損にしてから平均を取る
    placed_yo = out["placed"].where(out["is_yoshiba"] == 1)
    placed_no = out["placed"].where(out["is_noshiba"] == 1)
    out["_py"], out["_pn"] = placed_yo, placed_no
    out["h_yoshiba_place_rate"] = _prior_mean(out, horse, "_py")
    out["h_noshiba_place_rate"] = _prior_mean(out, horse, "_pn")
    out["h_yoshiba_runs"] = (
        out.groupby(horse, observed=True)["is_yoshiba"].transform(
            lambda s: s.shift(1).expanding().sum()
        )
    )
    out["h_turf_type_runs"] = _prior_count(out, ["horse_id", "turf_type"])
    out["h_turf_type_place"] = _prior_mean(out, ["horse_id", "turf_type"], "placed")

    # --- 条件ごとの走りの中身 -------------------------------------------
    cond = ["horse_id", "surface", "band"]
    out["h_cond_runs"] = _prior_count(out, cond)
    out["h_cond_best_last3f"] = _prior_min(out, cond, "last3f")
    out["h_cond_avg_last3f"] = _prior_mean(out, cond, "last3f")
    out["h_cond_best_finish"] = _prior_min(out, cond, "finish_pos")

    venue = ["horse_id", "venue"]
    out["h_venue_runs"] = _prior_count(out, venue)
    out["h_venue_best_last3f"] = _prior_min(out, venue, "last3f")
    out["h_venue_best_finish"] = _prior_min(out, venue, "finish_pos")

    # その条件で、その馬の「いちばん速かった上がり」に対してどれだけ近いか。
    # 全条件での最速と比べることで「ここでは脚が使えない」を表す。
    best_all = _prior_min(out, horse, "last3f")
    out["h_last3f_gap_here"] = out["h_cond_best_last3f"] - best_all

    # --- 経験の無い馬を血統で埋める --------------------------------------
    #
    # その競馬場で走った経験がある馬は半分しかいない（hv_place_rate の
    # 非欠損 49.1%）。条件ごとの出走数も中央値3走。**残りには自身の履歴が
    # 何も無い。**
    #
    # 父・母父の同条件成績を当てる。洋芝は血統に強く出るところで、
    # 2026-09-21 阪神10R の1番人気アルマデオロ（洋芝で1着1着2着 → 初阪神で
    # 11着）のような馬を、走る前に区別したい。
    #
    # **埋めたことを隠さない。** filled_from_pedigree で「いくつ推測で
    # 埋めたか」を渡し、どれだけ信用するかはモデルに決めさせる。自身の
    # 実績と血統の推測が同じ顔で入ると、後から区別できなくなる。
    if "sire" in out.columns:
        out["s_turf_type_place"] = _prior_mean(out, ["sire", "turf_type"], "placed")
        out["s_cond_last3f"] = _prior_mean(
            out, ["sire", "surface", "band"], "last3f"
        )
    if "damsire" in out.columns:
        out["ds_turf_type_place"] = _prior_mean(
            out, ["damsire", "turf_type"], "placed"
        )

    sire_cond = (
        out["ssb_place_rate"] if "ssb_place_rate" in out.columns
        else out.get("s_turf_type_place")
    )
    filled = 0
    for name, own, fallback in (
        ("cond_place_filled", "hsb_place_rate", sire_cond),
        ("turf_place_filled", "h_turf_type_place", out.get("s_turf_type_place")),
        ("cond_last3f_filled", "h_cond_best_last3f", out.get("s_cond_last3f")),
    ):
        mine = out[own] if own in out.columns else pd.Series(np.nan, index=out.index)
        back = fallback if fallback is not None else pd.Series(np.nan, index=out.index)
        out[name] = mine.where(mine.notna(), back)
        filled = filled + (mine.isna() & back.notna()).astype("int8")
    out["filled_from_pedigree"] = filled

    # --- そのコースそのもの・回りの向き ------------------------------------
    if "finish_ratio" in out.columns:
        all_ratio = _prior_mean(out, horse, "finish_ratio")
        course = ["horse_id", "venue", "surface", "distance"]
        out["h_course_runs"] = _prior_count(out, course)
        out["h_course_place_rate"] = _prior_mean(out, course, "placed")
        out["h_course_vs_all"] = _prior_mean(out, course, "finish_ratio") - all_ratio
        if "direction" in out.columns:
            way = ["horse_id", "surface", "direction"]
            out["h_dir_runs"] = _prior_count(out, way)
            out["h_dir_place_rate"] = _prior_mean(out, way, "placed")
            out["h_dir_vs_all"] = _prior_mean(out, way, "finish_ratio") - all_ratio
        if "sire" in out.columns:
            out["s_course_place"] = _prior_mean(
                out, ["sire", "venue", "surface", "distance"], "placed"
            )
        own = out["h_course_place_rate"]
        out["course_place_filled"] = own.where(own.notna(), out["s_course_place"])
        filled = filled + (own.isna() & out["s_course_place"].notna()).astype("int8")
        out["filled_from_pedigree"] = filled
        for col in ("course_place_filled", "h_course_vs_all"):
            g = out.groupby("race_id", observed=True)[col]
            out[f"{col}_z"] = (out[col] - g.transform("mean")) / g.transform(
                "std"
            ).replace(0, np.nan)

    return out.drop(columns=["is_yoshiba", "is_noshiba", "_py", "_pn"])


def coverage(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float(frame["h_cond_best_last3f"].notna().mean())
