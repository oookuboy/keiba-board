"""モデルの採点を「何を根拠にしたか」の6項目に分ける。

ユーザーが毎レース見る表（最近の着順・騎手・調教・タイム・適性・その他）。
LightGBM の寄与（pred_contrib）を特徴量ごとに出し、項目ごとに足す。
値はロジット上の足し算なので、項目どうし・馬どうしで大小を比べられる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.aptitude import APTITUDE_FEATURES

CATEGORIES = ("最近の着順", "騎手", "調教", "タイム", "適性", "その他")

_APTITUDE_PREFIX = (
    "hs_", "hsb_", "hv_", "s_", "ss", "ds_", "h_course", "h_dir", "course_",
    "h_cond", "h_venue", "turf", "h_yoshiba", "h_noshiba", "h_turf_type", "cond_",
)
_APTITUDE_EXTRA = {
    "c_distance_delta", "c_surface_switch", "c_switch_x_sire",
    "filled_from_pedigree", "h_last3f_gap_here",
}


def category(feature: str) -> str:
    """特徴量の名前から6項目のどれかを返す。"""
    if feature.startswith("w_"):
        return "調教"
    if feature.startswith(("j_", "jv_", "hj_")) or feature == "c_jockey_change":
        return "騎手"
    if feature.startswith("sp_") or feature == "h_last3f_r3":
        return "タイム"
    if (
        feature in APTITUDE_FEATURES
        or feature in _APTITUDE_EXTRA
        or feature.startswith(_APTITUDE_PREFIX)
    ):
        return "適性"
    if feature.startswith(
        ("h_finish", "h_prev_finish", "h_place_rate", "h_win_rate", "h_margin", "h_fr_")
    ) or feature == "h_runs":
        return "最近の着順"
    return "その他"


def breakdown(booster, frame: pd.DataFrame) -> pd.DataFrame:
    """行ごとに6項目の寄与を返す（列は CATEGORIES、行は frame と同じ並び）。"""
    names = booster.feature_name()
    contrib = booster.predict(frame[names], pred_contrib=True)[:, : len(names)]
    out = pd.DataFrame(0.0, index=frame.index, columns=list(CATEGORIES))
    for j, name in enumerate(names):
        out[category(name)] += contrib[:, j]
    return out


def ranks(values: dict[int, dict[str, float]]) -> dict[int, dict[str, int]]:
    """馬番 → 項目 → レース内の順位（1が一番良い）。"""
    if not values:
        return {}
    frame = pd.DataFrame.from_dict(values, orient="index")
    ranked = frame.rank(ascending=False, method="min").astype(int)
    return {int(u): {c: int(ranked.at[u, c]) for c in frame.columns} for u in frame.index}


def add_aptitude_view(values: pd.DataFrame, apt_log: pd.Series, share: float) -> pd.DataFrame:
    """適性の2割（aptitude_view.blend）の分も「適性」に足す。"""
    if share:
        values = values.copy()
        values["適性"] += share * np.nan_to_num(apt_log.to_numpy(dtype=float))
    return values
