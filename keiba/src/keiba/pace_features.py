"""展開・隊列を特徴量にする。

## 書いてあるのに使われていなかった

SKILL.md が「予想の核心」と書いている展開読みは features.py / engine.py の
**手置きの側**にしかない。学習モデルで採点する本番では、そちらは丸ごと
捨てられる。

    engine.run:
        if ml_scores is None:
            apply_floors(horses, weights)   # 教訓8「確逃げ馬を切らない」の床

本番では ml_scores があるので `apply_floors` は呼ばれない。想定ペースの
加点も score_horse 側なので同じく効かない。**印は展開を1ミリも見ていな
かった。**

厩舎コメントで踏んだのと同じ形。手置き側をいくら直しても届かないので、
列にする。

## 何を列にするか

モデルは既に `h_corner_ratio_r5`（過去5走の平均通過順位率）を持っている。
足りないのは「**このレースの中で自分がどこに位置するか**」。同じ通過順
0.2 の馬でも、他に前へ行く馬がいなければ単騎で、3頭いれば潰し合う。

レース内の相対で作る。先読みにはならない。材料は過去走の通過順だけで、
そのレースの結果は一切使わない。

## 脚質の境界は weights.yml と共有する

境界を二重に持つと、片方だけ直して食い違う。厩舎コメントの語の表で
同じことをやった。
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

# 脚質の境界と少頭数の判定は weights.yml に置いてある。features 側（手置きの
# 展開読み）と同じ値を見る。二重に持つと片方だけ直して食い違う。
DEFAULT_WEIGHTS = pathlib.Path(__file__).parents[2] / "config/weights.yml"


def load_pace_config(config_dir: pathlib.Path | None = None) -> dict:
    """weights.yml の pace 節を読む。学習側は config_dir を持ち回っていない
    ので既定を持たせる。置き場が1つであることのほうが大事。
    """
    import yaml

    path = (config_dir / "weights.yml") if config_dir else DEFAULT_WEIGHTS
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("pace", {})

PACE_FEATURES = [
    "pc_style_rank",     # レース内で何番目に前へ行くか（0=最前）
    "pc_style_z",        # 同じく z 値
    "pc_gap_ahead",      # 自分のすぐ前に行く馬との差。最前なら2番手との差
    "pc_front_runners",  # 逃げ相当の頭数
    "pc_front_share",    # 逃げ相当 / 出走頭数
    "pc_is_lone_front",  # 単騎逃げ濃厚か（教訓8の条件）
    "pc_projected",      # 想定ペース 0=slow 1=mid 2=fast
    "pc_draw_x_front",   # 枠の内外 × 前へ行くか
]

# 逃げとみなす通過順位率。weights.yml の pace.style_thresholds と同じ値を
# 既定にする。呼ぶ側が渡せるようにして、表が二重にならないようにする。
FRONT_THRESHOLD = 0.15


def attach(
    frame: pd.DataFrame,
    *,
    front_threshold: float = FRONT_THRESHOLD,
    small_field_max: int = 10,
    multi_front_runners: int = 2,
) -> pd.DataFrame:
    """出走の表に展開の列を足す。

    渡すものが無くても列はそろえる。**列の顔ぶれが学習と本番で変わらない
    ようにする。** 調教で一度踏んだ形。
    """
    out = frame.copy()
    for name in PACE_FEATURES:
        out[name] = np.nan

    if out.empty or "h_corner_ratio_r5" not in out.columns:
        return out

    style = out["h_corner_ratio_r5"]
    grouped = out.groupby("race_id", observed=True)["h_corner_ratio_r5"]

    # 小さいほど前。昇順の順位率をそのまま使う
    out["pc_style_rank"] = grouped.rank(pct=True, method="min")
    out["pc_style_z"] = (style - grouped.transform("mean")) / grouped.transform(
        "std"
    ).replace(0, np.nan)

    # 自分のすぐ前に行く馬との差。前が居なければ2番手との差（＝単騎の深さ）
    ordered = out.sort_values(["race_id", "h_corner_ratio_r5"])
    prev = ordered.groupby("race_id", observed=True)["h_corner_ratio_r5"].shift(1)
    nxt = ordered.groupby("race_id", observed=True)["h_corner_ratio_r5"].shift(-1)
    gap = (ordered["h_corner_ratio_r5"] - prev).fillna(
        nxt - ordered["h_corner_ratio_r5"]
    )
    out["pc_gap_ahead"] = gap.reindex(out.index)

    is_front = (style < front_threshold).astype(float)
    # 通過順が取れない馬（新馬・欠損）は「前へ行く」と数えない。分からない
    # ものを分かったことにすると、逃げ馬の頭数が水増しされる。
    is_front = is_front.where(style.notna())
    front_count = is_front.groupby(out["race_id"]).transform("sum")
    out["pc_front_runners"] = front_count
    out["pc_front_share"] = front_count / out["field_size"]

    # 単騎逃げ濃厚（教訓8）。逃げ相当が1頭で、2番手と明確に差がある
    lone = (front_count == 1) & (is_front == 1) & (out["pc_gap_ahead"] > 0.10)
    out["pc_is_lone_front"] = lone.astype(float).where(style.notna())

    # 想定ペース。少頭数は隊列がすんなり決まる前提で先行有利（教訓7）
    projected = np.where(
        out["field_size"] <= small_field_max, 0.0,
        np.where(front_count >= multi_front_runners, 2.0,
                 np.where(front_count == 1, 0.0, 1.0)),
    )
    out["pc_projected"] = projected

    # 枠の内外と脚質の噛み合わせ。内枠の逃げ先行は隊列を取りやすい
    out["pc_draw_x_front"] = out["draw_ratio"] * (1.0 - out["pc_style_rank"])
    return out


def coverage(frame: pd.DataFrame) -> float:
    """展開を組み立てられた割合。通過順の取れない馬ばかりだと立たない。"""
    if frame.empty:
        return 0.0
    return float(frame["pc_style_rank"].notna().mean())
