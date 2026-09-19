"""展開がモデルの列として届いていること。

## 手置きの側だけ直しても届かない

SKILL.md が「予想の核心」と書いている展開読みは features.py / engine.py の
手置きの側にしかなく、本番では捨てられていた。

    engine.run:
        if ml_scores is None:
            apply_floors(horses, weights)   # 教訓8「確逃げ馬を切らない」の床

本番は ml_scores があるので呼ばれない。想定ペースの加点も score_horse 側。
**印は展開を1ミリも見ていなかった。** 厩舎コメントとまったく同じ形で、
2026-09-19 に発覚した。

回顧の「教訓の効き」はこれを検出できない。reasons のテキストを数えている
だけで、テキストは features から素通しでコピーされるため。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.pace_features import PACE_FEATURES, attach


def frame(styles: list[float], field: int | None = None) -> pd.DataFrame:
    n = field or len(styles)
    return pd.DataFrame({
        "race_id": ["R"] * len(styles),
        "umaban": range(1, len(styles) + 1),
        "h_corner_ratio_r5": styles,
        "field_size": [n] * len(styles),
        "draw_ratio": [(i + 1) / n for i in range(len(styles))],
    })


def test_特徴量の一覧に入っている() -> None:
    """dataset.FEATURE_COLUMNS に載っていなければモデルは受け取らない。

    手置き側のテストでは代わりにならない、というのが今回の教訓そのもの。
    """
    from keiba.dataset import FEATURE_COLUMNS

    missing = [c for c in PACE_FEATURES if c not in FEATURE_COLUMNS]
    assert not missing, f"列は作られるのにモデルへ渡されていない: {missing}"


def test_脚質の境界はweights_ymlと同じ() -> None:
    """境界を二重に持つと、片方だけ直して食い違う。"""
    import pathlib

    import yaml

    from keiba import pace_features

    weights = yaml.safe_load(
        (pathlib.Path(__file__).parents[1] / "config/weights.yml").read_text()
    )["pace"]
    assert pace_features.load_pace_config()["style_thresholds"] == (
        weights["style_thresholds"]
    )


def test_同じ脚質でもレースによって意味が変わる() -> None:
    """通過順 0.10 の馬が、単騎か潰し合いかで別の値になること。

    これが無いなら h_corner_ratio_r5 だけで足りている（＝この列は要らない）。
    """
    lone = attach(frame([0.10, 0.45, 0.55, 0.65, 0.70, 0.80]))
    crowd = attach(frame([0.10, 0.11, 0.12, 0.65, 0.70, 0.80]))

    assert lone.loc[0, "pc_front_runners"] == 1
    assert crowd.loc[0, "pc_front_runners"] == 3
    assert lone.loc[0, "pc_is_lone_front"] == 1.0
    assert crowd.loc[0, "pc_is_lone_front"] == 0.0


def test_単騎逃げは2番手との差で決める() -> None:
    """逃げ馬が1頭でも、2番手がすぐ後ろなら単騎とみなさない（教訓8の条件）。"""
    clear = attach(frame([0.05, 0.40, 0.50, 0.60, 0.70, 0.80]))
    tight = attach(frame([0.05, 0.14, 0.50, 0.60, 0.70, 0.80]))

    assert clear.loc[0, "pc_is_lone_front"] == 1.0
    assert tight.loc[0, "pc_is_lone_front"] == 0.0, (
        "2番手が0.09差なのに単騎と判定している"
    )


def test_少頭数はハイペースにしない() -> None:
    """教訓7。少頭数では逃げ馬が複数いても隊列がすんなり決まる。

    大井4Rでこれを機械的に適用して確逃げ馬を切り、三連複を落としている。
    """
    small = attach(frame([0.05, 0.08, 0.5, 0.6, 0.7, 0.8], field=8))
    big = attach(frame([0.05, 0.08, 0.5, 0.6, 0.7, 0.8], field=16))

    assert small.loc[0, "pc_projected"] == 0.0, "少頭数をハイペース扱いしている"
    assert big.loc[0, "pc_projected"] == 2.0


def test_通過順の取れない馬を逃げに数えない() -> None:
    """新馬・欠損を「前へ行く」と数えると、逃げ馬の頭数が水増しされる。

    分からないものを分かったことにしない。
    """
    out = attach(frame([0.05, np.nan, np.nan, 0.6, 0.7, 0.8]))
    assert out.loc[0, "pc_front_runners"] == 1
    assert pd.isna(out.loc[1, "pc_is_lone_front"])


def test_渡すものが空でも列はそろう() -> None:
    """列の顔ぶれが学習と本番で変わらないこと。調教で踏んだ形を塞ぐ。"""
    out = attach(pd.DataFrame(columns=["race_id", "umaban", "field_size"]))
    for name in PACE_FEATURES:
        assert name in out.columns, f"{name} が作られていない"


def test_行数が増えない() -> None:
    """レース内の集計で merge を挟むと行が増えうる。増えると以降が全部ずれる。"""
    assert len(attach(frame([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))) == 6
