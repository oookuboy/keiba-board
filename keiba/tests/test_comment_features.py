"""厩舎コメントがモデルの列として届いていること。

## 手置きの側だけ直しても届かない

コメントは収集も判定コードもあった。にもかかわらず印は1頭も動いていない。

    features.condition_changes  手置きの重み側で加点していた
    engine.run                  ml_scores があるとその加点を丸ごと捨てる

根拠テキストだけが features から素通しで残るので、ボードには
「厩舎コメントに前向きな語」と出ていた。**説明文だけが出ていて、順位には
1ミリも効いていなかった。** 2026-09-11 に発覚。

同じ形をもう一度踏まないよう、「モデルの特徴量に入っている」ことを直接
固定する。手置き側のテストでは代わりにならない。
"""

from __future__ import annotations

import pandas as pd

from keiba import comment_features
from keiba.comment_features import COMMENT_FEATURES, _sides, attach


def _cfg() -> dict:
    return comment_features.load_keywords()


def test_特徴量の一覧に入っている() -> None:
    """dataset.FEATURE_COLUMNS に載っていなければ、モデルは受け取らない。"""
    from keiba.dataset import FEATURE_COLUMNS

    missing = [c for c in COMMENT_FEATURES if c not in FEATURE_COLUMNS]
    assert not missing, f"列は作られるのにモデルへ渡されていない: {missing}"


def test_語の表はfeaturesと同じものを見ている() -> None:
    """表が二重になると、片方だけ直して食い違う。"""
    import pathlib

    import yaml

    weights = yaml.safe_load(
        (pathlib.Path(__file__).parents[1] / "config/weights.yml").read_text()
    )["condition_change"]
    loaded = comment_features.load_keywords()
    assert loaded["positive_keywords"] == weights["positive_keywords"]
    assert loaded["negative_keywords"] == weights["negative_keywords"]


def test_同じ場所を二度数えない() -> None:
    """「動きは水準以上」は `動きは水準` と `水準以上` の両方に当たる。

    features._keyword_sides と同じ規則であること。規則が割れると、根拠
    テキストとモデルの入力が別のことを言い出す。
    """
    from keiba.features import _keyword_sides

    body = "動きは水準以上。ただ使ってからでは。"
    pos, neg = ["動きは水準", "水準以上"], ["使ってから"]
    assert _sides(body, pos, neg) == (1, 1)

    good, bad = _keyword_sides(body, pos, neg)
    assert (len(good), len(bad)) == _sides(body, pos, neg), "2つの規則が食い違っている"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"race_id": ["A", "A", "A", "B"], "umaban": [1, 2, 3, 1]}
    )


def test_コメントのある馬だけ値が入る() -> None:
    """出ない馬を 0 で埋めない。

    「コメントが無い」と「何も言っていないコメント」は別のこと。混ぜると
    後者を前者として学ぶ。調教の列と同じ方針。
    """
    comments = pd.DataFrame({
        "race_id": ["A", "A"],
        "umaban": [1, 2],
        "body": ["追うごとに動きは良化。反応良好。", "時計が詰まってこない。使ってから。"],
    })
    out = attach(_frame(), comments, _cfg())

    assert out.loc[out["umaban"].eq(1) & out["race_id"].eq("A"), "cm_net"].iloc[0] > 0
    assert out.loc[out["umaban"].eq(2), "cm_net"].iloc[0] < 0
    # コメントの無い馬は欠損のまま
    assert out.loc[out["race_id"].eq("B"), "cm_net"].isna().all()
    assert out.loc[out["race_id"].eq("B"), "cm_has"].iloc[0] == 0.0
    assert out.loc[out["race_id"].eq("A") & out["umaban"].eq(1), "cm_has"].iloc[0] == 1.0


def test_渡すものが空でも列はそろう() -> None:
    """列の有無で学習と本番が変わらないこと。

    調教で踏んだ形（本番だけ列が無い）をここでも塞ぐ。LightGBM は列の顔ぶれが
    変わると落ちるか、黙って別のものを学ぶ。
    """
    out = attach(_frame(), pd.DataFrame(), _cfg())
    for name in COMMENT_FEATURES:
        assert name in out.columns, f"{name} が作られていない"
    assert out["cm_has"].eq(0.0).all()


def test_レース内で相対化している() -> None:
    """絶対値より、同じレースの他馬と比べてどうかが効く（他の列と同じ作法）。"""
    comments = pd.DataFrame({
        "race_id": ["A", "A", "A"],
        "umaban": [1, 2, 3],
        "body": [
            "反応良好で好感触。動きも良化し態勢は整った。",
            "動きはまずまず。",
            "課題を残す。物足りない。時計が詰まってこない。",
        ],
    })
    out = attach(_frame(), comments, _cfg()).query("race_id == 'A'")
    ranks = out.set_index("umaban")["cm_net_rank"]
    assert ranks[1] > ranks[2] > ranks[3], f"レース内順位が付いていない: {ranks.to_dict()}"


def test_出走の行数が増えない() -> None:
    """merge で行が増えると、以降の集計が全部ずれる。

    コメントは (race_id, umaban) で一意のはずだが、崩れたときに黙って
    水増しされるのが一番怖い。
    """
    out = attach(_frame(), pd.DataFrame({
        "race_id": ["A"], "umaban": [1], "body": ["動きも良化。"],
    }), _cfg())
    assert len(out) == 4
