"""条件適性が、着順に潰されずにモデルへ届くこと。

## 複勝率だけに潰していた

2026-09-21 阪神10R の3着馬チルカーノ（15人気）。

    阪神 芝2000  5着  上がり33.9
    阪神 芝2200  4着  上がり33.9
    札幌 芝2000  5着  上がり35.9
    札幌 芝2000 12着  上がり38.5

阪神での複勝率は 0.0。4着と最下位が同じ扱いになるため、**「阪神では速い脚が
使えるが洋芝では使えない」という事実が 0 としか表現できていない。**

同じレースの1番人気アルマデオロは函館・札幌で 1着1着2着、今回が初の阪神で
11着。洋芝専用の馬を1番人気に支持し、こちらも ◎ を打っている。

着順に潰さず、その条件で出した上がり3F・最高着順・出走数をそのまま持つ。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba import aptitude
from keiba.aptitude import APTITUDE_FEATURES, attach, turf_type


def test_特徴量の一覧に入っている() -> None:
    """FEATURE_COLUMNS に載っていなければモデルは受け取らない。"""
    from keiba.dataset import FEATURE_COLUMNS

    missing = [c for c in APTITUDE_FEATURES if c not in FEATURE_COLUMNS]
    assert not missing, f"列は作られるのにモデルへ渡されていない: {missing}"


def test_洋芝と野芝を分ける() -> None:
    """札幌・函館は洋芝。中央の他場とは別の馬場。"""
    got = turf_type(
        pd.Series(["札幌", "函館", "阪神", "東京", "阪神"]),
        pd.Series(["芝", "芝", "芝", "芝", "ダ"]),
    )
    assert list(got) == [2, 2, 1, 1, 0]


def _horse(rows: list[dict]) -> pd.DataFrame:
    """1頭の履歴。古い順に並べる（dataset と同じ前提）。"""
    return pd.DataFrame([
        {"horse_id": "h1", "band": r.get("band", "middle"), "placed": r["placed"],
         **r} for r in rows
    ])


def test_着順が同じでも上がりの差が残る() -> None:
    """チルカーノの形。阪神33.9／札幌38.5 が区別されること。

    複勝率だけなら両方 0.0 で、差が消える。
    """
    df = _horse([
        {"venue": "阪神", "surface": "芝", "last3f": 33.9, "finish_pos": 5, "placed": 0},
        {"venue": "札幌", "surface": "芝", "last3f": 38.5, "finish_pos": 12, "placed": 0},
        {"venue": "阪神", "surface": "芝", "last3f": np.nan, "finish_pos": np.nan,
         "placed": np.nan},
    ])
    out = attach(df)
    # 3行目（今回の阪神）から見た、阪神での最速上がり
    assert out.loc[2, "h_venue_best_last3f"] == pytest.approx(33.9)
    # 洋芝の複勝率は 0 だが、出走数が別に立つので「1走しかない0」と分かる
    assert out.loc[2, "h_yoshiba_runs"] == 1


def test_薄いサンプルでも切り捨てない() -> None:
    """2走しかない馬の条件成績もそのまま持つ。

    どれだけ信用するかはモデルに決めさせる。こちらで足切りしない。
    """
    df = _horse([
        {"venue": "小倉", "surface": "芝", "last3f": 35.9, "finish_pos": 1, "placed": 1},
        {"venue": "阪神", "surface": "芝", "last3f": np.nan, "finish_pos": np.nan,
         "placed": np.nan},
    ])
    out = attach(df)
    assert out.loc[1, "h_cond_runs"] == 1, "1走ぶんの実績が落ちている"
    assert out.loc[1, "h_cond_best_finish"] == 1


def test_自分の結果は入らない() -> None:
    """先読み防止。今走の上がりが今走の特徴量に入らないこと。"""
    df = _horse([
        {"venue": "阪神", "surface": "芝", "last3f": 36.0, "finish_pos": 8, "placed": 0},
        {"venue": "阪神", "surface": "芝", "last3f": 33.0, "finish_pos": 1, "placed": 1},
    ])
    out = attach(df)
    # 2行目から見た最速は、1行目の 36.0 でなければならない（33.0 が入ったら先読み）
    assert out.loc[1, "h_venue_best_last3f"] == pytest.approx(36.0)
    assert out.loc[0, "h_venue_best_last3f"] != out.loc[0, "h_venue_best_last3f"]  # NaN


def test_得意な条件との差が出る() -> None:
    """全体の最速と、今回の条件での最速の差。

    「この条件では脚が使えない」を1つの数で表す。
    """
    df = _horse([
        {"venue": "阪神", "surface": "芝", "band": "middle", "last3f": 33.5,
         "finish_pos": 3, "placed": 1},
        {"venue": "札幌", "surface": "芝", "band": "long", "last3f": 38.0,
         "finish_pos": 10, "placed": 0},
        {"venue": "札幌", "surface": "芝", "band": "long", "last3f": np.nan,
         "finish_pos": np.nan, "placed": np.nan},
    ])
    out = attach(df)
    # long では 38.0、全体では 33.5 → 差 +4.5
    assert out.loc[2, "h_last3f_gap_here"] == pytest.approx(4.5)


def test_渡すものが空でも列はそろう() -> None:
    """列の顔ぶれが学習と本番で変わらないこと。"""
    out = attach(pd.DataFrame(columns=["horse_id", "venue", "surface", "band",
                                       "last3f", "finish_pos", "placed"]))
    for name in APTITUDE_FEATURES:
        assert name in out.columns


def test_作業用の列を残さない() -> None:
    """中間列が FEATURE_COLUMNS 外に漏れると、学習と本番で表がずれる。"""
    df = _horse([
        {"venue": "阪神", "surface": "芝", "last3f": 34.0, "finish_pos": 2, "placed": 1},
    ])
    out = attach(df)
    for tmp in ("is_yoshiba", "is_noshiba", "_py", "_pn"):
        assert tmp not in out.columns, f"{tmp} が残っている"


# --- 経験の無い馬を血統で埋める -----------------------------------------


def test_経験が無ければ父の成績で埋める() -> None:
    """その競馬場で走った経験がある馬は半分しかいない（非欠損 49.1%）。

    残りには自身の履歴が何も無いので、父の同条件成績を当てる。
    """
    df = pd.DataFrame([
        # 同じ父の別の馬が、先に洋芝を走っている
        {"horse_id": "other", "sire": "s1", "damsire": "d1", "venue": "札幌",
         "surface": "芝", "band": "middle", "last3f": 35.0, "finish_pos": 1,
         "placed": 1.0, "ssb_place_rate": np.nan, "h_turf_type_place": np.nan},
        # こちらは初出走。洋芝の経験が無い
        {"horse_id": "h1", "sire": "s1", "damsire": "d1", "venue": "札幌",
         "surface": "芝", "band": "middle", "last3f": np.nan, "finish_pos": np.nan,
         "placed": np.nan, "ssb_place_rate": np.nan, "h_turf_type_place": np.nan},
    ])
    out = attach(df)
    row = out.iloc[1]
    assert row["h_turf_type_place"] != row["h_turf_type_place"], "自身の実績は無いはず"
    assert row["turf_place_filled"] == pytest.approx(1.0), "父の洋芝成績で埋まっていない"
    assert row["filled_from_pedigree"] >= 1, "埋めたことが記録されていない"


def test_自身の実績があれば血統で上書きしない() -> None:
    """推測より実績を優先する。

    父の成績を意図的に低くし、自身の成績と取り違えていないかを見る。
    attach は自身の実績を履歴から計算し直すので、テスト側で値を渡しても
    上書きされる（それが正しい動作）。父側に差を付けて区別する。
    """
    df = pd.DataFrame([
        # 同じ父の別の馬が野芝で凡走 → 父の野芝成績は低くなる
        {"horse_id": "other", "sire": "s1", "damsire": "d1", "venue": "阪神",
         "surface": "芝", "band": "middle", "last3f": 36.0, "finish_pos": 10,
         "placed": 0.0, "ssb_place_rate": np.nan, "h_turf_type_place": np.nan},
        {"horse_id": "other", "sire": "s1", "damsire": "d1", "venue": "阪神",
         "surface": "芝", "band": "middle", "last3f": 36.2, "finish_pos": 12,
         "placed": 0.0, "ssb_place_rate": np.nan, "h_turf_type_place": np.nan},
        # 当該馬は野芝で好走している
        {"horse_id": "h1", "sire": "s1", "damsire": "d1", "venue": "阪神",
         "surface": "芝", "band": "middle", "last3f": 34.0, "finish_pos": 1,
         "placed": 1.0, "ssb_place_rate": np.nan, "h_turf_type_place": np.nan},
        {"horse_id": "h1", "sire": "s1", "damsire": "d1", "venue": "阪神",
         "surface": "芝", "band": "middle", "last3f": np.nan, "finish_pos": np.nan,
         "placed": np.nan, "ssb_place_rate": np.nan, "h_turf_type_place": np.nan},
    ])
    out = attach(df)
    row = out.iloc[3]
    assert row["turf_place_filled"] == pytest.approx(1.0), (
        "自身の実績（複勝率1.0）ではなく父の値を使っている"
    )
    assert row["filled_from_pedigree"] == 0, "実績があるのに埋めたと記録している"


def test_埋めたことを隠さない() -> None:
    """自身の実績と血統の推測が同じ顔で入ると、後から区別できない。

    いくつ推測で埋めたかを渡し、どれだけ信用するかはモデルに決めさせる。
    """
    assert "filled_from_pedigree" in APTITUDE_FEATURES
    from keiba.dataset import FEATURE_COLUMNS
    assert "filled_from_pedigree" in FEATURE_COLUMNS


def test_父が分からなければ埋めない() -> None:
    """血統が取れていない馬に、でたらめな値を入れない。"""
    df = pd.DataFrame([
        {"horse_id": "h1", "sire": None, "damsire": None, "venue": "札幌",
         "surface": "芝", "band": "middle", "last3f": np.nan, "finish_pos": np.nan,
         "placed": np.nan, "ssb_place_rate": np.nan, "h_turf_type_place": np.nan},
    ])
    out = attach(df)
    assert out.loc[0, "filled_from_pedigree"] == 0
