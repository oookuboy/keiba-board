"""適性評価（aptitude_view）の検証。

手で見ていた適性（そのコース・回り・道悪・休み明け・位置取り・枠）を、
数えた実績として出せること。自分の結果を先読みしないこと。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba import aptitude_view as av


def _runs(rows):
    base = dict(surface="芝", direction="右", going="良", sire="s", waku=1,
                h_days_since=30.0, corner_ratio=0.1, h_corner_ratio_r5=0.1,
                field_size=16)
    out = []
    for i, r in enumerate(rows):
        d = {**base, **r}
        d.setdefault("race_id", f"r{i:02d}")
        d["race_date"] = pd.Timestamp("2026-01-01") + pd.Timedelta(days=30 * i)
        fp = d.get("finish_pos")
        d["placed"] = np.nan if fp is None else float(fp <= 3)
        d["umaban"] = 1
        out.append(d)
    return pd.DataFrame(out)


def test_コースの成績と回りの成績を数える() -> None:
    """ジューンブレアの形: 中山芝1200で走り、他場で大敗。今走（最後）は数えない。"""
    df = _runs([
        dict(horse_id="h", venue="中山", distance=1200, finish_pos=1),
        dict(horse_id="h", venue="中京", distance=1200, direction="左", finish_pos=17),
        dict(horse_id="h", venue="中山", distance=1200, finish_pos=2),
        dict(horse_id="h", venue="阪神", distance=1400, finish_pos=11),
        dict(horse_id="h", venue="中山", distance=1200, finish_pos=None),  # 今走
    ])
    row = av.attach(df).iloc[-1]
    assert row["apt_course_n"] == 2 and row["apt_course_1"] == 1 and row["apt_course_2"] == 1
    assert row["apt_m_course"] > 1.15
    assert any("中山芝1200は1-1-0-0" in x for x in av.reasons(row))


def test_道悪は今走が稍重以上のときだけ効く() -> None:
    rows = [dict(horse_id="h", venue="中山", distance=1200, going="良", finish_pos=1),
            dict(horse_id="h", venue="中山", distance=1200, going="良", finish_pos=2),
            dict(horse_id="h", venue="中山", distance=1200, going="重", finish_pos=12),
            dict(horse_id="h", venue="中山", distance=1200, going="重", finish_pos=10)]
    dry = av.attach(_runs(rows + [dict(horse_id="h", venue="中山", distance=1200, going="良", finish_pos=None)]))
    wet = av.attach(_runs(rows + [dict(horse_id="h", venue="中山", distance=1200, going="稍重", finish_pos=None)]))
    assert dry.iloc[-1]["apt_m_soft"] == 1.0
    assert wet.iloc[-1]["apt_m_soft"] < 1.0


def test_休み明けで走った実績があれば割り引かない() -> None:
    """ルガルの形: 189日ぶりに勝っている馬は、211日の休み明けでも割り引かない。"""
    df = _runs([
        dict(horse_id="h", venue="中山", distance=1200, finish_pos=5),
        dict(horse_id="h", venue="中山", distance=1200, finish_pos=1, h_days_since=189.0),
        dict(horse_id="h", venue="中山", distance=1200, finish_pos=None, h_days_since=211.0),
        dict(horse_id="g", venue="中山", distance=1200, finish_pos=5),
        dict(horse_id="g", venue="中山", distance=1200, finish_pos=None, h_days_since=211.0),
    ])
    out = av.attach(df)
    assert out.iloc[2]["apt_m_layoff"] == 1.0
    # 休み明けで走った実績の無い馬は、長い休み明けの実測の割合で割り引く
    assert out.iloc[4]["apt_fresh_proven"] == 0


def test_掛けてもレース内の確率の合計は変わらない() -> None:
    frame = pd.DataFrame({"race_id": ["a", "a", "a"], "apt_log": [0.5, 0.0, -0.5]})
    p = pd.Series([0.3, 0.3, 0.3])
    adj = av.adjust(p, frame, 1.0)
    assert abs(adj.sum() - p.sum()) < 1e-9
    assert adj.iloc[0] > adj.iloc[1] > adj.iloc[2]
