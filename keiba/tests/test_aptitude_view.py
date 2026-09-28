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
        d.setdefault("band", "sprint" if d.get("distance", 1200) <= 1400 else "mile")
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
    assert any("中山芝1200（良）は1-1-0-0" in x for x in av.reasons(row))


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


def test_良馬場のコース成績で稍重の日の評価を打ち消さない() -> None:
    """ピューロマジックの形（2026-09-27 スプリンターズS 1着・8番人気）。

    良馬場の中山芝1200は3回走って全部4着以下。稍重では3勝。稍重の日に、
    良馬場のコース成績で割り引いてはいけない。
    """
    df = _runs([
        dict(horse_id="p", venue="中山", distance=1200, going="良", finish_pos=8),
        dict(horse_id="p", venue="中山", distance=1200, going="良", finish_pos=8),
        dict(horse_id="p", venue="中山", distance=1200, going="良", finish_pos=16),
        dict(horse_id="p", venue="京都", distance=1200, going="稍重", finish_pos=1),
        dict(horse_id="p", venue="小倉", distance=1200, going="稍重", finish_pos=1),
        dict(horse_id="p", venue="阪神", distance=1200, going="稍重", finish_pos=1),
        dict(horse_id="p", venue="阪神", distance=1200, going="重", finish_pos=2),
        dict(horse_id="p", venue="中山", distance=1200, going="稍重", finish_pos=None),  # 今走
    ])
    row = av.attach(df).iloc[-1]
    assert row["apt_m_course"] == 1.0, "良馬場のコース成績で稍重の日を割り引いている"
    assert row["apt_gc_n"] == 3 and row["apt_gc_1"] == 3, "稍重の成績を別に数えていない"
    assert row["apt_m_soft"] > 1.3
    assert any("稍重は3-0-0-0" in x for x in av.reasons(row))


def test_長い距離の凡走と位置取りで短距離を割り引かない() -> None:
    """マーゴットゲインの形（2026-09-27 中山12R 1着・6番人気）。

    ダ1800の右回りで2戦とも4着以下・後方。初めてのダ1200で1着（中団）。
    ダ1200の日に、1800mの右回りと後方の位置取りで割り引いてはいけない。
    """
    df = _runs([
        dict(horse_id="m", venue="中山", distance=1800, surface="ダ", finish_pos=7,
             corner_ratio=0.62, h_corner_ratio_r5=0.6),
        dict(horse_id="m", venue="新潟", distance=1800, surface="ダ", direction="左",
             finish_pos=6, corner_ratio=0.9, h_corner_ratio_r5=0.7),
        dict(horse_id="m", venue="東京", distance=1600, finish_pos=10,
             corner_ratio=1.0, h_corner_ratio_r5=0.8),
        dict(horse_id="m", venue="新潟", distance=1200, surface="ダ", direction="左",
             finish_pos=1, corner_ratio=0.6, h_corner_ratio_r5=0.8),
        dict(horse_id="m", venue="中山", distance=1200, surface="ダ", finish_pos=None,
             h_corner_ratio_r5=0.8),  # 今走
    ])
    row = av.attach(df).iloc[-1]
    assert row["apt_dir_n"] == 0, "1800mの右回りを1200mの向きの成績に数えている"
    assert row["apt_band_n"] == 1 and row["apt_band_1"] == 1
    assert row["apt_m_band"] > 1.15
    assert abs(row["apt_pos_ratio"] - 0.6) < 1e-9, "位置取りを別の距離帯から取っている"


def test_穴の印は適性の根拠がある人気薄に付く() -> None:
    """☆ は、距離・回り・コースのうち2つ以上に好走実績があり、弱点の無い人気薄へ。"""
    from keiba import predict
    from keiba.engine import ScoredHorse

    def h(u, mark, pop, score):
        return ScoredHorse(umaban=u, horse_id=str(u), horse_name=f"馬{u}", score=score,
                           style="先行", reasons=[], mark=mark, market_popularity=pop)
    horses = [h(1, "◎", 1, 30), h(2, "○", 2, 28), h(3, "☆", 9, 20), h(4, None, 11, 10), h(5, None, 7, 12)]
    apt = {
        3: {"plus": [], "minus": ["距離"]},
        4: {"plus": ["距離", "回り"], "minus": []},             # 候補
        5: {"plus": ["距離", "回り", "コース"], "minus": ["位置"]},  # 弱点ありで外す
    }
    predict.pick_longshot_by_aptitude(horses, apt)
    marks = {x.umaban: x.mark for x in horses}
    assert marks[4] == "☆" and marks[3] is None and marks[5] is None
    assert "適性で選んだ穴" in horses[3].reasons[0]
