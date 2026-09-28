"""適性評価。モデルの確率に、その馬の「このコース・この条件での走り」を掛ける。

## なぜ別に要るのか

モデルは16万走から「平均的に何が効くか」を学ぶ。近走の着順と騎手が強く、
コースの成績は平均すると効きが小さい。その結果、印は人気順とほぼ同じになる
（2026-09-26/27 の2日で人気との順位相関 0.75、スプリンターズSは印5頭＝人気
上位5頭）。人気と同じ結論では市場に勝てない。

市場と違う結論を出せるのは、手で見ていた適性の部分だった。

- 中山芝1200で4戦4連対なのに、他場の大敗2つで13番人気（ジューンブレア）
- 右回り 0-0-0-4 / 左回り 80%（レッドモンレーヴ）
- 道悪 3-1-1-1、父の産駒も重で走る（ピューロマジック）
- 父の産駒が重・不良で 2%（ペアポルックス）
- 半年以上の休み明けは3着内率が半分近くに落ちる。ただし休み明けで
  走った実績がある馬は別（ルガルは189日ぶりにスプリンターズSを勝った）
- 中山芝1200は最初のコーナーで前1/4にいた馬の3着内率 36%、後ろ1/4は 8%

これをモデルの外で、数えた実績として掛ける。

## 作り

どれも「この条件でのその馬の3着内率 ÷ その馬の普段の3着内率」を、
出走数が少ないほど 1 に寄せた倍率にする（サンプルが薄くても結果は結果として
使うが、1走で倍率が暴れないようにする）。経験の無い馬は父の産駒の値で補う。
倍率は掛け合わせ、重み w で効きを調整し、最後にレース内で合計を保つよう
正規化する。

すべて groupby().shift(1) を通した「そのレースより前」の数字だけを使う。
コースの有利不利（位置・枠）の表も、そのレースより前のレースから作る。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SOFT = {"稍重", "重", "不良"}
# 馬場を3つに分ける。稍重は良に近く、重・不良とは別物として数える
# （2026-09-27 スプリンターズSの勝ち馬ピューロマジックは道悪3勝がすべて稍重）。
GOING_CLASS = {"良": 0, "稍重": 1, "重": 2, "不良": 2}
GOING_LABEL = {0: "良", 1: "稍重", 2: "重・不良"}
BAND_LABEL = {"sprint": "短距離", "mile": "マイル", "middle": "中距離", "long": "長距離"}

# 1 に寄せる強さ（何走ぶんの「普段どおり」を足すか）
K_HORSE = 3.0
K_SIRE = 30.0
K_BIAS = 50.0
K_BASE = 5.0

# 倍率の上下限。1つの要素だけで順位を丸ごと入れ替えないため。
CLIP = (0.4, 2.5)

# ここより長い休みを「長い休み明け」とする（芝1200オープンで 181日以上は
# 3着内率 10%、121-180日は 25%）。
LONG_LAYOFF = 181
PROVEN_LAYOFF = 150


def _prior_sum(df: pd.DataFrame, keys: list[str], col: str) -> pd.Series:
    """そのレースより前の合計（同じ keys の中で）。"""
    # 累積和から自分の分を引く。shift→cumsum を lambda で回すより桁違いに速い。
    # キーが欠けた行（父が不明など）は 0 走扱い。
    total = df.groupby(keys, observed=True, dropna=True)[col].cumsum()
    return (total - df[col]).fillna(0.0)


def _ratio(placed, runs, base, k):
    """(placed + k*base) / (runs + k) / base。出走が少ないほど 1 に寄る。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (placed + k * base) / (runs + k) / base
    return pd.Series(r).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(*CLIP)


def _record(df: pd.DataFrame, keys: list[str], prefix: str) -> None:
    """1着・2着・3着・着外の、そのレースより前の数を列にする。"""
    for k, flag in ((1, "_w1"), (2, "_w2"), (3, "_w3")):
        df[f"{prefix}{k}"] = _prior_sum(df, keys, flag)
    df[f"{prefix}n"] = _prior_sum(df, keys, "_ran")


def _bias_table(hist: pd.DataFrame, keys: list[str], col: str) -> pd.DataFrame:
    """コース × col（位置の四分位・枠）ごとの3着内率と、コース全体の率。"""
    h = hist.dropna(subset=[col])
    by = h.groupby(keys + [col], observed=True)["placed"].agg(["sum", "count"])
    course = h.groupby(keys, observed=True)["placed"].mean().rename("course_rate")
    return by.reset_index().merge(course.reset_index(), on=keys)


def attach(df: pd.DataFrame, bias_before: str | None = None) -> pd.DataFrame:
    """適性の倍率と、その根拠の数を列として足した写しを返す。

    df は dataset.prepare の表（race_date, race_id で並んでいること）。
    bias_before を渡すと、コースの有利不利の表をその日より前だけで作る
    （検証用）。渡さなければ、各レースより前の全レースで作る近似として
    df 全体の完走行を使う（本番は予想する日のレースに結果が無いので同じ）。
    """
    out = df.copy()
    done = out["finish_pos"].notna()
    out["_ran"] = done.astype(float)
    out["_pl"] = (out["placed"].fillna(0) * done).astype(float)
    for k in (1, 2, 3):
        out[f"_w{k}"] = ((out["finish_pos"] == k) & done).astype(float)
    out["_soft"] = out["going"].astype(str).isin(SOFT)
    out["_gc"] = out["going"].astype(str).map(GOING_CLASS).fillna(0).astype(int)
    out["_wet"] = out["_gc"] > 0

    horse = ["horse_id"]
    g = float(out.loc[done, "placed"].mean())
    runs_all = _prior_sum(out, horse, "_ran")
    pl_all = _prior_sum(out, horse, "_pl")
    base = (pl_all + K_BASE * g) / (runs_all + K_BASE)
    out["apt_base"] = base

    # --- そのコース（場・芝ダ・距離）------------------------------------
    # 良馬場で走ったコースの成績で、道悪の日の評価を打ち消さない。
    # そのコースの成績は「今日と同じ側（良 / 道悪）」の走りだけで数える。
    # スプリンターズSで、良馬場の中山芝1200 0-0-0-3 を理由に、稍重で3勝の
    # ピューロマジックを消した（その馬が勝った）。
    course = ["horse_id", "venue", "surface", "distance", "_wet"]
    _record(out, course, "apt_course_")
    out["apt_m_course"] = _ratio(
        _prior_sum(out, course, "_pl"), out["apt_course_n"], base, K_HORSE
    ).values

    # --- 回りの向き（芝ダ・右左）-----------------------------------------
    # 向きは同じ距離帯で数える。1800mの右回りの凡走で1200mを割り引かない
    # （2026-09-27 中山12R 1着マーゴットゲインは「ダ右回り 0-0-0-2」が
    # すべて1800mだった）。
    way = ["horse_id", "surface", "direction", "band"]
    _record(out, way, "apt_dir_")
    out["apt_m_dir"] = _ratio(
        _prior_sum(out, way, "_pl"), out["apt_dir_n"], base, K_HORSE
    ).values

    # --- 同じ芝ダ・距離帯 ---------------------------------------------------
    # 距離を替えて一変する馬がいる。マーゴットゲインは初めてのダ1200で
    # 13番人気1着、2走目の中山12Rも6番人気1着。
    band = ["horse_id", "surface", "band"]
    _record(out, band, "apt_band_")
    out["apt_m_band"] = _ratio(
        _prior_sum(out, band, "_pl"), out["apt_band_n"], base, K_HORSE
    ).values

    # --- 道悪（今走が稍重以上のときだけ効かせる）-------------------------
    # 今日と同じ馬場（稍重 / 重・不良）での成績を優先し、無ければ道悪全体
    soft_key = ["horse_id", "surface", "_soft"]
    _record(out, soft_key, "apt_soft_")
    m_soft_all = _ratio(
        _prior_sum(out, soft_key, "_pl"), out["apt_soft_n"], base, K_HORSE
    ).values
    gc_key = ["horse_id", "surface", "_gc"]
    _record(out, gc_key, "apt_gc_")
    m_soft_same = _ratio(
        _prior_sum(out, gc_key, "_pl"), out["apt_gc_n"], base, K_HORSE
    ).values
    m_soft_horse = np.where(out["apt_gc_n"] > 0, m_soft_same, m_soft_all)
    # 経験の無い馬は父の産駒の「道悪 ÷ 全体」で補う
    sire_soft_pl = _prior_sum(out, ["sire", "surface", "_soft"], "_pl")
    sire_soft_n = _prior_sum(out, ["sire", "surface", "_soft"], "_ran")
    sire_all_pl = _prior_sum(out, ["sire", "surface"], "_pl")
    sire_all_n = _prior_sum(out, ["sire", "surface"], "_ran")
    sire_base = (sire_all_pl + K_BASE * g) / (sire_all_n + K_BASE)
    m_soft_sire = _ratio(sire_soft_pl, sire_soft_n, sire_base, K_SIRE).values
    out["apt_sire_soft_rate"] = (sire_soft_pl / sire_soft_n.replace(0, np.nan)).values
    out["apt_sire_soft_n"] = sire_soft_n.values
    m_soft = np.where(out["apt_soft_n"] > 0, m_soft_horse, m_soft_sire)
    out["apt_m_soft"] = np.where(out["_soft"], m_soft, 1.0)

    # --- 長い休み明け -----------------------------------------------------
    long_rows = out.loc[done & out["h_days_since"].notna()]
    rate_long = long_rows.loc[long_rows["h_days_since"] >= LONG_LAYOFF, "placed"].mean()
    m_long = float(rate_long / g) if g and rate_long == rate_long else 1.0
    out["_fresh_pl"] = (
        (out["h_days_since"] >= PROVEN_LAYOFF) & (out["_pl"] > 0)
    ).astype(float)
    out["apt_fresh_proven"] = _prior_sum(out, horse, "_fresh_pl")
    is_long = out["h_days_since"] >= LONG_LAYOFF
    out["apt_m_layoff"] = np.where(
        is_long & (out["apt_fresh_proven"] == 0), m_long, 1.0
    )

    # --- 前に行けるか × このコースの前有利 --------------------------------
    hist = out.loc[done]
    if bias_before is not None:
        hist = hist.loc[hist["race_date"] < pd.Timestamp(bias_before)]
    ckey = ["venue", "surface", "distance"]
    hist = hist.assign(_q=np.minimum((hist["corner_ratio"] * 4).fillna(-1), 3).astype(int))
    hist = hist.loc[hist["_q"] >= 0]
    pos = _bias_table(hist, ckey, "_q")
    # 位置取りは同じ芝ダ・距離帯の走りから。芝1600やダ1800で後ろにいた馬を、
    # ダ1200でも後ろと決めつけない（マーゴットゲインがそうだった）。
    # 同じ帯の走りが無ければ、全体の直近5走に落とす。
    out["_cr"] = out["corner_ratio"].fillna(0.0)
    out["_crn"] = out["corner_ratio"].notna().astype(float)
    cr_sum = _prior_sum(out, band, "_cr")
    cr_n = _prior_sum(out, band, "_crn")
    pos_here = (cr_sum / cr_n.replace(0, np.nan))
    out["apt_pos_ratio"] = pos_here.fillna(out["h_corner_ratio_r5"]).values
    out["_q"] = np.minimum((out["apt_pos_ratio"] * 4).fillna(-1), 3).astype(int)
    m = out[ckey + ["_q"]].merge(pos, on=ckey + ["_q"], how="left")
    m_pos = (m["sum"] + K_BIAS * m["course_rate"]) / (m["count"] + K_BIAS) / m["course_rate"]
    out["apt_m_pos"] = m_pos.fillna(1.0).clip(*CLIP).values
    out.loc[out["_q"] < 0, "apt_m_pos"] = 1.0

    # --- 枠 × このコース ----------------------------------------------------
    wk = _bias_table(hist.dropna(subset=["waku"]), ckey, "waku")
    m = out[ckey + ["waku"]].merge(wk, on=ckey + ["waku"], how="left")
    m_waku = (m["sum"] + 2 * K_BIAS * m["course_rate"]) / (
        m["count"] + 2 * K_BIAS
    ) / m["course_rate"]
    out["apt_m_waku"] = m_waku.fillna(1.0).clip(*CLIP).values

    parts = [
        "apt_m_course", "apt_m_dir", "apt_m_band", "apt_m_soft",
        "apt_m_layoff", "apt_m_pos", "apt_m_waku",
    ]
    out["apt_log"] = np.log(out[parts].astype(float)).sum(axis=1)
    out["_wet_label"] = out["_wet"]
    out["apt_gc_class"] = out["_gc"]
    out = out.drop(columns=[c for c in out.columns if c.startswith("_") and c != "_wet_label"])
    return out


def adjust(p: pd.Series, frame: pd.DataFrame, weight: float) -> pd.Series:
    """モデルの確率 p に適性の倍率を weight の強さで掛け、レース内の合計を保つ。"""
    raw = p * np.exp(weight * frame["apt_log"].fillna(0.0))
    total_before = p.groupby(frame["race_id"]).transform("sum")
    total_after = raw.groupby(frame["race_id"]).transform("sum")
    return (raw * total_before / total_after).clip(0, 0.95)


def _fmt(row: pd.Series, prefix: str) -> str:
    w = [int(row[f"{prefix}{k}"]) for k in (1, 2, 3)]
    n = int(row[f"{prefix}n"])
    return f"{w[0]}-{w[1]}-{w[2]}-{n - sum(w)}"


def reasons(row: pd.Series, threshold: float = 1.15) -> list[str]:
    """倍率が大きく動いた要素だけ、実績の数で理由にする。"""
    out: list[str] = []

    def mark(m: float) -> str:
        return "適性＋" if m > 1 else "適性－"

    def big(m: float) -> bool:
        return m >= threshold or m <= 1 / threshold

    course = f"{row['venue']}{row['surface']}{int(row['distance'])}"
    if big(row["apt_m_course"]) and row["apt_course_n"] > 0:
        side = "道悪" if row["_wet_label"] else "良"
        out.append(
            f"{course}（{side}）は{_fmt(row, 'apt_course_')}（{mark(row['apt_m_course'])}）"
        )
    band = BAND_LABEL.get(str(row["band"]), str(row["band"]))
    if big(row["apt_m_dir"]) and row["apt_dir_n"] > 0:
        out.append(
            f"{row['surface']}{band}の{row['direction']}回りは{_fmt(row, 'apt_dir_')}"
            f"（{mark(row['apt_m_dir'])}）"
        )
    if big(row["apt_m_band"]) and row["apt_band_n"] > 0:
        out.append(f"{row['surface']}{band}は{_fmt(row, 'apt_band_')}（{mark(row['apt_m_band'])}）")
    if big(row["apt_m_soft"]):
        if row["apt_gc_n"] > 0:
            out.append(
                f"{GOING_LABEL[int(row['apt_gc_class'])]}は{_fmt(row, 'apt_gc_')}"
                f"（{mark(row['apt_m_soft'])}）"
            )
        elif row["apt_soft_n"] > 0:
            out.append(f"道悪は{_fmt(row, 'apt_soft_')}（{mark(row['apt_m_soft'])}）")
        elif row["apt_sire_soft_n"] > 0:
            out.append(
                f"道悪は未経験。父の産駒は道悪で3着内{row['apt_sire_soft_rate']:.0%}"
                f"（{int(row['apt_sire_soft_n'])}走・{mark(row['apt_m_soft'])}）"
            )
    if row["apt_m_layoff"] < 1:
        out.append(f"{int(row['h_days_since'])}日の休み明けで、休み明けの好走歴なし（適性－）")
    if big(row["apt_m_pos"]):
        where = "前に行ける" if row["apt_m_pos"] > 1 else "後ろからになりやすい"
        out.append(f"{where}。このコースの位置取りの有利不利（{mark(row['apt_m_pos'])}）")
    if big(row["apt_m_waku"]):
        out.append(f"{int(row['waku'])}枠（このコースで{mark(row['apt_m_waku'])}）")
    return out


FACTOR_LABEL = {
    "apt_m_course": "コース", "apt_m_dir": "回り", "apt_m_band": "距離",
    "apt_m_soft": "馬場", "apt_m_layoff": "休み明け", "apt_m_pos": "位置",
    "apt_m_waku": "枠",
}


def summary(row: pd.Series, threshold: float = 1.15) -> dict:
    """予想の JSON に残す適性の要約。回顧で「なぜ走ったか」を見るために使う。

    total は倍率の積。plus / minus は大きく動いた要素の名前、facts は実績の数。
    """
    plus, minus = [], []
    for col, label in FACTOR_LABEL.items():
        m = float(row.get(col, 1.0) or 1.0)
        if m >= threshold:
            plus.append(label)
        elif m <= 1 / threshold:
            minus.append(label)
    return {
        "total": round(float(np.exp(row.get("apt_log", 0.0) or 0.0)), 2),
        "plus": plus,
        "minus": minus,
        "facts": reasons(row, threshold),
    }
