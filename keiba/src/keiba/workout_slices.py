"""調教が「どこで」効くかを見るための切り口。

全体平均で差が出なくても、特定の条件だけ効くことはある。むしろ調教は
そういう性質のデータだと考えるほうが自然で、たとえば休養明けの馬は
前走からの間隔が空いていて近走の形が読めないぶん、調教しか手がかりが無い。

## 切り口を先に決めておく理由

データを見てから切り口を探すと、**必ず良く見える切り口が見つかる**。
20個も試せば、効果がゼロでも1つは偶然「効いている」ように見える。それを
拾って本番に入れると、次の週に消える。

そこで、理由を説明できる切り口だけをここに固定し、測定はこの一覧に対して
だけ行う。思いついた切り口を後から足したくなったら、それは新しい仮説として
別途、別の期間で確かめる。

## 見つけたと思ったときの扱い

差が出た切り口は、**別の期間でもう一度確かめてから**でないと採用しない。
サンプルが小さいほど偶然の差は大きく出るので、行数も必ず併記する。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

# (名前, 理由, 判定関数)
#
# 理由が書けない切り口は入れない。「なんとなく効きそう」で足すと、
# 上の「20個試せば1つ当たる」に自分から突っ込むことになる。
SLICES: list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]] = [
    (
        "休養明け(90日以上)",
        "近走の形が読めない。調教しか状態の手がかりが無い",
        lambda d: d["h_days_since"] >= 90,
    ),
    (
        "小休養(60-89日)",
        "上と同じ理屈だが程度が軽い。効くなら休養明けより弱いはず",
        lambda d: (d["h_days_since"] >= 60) & (d["h_days_since"] < 90),
    ),
    (
        "キャリア2走以下",
        "過去走の統計が立たない。新馬・未勝利ほど調教の比重が上がる",
        lambda d: d["h_runs"] <= 2,
    ),
    (
        "馬場替わり",
        "芝⇔ダートは前走の内容がそのまま使えない。適性の手がかりが要る",
        lambda d: d["c_surface_switch"] == 1,
    ),
    (
        "クラス替わり",
        "相手が変わるので前走の着順の意味が変わる",
        lambda d: d["c_class_delta"].abs() >= 1,
    ),
    (
        "人気薄(6番人気以下)",
        "狙っている層そのもの。ここで効かないなら買い目には影響しない",
        lambda d: d["market_popularity"] >= 6,
    ),
]

# これ未満の行数しか無い切り口は「差が出た」と言わない。
# 小さい標本ほど偶然の差が大きく出るので、先に足切りしておく。
MIN_ROWS = 2000

# これ未満の AUC 差は誤差として扱う。
MIN_DELTA = 0.005


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC-AUC。実体は ml.roc_auc（同じ計算を2箇所に置かない）。"""
    from keiba.ml import roc_auc

    return roc_auc(labels, scores)


def compare(
    valid: pd.DataFrame,
    target: str,
    scores_without: np.ndarray,
    scores_with: np.ndarray,
) -> list[dict]:
    """切り口ごとに、調教あり／なしの AUC を並べる。

    返すのは事実だけ。「効いた」の判断は呼び出し側で、行数と差の大きさを
    見てから行う。
    """
    labels = valid[target].to_numpy()
    out: list[dict] = []

    for name, reason, predicate in SLICES:
        try:
            mask = predicate(valid).fillna(False).to_numpy()
        except KeyError:
            # 特徴量の名前が変わった場合。黙って飛ばさず、分かる形で残す
            out.append({"name": name, "reason": reason, "error": "列が無い"})
            continue

        rows = int(mask.sum())
        if rows == 0:
            out.append({"name": name, "reason": reason, "rows": 0})
            continue

        base = auc(labels[mask], scores_without[mask])
        test = auc(labels[mask], scores_with[mask])
        out.append(
            {
                "name": name,
                "reason": reason,
                "rows": rows,
                "positive_rate": float(labels[mask].mean()),
                "auc_without": base,
                "auc_with": test,
                "delta": None if base is None or test is None else test - base,
                "trustworthy": rows >= MIN_ROWS,
            }
        )
    return out


def format_slices(rows: list[dict], subject: str = "調教") -> str:
    """人が読む形にする。行数を必ず併記する（小さい標本を信じないため）。

    subject は何を足した比較かの名前。見出しを固定していたせいで、時計を
    測っているのに「調教なし → 調教あり」と表示していた。読み違えるので
    呼び出し側から渡す。
    """
    lines = [
        "=" * 78,
        f"切り口ごとの AUC（{subject}なし → {subject}あり）",
        "=" * 78,
        f"{'切り口':<20}{'行数':>8}{'3着内率':>8}{'なし':>9}{'あり':>9}{'差':>9}",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"{row['name']:<20}  {row['error']}")
            continue
        if not row.get("rows"):
            lines.append(f"{row['name']:<20}{0:>8}  該当なし")
            continue
        if row["auc_without"] is None or row["auc_with"] is None:
            lines.append(f"{row['name']:<20}{row['rows']:>8}  片方のクラスしか無い")
            continue

        mark = ""
        if row["trustworthy"] and abs(row["delta"]) >= MIN_DELTA:
            mark = "  ←差あり" if row["delta"] > 0 else "  ←悪化"
        lines.append(
            f"{row['name']:<20}{row['rows']:>8}{row['positive_rate']:>8.1%}"
            f"{row['auc_without']:>9.4f}{row['auc_with']:>9.4f}"
            f"{row['delta']:>+9.4f}{mark}"
        )

    lines += [
        "-" * 78,
        f"※ 行数 {MIN_ROWS:,} 未満、または差 {MIN_DELTA} 未満は誤差として扱う。",
        "※ 差が出た切り口は、別の期間で再現するまで採用しない。",
    ]
    return "\n".join(lines)
