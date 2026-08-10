"""「勝ちそうな馬を買う」を素直にやったら、いくらになるかを測る。

## なぜこれを作ったか

これまでの買い方には「想定配当50倍以上の組み合わせしか買わない」という
フィルタが入っていた。これは**勝ちそうな馬を買うのではなく、配当が大きい
組み合わせを探す**設定であり、順序が逆になっていた。

さらに根本的な取り違えがある。**モデルが予測しているのは「3着以内に入るか」
なのに、買っていたのは三連複だった**。三連複は3頭すべてを1つの組で当てる
必要があり、モデルの精度の大部分を捨てている。しかも控除率が高い。

    複勝・単勝  控除率 20%  → でたらめに買えば約 80%
    三連複      控除率 27.5% → でたらめに買えば約 72.5%

モデルの予測対象と賭式が一致しているのは複勝。まずそこを素直に測る。

## 穴を無視するわけではない

人気で足切りは一切しない。16番人気でもモデルが上位に置いたなら買う。
「穴を狙う」のではなく「良いと思った馬を買ったら、たまたま穴だった」を
そのまま実行する形にする。

## 実払戻で測る

payouts テーブルの実際の払戻を使う。少頭数で複勝が3着まで付かないレースも、
払戻が存在しないぶん自動的に0円になる。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# 単勝・複勝の控除率は20%、三連複は27.5%。でたらめに買った場合の水準。
RANDOM_ROI = {"単勝": 0.80, "複勝": 0.80, "三連複": 0.725}


def top_n_by_race(valid: pd.DataFrame, scores: np.ndarray, n: int) -> pd.DataFrame:
    """各レースでモデル上位 n 頭を取る。人気は一切見ない。"""
    work = valid[["race_id", "umaban", "market_popularity"]].copy()
    work["score"] = scores
    work["rank"] = work.groupby("race_id", observed=True)["score"].rank(
        ascending=False, method="first"
    )
    return work[work["rank"] <= n]


def evaluate(
    picks: pd.DataFrame,
    payouts: dict[tuple[str, int], int],
    label: str,
    bet_type: str,
) -> dict:
    """選んだ馬を100円ずつ買ったときの成績。"""
    returns = np.array(
        [
            payouts.get((race_id, int(umaban)), 0)
            for race_id, umaban in zip(picks["race_id"], picks["umaban"], strict=True)
        ],
        dtype=float,
    )
    spent = len(picks) * 100
    return {
        "買い方": label,
        "賭式": bet_type,
        "点数": len(picks),
        "的中": int((returns > 0).sum()),
        "的中率": float((returns > 0).mean()) if len(picks) else 0.0,
        "投資": spent,
        "払戻": int(returns.sum()),
        "回収率": float(returns.sum() / spent) if spent else 0.0,
        "平均人気": float(picks["market_popularity"].mean()),
        "基準": RANDOM_ROI[bet_type],
    }


def format_results(rows: list[dict]) -> str:
    lines = [
        "=" * 82,
        "「勝ちそうな馬を買う」を素直にやった場合の実回収率",
        "（人気での足切りは一切しない。モデルの評価順にそのまま買う）",
        "=" * 82,
        f"{'買い方':<26}{'賭式':<7}{'点数':>7}{'的中率':>8}"
        f"{'平均人気':>9}{'回収率':>9}{'基準':>7}",
    ]
    for row in rows:
        mark = ""
        if row["回収率"] > 1.0:
            mark = "  ←黒字"
        elif row["回収率"] > row["基準"]:
            mark = "  ←基準超"
        lines.append(
            f"{row['買い方']:<26}{row['賭式']:<7}{row['点数']:>7,}"
            f"{row['的中率']:>8.1%}{row['平均人気']:>9.1f}"
            f"{row['回収率']:>9.1%}{row['基準']:>7.0%}{mark}"
        )
    lines += [
        "-" * 82,
        "※ 基準 = でたらめに買った場合の回収率（控除率ぶん）。"
        " ここを超えて初めて意味がある。",
        "※ 100%を超えて初めて勝ち。",
    ]
    return "\n".join(lines)
