"""モデルと市場の食い違いに、妙味があるかを測る。

## なぜこれを最初に測るのか

モデルの順位付けは市場より下手だ（1位が3着以内 58.7% 対 65.6%）。しかし
**それは儲からないことを意味しない**。賭けで必要なのは平均的に上手いことでは
なく、市場が間違えている場所を見つけることだからだ。全体で劣っていても、
特定の場面で市場より正しければ、そこだけ買えば勝てる。

逆に、乖離のどこを見ても妙味が無いなら、能力モデルを積み増しても構造は
変わらない可能性が高い。次に何をするかがこの測定で決まる。

## 測り方

1. 学習期間だけを使って「単勝オッズ → 3着以内率」の実測カーブを作る。
   これが市場の織り込み。理論式ではなく実測なので、控除率も人気の偏りも
   最初から込みになっている。
2. 検証期間で、モデルの確率から市場の確率を引く。これが乖離。
3. 乖離の大きさで分けて、**実際の複勝払戻**で回収率を出す。

## シミュレーションしない

回収率は仮定を置いて計算するのではなく、payouts テーブルの**実払戻**を使う。
8頭立て未満で3着が付かないレースなども、払戻が存在しないぶん自動的に
0円として扱われる。自前で場合分けすると必ず間違える。

複勝の控除率は20%なので、**でたらめに買えば約80%**。ここを超えている区分が
あれば、そこには市場が取りこぼしている何かがある。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from keiba.store import Store

log = logging.getLogger(__name__)

# オッズの刻み。等間隔ではなく対数寄りにする。1倍台と50倍台を同じ幅で
# 切ると、人気馬側がすべて1つの箱に入ってしまう。
ODDS_BINS = [0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0,
             20.0, 30.0, 50.0, 100.0, 1e9]

# 単勝・複勝の控除率は20%。でたらめに買った場合の回収率がここ。
RANDOM_ROI = 0.80


def market_curve(train: pd.DataFrame, target: str) -> pd.Series:
    """学習期間の実測から「単勝オッズ → 3着以内率」を作る。

    検証期間のデータは一切使わない。ここに未来を混ぜると、市場が実際より
    賢く見えてしまい、モデルの乖離が過小評価される。
    """
    odds = train["market_odds"]
    usable = train[odds.notna()]
    bins = pd.cut(usable["market_odds"], ODDS_BINS)
    curve = usable.groupby(bins, observed=True)[target].mean()
    log.info("市場カーブを %d 区分で作成（学習 %d 行）", len(curve), len(usable))
    return curve


def apply_curve(frame: pd.DataFrame, curve: pd.Series) -> pd.Series:
    """検証期間の各行に、市場の織り込み確率を割り当てる。"""
    bins = pd.cut(frame["market_odds"], ODDS_BINS)
    return bins.map(curve).astype(float)


def payout_map(store: Store, bet_type: str) -> dict[tuple[str, int], int]:
    """(race_id, 馬番) → 払戻。実際に払われた金額だけを持つ。

    払戻の無い馬（＝外れ、あるいは複勝の付かない少頭数レースの3着）は
    このマップに現れない。呼び出し側は「無ければ0円」で扱えばよく、
    場合分けを自前で書かなくて済む。
    """
    out: dict[tuple[str, int], int] = {}
    for race_id, combination, payout in store.conn.execute(
        "SELECT race_id, combination, payout FROM payouts WHERE bet_type = ?",
        (bet_type,),
    ):
        try:
            out[(race_id, int(combination))] = int(payout)
        except (TypeError, ValueError):
            # 組番が単一の馬番でない券種が混ざった場合。黙って捨てない
            log.warning("払戻の組番を読めない: %s %s", race_id, combination)
    return out


def returns_for(frame: pd.DataFrame, payouts: dict[tuple[str, int], int]) -> np.ndarray:
    """各行を100円買ったときの払戻。"""
    return np.array(
        [
            payouts.get((race_id, int(umaban)), 0)
            for race_id, umaban in zip(frame["race_id"], frame["umaban"], strict=True)
        ],
        dtype=float,
    )


def by_edge(
    valid: pd.DataFrame,
    target: str,
    edge: np.ndarray,
    place_return: np.ndarray,
    win_return: np.ndarray,
    buckets: int = 10,
) -> pd.DataFrame:
    """乖離の大きさで区切って、実払戻の回収率を出す。"""
    work = valid[[target, "market_odds", "market_popularity"]].copy()
    work["edge"] = edge
    work["place"] = place_return
    work["win"] = win_return
    work = work[np.isfinite(work["edge"])]

    try:
        work["bucket"] = pd.qcut(work["edge"], buckets, duplicates="drop")
    except (ValueError, IndexError):
        work["bucket"] = "（全区分）"
    if work["bucket"].nunique() == 0:
        # 乖離が全行で同じだと区分が1つも作れない。空の表を返すと
        # 「データが無い」と読めてしまうので、1区分にまとめて返す。
        work["bucket"] = "（全区分）"
    grouped = work.groupby("bucket", observed=True)
    table = pd.DataFrame(
        {
            "行数": grouped.size(),
            "平均乖離": grouped["edge"].mean(),
            "平均人気": grouped["market_popularity"].mean(),
            "3着内率": grouped[target].mean(),
            "複勝回収率": grouped["place"].mean() / 100,
            "単勝回収率": grouped["win"].mean() / 100,
        }
    )
    return table


def format_table(table: pd.DataFrame, title: str) -> str:
    lines = [
        "=" * 78,
        title,
        "=" * 78,
        f"{'乖離':<22}{'行数':>8}{'平均人気':>9}{'3着内率':>9}"
        f"{'複勝回収':>10}{'単勝回収':>10}",
    ]
    for name, row in table.iterrows():
        mark = "  ←" if row["複勝回収率"] > 1.0 else ""
        lines.append(
            f"{str(name):<22}{int(row['行数']):>8}{row['平均人気']:>9.1f}"
            f"{row['3着内率']:>9.1%}{row['複勝回収率']:>10.1%}"
            f"{row['単勝回収率']:>10.1%}{mark}"
        )
    lines += [
        "-" * 78,
        f"※ 単複の控除率は20%。でたらめに買えば約 {RANDOM_ROI:.0%}。"
        " ここを超える区分にだけ意味がある。",
    ]
    return "\n".join(lines)
