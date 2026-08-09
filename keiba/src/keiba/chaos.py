"""レースが荒れるかどうかを、市場の形から読めるかを測る。

## なぜこれを測るのか

いまの自信度は「**自分の上位3頭の人気和**」で決めている。つまり測っている
のは「モデルが市場に同意しているか」であって、「レースが荒れるか」ではない。

2026-08-09 の新潟11R がその典型だった。モデルの上位3頭が 3・2・1番人気
だったので人気和6＝「堅い」と判断して見送ったが、実際は 10人気→8人気→1人気
で三連複36,980円。モデルが人気馬を推したことと、レースが堅く収まることは
別の話であり、それを混同していた。

荒れるかどうかは我々の意見ではなく**市場の形**に出ているはずで、それは
オッズ層で使ってよい情報（confidence.py の担当範囲）。

## 何を見るか

発走前に分かるものだけを使う。

    1番人気のオッズ … 抜けた馬がいるか
    頭数           … 多いほど紛れる
    上位人気の厚み   … 1〜3番人気に支持が集中しているか
    クラス         … 未勝利・新馬は実績が薄く読みにくい

これらで三連複の配当分布がどれだけ変わるかを見る。変わらなければ、荒れは
市場の形から読めないということで、見送り判定を作り直す意味は無い。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from keiba.store import Store

log = logging.getLogger(__name__)

# 「高配当」の線。三連複でこのあたりを超えると、狙う意味のある配当になる。
BIG = 10_000
HUGE = 30_000


def load(store: Store) -> pd.DataFrame:
    """レース単位で、市場の形と三連複の配当を並べた表を作る。"""
    races = pd.read_sql_query(
        """
        SELECT r.race_id, r.race_date, r.field_size, r.race_class, r.grade,
               r.surface, r.distance,
               p.payout AS trio
        FROM races r
        JOIN payouts p ON p.race_id = r.race_id AND p.bet_type = '三連複'
        """,
        store.conn,
    )

    odds = pd.read_sql_query(
        """
        SELECT race_id, market_odds
        FROM entries
        WHERE scratched = 0 AND market_odds IS NOT NULL
        """,
        store.conn,
    )
    grouped = odds.groupby("race_id")["market_odds"]

    shape = pd.DataFrame(
        {
            "fav_odds": grouped.min(),
            "runners_priced": grouped.size(),
            # 上位3頭の支持の厚み。オッズの逆数＝おおよその織り込み確率。
            # 1に近いほど「上位3頭で決まりそう」と市場が見ている。
            "top3_share": grouped.apply(
                lambda s: (1 / s.nsmallest(3)).sum() / (1 / s).sum()
            ),
        }
    )
    df = races.join(shape, on="race_id")
    df = df[df["fav_odds"].notna()]
    log.info("対象 %d レース（三連複の払戻があるもの）", len(df))
    return df


def summarise(df: pd.DataFrame, by: pd.Series, label: str) -> pd.DataFrame:
    """区分ごとの三連複配当の分布。平均ではなく中央値と超過確率で見る。

    配当は極端に歪んだ分布なので、平均は1本の大穴に引きずられる。
    「いくらくらいが普通か（中央値）」と「大きいのが出る確率」を見る。
    """
    work = df.assign(_bucket=by)
    grouped = work.groupby("_bucket", observed=True)["trio"]
    out = pd.DataFrame(
        {
            "R数": grouped.size(),
            "配当中央値": grouped.median(),
            f"{BIG//1000}千超": grouped.apply(lambda s: (s >= BIG).mean()),
            f"{HUGE//1000}千超": grouped.apply(lambda s: (s >= HUGE).mean()),
        }
    )
    out.index.name = label
    return out


def format_table(table: pd.DataFrame) -> str:
    lines = [
        "-" * 62,
        f"{table.index.name:<22}{'R数':>7}{'配当中央値':>11}"
        f"{'1万超':>8}{'3万超':>8}",
    ]
    for name, row in table.iterrows():
        lines.append(
            f"{str(name):<22}{int(row['R数']):>7}{int(row['配当中央値']):>11,}"
            f"{row[f'{BIG//1000}千超']:>8.1%}{row[f'{HUGE//1000}千超']:>8.1%}"
        )
    return "\n".join(lines)


def report(df: pd.DataFrame) -> str:
    """発走前に分かる要素ごとに、配当分布がどう動くかを並べる。"""
    parts = [
        "=" * 62,
        "レースの荒れやすさは市場の形から読めるか",
        f"（三連複の払戻がある {len(df):,} レース）",
        "=" * 62,
    ]

    fav = pd.cut(
        df["fav_odds"],
        [0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 1e9],
        labels=["〜1.5倍", "1.5-2.0", "2.0-2.5", "2.5-3.0",
                "3.0-4.0", "4.0-5.0", "5.0倍〜"],
    )
    parts.append(format_table(summarise(df, fav, "1番人気のオッズ")))

    size = pd.cut(
        df["field_size"],
        [0, 9, 12, 14, 16, 99],
        labels=["〜9頭", "10-12頭", "13-14頭", "15-16頭", "17頭〜"],
    )
    parts.append("")
    parts.append(format_table(summarise(df, size, "頭数")))

    share = pd.qcut(df["top3_share"], 5, labels=["薄い", "やや薄", "中", "やや厚", "厚い"])
    parts.append("")
    parts.append(format_table(summarise(df, share, "上位3頭への支持の厚み")))

    is_low = df["race_class"].fillna("").str.contains("未勝利|新馬")
    parts.append("")
    parts.append(format_table(
        summarise(df, np.where(is_low, "未勝利・新馬", "それ以外"), "クラス")
    ))

    # 実際に使うのは組み合わせ。1番人気が緩く、頭数が多い形を切り出す。
    chaotic = (df["fav_odds"] >= 3.0) & (df["field_size"] >= 14)
    solid = (df["fav_odds"] < 2.0) & (df["field_size"] <= 12)
    parts.append("")
    parts.append(format_table(summarise(
        df,
        np.select([chaotic, solid], ["荒れ型(3倍以上×14頭以上)", "堅型(2倍未満×12頭以下)"],
                  default="その他"),
        "組み合わせ",
    )))
    parts.append("-" * 62)
    return "\n".join(parts)
