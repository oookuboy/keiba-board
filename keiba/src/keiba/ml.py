"""能力モデルの学習と推論。

ルールベースの手置き重みでは、能力順位が市場の半分しか当たらなかった
（エンジン1位の3着内率 40.3% 対 1番人気 66.8%）。穴を狙うには、まず市場と
同等以上に順位を当てられる必要がある。そこを機械学習に置き換える。

**SKILL.md の教訓は捨てていない。** 血統適性・条件一変・展開・降級といった
観点は dataset.py で特徴量として残してあり、モデルがそれぞれの重みを
データから決める。手で置いた重みが、実測に基づく重みに変わっただけ。

**オッズは依然として入力に入れない。** 学習データからも除いてある
（dataset.assert_no_market_leakage）。人気を見るのは confidence.py だけ。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from keiba.dataset import CATEGORICAL, FEATURE_COLUMNS, TARGET

log = logging.getLogger(__name__)

MODEL_PATH = Path("keiba/config/model.txt")
META_PATH = Path("keiba/config/model.json")

PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,
    # 乱数を固定する。feature_fraction と bagging_fraction が毎回別の抽選に
    # なるため、**同じデータで同じ比較をしても結果が動く**。
    #
    # 実際に踏んだ。調教あり／なしを同じ条件で2度測ったところ、
    #
    #     1回目  0.75422 → 0.75717   (+0.00295)
    #     2回目  0.75905 → 0.76246   (+0.00341)
    #
    # 差（+0.003）は再現したが、**絶対値が 0.005 動いた**。差より揺れのほうが
    # 大きい。切り口ごとに見ると人気薄が +0.0083 → +0.0044 で、採否の閾値
    # (0.005) をまたいでしまった。固定しないと「効いた」の判断が抽選になる。
    "seed": 20260815,
    "deterministic": True,
}


@dataclass
class TrainResult:
    booster: lgb.Booster
    auc: float
    best_iteration: int
    train_rows: int
    valid_rows: int
    importance: list[tuple[str, int]]

    def report(self) -> str:
        lines = [
            "=" * 58,
            "能力モデルの学習結果",
            "=" * 58,
            f"  学習 {self.train_rows:,} 行 / 検証 {self.valid_rows:,} 行",
            f"  AUC {self.auc:.4f}  （最良反復 {self.best_iteration}）",
            "-" * 58,
            "  効いている特徴量（上位15）",
        ]
        for name, gain in self.importance[:15]:
            lines.append(f"    {name:<28} {gain:>10,}")
        lines.append("=" * 58)
        return "\n".join(lines)


def split(df: pd.DataFrame, valid_from: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """時系列で切る。

    ランダム分割は絶対にしない。同じレースの馬が学習と検証に散らばると、
    レース内の相対特徴量を通じて答えが漏れる。
    """
    boundary = pd.Timestamp(valid_from)
    return df[df["race_date"] < boundary], df[df["race_date"] >= boundary]


def train(
    df: pd.DataFrame,
    valid_from: date,
    rounds: int = 2000,
    features: list[str] | None = None,
) -> TrainResult:
    """3着以内に入る確率を学習する。

    features を渡すと特徴量の集合を差し替えられる。既定は FEATURE_COLUMNS。
    調教を足した／足さないを同じ分割で比べるためだけの引数で、本番の学習は
    既定のまま使う。ここを本番で使い分けると、保存したモデルと predict の
    列がずれる（62列のコードで56列のモデルを読む形を既に踏んでいる）。
    """
    features = features or FEATURE_COLUMNS
    if "finished" in df.columns and not df["finished"].all():
        # まだ走っていない行は目的変数が 0 に潰れている（着順が無いため）。
        # 混ぜると「今週の全馬は3着以内に入らなかった」を学習する。
        dropped = int((~df["finished"]).sum())
        log.info("未走 %d 行を学習から外す", dropped)
        df = df[df["finished"]]
    train_df, valid_df = split(df, valid_from)
    if train_df.empty or valid_df.empty:
        raise ValueError(f"分割が空になった（境界 {valid_from}）")

    log.info(
        "学習 %s〜%s / 検証 %s〜%s",
        train_df["race_date"].min().date(), train_df["race_date"].max().date(),
        valid_df["race_date"].min().date(), valid_df["race_date"].max().date(),
    )

    train_set = lgb.Dataset(
        train_df[features], label=train_df[TARGET],
        categorical_feature=CATEGORICAL, free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid_df[features], label=valid_df[TARGET],
        categorical_feature=CATEGORICAL, reference=train_set, free_raw_data=False,
    )

    booster = lgb.train(
        PARAMS, train_set, num_boost_round=rounds,
        valid_sets=[valid_set], valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(200),
        ],
    )

    gains = booster.feature_importance("gain")
    importance = sorted(
        zip(booster.feature_name(), (int(g) for g in gains)),
        key=lambda kv: -kv[1],
    )
    return TrainResult(
        booster=booster,
        auc=booster.best_score["valid"]["auc"],
        best_iteration=booster.best_iteration,
        train_rows=len(train_df),
        valid_rows=len(valid_df),
        importance=importance,
    )


def save(result: TrainResult, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    result.booster.save_model(str(model_path), num_iteration=result.best_iteration)
    meta_path.write_text(
        json.dumps(
            {
                "auc": result.auc,
                "best_iteration": result.best_iteration,
                "train_rows": result.train_rows,
                "valid_rows": result.valid_rows,
                "features": FEATURE_COLUMNS,
                "importance": dict(result.importance[:40]),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    log.info("モデルを保存: %s", model_path)


def load(model_path: Path = MODEL_PATH) -> lgb.Booster | None:
    if not model_path.exists():
        log.warning("%s が無い。ルールベースのスコアで動く", model_path)
        return None
    return lgb.Booster(model_file=str(model_path))


def predict(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    """3着以内に入る確率を返す。0〜100 のスコアに写して使う。

    列は **モデル自身が持っている名前** で選ぶ。FEATURE_COLUMNS で選ぶと、
    特徴量を足した直後に「74列のコードで65列のモデルを読む」形になって落ちる。
    再学習は月1なので、足してから差し替わるまでの間が必ず存在する。

    足りない列があるときだけ落とす。そちらは黙って進むと、モデルが見て
    いるはずの情報が欠けたまま予想が出てしまう。
    """
    names = booster.feature_name()
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"モデルが要求する列が表に無い: {missing}")

    extra = [c for c in FEATURE_COLUMNS if c not in names]
    if extra:
        # 特徴量を足したがモデルがまだ古い状態。予想は出せるので止めない。
        log.info("モデルがまだ使っていない特徴量: %s（再学習で入る）", extra)
    return booster.predict(df[names], num_iteration=booster.best_iteration)


def evaluate_ranking(df: pd.DataFrame, scores: np.ndarray) -> dict:
    """モデルの順位付けを、市場（人気）と同じ土俵で比べる。

    ここが本題。回収率の前に、まず市場を上回れているかを見る。
    """
    work = df[["race_id", "umaban", TARGET, "finish_pos", "market_popularity"]].copy()
    work["score"] = scores

    model_top1 = market_top1 = 0
    model_win = market_win = 0
    model_top3 = market_top3 = 0
    races = 0

    for _, group in work.groupby("race_id", observed=True):
        pops = group.dropna(subset=["market_popularity"])
        if len(pops) < 3 or group[TARGET].sum() < 3:
            continue
        races += 1

        placed = set(group.loc[group[TARGET] == 1, "umaban"])
        winner = group.loc[group["finish_pos"] == 1, "umaban"]
        winner = winner.iloc[0] if len(winner) else None

        best = group.sort_values("score", ascending=False)
        fav = pops.sort_values("market_popularity")

        model_top1 += best.iloc[0]["umaban"] in placed
        market_top1 += fav.iloc[0]["umaban"] in placed
        model_win += best.iloc[0]["umaban"] == winner
        market_win += fav.iloc[0]["umaban"] == winner
        model_top3 += len(set(best.head(3)["umaban"]) & placed)
        market_top3 += len(set(fav.head(3)["umaban"]) & placed)

    if not races:
        return {}
    return {
        "races": races,
        "model_top1_in3": model_top1 / races,
        "market_top1_in3": market_top1 / races,
        "model_win": model_win / races,
        "market_win": market_win / races,
        "model_top3_hits": model_top3 / races,
        "market_top3_hits": market_top3 / races,
    }


def format_ranking(stats: dict) -> str:
    if not stats:
        return "評価できるレースがない"
    beat = stats["model_top1_in3"] > stats["market_top1_in3"]
    return "\n".join([
        "=" * 58,
        f"順位付けの精度比較（{stats['races']:,} レース）",
        "=" * 58,
        f"                     モデル      1番人気",
        f"  1位が3着以内     {stats['model_top1_in3']:7.1%}     {stats['market_top1_in3']:7.1%}",
        f"  1位が勝利        {stats['model_win']:7.1%}     {stats['market_win']:7.1%}",
        f"  上位3頭の的中    {stats['model_top3_hits']:7.2f}/3   {stats['market_top3_hits']:7.2f}/3",
        "-" * 58,
        "  " + ("モデルが市場を上回っている" if beat else "モデルはまだ市場に届いていない"),
        "=" * 58,
    ])
