"""厩舎コメントを特徴量にする。

## なぜモジュールを分けるか

コメントは収集も判定コードもあったのに、予想に**1度も効いていなかった**。

    features.condition_changes  手置きの重み側で加点していた
    engine.run                  学習モデルで採点するとき、その加点を丸ごと捨てる

印は1頭も動いていなかった。根拠テキストだけが features から素通しで残るので、
ボード上は「使っている」ように見えていた。2026-09-11 に発覚。

直し方は「モデルの列にする」しかない。手置き側をいくら直しても、そちらは
使われていないので届かない。

## 欠損のままにする

netkeiba は全レースにコメントを出さない。2026-09-12 は24R中10R（新馬と
9R以降の特別戦）だけで、未勝利・1勝クラスには1頭も無い。出る／出ないは
こちらでは決められない。

無い馬を 0 で埋めない。「コメントが無い」と「何も言っていないコメント」は
別のことで、混ぜると後者を前者として学ぶ。調教の列と同じ方針。

## 語の向き

前向きな語と後ろ向きな語を数え、差し引きで向きを決める。語は実文から
起こしたもので、weights.yml に置いてある（features 側と共有する。表が
二重になると、片方だけ直して食い違う）。
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

# 語の表は weights.yml に置いてある。features 側（根拠テキストを書く側）と
# 同じ表を見る。二重に持つと、片方だけ直して食い違う。
DEFAULT_WEIGHTS = pathlib.Path(__file__).parents[2] / "config/weights.yml"


def load_keywords(config_dir: pathlib.Path | None = None) -> dict:
    """語の表を読む。渡されなければ同梱の weights.yml から。

    学習側は config_dir を持ち回っていないので、既定を持たせる。表の置き場が
    1つであることのほうが、引数で渡せることより大事。
    """
    import yaml

    path = (config_dir / "weights.yml") if config_dir else DEFAULT_WEIGHTS
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")).get(
        "condition_change", {}
    )

COMMENT_FEATURES = [
    "cm_has",       # コメントがあるか。無いレースが半分以上あるので、これ自体が情報
    "cm_positive",  # 前向きな語の数
    "cm_negative",  # 後ろ向きな語の数
    "cm_net",       # 差し引き
    "cm_len",       # 本文の長さ。書くことが多い馬ほど長い、という仮説
    "cm_net_rank",  # レース内での相対。絶対値より効くのは他の列と同じ
]


def load_comments(store) -> pd.DataFrame:
    """comments テーブルを読む。**本文は保存していない。**

    数え上げは収集の時点で済ませてある（collect._score_comments）。本文は
    netkeiba の有料会員向けの文章で、raw は公開リポジトリに入るので持たない。
    ここが読むのは数だけ。
    """
    return pd.read_sql_query(
        "SELECT race_id, umaban, positive, negative, length FROM comments",
        store.conn,
    )


def _sides(body: str, positive: list[str], negative: list[str]) -> tuple[int, int]:
    """本文の同じ場所を二度数えずに、前向き／後ろ向きの数を返す。

    語の表には重なりがある。「動きは水準以上」は `動きは水準` と `水準以上` の
    両方に当たり、素直に数えると前向き2として出る。重なりの多い言い回しほど
    強く出るという歪みになるので、当たった場所を前から見て重ならないものだけ
    採る。同じ場所に複数当たったときは長いほうを採る。

    features._keyword_sides と同じ規則。あちらは根拠テキストを書くために語その
    ものを返し、こちらは数だけ要る。規則が2つに割れないよう、変えるときは
    両方のテストを見ること。
    """
    spans: list[tuple[int, int, bool]] = []
    for words, good in ((positive, True), (negative, False)):
        for word in words:
            at = body.find(word)
            while at >= 0:
                spans.append((at, at + len(word), good))
                at = body.find(word, at + 1)

    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    pos = neg = 0
    end = -1
    for start, stop, good in spans:
        if start < end:
            continue
        end = stop
        if good:
            pos += 1
        else:
            neg += 1
    return pos, neg


def attach(frame: pd.DataFrame, comments: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """出走の表にコメントの列を足す。

    渡すものが空でも列はそろえる。**列の有無で学習と本番が変わらないように
    する。** 調教で踏んだ形（本番だけ列が無い）をここでも塞ぐ。
    """
    out = frame.copy()
    for name in COMMENT_FEATURES:
        out[name] = np.nan

    if comments is None or comments.empty:
        out["cm_has"] = 0.0
        return out

    scored = comments.copy()
    if "body" in scored.columns:
        # 収集の出口を通っていない古い raw から来た行だけ、ここで数える。
        # 新しい経路では body 列そのものが無い。
        positive = list(cfg.get("positive_keywords", []))
        negative = list(cfg.get("negative_keywords", []))
        counted = scored["body"].fillna("").map(
            lambda b: _sides(str(b), positive, negative)
        )
        scored["positive"] = [p for p, _ in counted]
        scored["negative"] = [n for _, n in counted]
        scored["length"] = scored["body"].fillna("").str.len()
        scored = scored.drop(columns=["body"])

    scored["cm_positive"] = scored["positive"].fillna(0)
    scored["cm_negative"] = scored["negative"].fillna(0)
    scored["cm_net"] = scored["cm_positive"] - scored["cm_negative"]
    scored["cm_len"] = scored["length"].fillna(0)
    scored = scored.drop(columns=["positive", "negative", "length"])

    out = out.drop(columns=COMMENT_FEATURES).merge(
        scored, on=["race_id", "umaban"], how="left"
    )
    out["cm_has"] = out["cm_net"].notna().astype(float)

    # レース内での相対化。同じレースの他馬と比べてどうか。
    # コメントの無い馬は順位をつけない（欠損のまま）。
    grouped = out.groupby("race_id", observed=True)["cm_net"]
    out["cm_net_rank"] = grouped.rank(pct=True)
    return out


def coverage(frame: pd.DataFrame) -> float:
    """コメントの入っている割合。収集が壊れたことに気づくための数。"""
    if frame.empty:
        return 0.0
    return float(frame["cm_net"].notna().mean())
