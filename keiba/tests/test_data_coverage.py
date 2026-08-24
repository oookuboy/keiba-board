"""集めているのに予想に届いていない列を、機械で見つける。

## なぜこのテストが要るか

このプロジェクトで一番よく起きた壊れ方が「集めたのに使っていない」だった。

  - 学習モデルを作ったのに、ボードの予想は手置きの重みで動いていた
  - 調教を10時間かけて集めたのに、特徴量に入っていなかった
  - 天候・開催回・着差・所属は、DB に入っているのに SELECT すらしていなかった

どれも型でも例外でも捕まらない。予想は普通に出るし、ログにも何も出ない。
気づいたのは毎回、利用者に「それ使ってるの？」と聞かれたときだった。

そこで、DB のスキーマと dataset.py を突き合わせて機械で見張る。列を足したら
「特徴量にする」か「使わない理由を書く」かのどちらかを強制する。
"""

from __future__ import annotations

import pathlib
import re

from keiba.store import SCHEMA

SRC = pathlib.Path(__file__).parents[1] / "src/keiba"

# 予想に使わない列と、その理由。**理由が書けないなら使うべき。**
# ここに足すのは「使わない」という判断であって、忘れていることの言い訳ではない。
EXCLUDED = {
    # --- 識別子・表示名 ---
    "race_id": "キー",
    "umaban": "キー（別途 draw_ratio として使用）",
    "horse_id": "キー",
    "horse_name": "表示用。能力とは無関係",
    "jockey": "表示用。同定は jockey_id",
    "trainer": "表示用。同定は trainer_id",
    "venue_code": "venue と一対一。重複",
    "name": "レース名のテキスト。格は grade と race_class で持っている",
    # --- 意図的に隔離しているもの ---
    "market_odds": "オッズ隔離。能力評価に人気を入れない（confidence だけが見る）",
    "market_popularity": "同上",
    # --- 中身が無い ---
    "prize": "実データで 0.7% しか入っていない",
    "dam": "母名は単体では効かない。血統は sire / damsire で拾う",
    # --- 別の形にして使っている ---
    "scratched": "取消の除外条件として WHERE で使用",
    "corners": "1コーナー通過順として corner_ratio に変換",
    "margin": "馬身の数値 margin_len に変換",
    "post_time": "時台だけ取って post_hour に変換",
    "time_sec": "タイム指数（speed.py）の材料",
    "finish_pos": "目的変数そのもの（top3 / won / finish_ratio）",
    "race_class": "class_rank に変換",
    "grade": "class_rank に変換",
    "body_weight_diff": "絶対値 body_weight_abs_diff も併用",
}

# 予想に使うテーブル。払戻や予想結果の記録テーブルは対象外。
TABLES = ("races", "entries", "results")


def columns_of(table: str) -> list[str]:
    """CREATE TABLE から列名を取り出す。

    1行に複数列が並ぶ書き方（`race_id TEXT NOT NULL, umaban INTEGER NOT NULL,`）
    をしているので、行ではなくカンマで割る。PRIMARY KEY (a, b) の中の
    カンマを拾わないよう、括弧の中身は先に落とす。
    """
    body = SCHEMA.split(f"CREATE TABLE IF NOT EXISTS {table} (", 1)[1]
    body = body.split(");", 1)[0]
    body = re.sub(r"--[^\n]*", "", body)          # コメント
    body = re.sub(r"\([^)]*\)", "", body)         # PRIMARY KEY (…) の中身
    out = []
    for part in body.split(","):
        part = part.strip()
        if not part or part.upper().startswith(("PRIMARY KEY", "UNIQUE", "FOREIGN")):
            continue
        out.append(part.split()[0])
    return out


def dataset_source() -> str:
    return (SRC / "dataset.py").read_text(encoding="utf-8")


def test_every_stored_column_is_used_or_explained() -> None:
    """DB に持っている列が、特徴量になっているか理由付きで除外されていること。

    落ちたときの直し方は2つ。
      1. その列を特徴量にする（dataset.py で参照する）
      2. 使わない理由を EXCLUDED に書く

    「忘れていた」を通さないのがこのテストの目的なので、**理由を書けないなら
    使うべき**。天候・開催回・着差・所属はこれで見つかった。
    """
    source = dataset_source()
    forgotten: list[str] = []

    for table in TABLES:
        for column in columns_of(table):
            if column in EXCLUDED:
                continue
            # SELECT していて、かつどこかで参照していれば使っている
            if re.search(rf'\b(?:r|e|res)\.{column}\b', source) and re.search(
                rf'["\']{column}["\']', source
            ):
                continue
            forgotten.append(f"{table}.{column}")

    assert not forgotten, (
        "DB に持っているのに予想へ届いていない列がある: "
        + ", ".join(forgotten)
        + "\n特徴量にするか、EXCLUDED に理由を書くこと"
    )


def test_the_check_actually_catches_a_forgotten_column() -> None:
    """この検証自体が機能していることを確かめる。

    実在の列を EXCLUDED から外し、参照も無い状態を作って検出できること。
    """
    source = "SELECT r.going FROM races"  # weather を参照しない偽のソース
    missed = [
        c for c in columns_of("races")
        if c not in EXCLUDED and not re.search(rf"\br\.{c}\b", source)
    ]
    assert "weather" in missed, "見落としを検出できていない"


def test_excluded_columns_still_exist() -> None:
    """使わない理由を書いた列が、実際にスキーマにあること。

    列名を変えたのに EXCLUDED が古いままだと、新しい名前が野放しになる。
    """
    known = {c for t in TABLES for c in columns_of(t)}
    stale = [c for c in EXCLUDED if c not in known]
    assert not stale, f"スキーマに無い列が EXCLUDED に残っている: {stale}"
