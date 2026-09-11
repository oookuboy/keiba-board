"""集めたものが予想に届いているかを、**一覧で**確かめる。

## なぜ毎週1個ずつ見つかるのか

このプロジェクトは「集めたのに使っていない」を繰り返し踏んでいる。

    モデルを学習したのにボードに届いていない
    10時間かけた調教が特徴量に入っていない
    厩舎コメントの収集経路が無い
    集めたコメントを次の手順が上書きして消していた
    新馬のコメントが past_runs の門で止まっていた
    コメントの加点がモデル採点で丸ごと捨てられていた

**どれも見つけ方が同じだった。** 何かの拍子に1つ気づいて、直して、報告して、
次の週にまた別の1つが出てくる。見つけ方が「気づく」である限りこれは終わらない。

突き合わせる。集めたものを全部並べ、モデルが読んでいるかを機械的に照合する。
読んでいないものは **理由を書かないと通らない**。新しい収集経路を足したときに
黙って落ちることが無くなる。

ここが落ちたときの直し方は2つしかない。

  1. 特徴量にする（dataset.py で読む）
  2. 使わない理由を UNUSED に書く

「あとで直す」は選択肢に無い。書けば残るので、次に読んだ人が期限を決められる。
"""

from __future__ import annotations

import pathlib
import re

SRC = pathlib.Path(__file__).parents[1] / "src/keiba"

# 学習の入口。ここから読まれていなければ、そのデータは予想に効かない。
PIPELINE = ("dataset.py", "workout_features.py")

# 使わないものと、その理由。**理由の無い除外は書けない。**
UNUSED: dict[str, str] = {
    "bets": "予想の出力。学習の入力にすると自分の買い目を学ぶことになる",
    "predictions": "予想の記録。これも出力なので、学習の入力に戻すと自分の予想を追認するだけになる",
    "reviews": "回顧の記録。着順は results から取る",
    "workouts": "レース単位の調教。horse_workouts に統合済みで、こちらは空のまま",
    "payouts": (
        "払戻。発走前には存在しないので、その馬の特徴量にはできない。"
        "コース別の荒れやすさとして使う案はあるが、まだ測っていない"
    ),
    "horses": (
        "血統マスタ。sire/damsire は entries 側の列から読んでいるので"
        "経路としては届いている。sire_line（系統）だけ未使用"
    ),
    "past_runs": "0行。地方・海外・期間外の戦績で、収集経路がまだ無い",
}

# ここに名前がある限り、そのデータは予想に効いていない。**空にするのが目標。**
#
# comments: 2026-09-11 に発覚。収集も判定コードもあるのに、判定は score_horse
#   （手置きの重み）側にしかなく、engine.run は ml_scores があるとそちらを
#   丸ごと捨てる。つまり印は1つも動いていなかった。根拠テキストだけ残るので
#   ボード上は「使っているように見える」状態だった。
#   学習データ側のコメントが814行しか無いのが直せていない理由。まず何年ぶん
#   遡って取れるかを測る。
KNOWN_GAPS = {"comments"}


def _pipeline_source() -> str:
    return "\n".join((SRC / name).read_text(encoding="utf-8") for name in PIPELINE)


def _tables() -> list[str]:
    """schema.sql に書いてある表を全部。DB を作らずに読む。"""
    store = (SRC / "store.py").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", store)))


def _is_read(table: str, source: str) -> bool:
    """SQL として読まれているか。

    素直に `FROM x` だけ見ると取りこぼす。dataset.py は JOIN の種類を
    `{join}` で差し替えているので、キーワードが本文に無い行がある。
    別名つきの `results res ON` の形も拾う。
    """
    patterns = (
        rf"(?:FROM|JOIN)\s+{table}\b",   # FROM entries / JOIN races
        rf"\}}\s*{table}\b",            # {join} results
        rf"\b{table}\s+\w+\s+ON\b",    # results res ON …
    )
    return any(re.search(p, source, re.I) for p in patterns)


def test_集めた表はすべて説明がついている() -> None:
    """新しい表を足したら、特徴量にするか理由を書くかを迫る。

    どちらもしないと通らない。**黙って増やせないことがこのテストの本体。**
    """
    source = _pipeline_source()
    unexplained = [
        t for t in _tables()
        if not _is_read(t, source) and t not in UNUSED and t not in KNOWN_GAPS
    ]
    assert not unexplained, (
        f"集めているのにモデルが読まず、理由も書かれていない表: {unexplained}。"
        " dataset.py で読むか、UNUSED に理由を書くこと"
    )


def test_使わない理由が書いてある() -> None:
    """理由が空文字や「あとで」では通らない。"""
    for table, why in UNUSED.items():
        assert len(why) >= 20, f"{table} の除外理由が短すぎる: {why!r}"
        assert "あとで" not in why, f"{table}: 「あとで」は理由ではない"


def test_使うと書いた表は本当に読まれている() -> None:
    """UNUSED にも KNOWN_GAPS にも無い表は、実際に読まれていること。

    表を消したり名前を変えたりしたときに、この一覧が嘘になるのを防ぐ。
    """
    source = _pipeline_source()
    for table in _tables():
        if table in UNUSED or table in KNOWN_GAPS:
            continue
        assert _is_read(table, source), (
            f"{table} は使う側に分類されているのに dataset.py が読んでいない"
        )


def test_届いていないものが一覧に残っている() -> None:
    """KNOWN_GAPS は「予想に効いていない」と分かっているもの。

    ここが空になったとき、集めたものが全部予想に届いている。**空にするのが
    目標なので、増えたら気づけるように数を固定しておく。**
    """
    assert KNOWN_GAPS == {"comments"}, (
        "予想に届いていないものが増減した。README と併せて見直すこと: "
        f"{KNOWN_GAPS}"
    )


def test_厩舎コメントがモデルの列になったら一覧から外す() -> None:
    """comments を dataset.py で読んだ瞬間、このテストが KNOWN_GAPS を外させる。

    直したのに一覧が古いまま、という形を塞ぐ。
    """
    if _is_read("comments", _pipeline_source()):
        assert "comments" not in KNOWN_GAPS, (
            "comments を特徴量にしたなら KNOWN_GAPS から外すこと"
        )
