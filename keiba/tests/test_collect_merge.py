"""出馬表を取り直しても、別経路で集めた情報を捨てないこと。

## 集めたものを次の手順が捨てていた

collect_upcoming は race_id をキーにカードを**まるごと**差し替えていた。
JRA公式の出馬表には厩舎コメントも調教も無いので、当日朝に馬体重とオッズを
取り直すたびに、金曜に集めたコメントが消えていた。

2026-09-05/06 が実際にそうなった。収集は4分7秒かけて正常に走り、72レース
ぶん取れていたのに、raw に残っていたのは **0件**。取れていないのではなく、
取ったあとに消していた。

この開発で「集めたのに使っていない」を何度も踏んでいる。今回はその変種で、
**集めて、書いて、次のステップが上書きしていた**。上書きの向きをテストで
固定する。
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

from keiba import collect
from keiba.models import Entry, Race, RaceCard, TrainerComment
from keiba.store import read_jsonl, write_jsonl


def race(race_id: str = "202604030408") -> Race:
    return Race(
        race_id=race_id,
        race_date=dt.date(2026, 9, 5),
        venue="新潟",
        venue_code="04",
        kai=3,
        nichi=4,
        race_no=8,
        name="テスト",
        surface="芝",
        distance=2000,
    )


def card(*, comments: int = 0, odds: float | None = None) -> RaceCard:
    return RaceCard(
        race=race(),
        entries=[
            Entry(race_id=race().race_id, umaban=1, horse_name="ウマ", horse_id="1",
                  market_odds=odds)
        ],
        comments=[
            TrainerComment(race_id=race().race_id, umaban=i + 1, body=f"コメント{i}")
            for i in range(comments)
        ],
    )


def test_出馬表の取り直しでコメントを消さない() -> None:
    """新しいカードにコメントが無ければ、古いほうを残す。"""
    kept = collect._keep_extras(card(comments=3), card(comments=0, odds=4.2))

    assert len(kept.comments) == 3, "出馬表を取り直すたびにコメントが消える"
    # 出馬表そのもの（オッズ・馬体重）は新しいほうで更新されるのが正しい
    assert kept.entries[0].market_odds == 4.2


def test_新しいコメントがあればそちらを使う() -> None:
    kept = collect._keep_extras(card(comments=3), card(comments=5))
    assert len(kept.comments) == 5


def test_初回はそのまま通す() -> None:
    assert collect._keep_extras(None, card(comments=2)).comments


def test_引き継ぐ対象に結果と払戻も入っている() -> None:
    """回顧のあとに出馬表を取り直しても、着順と払戻を消さないこと。

    コメントと同じ経路で消えうる。実際に踏む前に塞いでおく。
    """
    assert "results" in collect.EXTRA_FIELDS
    assert "payouts" in collect.EXTRA_FIELDS
    assert "comments" in collect.EXTRA_FIELDS


def test_書き出して読み直してもコメントが残る(tmp_path: pathlib.Path) -> None:
    """ファイルへの往復で落ちないこと。

    raw は gzip JSONL で、モデルの to_dict / from_dict を通る。ここで
    落ちると、メモリ上は正しいのに保存された時点で消える。
    """
    path = tmp_path / "2026-09-05.jsonl.gz"
    write_jsonl([card(comments=3)], path)
    assert len(list(read_jsonl(path))[0].comments) == 3


# --- 被覆率の分母 -------------------------------------------------------


def test_有料データが出ているレースだけを分母にする(tmp_path: pathlib.Path) -> None:
    """netkeiba が出さないレースを「取りこぼし」と数えないこと。

    2026-09-12 を数えるとこうなった。

        5R 6R（新馬） 9R 10R 11R  → 全頭ぶんある
        未勝利・1勝クラス          → 1頭も無い

    全出走馬を分母にすると 126/316 = 39.9% で、取りこぼしゼロでも「半分未満」
    として落ちる。毎週かならず鳴る警報がもう一つできあがっていた。

    出る／出ないはこちらでは決められない。決められるのは**出ているレースを
    取りこぼさないこと**だけなので、そこを分母にする。
    """
    import argparse
    import datetime as dt

    from keiba.cli import cmd_health
    from keiba.store import Store

    db = tmp_path / "t.db"
    with Store(db) as store:
        store.conn.executescript(
            """
            INSERT INTO races (race_id, race_date, venue, venue_code, race_no)
                VALUES ('A', '2026-09-12', '中山', '06', 9),
                       ('B', '2026-09-12', '中山', '06', 1);
            INSERT INTO entries (race_id, umaban, horse_id)
                VALUES ('A',1,'h1'), ('A',2,'h2'), ('B',1,'h3'), ('B',2,'h4');
            INSERT INTO comments (race_id, umaban, body)
                VALUES ('A',1,'上向き'), ('A',2,'変わらず');
            -- 調教はここでは論点でないので、全頭ぶん埋めておく
            INSERT INTO horse_workouts (horse_id, workout_date, course, rank)
                VALUES ('h1','2026-09-09','美浦W',1), ('h2','2026-09-09','美浦W',1),
                       ('h3','2026-09-09','美浦W',1), ('h4','2026-09-09','美浦W',1);
            """
        )
        store.conn.commit()

    args = argparse.Namespace(
        day=dt.date(2026, 9, 12), db=db, data_dir=tmp_path, floor=0.5, strict=True,
    )
    assert cmd_health(args) == 0, "出ているレースは全頭ぶんあるのに落ちている"

    health = json.loads((tmp_path / "health.json").read_text(encoding="utf-8"))
    c = health["comments"]
    assert c["rate"] == 0.5, "全体の被覆率は事実として残す"
    assert c["served_rate"] == 1.0, "出ているレースでは取りこぼしゼロ"
    assert (c["served_races"], c["races"]) == (1, 2)


def test_1レースも出ていなければ落とす(tmp_path: pathlib.Path) -> None:
    """分母を狭めたせいで「全滅」を見逃さないこと。

    出ているレースだけを分母にすると、0レースのときは 0/0 になる。素直に
    書くと率が計算できず素通りする。**経路が死んだ**のはいちばん重い失敗
    なので、ここは必ず鳴らす。
    """
    import argparse
    import datetime as dt

    from keiba.cli import cmd_health
    from keiba.store import Store

    db = tmp_path / "t.db"
    with Store(db) as store:
        store.conn.executescript(
            """
            INSERT INTO races (race_id, race_date, venue, venue_code, race_no)
                VALUES ('A', '2026-09-13', '中山', '06', 9);
            INSERT INTO entries (race_id, umaban, horse_id)
                VALUES ('A',1,'h1'), ('A',2,'h2');
            """
        )
        store.conn.commit()

    args = argparse.Namespace(
        day=dt.date(2026, 9, 13), db=db, data_dir=tmp_path, floor=0.5, strict=True,
    )
    assert cmd_health(args) == 1, "厩舎コメントが全滅しているのに通している"


def test_調教が何本あっても出走頭数は水増しされない(tmp_path: pathlib.Path) -> None:
    """1頭に調教が何本もぶら下がる。素直に COUNT(*) と書くと分母が壊れる。"""
    import argparse
    import datetime as dt

    from keiba.cli import cmd_health
    from keiba.store import Store

    db = tmp_path / "t.db"
    with Store(db) as store:
        store.conn.executescript(
            """
            INSERT INTO races (race_id, race_date, venue, venue_code, race_no)
                VALUES ('A', '2026-09-12', '中山', '06', 9);
            INSERT INTO entries (race_id, umaban, horse_id)
                VALUES ('A',1,'h1'), ('A',2,'h2');
            INSERT INTO comments (race_id, umaban, body)
                VALUES ('A',1,'上向き'), ('A',2,'変わらず');
            INSERT INTO horse_workouts (horse_id, workout_date, course, rank)
                VALUES ('h1','2026-09-09','美浦W',1),
                       ('h1','2026-09-05','美浦W',2),
                       ('h1','2026-09-02','美浦坂',3),
                       ('h2','2026-09-09','栗東CW',1);
            """
        )
        store.conn.commit()

    args = argparse.Namespace(
        day=dt.date(2026, 9, 12), db=db, data_dir=tmp_path, floor=0.5, strict=True,
    )
    assert cmd_health(args) == 0

    health = json.loads((tmp_path / "health.json").read_text(encoding="utf-8"))
    assert health["workouts"]["runners"] == 2, "調教の本数ぶん出走頭数が増えている"
    assert health["comments"]["runners"] == 2
    assert health["workouts"]["served_rate"] == 1.0


# --- 予想に効いたかを数える ---------------------------------------------


def health_args(tmp: pathlib.Path):
    import argparse
    return argparse.Namespace(data_dir=tmp)


def prediction(
    tmp: pathlib.Path, *, scored_by: str, cited: int, provisional: bool = False
) -> None:
    """印5頭のうち cited 頭の根拠に厩舎コメントが出る予想を書く。"""
    import json
    horses = [
        {
            "umaban": i + 1,
            "mark": "◎",
            "reasons": (
                ["厩舎コメントに前向きな語（上向き）"] if i < cited else ["父の複勝率"]
            ),
        }
        for i in range(5)
    ]
    (tmp / "2026-09-11.json").write_text(
        json.dumps({
            "scored_by": scored_by,
            "provisional": provisional,
            "races": [{"horses": horses}],
        }),
        encoding="utf-8",
    )


def test_予想がコメントを1頭も使っていなければ検出する(tmp_path: pathlib.Path) -> None:
    """9/5 に実際に起きた形。

    印を175頭に打って、厩舎コメントを根拠にしたのは0頭だった。DB の行数を
    数えるだけでは分からない。**出力を見る。**
    """
    from keiba.cli import _paid_data_reached_the_prediction

    prediction(tmp_path, scored_by="model", cited=0)
    bad = _paid_data_reached_the_prediction(health_args(tmp_path), "2026-09-11", {})
    assert any("厩舎コメント" in b for b in bad)


def test_使っていれば通す(tmp_path: pathlib.Path) -> None:
    from keiba.cli import _paid_data_reached_the_prediction

    prediction(tmp_path, scored_by="model", cited=2)
    assert _paid_data_reached_the_prediction(health_args(tmp_path), "2026-09-11", {}) == []


def test_暫定予想は裁かない(tmp_path: pathlib.Path) -> None:
    """木曜のプレビューを「有料データを使っていない」と数えない。

    有料データを引くのは金曜。収集の直後に置いてある予想は必ず木曜の暫定に
    なるので、ここで赤にすると毎週かならず鳴る警報ができあがる。2026-09-11
    の収集は実際にこれで落ちた（weights / コメント根拠0頭）。裁くのは土日朝の
    本予想だけにする。
    """
    from keiba.cli import _paid_data_reached_the_prediction

    prediction(tmp_path, scored_by="weights", cited=0, provisional=True)
    health: dict = {}
    assert _paid_data_reached_the_prediction(
        health_args(tmp_path), "2026-09-11", health
    ) == []
    assert health["prediction"]["provisional"] is True


def test_本予想になったら同じ内容でも裁く(tmp_path: pathlib.Path) -> None:
    """暫定を外した瞬間に、同じ中身が失敗として出ること。

    「暫定は見逃す」を入れたせいで本予想まで見逃す、では意味がない。
    """
    from keiba.cli import _paid_data_reached_the_prediction

    prediction(tmp_path, scored_by="weights", cited=0, provisional=False)
    bad = _paid_data_reached_the_prediction(health_args(tmp_path), "2026-09-11", {})
    assert any("厩舎コメント" in b for b in bad)
    assert any("学習モデル" in b for b in bad)


def test_手置きの重みで採点していたら検出する(tmp_path: pathlib.Path) -> None:
    """モデルがボードに届いていない、という壊れ方を過去にしている。"""
    from keiba.cli import _paid_data_reached_the_prediction

    prediction(tmp_path, scored_by="weights", cited=2)
    bad = _paid_data_reached_the_prediction(health_args(tmp_path), "2026-09-11", {})
    assert any("学習モデル" in b for b in bad)
