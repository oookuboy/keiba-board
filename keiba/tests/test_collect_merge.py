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


# --- 予想に効いたかを数える ---------------------------------------------


def health_args(tmp: pathlib.Path):
    import argparse
    return argparse.Namespace(data_dir=tmp)


def prediction(tmp: pathlib.Path, *, scored_by: str, cited: int) -> None:
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
        json.dumps({"scored_by": scored_by, "races": [{"horses": horses}]}),
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


def test_手置きの重みで採点していたら検出する(tmp_path: pathlib.Path) -> None:
    """モデルがボードに届いていない、という壊れ方を過去にしている。"""
    from keiba.cli import _paid_data_reached_the_prediction

    prediction(tmp_path, scored_by="weights", cited=2)
    bad = _paid_data_reached_the_prediction(health_args(tmp_path), "2026-09-11", {})
    assert any("学習モデル" in b for b in bad)
