"""回顧が止まらないこと、そして回顧の数字が読めることの検証。

このプロジェクトで一番よく起きた壊れ方は「集めたのに使っていない」だった。
回顧はその親戚で、**回りはするが何も学べない**という形で壊れる。実際に:

  1. 当日回顧の cron が4時間遅れ、JST の日付が翌日へ回って、まだ走っていない
     8/30 を回顧した。「対象36R・買い0R・結果未照合」という、成績に見えるが
     成績ではない記録が残った（2026-08-29）。
  2. 着順が取れなかった日は実戦ログに「翌日に再実行すること」と書くだけで、
     人が忘れれば成績が永久に残らなかった。
  3. 教訓の効き目を「効いた回数／外した回数」で出していた。よく発火する教訓が
     両方の一覧で1位になるだけで、効き目は何も分からない。
  4. reviews テーブルに書く教訓の数え上げが、レース単位ではなく
     「その日のそこまでの累計」になっていた。

どれも例外にならず、ログも緑のまま通る。ここで機械に見張らせる。
"""

from __future__ import annotations

import json
import pathlib

import pytest

from keiba import review
from keiba.cli import _unreviewed_days, build_parser


def horse(umaban: int, mark: str, score: float, reasons: list[str] | None = None) -> dict:
    return {
        "umaban": umaban,
        "name": f"ウマ{umaban}",
        "mark": mark,
        "score": score,
        "reasons": reasons or [],
    }


def race(marks: list[int], *, bets: int = 1) -> dict:
    """印を5頭、評価順に打ったレース。"""
    return {
        "race_id": "202601010101",
        "venue": "札幌",
        "race_no": 1,
        "confidence": "◎",
        "confidence_reason": "",
        "spend": bets * 100,
        "horses": [horse(n, "◎", 60 - i) for i, n in enumerate(marks)],
        "bets": [
            {"type": "三連複", "combination": "1-2-3", "amount": 100, "why": ""}
        ]
        * bets,
    }


class FakeCard:
    def __init__(self, payouts):
        self.payouts = payouts


class Payout:
    def __init__(self, bet_type, combination, payout):
        self.bet_type = bet_type
        self.combination = combination
        self.payout = payout


# --- 対案の計算 ---------------------------------------------------------


def test_box_hits_exactly_when_the_top3_are_all_marked() -> None:
    """ボックスは印の3頭組を全部買うので、的中条件は「1〜3着が全員印」。

    買い目を組み直さずに判定できるのが肝心。見送ったレース（買い目0点）にも
    同じ計算が効くので、「見送りが正しかったか」を後から測れる。
    """
    card = FakeCard([Payout("三連複", "2-5-9", 4520)])
    marked = race([9, 5, 2, 7, 3])

    alt = review._alternatives(marked, card, [5, 9, 2])
    # 5頭ボックス＝10点＝1,000円
    assert alt["box_all"] == (1000, 4520)

    # 3着が印の外なら当たらない
    miss = review._alternatives(marked, FakeCard([]), [5, 9, 11])
    assert miss["box_all"] == (1000, 0)


def test_skipped_races_are_measured_but_not_counted_as_bought() -> None:
    """見送ったレースも対案では測る。ただし「買ったRだけ」には入れない。"""
    card = FakeCard([Payout("三連複", "2-5-9", 4520)])
    skipped = race([9, 5, 2, 7, 3], bets=0)
    skipped["bets"] = []

    alt = review._alternatives(skipped, card, [5, 9, 2])
    assert "box_all" in alt
    assert "box_bet" not in alt, "見送りの損得が測れなくなる"


def test_place_bet_uses_the_top_three_by_score() -> None:
    """複勝は評価上位3頭。印の順ではなくスコア順で決める。"""
    card = FakeCard(
        [Payout("複勝", "9", 150), Payout("複勝", "5", 320), Payout("複勝", "1", 110)]
    )
    alt = review._alternatives(race([9, 5, 2, 7, 3]), card, [9, 5, 1])
    # 9 と 5 は印上位3頭に入っているが、1 は入っていない
    assert alt["place3"] == (300, 150 + 320)


# --- 教訓を率で出す -----------------------------------------------------


def test_lessons_are_reported_as_rates_against_the_base() -> None:
    """回数ではなく率。母数と印全体の率を必ず添える。

    回数だけだと、よく発火する教訓が「効いた」「外した」の両方で1位になる。
    実際 教訓2 展開読み がずっとそうなっていた。
    """
    lines = review._lesson_lines(
        {
            "lessons_worked": {"教訓A": 9, "教訓B": 1},
            "lessons_failed": {"教訓A": 18, "教訓B": 1},
            "marks": {"total": 180, "placed": 64},
        }
    )
    text = "\n".join(lines)
    assert "印全体の3着内率 35.6%" in text
    assert "n=180" in text
    assert "教訓A 33.3% (9/27" in text
    # 印全体を上回るほうが先に出る
    assert text.index("教訓B") < text.index("教訓A")


def test_no_lesson_lines_without_a_denominator() -> None:
    """母数が無いのに率を出さない。"""
    assert review._lesson_lines({"lessons_worked": {"教訓A": 3}, "marks": {}}) == []


# --- 通算台帳 -----------------------------------------------------------


def day_file(tmp_path: pathlib.Path, date: str, summary: dict) -> None:
    (tmp_path / f"{date}.json").write_text(
        json.dumps({"date": date, "races": [], "summary": summary}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_the_ledger_shows_what_one_big_payout_is_carrying() -> None:
    """回収率の右に「最高配当を除くと」を必ず出す。

    三連複は的中が少なく配当の幅が大きいので、数十日ぶんでも1本で全体が立つ。
    実際、印ボックスの通算 112.5% は 8/16 札幌11R の43,170円1本で立っていて、
    それを引くと 83.1% になる。この列が無いと、偶然の1本に合わせて設定を
    いじることになる。
    """
    tmp = pathlib.Path(pytest.importorskip("tempfile").mkdtemp())
    day_file(
        tmp,
        "2026-08-16",
        {
            "races": 36,
            "graded": 36,
            "spend": 10000,
            "returned": 12000,
            "best_race": 9000,
            "alternatives": {"box_bet": {"spend": 10000, "returned": 12000, "best": 9000}},
        },
    )
    text = "\n".join(review.build_ledger(tmp))
    assert "120.0%" in text          # 見たままの回収率
    assert "30.0%" in text           # 最高配当を除いた回収率
    assert "最高配当を除くと" in text


def test_the_ledger_counts_only_reviewed_days() -> None:
    """未照合の日を「投資あり・払戻0円」として混ぜない。"""
    tmp = pathlib.Path(pytest.importorskip("tempfile").mkdtemp())
    day_file(tmp, "2026-08-29", {"races": 36, "graded": 36, "spend": 1000, "returned": 1500})
    day_file(tmp, "2026-08-30", {"races": 36, "graded": 0, "spend": 1000})
    text = "\n".join(review.build_ledger(tmp))
    assert "回顧済み 1日" in text
    assert "150.0%" in text


def test_the_ledger_block_is_replaced_not_appended(tmp_path: pathlib.Path) -> None:
    """毎回の回顧で積み直す。追記していくと通算が何本も並ぶ。"""
    data = tmp_path / "data"
    data.mkdir()
    lessons = tmp_path / "LESSONS.md"
    day_file(data, "2026-08-29", {"races": 1, "graded": 1, "spend": 1000, "returned": 1500})

    review.write_ledger(data, lessons)
    day_file(data, "2026-08-30", {"races": 1, "graded": 1, "spend": 1000, "returned": 500})
    review.write_ledger(data, lessons)

    text = lessons.read_text(encoding="utf-8")
    assert text.count(review.LEDGER_START) == 1
    assert text.count("## 通算") == 1
    assert "回顧済み 2日" in text
    assert "100.0%" in text


# --- 取りこぼしを拾い直す -----------------------------------------------


def test_catch_up_finds_days_that_never_got_graded(tmp_path: pathlib.Path) -> None:
    """着順が取れなかった日を機械が拾い直せること。

    db.netkeiba への反映は遅れる。当日夜の回顧が0件で終わることがあり、
    以前は実戦ログに「翌日に再実行すること」と書いて人に投げていた。
    """
    day_file(tmp_path, "2026-08-22", {"races": 36, "graded": 36, "spend": 1000})
    day_file(tmp_path, "2026-08-23", {"races": 36, "graded": 0, "spend": 1000})
    day_file(tmp_path, "2026-08-28", {"races": 36, "graded": 20, "spend": 1000})

    assert _unreviewed_days(tmp_path, before="2026-08-29") == ["2026-08-23", "2026-08-28"]
    # 対象日そのものは含めない（呼び出し側で先に回顧している）
    assert "2026-08-28" not in _unreviewed_days(tmp_path, before="2026-08-28")


def test_provisional_days_are_never_caught_up(tmp_path: pathlib.Path) -> None:
    """暫定予想は買い目を組んでいないので、回顧の対象ではない。"""
    (tmp_path / "2026-08-30.json").write_text(
        json.dumps(
            {"date": "2026-08-30", "provisional": True, "summary": {"races": 36, "graded": 0}}
        ),
        encoding="utf-8",
    )
    assert _unreviewed_days(tmp_path, before="2026-09-01") == []


def test_review_accepts_catch_up() -> None:
    """ワークフローが渡す引数を argparse が解釈できること。"""
    args = build_parser().parse_args(["review", "--date", "2026-08-29", "--catch-up"])
    assert args.catch_up is True
    assert build_parser().parse_args(["review", "--date", "2026-08-29"]).catch_up is False
