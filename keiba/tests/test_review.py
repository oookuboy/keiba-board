"""回顧の記録が「結果未照合」と「全部外した」を取り違えないこと。

2026-08-08 の初回運用で実際に起きた事故。開催当日の夜に回顧を走らせたが、
db.netkeiba にまだ着順が反映されておらず1件も照合できなかった。それなのに
実戦ログには

    投資 12,200円 → 払戻 0円 （回収率 0.0%・的中 0R）

と書かれ、20レース全部外したようにしか読めなかった。成績の記録としては
最悪の壊れ方なので、ここで固定する。
"""

from __future__ import annotations

from pathlib import Path

from keiba.review import append_lessons


def payload(graded: int, races: int = 3, **summary) -> dict:
    return {
        "date": "2026-08-08",
        "summary": {
            "races": races, "bet": races, "skipped": 0, "spend": 1000,
            "graded": graded, "returned": 0, "hits": 0, "roi": None,
            **summary,
        },
        "races": [],
    }


def test_ungraded_day_is_not_written_as_a_loss(tmp_path: Path) -> None:
    """1件も照合できていない日を、回収率0%の成績として書かない。"""
    log = tmp_path / "LESSONS.md"
    append_lessons(payload(graded=0), log)
    text = log.read_text(encoding="utf-8")

    assert "結果未照合" in text
    assert "回収率" not in text, "照合できていないのに回収率を書いている"
    assert "的中なし" not in text, "外したように読める書き方になっている"


def test_graded_day_records_the_numbers(tmp_path: Path) -> None:
    """照合できた日は、これまでどおり数字を書く。"""
    log = tmp_path / "LESSONS.md"
    append_lessons(payload(graded=3, returned=1500, hits=1, roi=150.0), log)
    text = log.read_text(encoding="utf-8")

    assert "結果未照合" not in text
    assert "回収率 150.0%" in text
    assert "的中 1R" in text


def test_partial_grading_is_stated(tmp_path: Path) -> None:
    """一部しか照合できていないなら、そう書く。黙って全体の成績にしない。"""
    log = tmp_path / "LESSONS.md"
    append_lessons(payload(graded=1, races=3, returned=500, hits=1, roi=50.0), log)
    assert "照合できたのは 1/3R" in log.read_text(encoding="utf-8")


def test_ungraded_record_is_replaced_when_results_arrive(tmp_path: Path) -> None:
    """当日夜に未照合で書いた記録を、翌日の再実行で成績に差し替える。

    差し替えないと、未照合の行が残ったまま成績が二度と記録されない。
    """
    log = tmp_path / "LESSONS.md"
    append_lessons(payload(graded=0), log)
    append_lessons(payload(graded=3, returned=1500, hits=1, roi=150.0), log)
    text = log.read_text(encoding="utf-8")

    assert text.count("## 2026-08-08") == 1, "同じ日が二重に書かれている"
    assert "結果未照合" not in text
    assert "回収率 150.0%" in text


def test_record_written_by_older_code_is_replaced(tmp_path: Path) -> None:
    """照合数の印が無い古い記録を、照合済みの内容で置き換える。

    「前回が『結果未照合』なら置き換える」という文面依存の判定にしていたため、
    修正前のコードが書いた記録（その文言を含まない）が残り続け、36レース
    照合できたのにログが更新されなかった。2026-08-08 に実際そうなった。
    """
    log = tmp_path / "LESSONS.md"
    log.write_text(
        "# 実戦ログ\n\n## 2026-08-08\n\n"
        "- 対象 36R / 買い 20R / 見送り 16R\n"
        "- 投資 12,200円 → 払戻 0円 （回収率 0.0%・的中 0R）\n"
        "- 的中なし\n",
        encoding="utf-8",
    )
    append_lessons(payload(graded=36, races=36, returned=5820, hits=2, roi=47.7), log)
    text = log.read_text(encoding="utf-8")

    assert text.count("## 2026-08-08") == 1
    assert "回収率 47.7%" in text
    assert "的中 2R" in text
    assert "払戻 0円" not in text


def test_partial_record_is_upgraded_when_more_races_are_graded(tmp_path: Path) -> None:
    """一部しか照合できていない記録は、全部揃った時点で置き換える。"""
    log = tmp_path / "LESSONS.md"
    append_lessons(payload(graded=12, races=36, returned=500, hits=1, roi=50.0), log)
    append_lessons(payload(graded=36, races=36, returned=5820, hits=2, roi=47.7), log)
    text = log.read_text(encoding="utf-8")

    assert text.count("## 2026-08-08") == 1
    assert "照合できたのは 12/36R" not in text
    assert "回収率 47.7%" in text


def test_graded_record_is_not_overwritten(tmp_path: Path) -> None:
    """一度ちゃんと記録できた日は、後から上書きしない。"""
    log = tmp_path / "LESSONS.md"
    append_lessons(payload(graded=3, returned=1500, hits=1, roi=150.0), log)
    append_lessons(payload(graded=3, returned=9999, hits=9, roi=999.0), log)
    text = log.read_text(encoding="utf-8")

    assert "回収率 150.0%" in text
    assert "999.0" not in text
