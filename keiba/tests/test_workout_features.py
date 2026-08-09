"""調教特徴量の検証。

一番大事なのは先読みが無いこと。調教は「レース直前の状態」を表すデータで、
うっかりレース後の調教まで混ぜると、そのレースの結果を知った状態で予想する
ことになる。バックテストの数字だけが良くなり、本番では効かない。
"""

from __future__ import annotations

import datetime

import pandas as pd

from keiba.models import HorseWorkout
from keiba.store import Store
from keiba.workout_features import attach, coverage, load_workouts


def _workout(day: str, t3f: float, course: str = "栗坂", **kw) -> dict:
    return HorseWorkout(
        horse_id="H1",
        workout_date=datetime.date.fromisoformat(day),
        course=course,
        times=[None, None, t3f + 13.0, t3f, 13.0],
        **kw,
    ).to_dict()


def _store(tmp_path, rows: list[dict]) -> Store:
    store = Store(tmp_path / "w.db")
    store.upsert_horse_workouts(rows)
    return store


def test_only_workouts_before_the_race_are_used(tmp_path) -> None:
    """レース後の調教を混ぜないこと。

    ここが漏れると、結果を知った状態で予想することになる。バックテストの
    数字だけが良くなって本番で効かない、という最悪の壊れ方をする。
    """
    with _store(tmp_path, [
        _workout("2026-01-07", 38.0),
        _workout("2026-01-21", 20.0),   # レース後。混ざれば極端な値で分かる
    ]) as store:
        workouts = load_workouts(store)

    entries = pd.DataFrame(
        {"race_date": [pd.Timestamp("2026-01-10")], "horse_id": ["H1"]}
    )
    got = attach(entries, workouts)
    assert got.loc[0, "w_last_3f"] == 38.0
    assert got.loc[0, "w_days_since"] == 3


def test_same_day_workout_is_not_used(tmp_path) -> None:
    """当日のものも使わない。最終追い切りは水木で、当日には存在しない。"""
    with _store(tmp_path, [_workout("2026-01-10", 38.0)]) as store:
        workouts = load_workouts(store)

    entries = pd.DataFrame(
        {"race_date": [pd.Timestamp("2026-01-10")], "horse_id": ["H1"]}
    )
    assert pd.isna(attach(entries, workouts).loc[0, "w_last_3f"])


def test_comparison_is_against_the_horses_own_history(tmp_path) -> None:
    """自己平均との差が取れること。

    コースをまたいだ絶対値の比較は「速い馬」ではなく「速いコースで追われた
    馬」を選んでしまう。効くのはその馬自身の過去との差。
    """
    with _store(tmp_path, [
        _workout("2025-12-01", 40.0),
        _workout("2025-12-15", 40.0),
        _workout("2026-01-07", 38.0),   # いつもより2秒速い
    ]) as store:
        workouts = load_workouts(store)

    entries = pd.DataFrame(
        {"race_date": [pd.Timestamp("2026-01-10")], "horse_id": ["H1"]}
    )
    got = attach(entries, workouts)
    assert got.loc[0, "w_3f_vs_self"] == -2.0, "自己平均より速いので負になるはず"


def test_self_average_excludes_the_workout_itself(tmp_path) -> None:
    """自己平均に自分自身を入れないこと。入れると差が薄まって信号が消える。"""
    with _store(tmp_path, [
        _workout("2025-12-01", 40.0),
        _workout("2026-01-07", 38.0),
    ]) as store:
        workouts = load_workouts(store)
    latest = workouts.sort_values("workout_date").iloc[-1]
    assert latest["t3f_self"] == 40.0


def test_courses_are_compared_separately(tmp_path) -> None:
    """坂路とコースを混ぜて平均しないこと。

    坂路は800m、ウッドは長い。混ぜた平均との差は、状態ではなく
    「今回どちらで追ったか」を表すだけになる。
    """
    with _store(tmp_path, [
        _workout("2025-12-01", 52.0, course="栗ＣＷ"),
        _workout("2025-12-08", 40.0, course="栗坂"),
        _workout("2026-01-07", 38.0, course="栗坂"),
    ]) as store:
        workouts = load_workouts(store)

    entries = pd.DataFrame(
        {"race_date": [pd.Timestamp("2026-01-10")], "horse_id": ["H1"]}
    )
    assert attach(entries, workouts).loc[0, "w_3f_vs_self"] == -2.0


def test_stale_workouts_are_left_missing(tmp_path) -> None:
    """古い調教で埋めないこと。

    「調教が無い」と「普通の調教だった」を同じ値にすると、休養明けの馬が
    平凡な状態として扱われる。欠損は欠損のまま渡して木に判断させる。
    """
    with _store(tmp_path, [_workout("2025-10-01", 38.0)]) as store:
        workouts = load_workouts(store)

    entries = pd.DataFrame(
        {"race_date": [pd.Timestamp("2026-01-10")], "horse_id": ["H1"]}
    )
    got = attach(entries, workouts)
    assert pd.isna(got.loc[0, "w_last_3f"])
    assert pd.isna(got.loc[0, "w_3f_vs_self"])


def test_recent_count_only_looks_backwards(tmp_path) -> None:
    """本数の集計も過去だけを見ること。"""
    with _store(tmp_path, [
        _workout("2026-01-01", 40.0),
        _workout("2026-01-07", 39.0),
        _workout("2026-01-21", 20.0),   # レース後
    ]) as store:
        workouts = load_workouts(store)

    entries = pd.DataFrame(
        {"race_date": [pd.Timestamp("2026-01-10")], "horse_id": ["H1"]}
    )
    assert attach(entries, workouts).loc[0, "w_count_21d"] == 2


def test_missing_workouts_do_not_break_the_pipeline(tmp_path) -> None:
    """調教が1件も無くても落ちないこと。

    バックフィルは途中で止まる前提で走らせる。揃っていない状態でも学習
    そのものは回り、被覆率で判断できるようにしておく。
    """
    with Store(tmp_path / "empty.db") as store:
        workouts = load_workouts(store)

    entries = pd.DataFrame(
        {"race_date": [pd.Timestamp("2026-01-10")], "horse_id": ["H1"]}
    )
    got = attach(entries, workouts)
    assert pd.isna(got.loc[0, "w_last_3f"])
    assert coverage(got) == 0.0


def test_concatenated_gzip_chunks_read_back_as_one(tmp_path) -> None:
    """分割して集めた調教を、繋ぐだけで1本として読めること。

    収集は3チャンクに分かれて別々の artifact に出る。ワークフローは
    `cat workouts-*.jsonl.gz > workouts.jsonl.gz` で繋いでいるだけで、
    これは gzip が連結を1本のストリームとして読める仕様に頼っている。

    ここが崩れると、集め終わってから「読めない」と分かる。10時間かけた
    あとで気づくのは高いので、先に固定しておく。
    """
    import gzip

    from keiba.backfill import load_workout_file
    from keiba.store import Store

    chunks = []
    for chunk in range(3):
        path = tmp_path / f"workouts-{chunk}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for i in range(2):
                fh.write(
                    '{"horse_id": "H%d%d", "workout_date": "2026-01-0%d",'
                    ' "course": "栗坂", "times": "[null, null, 52.0, 38.0, 13.0]",'
                    ' "going": "良", "rider": null, "position": null,'
                    ' "leg": null, "evaluation": null, "rank": "B"}\n'
                    % (chunk, i, i + 1)
                )
        chunks.append(path)

    merged = tmp_path / "workouts.jsonl.gz"
    merged.write_bytes(b"".join(p.read_bytes() for p in chunks))

    with Store(tmp_path / "m.db") as store:
        assert load_workout_file(store, merged) == 6, "連結したぶんが全部読めていない"
        rows = store.conn.execute("SELECT COUNT(*) FROM horse_workouts").fetchone()[0]
    assert rows == 6
