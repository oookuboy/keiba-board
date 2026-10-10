"""JRA公式の入口ページから重賞の格を拾う。

出走表には格が書かれていないので、JRA公式に切り替えたあとの重賞は
2026-09-19〜10-12 の間ずっと grade が空だった（スプリンターズS・毎日王冠も）。
"""
from __future__ import annotations

import gzip
import json
import pathlib

from keiba import collect
from keiba.sources import jra

PROBE = pathlib.Path(__file__).parent / "fixtures/probe"


def test_出馬表の入口から重賞の格を拾う() -> None:
    html = (PROBE / "card_L1_kaisai__www.jra.go.jp_JRADB_accessD.html.html").read_text(encoding="utf-8")
    grades = jra.parse_grade_races(html)
    assert grades["202601010511"] == "GIII"   # エルムS（札幌11R）
    assert grades["202604020607"] == "GIII"   # レパードS
    assert grades["202607020607"] == "GIII"   # CBC賞


def test_結果の入口から重賞の格を拾う() -> None:
    html = (PROBE / "result_L1_kaisai__www.jra.go.jp_JRADB_accessS.html.html").read_text(encoding="utf-8")
    grades = jra.parse_grade_races(html)
    assert grades["202601010411"] == "GIII"   # クイーンS
    assert grades["202610020811"] == "GIII"   # 小倉記念
    assert grades["202609030411"] == "GI"
    assert grades["202605030304"] == "J.GIII"


def test_格の無いリンクに隣の重賞の格を付けない() -> None:
    html = (
        '<a href="/JRADB/accessD.html?CNAME=pw01dde1005202604020120261010/AA">東京1R</a>'
        '<a href="/JRADB/accessD.html?CNAME=pw01dde1005202604021120261010/BB">東京11R'
        '<span class="grade_icon lg"><img src="/JRADB/img/grade/icon_grade_g3.png" /></span></a>'
    )
    assert jra.parse_grade_races(html) == {"202605040211": "GIII"}


def test_rawの空欄だけに格を入れる(tmp_path: pathlib.Path) -> None:
    day = tmp_path / "2026"
    day.mkdir()
    src = pathlib.Path(__file__).parents[1] / "raw/2026/2026-08-09.jsonl.gz"
    rows = [json.loads(line) for line in gzip.open(src, "rt", encoding="utf-8")]
    for row in rows:
        (row.get("race") or row)["grade"] = None
    with gzip.open(day / "2026-08-09.jsonl.gz", "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = collect.apply_grades(tmp_path, {"202604020607": "GIII", "209901010101": "GI"})
    assert n == 1
    cards = {c.race.race_id: c for c in collect.read_jsonl(day / "2026-08-09.jsonl.gz")}
    assert cards["202604020607"].race.grade == "GIII"
    # 2回目は何もしない（既にある格は書き換えない）
    assert collect.apply_grades(tmp_path, {"202604020607": "GII"}) == 0
