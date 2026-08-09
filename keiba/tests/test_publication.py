"""公開してはいけないものが公開されていないことの検証。

このリポジトリは公開されている。しかも GitHub Pages がルートから配信して
いるので、**リポジトリ内の全ファイル**が URL で取れる状態にある。ボードが
keiba/data/index.json を取れているのがまさにその証拠。

つまり「コミットしない」がそのまま「公開しない」になる。ここを守る。
"""

from __future__ import annotations

import pathlib
import subprocess

REPO = pathlib.Path(__file__).parents[2]


def _tracked(pattern: str) -> list[str]:
    return subprocess.run(
        ["git", "ls-files", pattern],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split()


def test_paid_workout_data_is_never_tracked() -> None:
    """調教タイムがコミットされていないこと。

    netkeiba の有料コンテンツで、2.4万頭ぶんを公開リポジトリに置けば
    課金者向けのデータを誰でも取れる形にしてしまう。収集結果は Actions の
    artifact に出すだけにして、学習時にそこから取る。
    """
    tracked = _tracked("keiba/raw/workouts*")
    assert not tracked, (
        f"調教データがコミットされている: {tracked[:3]}。"
        " 有料コンテンツを公開リポジトリに置かないこと"
    )


def test_the_workout_path_is_ignored() -> None:
    """うっかり git add -A しても入らないこと。

    バックフィルは Actions で走り、他のジョブが git add -A でコミットする。
    gitignore が効いていないと、意図せず混ざる。
    """
    result = subprocess.run(
        ["git", "check-ignore", "keiba/raw/workouts-0.jsonl.gz"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, "keiba/raw/workouts*.jsonl.gz が gitignore されていない"


def test_the_workout_path_is_outside_the_board() -> None:
    """調教データの置き場所が、ボードが読む場所と混ざっていないこと。"""
    from keiba.cli import DATA_DIR, WORKOUT_PATH

    assert DATA_DIR not in WORKOUT_PATH.parents, (
        f"{WORKOUT_PATH} がボードの配信対象 {DATA_DIR} の中にある"
    )
