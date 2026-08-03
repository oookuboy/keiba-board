"""ワークフローが叩く CLI のコマンドが実際に解釈できるか検証する。

血統収集を6時間空回りさせた原因がこれ。`--pedigree` はトップレベル引数
なのにサブコマンドの後ろに置いており、argparse が起動49秒で
`unrecognized arguments` を返して4チャンクとも即死した。

コマンドの綴りは Actions を実際に走らせるまで誰も確かめない。長時間ジョブでは
その代償が大きいので、YAML から `python -m keiba.cli ...` を抜き出して
パーサに通しておく。
"""

from __future__ import annotations

import pathlib
import re
import shlex

import pytest

from keiba.cli import build_parser

WORKFLOWS = pathlib.Path(__file__).parents[2] / ".github/workflows"

# ${{ ... }} は実行時に埋まる。型が合う適当な値へ置き換えて構文だけを見る。
SUBSTITUTIONS = [
    (r"\$\(\(\s*\$\{\{[^}]*\}\}[^)]*\)\)", "0"),   # $(( ${{ matrix.chunk }} * 7000 ))
    (r"\$\{\{\s*matrix\.year[^}]*\}\}", "2024"),
    (r"\$\{\{\s*matrix\.chunk[^}]*\}\}", "0"),
    (r"\$\{\{\s*steps\.plan\.outputs\.date[^}]*\}\}", "2026-08-08"),
    (r"\$\{\{[^}]*\}\}", "x"),
]


def cli_commands() -> list[tuple[str, str]]:
    """ワークフローに書かれた keiba.cli の呼び出しを (ファイル名, 引数列) で返す。"""
    found: list[tuple[str, str]] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        # 行継続を畳んでから1コマンドずつ拾う
        text = re.sub(r"\\\s*\n\s*", " ", text)
        for line in text.splitlines():
            if "keiba.cli" not in line:
                continue
            command = line.split("python -m keiba.cli", 1)[1]
            # シェルの後置き（|| true、リダイレクトなど）を落とす
            command = re.split(r"\|\||&&|[|>;]", command)[0]
            for pattern, replacement in SUBSTITUTIONS:
                command = re.sub(pattern, replacement, command)
            found.append((path.name, command.strip()))
    return found


def test_workflows_reference_the_cli() -> None:
    """抜き出しそのものが壊れていないことの確認。"""
    commands = cli_commands()
    assert commands, "ワークフローから keiba.cli の呼び出しを1つも拾えていない"
    assert any("backfill-pedigree" in c for _, c in commands)
    assert any("predict" in c for _, c in commands)


@pytest.mark.parametrize(
    ("workflow", "command"), cli_commands(), ids=lambda v: str(v)[:60]
)
def test_workflow_cli_invocation_parses(workflow: str, command: str) -> None:
    """引数の順序と綴りが argparse を通ること。

    parse_args を直接呼ぶ。main に --help を足して確かめる方法だと argparse が
    そこで正常終了してしまい、引数の誤りに到達しない（実際それで見逃した）。
    SystemExit(2) が argparse の使い方エラーで、Actions でも同じように落ちる。
    """
    try:
        build_parser().parse_args(shlex.split(command))
    except SystemExit as exc:
        pytest.fail(
            f"{workflow}: 引数を解釈できない（exit {exc.code}）\n"
            f"  python -m keiba.cli {command}"
        )


def test_detects_argument_order_mistakes() -> None:
    """この検証自体が機能していることを確かめる。

    トップレベル引数をサブコマンドの後ろに置いた形を、ちゃんと弾けること。
    """
    broken = "backfill-pedigree --offset 0 --limit 7000 --pedigree keiba/raw/p.jsonl.gz"
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(shlex.split(broken))
    assert exc.value.code == 2

    fixed = "--pedigree keiba/raw/p.jsonl.gz backfill-pedigree --offset 0 --limit 7000"
    args = build_parser().parse_args(shlex.split(fixed))
    assert str(args.pedigree) == "keiba/raw/p.jsonl.gz"
