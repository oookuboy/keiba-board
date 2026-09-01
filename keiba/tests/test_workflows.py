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
    # 日付を受ける inputs は日付に置く。汎用の "x" を入れると、綴りは
    # 正しいのに型で落ちて偽の失敗になる。
    (r"\$\{\{\s*inputs\.(since|valid_from|date|from|to)[^}]*\}\}", "2026-08-08"),
    # 数を受ける inputs も同じ理由で数に置く（"x" だと型で落ちて偽の失敗になる）
    (r"\$\{\{\s*inputs\.(limit|offset|chunk|days|count)[^}]*\}\}", "5"),
    (r"\$\{\{[^}]*\}\}", "x"),
    # 素のシェル変数も実行時に埋まる。日付を入れる変数は日付に、それ以外は
    # 空に落とす。$FORCE のように「空か --force のどちらか」という変数を
    # 日付で置き換えると、ありもしない引数エラーを検出してしまう。
    (r"\$(SAT|SUN|VF|DATE|D)\b", "2026-08-08"),
    (r"\$[A-Z_][A-Z0-9_]*", ""),
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


def test_every_cron_maps_to_a_job() -> None:
    """cron を足したのに case 文へ書き忘れていないこと。

    一致しない cron は default の predict に落ちる。落ちても何かは動くので
    エラーにならず、「回顧を足したのに走らない」形で静かに壊れる。

    実際に当日回顧の cron を足したときにここを間違えかけた。
    """
    text = (WORKFLOWS / "keiba-weekend.yml").read_text(encoding="utf-8")
    crons = set(re.findall(r'- cron: "([^"]+)"', text))
    handled = {c for c, _ in re.findall(r'"([^"]+)"\)\s+JOB=(\w+)', text)}

    missing = crons - handled
    assert not missing, f"case 文に無い cron: {sorted(missing)}（predict に落ちる）"


def test_the_review_runs_on_the_day_of_the_races() -> None:
    """当日中に回顧が走ること。

    利用者の要望が「当日中がうれしい」なので、翌朝だけでは足りない。
    一度は当日22時で着順が取れず翌朝に倒したが、原因は時刻ではなく
    経路（db.netkeiba の反映待ち）だった。JRA公式を先に見る今なら回る。
    """
    text = (WORKFLOWS / "keiba-weekend.yml").read_text(encoding="utf-8")
    same_day = re.search(r'"([^"]+)"\)\s+JOB=review;\s*SAMEDAY=1', text)
    assert same_day, "当日ぶんを対象にする review の cron が無い"

    minute, hour, _, _, dow = same_day.group(1).split()
    assert int(hour) < 14, "JST の深夜になる。UTC 13時=JST 22時あたりに置くこと"
    assert set(dow.split(",")) == {"6", "0"}, "土日に走らせること"


def test_the_review_target_day_survives_a_late_cron() -> None:
    """回顧の対象日を JST の実行時刻で決めていないこと。

    GitHub Actions の cron は遅れて発火する。当日回顧は 13:00 UTC（＝22:00 JST）
    に置いてあり、JST の日付が変わるまで2時間しか余裕が無い。

    2026-08-29 に実際そうなった。cron が4時間遅れて 16:55 UTC（＝翌 01:55 JST）
    に発火し、`TZ=Asia/Tokyo date +%F` が 8/30 を返した。まだ1レースも走って
    いない 8/30 に対して回顧が走り、「対象36R・買い0R・結果未照合」という、
    成績に見えるが成績ではない記録が実戦ログとボードに残った。

    UTC で決めれば 00:00 UTC まで11時間の余裕がある。
    """
    text = (WORKFLOWS / "keiba-weekend.yml").read_text(encoding="utf-8")
    block = text.split("DATE=\"${{ inputs.date }}\"", 1)[1].split("preview は木曜")[0]

    review_lines = [
        line for line in block.splitlines() if "DATE=$(" in line
    ]
    assert review_lines, "対象日を決めている行が見つからない"
    # review の分岐（前半2つ）は UTC で決める
    for line in review_lines[:2]:
        assert "date -u" in line, f"回顧の対象日を JST で決めている: {line.strip()}"


def test_the_review_picks_up_days_it_could_not_grade() -> None:
    """回顧が取りこぼしを自分で拾い直すこと。

    db.netkeiba への着順の反映は遅れる。当日夜の回顧は0件で終わることがあり、
    以前は実戦ログに「翌日に再実行すること」と書いて人に投げていた。人が
    忘れればその日の成績は永久に残らない。--catch-up が無ければ元に戻る。
    """
    text = (WORKFLOWS / "keiba-weekend.yml").read_text(encoding="utf-8")
    assert "review --date" in text
    assert "--catch-up" in text, (
        "着順を取れなかった日を拾い直さない。成績が永久に欠けたままになる"
    )


def test_this_weeks_paid_data_comes_from_the_race_pages() -> None:
    """今週の追い切りと厩舎コメントを、レース単位のページから取っていること。

    馬別の調教ページ（backfill-workouts）は**既に走ったレースに紐づく調教しか
    持たない**。2026-08-28 に今週の出走馬514頭を引いて 10,981本は取れたのに、
    直近21日の調教を持つ馬は93頭（18.1%）だった。最新が11日前・25日前・39日前で、
    どれも「その馬の前走の直前」。

    学習側は当該レースの追い切りが入った状態で学習しているので、これは
    「モデルが本番に存在しない列を当てにする」形になる。gain 上位10個のうち
    3つが調教なので影響は小さくない。

    馬単位の収集に戻すとこれが再発する。レース単位の経路を必須にする。
    """
    text = (WORKFLOWS / "keiba-weekend.yml").read_text(encoding="utf-8")
    assert "collect-paid" in text, (
        "今週の追い切りと厩舎コメントを取っていない。馬別ページには載らないので、"
        " backfill-workouts だけでは本番の調教が欠けたままになる"
    )


def test_the_workout_artifact_chain_renews_itself() -> None:
    """調教を取り込むワークフローが、上げ直しもすること。

    調教は netkeiba の有料データで、公開リポジトリにはコミットできない。
    かわりに artifact で持ち回っているが、artifact は90日で消える。

    取り込むだけで上げ直さないと、90日後に静かに消えて「有料データを使って
    いるつもりで使っていない」状態に戻る。しかも予想も学習もそのまま動くので
    （欠損として扱われるだけ）、気づく手がかりが無い。

    そこで、週次で回るワークフローには取り込みと上げ直しの両方を求める。
    """
    import yaml

    weekend = (WORKFLOWS / "keiba-weekend.yml").read_text(encoding="utf-8")
    assert "./.github/actions/workouts" in weekend, "調教を取り込んでいない"
    assert "name: workouts" in weekend, (
        "取り込むだけで上げ直していない。90日で消えて静かに調教なしに戻る"
    )

    # API を引くので actions:read が要る。無いと 403 で毎回「artifact が無い」
    # 扱いになり、これも静かに調教なしへ落ちる。
    for name in ("keiba-weekend.yml", "keiba-retrain.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        if "./.github/actions/workouts" not in text:
            continue
        permissions = yaml.safe_load(text).get("permissions") or {}
        assert permissions.get("actions") == "read", f"{name}: actions:read が無い"


def test_training_and_prediction_see_the_same_workouts() -> None:
    """学習する側も予想する側も調教を戻していること。

    片方だけだと、モデルが使っている列が本番で全行欠損になる（またはその逆）。
    学習と本番で特徴量の中身がずれるのは、精度が落ちるだけでなく、
    バックテストの数字が本番を説明しなくなるという意味で質が悪い。
    """
    retrain = (WORKFLOWS / "keiba-retrain.yml").read_text(encoding="utf-8")
    weekend = (WORKFLOWS / "keiba-weekend.yml").read_text(encoding="utf-8")
    assert "./.github/actions/workouts" in retrain, "学習側が調教を戻していない"
    assert "./.github/actions/workouts" in weekend, "予想側が調教を戻していない"


def test_data_committing_workflows_can_trigger_a_redeploy() -> None:
    """結果を push するワークフローが、配信を起こせること。

    Actions の標準トークンで行った push は、他のワークフローを起動しない
    （GitHub の無限ループ防止の仕様）。そのため keiba-weekend が回顧結果を
    コミットしても、paths を見ている push トリガでは配信が走らない。実際に
    これで、回顧が終わっているのにサイトが前日のままになった。

    workflow_run で拾えば push を経由しないので、この制限を受けない。
    """
    import yaml

    pages = yaml.safe_load((WORKFLOWS / "keiba-pages.yml").read_text(encoding="utf-8"))
    triggers = pages[True]           # YAML の `on:` は真偽値 True として読まれる
    assert "workflow_run" in triggers, (
        "配信が workflow_run で起動しない。標準トークンの push では"
        " 起動しないので、これが無いと結果がサイトに出ない"
    )

    watched = triggers["workflow_run"]["workflows"]
    committers = []
    for path in WORKFLOWS.glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        if "keiba/data" in text and "git push" in text:
            committers.append(yaml.safe_load(text)["name"])

    missing = [name for name in committers if name not in watched]
    assert not missing, (
        f"データをコミットするのに配信を起こさないワークフロー: {missing}"
    )
