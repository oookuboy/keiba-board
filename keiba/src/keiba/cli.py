"""コマンドライン入口。

    keiba backfill-races    --from 2023-08-01 --to 2024-07-31
    keiba backfill-pedigree --limit 6000
    keiba build             # raw/*.jsonl.gz → SQLite → sire_aptitude.json
    keiba stats

GitHub Actions からはすべてこの入口を叩く。
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import yaml

from keiba import backfill, backtest, collect, predict, review
from keiba.sources.http import Fetcher
from keiba.store import Store, rebuild

log = logging.getLogger(__name__)

RAW_DIR = Path("keiba/raw")
CONFIG_DIR = Path("keiba/config")
DB_PATH = Path("keiba/keiba.db")
PEDIGREE_PATH = RAW_DIR / "pedigree.jsonl.gz"
WEIGHTS_PATH = CONFIG_DIR / "weights.yml"
SIRE_PATH = CONFIG_DIR / "sire_aptitude.json"
DATA_DIR = Path("keiba/data")
LESSONS_PATH = Path("keiba/LESSONS.md")


def _date(value: str) -> date:
    return date.fromisoformat(value)


def cmd_backfill_races(args: argparse.Namespace) -> int:
    fetcher = Fetcher(cache_dir=args.cache)
    written, failed = backfill.collect_races(
        fetcher, args.start, args.end, args.raw_dir
    )
    log.info("レース %d件を書き出し（失敗 %d件）cache=%s", written, failed, fetcher.stats)
    return 0


def cmd_backfill_pedigree(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        rebuild(store, args.raw_dir)
        backfill.load_pedigree_file(store, args.pedigree)
        fetched = backfill.collect_pedigrees(
            Fetcher(cache_dir=args.cache),
            store,
            args.pedigree,
            args.limit,
            args.offset,
        )
        remaining = len(store.horse_ids_without_pedigree())
    log.info("血統 %d頭を取得。未取得の残り %d頭", fetched, remaining)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """raw から SQLite を作り直し、種牡馬適性テーブルを吐く。"""
    args.db.unlink(missing_ok=True)
    with Store(args.db) as store:
        races = rebuild(store, args.raw_dir)
        backfill.load_pedigree_file(store, args.pedigree)
        updated = store.apply_pedigree()
        counts = store.counts()

        table = store.sire_aptitude(min_runs=args.min_sire_runs)
        args.config_dir.mkdir(parents=True, exist_ok=True)
        out = args.config_dir / "sire_aptitude.json"
        out.write_text(
            json.dumps(table, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )

    log.info("レース %d件を投入 / 血統反映 %d行", races, updated)
    log.info("件数: %s", counts)
    log.info("種牡馬適性: %d頭ぶん → %s", len(table), out)
    return 0


def _load_config(args: argparse.Namespace) -> tuple[dict, dict]:
    """重みと種牡馬適性表を読む。適性表が無ければ空で動かす（中立値になる）。"""
    weights = yaml.safe_load((args.config_dir / "weights.yml").read_text())
    sire_path = args.config_dir / "sire_aptitude.json"
    if sire_path.exists():
        sire_table = json.loads(sire_path.read_text())
    else:
        log.warning(
            "%s が無い。血統評価は中立値に潰れる（backfill-pedigree → build で生成）",
            sire_path,
        )
        sire_table = {}
    return weights, sire_table


def cmd_collect(args: argparse.Namespace) -> int:
    """開催日まわりのレースを取り込む。"""
    fetcher = Fetcher(cache_dir=args.cache)
    kept, failed = collect.collect_range(
        fetcher, args.day, args.days_ahead, args.raw_dir, results_only=args.results
    )
    log.info("取り込み %d レース（失敗 %d）cache=%s", kept, failed, fetcher.stats)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """指定日の全レースを予想し、ボード用 JSON を書き出す。"""
    weights, sire_table = _load_config(args)
    with Store(args.db) as store:
        payload = predict.predict_day(
            store, weights, sire_table, args.day, provisional=args.provisional
        )
        predict.save_predictions(store, payload)
    out = predict.write_day(payload, args.data_dir)
    s = payload["summary"]
    if not s["races"]:
        # 収集が空だと予想も空になる。静かに0件で終わると「動いたのに中身が無い」
        # ことに気づけないので、ここで明示的に警告する。
        log.warning(
            "%s のレースが1件も無い。収集できていない可能性が高い"
            "（発走前の出馬表が取れているか確認すること）",
            args.day,
        )
        return 0
    log.info(
        "%s: %d R（買い %d / 見送り %d）投資 %d円 → %s",
        args.day, s["races"], s["bet"], s["skipped"], s["spend"], out,
    )
    log.info("自信度内訳: %s", s["by_confidence"])
    if args.provisional:
        log.info("暫定予想。買い目は組んでいない。当日の本予想で上書きされる")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """指定日の予想に実結果を突き合わせ、成績と教訓ログを更新する。"""
    path = args.data_dir / f"{args.day.isoformat()}.json"
    if not path.exists():
        log.error("予想ファイルが無い: %s（先に predict を実行すること）", path)
        return 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    with Store(args.db) as store:
        payload = review.review_day(store, payload)
        overall = review.totals(store)

    predict.write_day(payload, args.data_dir)
    review.append_lessons(payload, args.lessons)
    s = payload["summary"]
    log.info(
        "%s: 的中 %s R / 投資 %s円 → 払戻 %s円（回収率 %s%%）",
        args.day, s.get("hits"), s["spend"], s.get("returned"), s.get("roi"),
    )
    log.info("通算: %s", overall)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """能力モデルを学習する。

    手置きの重みでは能力順位が市場の半分しか当たらなかったため、順位付けは
    学習に任せる。SKILL.md の観点は dataset.py の特徴量として残っている。
    """
    from keiba import dataset, ml

    with Store(args.db) as store:
        df = dataset.prepare(store)
        result = ml.train(df, valid_from=args.valid_from)
        print(result.report())

        valid = df[df["race_date"] >= str(args.valid_from)]
        scores = ml.predict(result.booster, valid)
        print()
        print(ml.format_ranking(ml.evaluate_ranking(valid, scores)))
        ml.save(result, args.config_dir / "model.txt", args.config_dir / "model.json")
    return 0


def _ml_scores(store, config_dir, start, end) -> dict | None:
    """期間内の全レースぶんの ML スコアを作る。モデルが無ければ None。"""
    from keiba import dataset, ml

    booster = ml.load(config_dir / "model.txt")
    if booster is None:
        return None
    df = dataset.prepare(store)
    window = df[(df["race_date"] >= str(start)) & (df["race_date"] <= str(end))].copy()
    if window.empty:
        return None
    window["p"] = ml.predict(booster, window)
    return {
        rid: dict(zip(g["umaban"], g["p"]))
        for rid, g in window.groupby("race_id", observed=True)
    }


def cmd_backtest(args: argparse.Namespace) -> int:
    """過去データに対してエンジンを回し、的中率と回収率を出す。"""
    weights = yaml.safe_load((args.config_dir / "weights.yml").read_text())
    with Store(args.db) as store:
        scores = None if args.no_model else _ml_scores(
            store, args.config_dir, args.start, args.end
        )
        if scores:
            log.info("学習モデルのスコアで採点する（%d レース）", len(scores))
        result = backtest.run(store, weights, args.start, args.end, ml_scores=scores)
    print(result.report())
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        print(json.dumps(store.counts(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """引数パーサを組み立てる。

    実行から切り離してあるのは、ワークフローに書かれたコマンドが解釈できるかを
    テストから確かめるため。--help を足して確認する方法だと argparse が
    そこで正常終了してしまい、引数の誤りに到達しない。
    """
    parser = argparse.ArgumentParser(prog="keiba", description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--pedigree", type=Path, default=PEDIGREE_PATH)
    parser.add_argument("--cache", type=Path, default=Path(".cache/keiba"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--lessons", type=Path, default=LESSONS_PATH)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backfill-races", help="期間内の中央レースを収集する")
    p.add_argument("--from", dest="start", type=_date, required=True)
    p.add_argument("--to", dest="end", type=_date, required=True)
    p.set_defaults(func=cmd_backfill_races)

    p = sub.add_parser("backfill-pedigree", help="出走馬の血統を収集する")
    p.add_argument("--limit", type=int, default=None, help="1回で引く頭数の上限")
    p.add_argument(
        "--offset", type=int, default=0, help="対象リストの先頭から飛ばす頭数（並列分割用）"
    )
    p.set_defaults(func=cmd_backfill_pedigree)

    p = sub.add_parser("build", help="raw から SQLite と種牡馬適性を作る")
    p.add_argument("--min-sire-runs", type=int, default=30)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("collect", help="開催日まわりのレースを取り込む")
    p.add_argument("--date", dest="day", type=_date, required=True)
    p.add_argument("--days-ahead", type=int, default=0, help="何日先まで取るか")
    p.add_argument(
        "--results", action="store_true", help="結果の入ったレースだけ残す（回顧用）"
    )
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("predict", help="指定日の全レースを予想する")
    p.add_argument("--date", dest="day", type=_date, required=True)
    p.add_argument(
        "--provisional",
        action="store_true",
        help="暫定予想。オッズ確定前（木曜など）に能力評価だけを出す。"
        " 買い目は組まず、当日の本予想で上書きする前提",
    )
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("review", help="指定日の予想に実結果を突き合わせる")
    p.add_argument("--date", dest="day", type=_date, required=True)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("train", help="能力モデルを学習する")
    p.add_argument(
        "--valid-from", type=_date, required=True,
        help="この日以降を検証に回す（時系列で切る。ランダム分割はしない）",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("backtest", help="過去データで的中率・回収率を出す")
    p.add_argument(
        "--no-model", action="store_true", help="学習モデルを使わずルールベースで測る"
    )
    p.add_argument("--from", dest="start", type=_date, required=True)
    p.add_argument("--to", dest="end", type=_date, required=True)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("stats", help="投入済みデータの件数を表示する")
    p.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
