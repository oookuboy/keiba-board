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

from keiba import backfill, backtest
from keiba.sources.http import Fetcher
from keiba.store import Store, rebuild

log = logging.getLogger(__name__)

RAW_DIR = Path("keiba/raw")
CONFIG_DIR = Path("keiba/config")
DB_PATH = Path("keiba/keiba.db")
PEDIGREE_PATH = RAW_DIR / "pedigree.jsonl.gz"
WEIGHTS_PATH = CONFIG_DIR / "weights.yml"
SIRE_PATH = CONFIG_DIR / "sire_aptitude.json"


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


def cmd_backtest(args: argparse.Namespace) -> int:
    """過去データに対してエンジンを回し、的中率と回収率を出す。"""
    weights = yaml.safe_load((args.config_dir / "weights.yml").read_text())
    with Store(args.db) as store:
        result = backtest.run(store, weights, args.start, args.end)
    print(result.report())
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        print(json.dumps(store.counts(), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="keiba", description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--pedigree", type=Path, default=PEDIGREE_PATH)
    parser.add_argument("--cache", type=Path, default=Path(".cache/keiba"))
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

    p = sub.add_parser("backtest", help="過去データで的中率・回収率を出す")
    p.add_argument("--from", dest="start", type=_date, required=True)
    p.add_argument("--to", dest="end", type=_date, required=True)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("stats", help="投入済みデータの件数を表示する")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
