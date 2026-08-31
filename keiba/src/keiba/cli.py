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
import re
from datetime import date, timedelta
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
WORKOUT_PATH = RAW_DIR / "workouts.jsonl.gz"
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


def cmd_backfill_workouts(args: argparse.Namespace) -> int:
    """netkeiba 有料プランの調教タイムを収集する。

    認証情報が無ければ collect_workouts が落とす。未ログインでも netkeiba は
    200 で案内ページを返すので、黙って空を積ませないため。
    """
    with Store(args.db) as store:
        rebuild(store, args.raw_dir)
        backfill.load_workout_file(store, args.workouts)

        targets = None
        if args.upcoming:
            # 今週の出走馬だけを引き直す。過去の一括収集と違い、一度引いた馬も
            # 対象に入れる。調教は毎週更新されるうえ、特徴量は21日より古い
            # ものを捨てるので、8月に引いた履歴は10月には1本も残らない。
            today = date.today()
            fresh_since = today - timedelta(days=args.fresh_days)
            targets = store.horse_ids_with_stale_workouts(
                today.isoformat(), fresh_since.isoformat()
            )
            log.info(
                "今週の出走馬のうち %s 以降の調教が無い: %d頭",
                fresh_since, len(targets),
            )

        fetched = backfill.collect_workouts(
            Fetcher(cache_dir=args.cache),
            store,
            args.workouts,
            args.limit,
            args.offset,
            targets=targets,
            refresh=args.upcoming,
        )
        remaining = len(store.horse_ids_without_workouts())
    log.info("調教 %d本を取得。未取得の残り %d頭", fetched, remaining)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """raw から SQLite を作り直し、種牡馬適性テーブルを吐く。"""
    args.db.unlink(missing_ok=True)
    with Store(args.db) as store:
        races = rebuild(store, args.raw_dir)
        backfill.load_pedigree_file(store, args.pedigree)
        backfill.load_workout_file(store, args.workouts)
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
    """開催日まわりのレースを取り込む。

    発走前と発走後で経路が違う。db.netkeiba は結果データベースなので、
    まだ行われていないレースは race_id ごと存在しない（2026-08-06 実測）。
    出馬表は JRA公式、結果は db.netkeiba という分担になる。
    """
    fetcher = Fetcher(cache_dir=args.cache)

    if args.upcoming:
        counts = collect.collect_upcoming(fetcher, args.raw_dir)
        if not counts:
            log.warning(
                "JRA公式に開催が公開されていない（出馬表は木曜公開）cache=%s",
                fetcher.stats,
            )
            return 0
        total = sum(counts.values())
        log.info(
            "出馬表 %d レース（%s）cache=%s",
            total,
            " / ".join(f"{d}:{n}" for d, n in sorted(counts.items())),
            fetcher.stats,
        )
        return 0

    if args.day is None:
        log.error("--date か --upcoming のどちらかを指定すること")
        return 2

    if args.results and args.from_jra:
        filled = collect.collect_results_from_jra(fetcher, args.day, args.raw_dir)
        if not filled:
            log.warning(
                "%s の結果をJRA公式から取得できなかった（まだ確定前の可能性）cache=%s",
                args.day, fetcher.stats,
            )
            return 0
        log.info("結果 %d レース cache=%s", filled, fetcher.stats)
        return 0

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
            store, weights, sire_table, args.day,
            provisional=args.provisional, raw_dir=args.raw_dir,
            config_dir=args.config_dir,
        )
        predict.save_predictions(store, payload)

    s = payload["summary"]
    if not s["races"]:
        # 収集が空だと予想も空になる。**空の予想を書き出してはいけない。**
        # 書き出しが先だったせいで、収集に失敗しただけで前回の正しい予想を
        # 空ファイルで潰していた（実際に手元で踏んだ）。当日朝の再生成が
        # 一度でも空振りすれば、前夜に出した買い目がボードから消える。
        existing = args.data_dir / f"{args.day.isoformat()}.json"
        log.error(
            "%s のレースが1件も無い。収集できていない可能性が高い"
            "（発走前の出馬表が取れているか確認すること）。%s",
            args.day,
            "既存の予想は残した" if existing.exists() else "書き出すものが無い",
        )
        return 1

    out = predict.write_day(payload, args.data_dir)
    log.info(
        "%s: %d R（買い %d / 見送り %d）投資 %d円 → %s",
        args.day, s["races"], s["bet"], s["skipped"], s["spend"], out,
    )
    log.info("自信度内訳: %s", s["by_confidence"])
    if args.provisional:
        log.info("暫定予想。買い目は組んでいない。当日の本予想で上書きされる")
    return 0


def cmd_probe_workouts(args: argparse.Namespace) -> int:
    """今週の追い切りが取れない原因を切り分ける。

    2026-08-28 の収集で、今週の出走馬514頭ぶんのページを引いて 10,981本の
    調教レコードは取れたのに、**直近21日の調教を持つ馬は93頭（18.1%）**しか
    いなかった。量は取れていて新しいぶんだけ来ていない、という形。

    有力な仮説は「最新の追い切りが別の表に載っていて、パーサが弾いている」。
    parse_horse_training は見出しに「調教タイム」を含む表しか読まないので、
    見出しの違う表があれば、そこに何行あっても静かに落ちる。

    ## ログに何を出すか

    Actions のログは公開される。調教は有料データなので、**日付・タイム・
    評価などのセルの中身は一切出さない**。出すのは表の見出しと件数、そして
    そこから計算した「最新が何日前か」だけにする。見出しはページの構造で
    あってデータではないため、これは出してよい（他の切り分けでも同じ線を
    引いてきた）。
    """
    import datetime as dt

    from bs4 import BeautifulSoup

    from keiba import backfill
    from keiba.sources import netkeiba, netkeiba_auth

    fetcher = Fetcher(cache_dir=args.cache)
    netkeiba_auth.login(fetcher, required=True)

    with Store(args.db) as store:
        rebuild(store, args.raw_dir)
        today = dt.date.today()
        horses = [
            r[0]
            for r in store.conn.execute(
                "SELECT DISTINCT e.horse_id FROM entries e"
                " JOIN races r ON r.race_id = e.race_id"
                " WHERE r.race_date >= ? AND e.horse_id <> ''"
                " ORDER BY e.horse_id LIMIT ?",
                (today.isoformat(), args.limit),
            )
        ]

    log.info("対象 %d頭（今週の出走馬から）", len(horses))
    accepted_total = ignored_total = 0
    unknown_headers: dict[str, int] = {}
    fresh = 0

    for i, horse_id in enumerate(horses, 1):
        html = fetcher.fetch(
            backfill.TRAINING_URL.format(horse_id=horse_id), force=True
        )
        soup = BeautifulSoup(html, "lxml")
        log.info("--- %d/%d  html %d bytes", i, len(horses), len(html))

        for table in soup.select("table"):
            rows = table.find_all("tr")
            if not rows:
                continue
            headers = [
                netkeiba._text(c) for c in rows[0].find_all(["th", "td"])
            ]
            body = len(rows) - 1
            ok = "調教タイム" in headers
            accepted_total += body if ok else 0
            if not ok:
                ignored_total += body
                key = "|".join(headers)[:120]
                unknown_headers[key] = unknown_headers.get(key, 0) + body
            log.info(
                "    表 class=%s 行=%d 採用=%s 見出し=%s",
                ".".join(table.get("class") or []) or "-", body, ok,
                "|".join(headers)[:160],
            )

        parsed = netkeiba.parse_horse_training(html, horse_id)
        if parsed:
            newest = max(w.workout_date for w in parsed)
            age = (today - newest).days
            recent = sum(1 for w in parsed if (today - w.workout_date).days <= 21)
            fresh += int(age <= 21)
            log.info("    解析 %d本 / 最新は %d日前 / 直近21日 %d本",
                     len(parsed), age, recent)
        else:
            log.info("    解析 0本")

    log.info("=" * 60)
    log.info("採用された行 %d / 弾いた行 %d", accepted_total, ignored_total)
    log.info("直近21日の調教を持つ馬 %d/%d", fresh, len(horses))
    if unknown_headers:
        log.info("パーサが読んでいない表の見出し（行数の多い順）:")
        for headers, n in sorted(unknown_headers.items(), key=lambda kv: -kv[1])[:10]:
            log.info("  %4d行  %s", n, headers)

    # 馬ごとのページに今週ぶんが無いなら、レース単位のページを見るしかない。
    # 中身は出さず、サーバ側で描かれているか（＝requests で取れるか）だけを見る。
    if not args.race_id:
        return 0
    log.info("=" * 60)
    log.info("レース単位のページを見る: %s", args.race_id)

    # 新しいパーサが実ページで動くかを、件数だけで確かめる。
    # 中身は出さない（有料データ・公開ログ）。
    oikiri = f"https://race.netkeiba.com/race/oikiri.html?race_id={args.race_id}"
    try:
        rows = netkeiba.parse_race_oikiri(fetcher.fetch(oikiri, force=True))
    except Exception as exc:  # noqa: BLE001
        log.info("parse_race_oikiri が落ちた: %r", exc)
    else:
        horses = {w.horse_id for w in rows}
        timed = sum(1 for w in rows if any(t is not None for t in w.times))
        newest = min(((date.today() - w.workout_date).days for w in rows), default=None)
        log.info(
            "parse_race_oikiri → %d本 / %d頭 / タイムあり %d本 / 最新 %s日前",
            len(rows), len(horses), timed, newest,
        )

    comment_url = f"https://race.netkeiba.com/race/comment.html?race_id={args.race_id}"
    try:
        comments = netkeiba.parse_race_comments(
            fetcher.fetch(comment_url, force=True), args.race_id
        )
    except Exception as exc:  # noqa: BLE001
        log.info("parse_race_comments が落ちた: %r", exc)
    else:
        lengths = [len(c.body) for c in comments]
        log.info(
            "parse_race_comments → %d頭 / 本文の文字数 最小%s 中央%s 最大%s",
            len(comments),
            min(lengths, default=0),
            sorted(lengths)[len(lengths) // 2] if lengths else 0,
            max(lengths, default=0),
        )

    for url in (
        oikiri,
        f"https://race.netkeiba.com/race/newspaper.html?race_id={args.race_id}",
        # 厩舎コメントの在り処を探す。馬別の kyusya_comment は課金の壁で
        # 本文が伏せられていた（提供元はデイリースポーツ）。調教と同じく
        # レース単位のページなら出るかもしれない。
        f"https://race.netkeiba.com/race/comment.html?race_id={args.race_id}",
        f"https://race.netkeiba.com/race/danwa.html?race_id={args.race_id}",
    ):
        try:
            html = fetcher.fetch(url, force=True)
        except Exception as exc:  # noqa: BLE001 — 切り分けなので種類を問わず出す
            log.info("  %s → 取得できない: %s", url.split("/")[-1], exc)
            continue
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.get_text(strip=True) if soup.title else "")[:80]
        tables = soup.select("table")
        from keiba.sources.netkeiba_auth import WALL_PAGE_MARKERS
        walled = [m for m in WALL_PAGE_MARKERS if m in html]
        log.info("  %s", url.split("/")[-1])
        log.info(
            "    %d bytes / title=%s / 表 %d個 / 課金の壁=%s",
            len(html), title, len(tables), walled or "なし",
        )
        for table in tables[:6]:
            rows = table.find_all("tr")
            headers = (
                [netkeiba._text(c) for c in rows[0].find_all(["th", "td"])]
                if rows else []
            )
            log.info(
                "      class=%s 行=%d 見出し=%s",
                ".".join(table.get("class") or []) or "-",
                max(len(rows) - 1, 0),
                "|".join(headers)[:160],
            )
            _dump_row_shape(rows[1:])
    return 0


# セルの中身は出さずに、行の形だけを出すための道具。
# 有料データを公開ログへ出さずにパーサを直すために要る。
_DATE_PATTERNS = {
    "yyyy/m/d": re.compile(r"\d{4}/\d{1,2}/\d{1,2}"),
    "m/d": re.compile(r"^\d{1,2}/\d{1,2}"),
    "m月d日": re.compile(r"\d{1,2}月\d{1,2}日"),
}


def _dump_row_shape(rows: list) -> None:
    """行あたりのセル数の分布と、日付らしき列の位置を出す。

    rowspan で列が省かれる表は、行によってセル数が変わる。どの行がどれだけ
    ずれているかが分からないと列を引き当てられない。中身は出さず、
    セル数・文字数・日付の書式に一致するか、だけを見る。
    """
    from collections import Counter

    from keiba.sources import netkeiba

    shapes = Counter(len(r.find_all(["td", "th"])) for r in rows)
    linked = sum(1 for r in rows if r.find("a", href=re.compile(r"/horse/\d+")))
    log.info("        セル数の分布=%s / 馬リンクを持つ行=%d/%d",
             dict(sorted(shapes.items())), linked, len(rows))

    for label, row in (("先頭行", rows[0] if rows else None),
                       ("2行目", rows[1] if len(rows) > 1 else None)):
        if row is None:
            continue
        marks = []
        for i, cell in enumerate(row.find_all(["td", "th"])):
            text = netkeiba._text(cell)
            hit = [name for name, rx in _DATE_PATTERNS.items() if rx.search(text)]
            marks.append(f"{i}:len{len(text)}{'=' + '/'.join(hit) if hit else ''}")
        log.info("        %s %s", label, " ".join(marks))


def _review_one(store: Store, day: str, args: argparse.Namespace) -> int | None:
    """1日ぶんの回顧。照合できたレース数を返す。対象外なら None。"""
    path = args.data_dir / f"{day}.json"
    if not path.exists():
        log.error("予想ファイルが無い: %s（先に predict を実行すること）", path)
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    # 暫定予想は買い目を組んでいない（発走前に出す印だけの版）。これを回顧すると
    # 「対象36R・買い0R・見送り36R・結果未照合」という、成績に見えるが成績では
    # ない記録が残る。2026-08-29 に、当日回顧の cron が4時間遅れて JST の日付が
    # 翌日へ回り、まだ走っていない 8/30 に対してこれが起きた。
    if payload.get("provisional"):
        log.error("%s は暫定予想（買い目なし）。回顧の対象ではない", day)
        return None

    payload = review.review_day(store, payload)
    predict.write_day(payload, args.data_dir)
    review.append_lessons(payload, args.lessons)
    s = payload["summary"]
    log.info(
        "%s: 照合 %s/%sR・的中 %s R / 投資 %s円 → 払戻 %s円（回収率 %s%%）",
        day, s.get("graded"), s["races"], s.get("hits"),
        s["spend"], s.get("returned"), s.get("roi"),
    )
    return s.get("graded", 0)


def _unreviewed_days(data_dir: Path, before: str) -> list[str]:
    """まだ全レースを照合できていない過去の開催日。

    db.netkeiba への着順の反映は遅れる。当日夜に走らせると0件のことがあり、
    これまでは実戦ログに「翌日に再実行すること」と書いて人に投げていた。
    書いた本人が忘れれば、その日の成績は永久に残らない。機械が拾い直す。
    """
    days = []
    for path in sorted(data_dir.glob("[0-9]*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["date"] >= before or data.get("provisional"):
            continue
        summary = data.get("summary") or {}
        if summary.get("graded", 0) < summary.get("races", 0):
            days.append(data["date"])
    return days


def cmd_review(args: argparse.Namespace) -> int:
    """指定日の予想に実結果を突き合わせ、成績と教訓ログを更新する。"""
    day = args.day.isoformat()
    with Store(args.db) as store:
        graded = _review_one(store, day, args)
        if graded is None:
            return 1

        # 取りこぼした過去日を拾い直す。着順が既に取れていれば埋まる。
        if args.catch_up:
            for missed in _unreviewed_days(args.data_dir, before=day):
                log.info("%s は未照合のぶんが残っている。拾い直す", missed)
                _review_one(store, missed, args)

        overall = review.totals(store)

    review.write_ledger(args.data_dir, args.lessons)
    log.info("通算: %s", overall)
    if graded == 0:
        log.warning(
            "%s は着順を1件も取得できていない。次回の回顧で拾い直す", day
        )
    return 0


def cmd_strategy(args: argparse.Namespace) -> int:
    """「勝ちそうな馬を買う」を素直にやったら、いくらになるかを測る。

    人気での足切りはしない。16番人気でもモデルが上位に置いたなら買う。
    穴を狙うのではなく、良いと思った馬を買った結果として穴が混ざる形。
    """
    import numpy as np
    import pandas as pd

    from keiba import dataset, edge, ml, strategy

    with Store(args.db) as store:
        df = dataset.prepare(store, speed_before=pd.Timestamp(str(args.valid_from)))
        result = ml.train(df, valid_from=args.valid_from)
        valid = df[df["race_date"] >= str(args.valid_from)].copy()
        scores = ml.predict(result.booster, valid)

        place = edge.payout_map(store, "複勝")
        win = edge.payout_map(store, "単勝")

    log.info("検証 %d レース / AUC %.5f", valid["race_id"].nunique(), result.auc)

    rows = []
    for n in (1, 2, 3):
        picks = strategy.top_n_by_race(valid, scores, n)
        rows.append(strategy.evaluate(picks, place, f"モデル上位{n}頭", "複勝"))
    rows.append(
        strategy.evaluate(
            strategy.top_n_by_race(valid, scores, 1), win, "モデル1位", "単勝"
        )
    )

    # 比較のため、市場のとおりに買った場合も出す。モデルがこれを下回るなら
    # 「モデルを使う意味が無い」ということになる。
    market = -valid["market_popularity"].to_numpy(dtype=float)
    for n in (1, 3):
        picks = strategy.top_n_by_race(valid, market, n)
        rows.append(strategy.evaluate(picks, place, f"人気上位{n}頭", "複勝"))
    rows.append(
        strategy.evaluate(
            strategy.top_n_by_race(valid, market, 1), win, "1番人気", "単勝"
        )
    )
    print()
    print(strategy.format_results(rows))

    # ここからが本題。別々の測定で「市場が緩んでいるかもしれない」と出た
    # 2つを掛け合わせる。荒れ型のレース × モデルが市場より高く評価した馬。
    # 人気では一切切らない。
    curve = edge.market_curve(df[df["race_date"] < str(args.valid_from)], dataset.TARGET)
    gap = scores - edge.apply_curve(valid, curve).to_numpy()
    shape = edge.race_shape(valid).to_numpy()

    focused = []
    for name in ("荒れ型", "堅型", "その他"):
        in_shape = shape == name
        if in_shape.sum() < 500:
            continue
        # そのレース形の中で、乖離の上位1/3を買う
        threshold = pd.Series(gap[in_shape]).quantile(2 / 3)
        picked = in_shape & (gap >= threshold) & np.isfinite(gap)
        if picked.sum() < 200:
            continue
        focused.append(
            strategy.evaluate(
                valid[picked], place, f"{name}・乖離上位1/3", "複勝"
            )
        )
        # 比べる相手として、同じレース形の全馬も出す
        focused.append(
            strategy.evaluate(valid[in_shape], place, f"{name}・全馬", "複勝")
        )
    if focused:
        print()
        print(strategy.format_results(focused))
    return 0


def cmd_chaos(args: argparse.Namespace) -> int:
    """レースが荒れるかどうかを、市場の形から読めるかを測る。

    いまの自信度は「自分の上位3頭の人気和」で決めており、測っているのは
    「モデルが市場に同意しているか」であって「レースが荒れるか」ではない。
    2026-08-09 新潟11R はモデルが人気馬を推したので見送ったが、実際は
    三連複36,980円の波乱だった。荒れは市場の形に出ているはずで、それは
    オッズ層で使ってよい情報。
    """
    from keiba import chaos

    with Store(args.db) as store:
        df = chaos.load(store)
    if df.empty:
        log.error("三連複の払戻を持つレースが無い。build を先に走らせること")
        return 1
    print(chaos.report(df))
    return 0


def cmd_eval_speed(args: argparse.Namespace) -> int:
    """タイム指数を足すと精度が動くかを、同じ分割で並べて測る。

    調教のときと同じ扱い。効くと分かってから本番の特徴量に入れる。
    基準タイムと馬場差は検証期間より前のデータだけで作る（種牡馬適性表と
    同じ）。ここに未来を混ぜると、そのコースの基準を未来から知っている
    ことになる。
    """
    import pandas as pd

    from keiba import dataset, ml, speed, workout_slices

    with Store(args.db) as store:
        df = dataset.prepare(store, speed_before=pd.Timestamp(str(args.valid_from)))
    covered = speed.coverage(df[df["race_date"] >= str(args.valid_from)])
    log.info("指数あり %.1f%%（検証期間）", covered * 100)

    valid = df[df["race_date"] >= str(args.valid_from)]
    scores: dict[str, object] = {}
    importance: list[tuple[str, int]] = []
    for label, features in (
        ("時計なし", [c for c in dataset.FEATURE_COLUMNS
                    if c not in speed.SPEED_FEATURES]),
        ("時計あり", dataset.FEATURE_COLUMNS),
    ):
        result = ml.train(df, valid_from=args.valid_from, features=features)
        scores[label] = result.booster.predict(
            valid[features], num_iteration=result.booster.best_iteration
        )
        print()
        print(f"【{label}】特徴量 {len(features)}個  AUC {result.auc:.5f}")
        print(ml.format_ranking(ml.evaluate_ranking(valid, scores[label])))
        if label == "時計あり":
            importance = result.importance

    print()
    print(workout_slices.format_slices(
        workout_slices.compare(
            valid, dataset.TARGET, scores["時計なし"], scores["時計あり"]
        ),
        subject="時計",
    ))

    used = [(n, g) for n, g in importance if n in speed.SPEED_FEATURES]
    total = sum(g for _, g in importance) or 1
    print()
    print("時計の特徴量の寄与（gain・全体に占める割合）")
    for name, gain in used:
        print(f"  {name:<18}{gain:>12,}{gain / total:>8.2%}")
    print(f"  {'合計':<18}{sum(g for _, g in used):>12,}"
          f"{sum(g for _, g in used) / total:>8.2%}")
    return 0


def cmd_edge(args: argparse.Namespace) -> int:
    """モデルと市場の食い違いに妙味があるかを、実払戻で測る。

    モデルの順位付けは市場より下手だが、それは儲からないことを意味しない。
    賭けで必要なのは平均的に上手いことではなく、市場が間違えている場所を
    見つけること。ここで妙味がゼロなら、能力モデルを積み増しても構造は
    変わらない可能性が高い。次に何をするかがこの測定で決まる。
    """
    import numpy as np
    import pandas as pd

    from keiba import dataset, edge, ml

    with Store(args.db) as store:
        df = dataset.prepare(store, speed_before=pd.Timestamp(str(args.valid_from)))
        train = df[df["race_date"] < str(args.valid_from)]
        valid = df[df["race_date"] >= str(args.valid_from)].copy()
        if train.empty or valid.empty:
            log.error("分割が空になった（境界 %s）", args.valid_from)
            return 1

        # 市場カーブは学習期間だけで作る。未来を混ぜると市場が実際より
        # 賢く見え、モデルの乖離が過小評価される。
        curve = edge.market_curve(train, dataset.TARGET)
        log.info("市場の織り込み（単勝オッズ帯 → 3着内率）:\n%s", curve.to_string())

        result = ml.train(df, valid_from=args.valid_from)
        p_model = ml.predict(result.booster, valid)
        p_market = edge.apply_curve(valid, curve).to_numpy()

        place = edge.returns_for(valid, edge.payout_map(store, "複勝"))
        win = edge.returns_for(valid, edge.payout_map(store, "単勝"))

    print()
    print(f"モデル AUC {result.auc:.5f} / 検証 {len(valid):,} 行")
    table = edge.by_edge(valid, dataset.TARGET, p_model - p_market, place, win)
    print(edge.format_table(table, "モデルと市場の乖離ごとの実回収率"))

    # 全体の表は人気-穴のバイアスと混ざる。人気馬は元々回収率が高く、
    # モデルが市場より低く見るのは主に人気馬なので、「乖離が負のほうが
    # 儲かる」という見かけの関係が勝手に出る。オッズ帯を固定して確かめる。
    print()
    print(edge.format_bands(
        edge.by_odds_band(valid, dataset.TARGET, p_model - p_market, place)
    ))

    # 配当が大きい場所（荒れ型）が分かっても、そこで市場が正確なら期待値は
    # 変わらない。荒れ型でだけ市場が緩んでいるかを確かめる。ここが本題。
    shape = edge.race_shape(valid)
    for name in ("荒れ型", "堅型"):
        mask = (shape == name).to_numpy()
        if mask.sum() < 2000:
            continue
        print()
        print(f"■ {name}（{mask.sum():,} 行 / "
              f"{valid.loc[mask, 'race_id'].nunique():,} レース）")
        print(edge.format_bands(
            edge.by_odds_band(
                valid[mask], dataset.TARGET, (p_model - p_market)[mask], place[mask]
            )
        ))

    # 狙いは穴なので、人気薄に絞っても同じことを見る。全体で妙味が無くても
    # 人気薄だけにあるなら、そこが買い場になる。
    longshot = valid["market_popularity"] >= args.longshot_from
    if longshot.sum() > 1000:
        mask = longshot.to_numpy()
        print()
        print(edge.format_table(
            edge.by_edge(
                valid[mask], dataset.TARGET,
                (p_model - p_market)[mask], place[mask], win[mask],
                buckets=5,
            ),
            f"人気薄（{args.longshot_from}番人気以下）だけで見た場合",
        ))
    return 0


def cmd_eval_workouts(args: argparse.Namespace) -> int:
    """調教を足すと精度が動くかを、同じ分割で並べて測る。

    本番の学習には手を入れない。効くと分かってから FEATURE_COLUMNS に
    入れる。効かなければ入れない。10時間かけて集めたから使う、では
    順序が逆になる。

    調教が揃っているのは直近だけ（新しい馬から引いているため）なので、
    被覆率の高い期間に絞って比べる。揃っていない期間を混ぜると、調教が
    効かないのかデータが無いのか区別が付かなくなる。
    """
    import numpy as np  # noqa: F401  （型注釈で使う）

    from keiba import dataset, ml, workout_features, workout_slices

    with Store(args.db) as store:
        df = dataset.prepare(store)
        workouts = workout_features.load_workouts(store)

    if workouts.empty:
        log.error("調教が1件も入っていない。backfill-workouts を先に走らせること")
        return 1

    # 貼り付けは dataset.prepare が済ませている（本番の特徴量に入ったため）
    df = df[df["race_date"] >= str(args.since)]
    covered = workout_features.coverage(df)
    log.info(
        "調教あり %.1f%%（%s 以降 %d行 / %d レース）",
        covered * 100, args.since, len(df), df["race_id"].nunique(),
    )
    if covered < args.min_coverage:
        log.error(
            "被覆率 %.1f%% は低すぎる。この状態で測っても、効かないのか"
            " データが足りないのか分からない。--since を新しくするか"
            " バックフィルの続きを走らせること",
            covered * 100,
        )
        return 1

    valid = df[df["race_date"] >= str(args.valid_from)]
    scores: dict[str, "np.ndarray"] = {}
    importance: list[tuple[str, int]] = []

    # 調教は本番の FEATURE_COLUMNS に入れたので、「なし」は引き算で作る。
    # 足し算のままだと、同じ列を二度渡して「なし」と「あり」が同じものになる。
    without = [
        c for c in dataset.FEATURE_COLUMNS
        if c not in workout_features.WORKOUT_FEATURES
    ]
    for label, features in (
        ("調教なし", without),
        ("調教あり", dataset.FEATURE_COLUMNS),
    ):
        result = ml.train(df, valid_from=args.valid_from, features=features)
        scores[label] = result.booster.predict(
            valid[features], num_iteration=result.booster.best_iteration
        )
        print()
        print(f"【{label}】特徴量 {len(features)}個  AUC {result.auc:.5f}")
        print(ml.format_ranking(ml.evaluate_ranking(valid, scores[label])))
        if label == "調教あり":
            importance = result.importance

    # 全体で潰れても、特定の条件だけ効くことはある。むしろ調教はそういう
    # 性質のデータなので、切り口ごとに見る。切り口は先に決めてあり、
    # データを見てから増やさない（増やせば必ず「効いた」切り口が見つかる）。
    print()
    print(workout_slices.format_slices(
        workout_slices.compare(
            valid, dataset.TARGET, scores["調教なし"], scores["調教あり"]
        )
    ))

    # モデルがそもそも調教を使っているか。使っていないなら、切り口を
    # 探す以前の問題（データが薄いか、特徴量の作り方が悪い）。
    used = [(n, g) for n, g in importance if n in workout_features.WORKOUT_FEATURES]
    total = sum(g for _, g in importance) or 1
    print()
    print("調教特徴量の寄与（gain・全体に占める割合）")
    for name, gain in used:
        print(f"  {name:<16}{gain:>12,}{gain / total:>8.2%}")
    print(f"  {'合計':<16}{sum(g for _, g in used):>12,}"
          f"{sum(g for _, g in used) / total:>8.2%}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """能力モデルを学習する。

    手置きの重みでは能力順位が市場の半分しか当たらなかったため、順位付けは
    学習に任せる。SKILL.md の観点は dataset.py の特徴量として残っている。
    """
    from keiba import dataset, ml

    meta_path = args.config_dir / "model.json"
    # 前回の AUC を model.json から読んで比べる形にしていたが、**あれは別の
    # 期間で測った数字**だった。検証窓は毎回ずらすので、比べているのは
    # 「モデルの良し悪し」ではなく「その期間の当てやすさ」になる。
    #
    # 実際に踏んだ。調教を入れた再学習が 0.78502 → 0.77478 で弾かれたが、
    # 旧モデルの検証は 25,895行、新しいほうは 11,240行と 2.3倍違っていて、
    # そもそも比較になっていなかった。
    #
    # 前のモデルを**今回と同じ検証行**で採点し直して比べる。
    previous_booster = ml.load(args.config_dir / "model.txt")

    import pandas as pd

    with Store(args.db) as store:
        # 検証期間より前だけで指数の基準を作る。ここを全期間にすると、
        # 報告する AUC が本番より良く出てしまう。
        df = dataset.prepare(store, speed_before=pd.Timestamp(str(args.valid_from)))
        result = ml.train(df, valid_from=args.valid_from)
        print(result.report())

        valid = df[df["race_date"] >= str(args.valid_from)]
        scores = ml.predict(result.booster, valid)
        print()
        print(ml.format_ranking(ml.evaluate_ranking(valid, scores)))

        # 定期再学習で黙って悪いモデルに差し替わらないようにする。
        # 月1で自動実行する以上、誰も数字を見ないまま入れ替わる回が必ず来る。
        #
        # 比較は同じ検証行の上で行う。前のモデルにも今回の valid を採点させ、
        # そこで負けているときだけ止める。列は ml.predict がモデル自身の
        # feature_name() で選ぶので、特徴量を足した直後でもそのまま通る。
        if previous_booster is not None and not args.force:
            previous = ml.roc_auc(
                valid[dataset.TARGET].to_numpy(),
                ml.predict(previous_booster, valid),
            )
            if previous is None:
                log.warning("前のモデルを採点できなかった。比較を飛ばす")
            else:
                drop = previous - result.auc
                if drop > args.max_auc_drop:
                    log.error(
                        "同じ検証行で AUC %.5f（前）→ %.5f（新）。%.5f を超える"
                        "低下なので差し替えない。意図的に入れ替えるなら --force",
                        previous, result.auc, drop, args.max_auc_drop,
                    )
                    return 1
                log.info(
                    "同じ検証行で AUC %.5f（前）→ %.5f（新）／%+.5f",
                    previous, result.auc, -drop,
                )

        _check_score_separation(valid, scores, args.config_dir)
        ml.save(result, args.config_dir / "model.txt", meta_path)
        log.info("モデルを差し替えた: %s", args.config_dir / "model.txt")
    return 0


def _check_score_separation(valid, scores, config_dir) -> None:
    """◎ の閾値がモデルのスケールに合っているかを確かめる。

    ## なぜ要るか

    min_score_separation は「スコア1位と4位の差」で、**モデルのスコアの
    スケールに依存する**。手置きスコア（中央値50前後）から学習モデルの
    確率×100（中央値20前後）へ切り替えたとき、この値だけ 6.0 のまま残り、
    **97%のレースが ◎** になっていた。その1位の勝率は全体平均と同じ。

    型でも例外でも捕まらず、予想は普通に出るので誰も気づかない。モデルを
    入れ替えるたびに起こりうるので、再学習のたびに機械で見張る。

    狙いは「◎ は4レースに1つくらい」。実測の75%点から外れていたら鳴らす。
    """
    import numpy as np
    import yaml

    work = valid[["race_id"]].copy()
    work["score"] = scores * 100
    seps = [
        s.iloc[0] - s.iloc[3]
        for _, g in work.groupby("race_id", observed=True)
        if len(s := g["score"].sort_values(ascending=False)) >= 4
    ]
    if not seps:
        return

    fitted = float(np.quantile(seps, 0.75))
    cfg = yaml.safe_load((config_dir / "weights.yml").read_text())
    current = float(cfg["confidence"]["min_score_separation"])
    share = float(np.mean(np.array(seps) >= current))

    log.info(
        "◎ の閾値: 設定 %.2f → ◎になる割合 %.0f%%（実測の75%%点は %.2f）",
        current, share * 100, fitted,
    )
    # 4レースに1つのつもりが半分以上、あるいは1割未満なら、もう区別として
    # 機能していない。数字を出すだけでは見落とすので警告にする。
    if not 0.10 <= share <= 0.50:
        log.warning(
            "::warning::◎ が %.0f%% のレースに付いている。モデルのスケールと"
            " 合っていない可能性が高い。weights.yml の min_score_separation を"
            " %.1f 付近へ直すこと",
            share * 100, fitted,
        )


def _ml_scores(store, config_dir, start, end) -> dict | None:
    """期間内の全レースぶんの ML スコアを作る。モデルが無ければ None。"""
    from keiba import dataset, ml

    import pandas as pd

    booster = ml.load(config_dir / "model.txt")
    if booster is None:
        return None
    # 指数の基準も期間開始より前だけで作る。種牡馬適性表と同じ理屈で、
    # 全期間から作ると「未来のコース水準を知って過去を予想する」ことになる。
    df = dataset.prepare(store, speed_before=pd.Timestamp(str(start)))
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
    if args.trio_shape:
        # 買い目の組み方を測るための上書き。weights.yml は触らない。
        weights["betting"]["trio_shape"] = args.trio_shape
        log.info("三連複の組み方: %s", args.trio_shape)
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
    parser.add_argument("--workouts", type=Path, default=WORKOUT_PATH)
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

    p = sub.add_parser(
        "probe-workouts",
        help="今週の追い切りが取れない原因を切り分ける（表の見出しと件数だけを出す）",
    )
    p.add_argument("--limit", type=int, default=5, help="調べる頭数（既定 5）")
    p.add_argument(
        "--race-id", default=None,
        help="このレースの調教ページも見る（レース単位のページに今週ぶんが"
        " 載っているかの確認）",
    )
    p.set_defaults(func=cmd_probe_workouts)

    p = sub.add_parser("backfill-workouts", help="調教タイムを収集する（要 netkeiba 有料）")
    p.add_argument("--limit", type=int, default=None, help="1回で引く頭数の上限")
    p.add_argument(
        "--offset", type=int, default=0, help="対象リストの先頭から飛ばす頭数（並列分割用）"
    )
    p.add_argument(
        "--upcoming",
        action="store_true",
        help="今週の出走馬だけを引き直す（毎週の運用）。過去の一括収集と違い、"
        " 一度引いた馬も対象にする。調教は毎週更新され、21日より古いものは"
        " 特徴量として使われないため",
    )
    p.add_argument(
        "--fresh-days", type=int, default=10,
        help="--upcoming のとき、この日数以内の調教があれば引き直さない（既定 10）",
    )
    p.set_defaults(func=cmd_backfill_workouts)

    p = sub.add_parser(
        "eval-speed", help="タイム指数を足すと精度が動くかを並べて測る"
    )
    p.add_argument("--valid-from", type=_date, required=True)
    p.set_defaults(func=cmd_eval_speed)

    p = sub.add_parser(
        "strategy", help="勝ちそうな馬を素直に買った場合の実回収率を測る"
    )
    p.add_argument("--valid-from", type=_date, required=True)
    p.set_defaults(func=cmd_strategy)

    p = sub.add_parser(
        "chaos", help="荒れやすさが市場の形から読めるかを測る"
    )
    p.set_defaults(func=cmd_chaos)

    p = sub.add_parser(
        "edge", help="モデルと市場の乖離に妙味があるかを実払戻で測る"
    )
    p.add_argument("--valid-from", type=_date, required=True)
    p.add_argument(
        "--longshot-from", type=int, default=6, help="人気薄とみなす人気順"
    )
    p.set_defaults(func=cmd_edge)

    p = sub.add_parser(
        "eval-workouts", help="調教を足すと精度が動くかを並べて測る"
    )
    p.add_argument("--since", type=_date, required=True, help="測定に使う期間の開始日")
    p.add_argument("--valid-from", type=_date, required=True)
    p.add_argument(
        "--min-coverage", type=float, default=0.7,
        help="この割合まで調教が揃っていなければ測らない",
    )
    p.set_defaults(func=cmd_eval_workouts)

    p = sub.add_parser("build", help="raw から SQLite と種牡馬適性を作る")
    p.add_argument("--min-sire-runs", type=int, default=30)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("collect", help="開催日まわりのレースを取り込む")
    # --upcoming は JRA公式が公開している開催を全部取るので日付を取らない
    p.add_argument("--date", dest="day", type=_date, default=None)
    p.add_argument("--days-ahead", type=int, default=0, help="何日先まで取るか")
    p.add_argument(
        "--results", action="store_true", help="結果の入ったレースだけ残す（回顧用）"
    )
    p.add_argument(
        "--upcoming",
        action="store_true",
        help="JRA公式から発走前の出馬表を取る（--date は無視する）。"
        " db.netkeiba には発走前の race_id が存在しないため",
    )
    p.add_argument(
        "--from-jra",
        action="store_true",
        help="--results と併用。着順と払戻をJRA公式から取る。"
        " db.netkeiba は反映が遅く、開催当日には結果が出ていない",
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
    p.add_argument(
        "--catch-up", action="store_true",
        help="照合しきれていない過去の開催日も拾い直す",
    )
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("train", help="能力モデルを学習する")
    p.add_argument(
        "--valid-from", type=_date, required=True,
        help="この日以降を検証に回す（時系列で切る。ランダム分割はしない）",
    )
    p.add_argument(
        "--max-auc-drop", type=float, default=0.005,
        help="前回モデルからのAUC低下がこれを超えたら差し替えない（既定 0.005）",
    )
    p.add_argument(
        "--force", action="store_true",
        help="AUCが下がっていても差し替える",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("backtest", help="過去データで的中率・回収率を出す")
    p.add_argument(
        "--no-model", action="store_true", help="学習モデルを使わずルールベースで測る"
    )
    p.add_argument(
        "--trio-shape", choices=["axis", "box"], default=None,
        help="三連複の組み方を上書きして測る（既定は weights.yml）",
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
