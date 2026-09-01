"""開催日まわりの日常収集。

バックフィルとの違いは「まだ結果が出ていないレース」を扱うこと。予想には
発走前の出馬表が要るが、そこが取れるかどうかは経路ごとに事情が違う。

  db.netkeiba.com/race/{race_id}/
      過去のレースはサーバーレンダリングで確実に取れる（検証済み）。
      発走前にどこまで見えるかは未検証。結果テーブルの列構成が違えば
      パースに失敗するので、そのときは警告を出して次へ進む。

  www.jra.go.jp（JRA公式）
      robots.txt が全面許可で、規約上いちばん安全な経路。ただし出馬表は
      木曜公開のため、週明けには開催選択ページに開催リンクが存在しない。
      実経路の確定は木曜以降のプローブ待ち（sources/jra.py 参照）。

したがって現状は db.netkeiba を主経路とし、取れなかったレースは記録に残して
飛ばす。黙って空の予想を出すより、取れていないことが分かるほうがよい。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from collections import defaultdict

from keiba.models import RaceCard
from keiba.sources import jra, netkeiba
from keiba.sources.http import Fetcher, FetchError
from keiba.store import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

LIST_URL = "https://db.netkeiba.com/race/list/{stamp}/"
RACE_URL = "https://db.netkeiba.com/race/{race_id}/"


def collect_day(
    fetcher: Fetcher,
    day: date,
    out_dir: Path,
    *,
    results_only: bool = False,
    refresh: bool = True,
) -> tuple[int, int]:
    """1日ぶんのレースを取り、raw/YYYY/YYYY-MM-DD.jsonl.gz に書く。

    results_only=True なら結果の入っているレースだけを残す（回顧用）。
    refresh=True なら既存ファイルがあっても取り直す。当日は馬場状態・取消・
    馬体重が更新されるので、予想前には取り直す必要がある。
    """
    stamp = day.strftime("%Y%m%d")
    out_path = out_dir / str(day.year) / f"{day.isoformat()}.jsonl.gz"

    existing: dict[str, RaceCard] = {}
    if out_path.exists():
        if not refresh:
            return 0, 0
        existing = {c.race.race_id: c for c in read_jsonl(out_path)}

    try:
        html = fetcher.fetch(LIST_URL.format(stamp=stamp), force=refresh)
    except FetchError as exc:
        log.warning("%s の一覧を取得できない: %s", stamp, exc)
        return 0, 0

    race_ids = netkeiba.parse_race_list(html, jra_only=True)
    if not race_ids:
        log.info("%s: 中央の開催なし", day)
        return 0, 0

    kept, failed = 0, 0
    for race_id in race_ids:
        try:
            page = fetcher.fetch(RACE_URL.format(race_id=race_id), force=refresh)
            card = netkeiba.parse_race_page(page, race_id)
        except (FetchError, ValueError) as exc:
            # 発走前は結果テーブルの列構成が違う可能性がある。落とさず記録する
            log.warning("取得・解析できない %s: %s", race_id, exc)
            failed += 1
            continue

        if card.race.race_date != day:
            continue
        if results_only and not card.results:
            continue

        existing[race_id] = card
        kept += 1

    if existing:
        write_jsonl(existing.values(), out_path)
        log.info(
            "%s: %d レース（新規・更新 %d / 失敗 %d）→ %s",
            day, len(existing), kept, failed, out_path.name,
        )
    return kept, failed


def collect_upcoming(fetcher: Fetcher, out_dir: Path) -> dict[date, int]:
    """JRA公式から、公開中の全開催ぶんの出馬表を取って日付ごとに書く。

    db.netkeiba は結果データベースなので、発走前のレースは race_id ごと
    存在しない（2026-08-06 実測）。よって発走前はこちらが唯一の経路。

    既存ファイルとは race_id 単位でマージする。当日に取り直すと馬体重や
    オッズが埋まるので、上書きされるのは正しい。

    戻り値は日付ごとのレース数。空なら「まだ公開されていない」。
    """
    cards = jra.collect_racecards(fetcher)
    if not cards:
        return {}

    by_day: dict[date, list[RaceCard]] = defaultdict(list)
    for card in cards:
        by_day[card.race.race_date].append(card)

    counts: dict[date, int] = {}
    for day, day_cards in sorted(by_day.items()):
        out_path = out_dir / str(day.year) / f"{day.isoformat()}.jsonl.gz"
        merged: dict[str, RaceCard] = {}
        if out_path.exists():
            merged = {c.race.race_id: c for c in read_jsonl(out_path)}
        for card in day_cards:
            merged[card.race.race_id] = card
        write_jsonl(merged.values(), out_path)

        confirmed = sum(1 for c in day_cards if jra.post_positions_confirmed(c))
        counts[day] = len(day_cards)
        log.info(
            "%s: %d レース（枠順確定 %d）→ %s",
            day, len(day_cards), confirmed, out_path.name,
        )
        if confirmed == 0:
            log.info("  枠順は金曜確定。馬番が入るまで買い目は組めない")
    return counts


def collect_results_from_jra(fetcher: Fetcher, day: date, out_dir: Path) -> int:
    """JRA公式から着順と払戻を取り、既存の出馬表に合流させる。

    db.netkeiba は結果データベースだが反映が遅く、開催当日の夜になっても
    着順を1件も出していなかった（2026-08-08 実測）。当日中に回顧を回すには
    こちらが要る。翌日以降は netkeiba 側でも取れるので、両方を残しておく。

    戻り値は結果を埋められたレース数。0 なら「まだ出ていない」。
    """
    out_path = out_dir / str(day.year) / f"{day.isoformat()}.jsonl.gz"
    if not out_path.exists():
        log.warning("%s の出馬表が無い。先に collect --upcoming が要る", day)
        return 0

    found = jra.collect_results(fetcher, day)
    if not found:
        return 0

    cards = {c.race.race_id: c for c in read_jsonl(out_path)}
    filled = 0
    for race_id, (results, payouts) in found.items():
        card = cards.get(race_id)
        if card is None:
            log.warning("%s の出馬表が手元に無い（結果だけ取れた）", race_id)
            continue
        card.results = results
        card.payouts = payouts
        filled += 1

    if filled:
        write_jsonl(cards.values(), out_path)
        log.info("%s: %d レースに着順と払戻を入れた → %s", day, filled, out_path.name)
    return filled


def collect_range(
    fetcher: Fetcher,
    start: date,
    days_ahead: int,
    out_dir: Path,
    *,
    results_only: bool = False,
) -> tuple[int, int]:
    """start から days_ahead 日先までを収集する。

    予想では未来の開催日を、回顧では当日だけを対象にする。中央は土日中心だが
    月曜が祝日の3連休には開催があるので、曜日で決め打ちせず一覧の空振りに任せる。
    """
    total_kept = total_failed = 0
    for offset in range(days_ahead + 1):
        day = start + timedelta(days=offset)
        kept, failed = collect_day(
            fetcher, day, out_dir, results_only=results_only
        )
        total_kept += kept
        total_failed += failed
    return total_kept, total_failed


# 今週の追い切りと厩舎コメント。どちらもレース単位のページにしか無い。
OIKIRI_URL = "https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
COMMENT_URL = "https://race.netkeiba.com/race/comment.html?race_id={race_id}"


def collect_paid(
    fetcher: Fetcher, raw_dir: Path, workouts_path: Path, days: list[date]
) -> dict[str, int]:
    """今週の追い切りと厩舎コメントを、レース単位のページから取る。

    ## 馬単位では今週ぶんが取れない

    db.netkeiba の馬別調教ページは**既に走ったレースに紐づく調教しか持たない**。
    2026-08-28 に今週の出走馬514頭を引いて 10,981本は取れたのに、直近21日の
    調教を持つ馬は93頭（18.1%）だった。最新が11日前・25日前・39日前で、どれも
    「その馬の前走の直前」。今週の追い切りは、そのレースが終わるまで出てこない。

    これは学習と本番のズレとして効いていた。過去のレースを学習するときは当該
    レースの追い切りが（走り終わっているので）入っているのに、本番はこれから
    走るレースなので同じ列が空になる。モデルの gain 上位10個のうち3つが調教で、
    それを当てにして学習して本番では空、という状態だった。

    レース単位のページには今週ぶんが載っている（実測で最新5日前）。しかも
    36リクエストで全頭ぶん揃う。馬単位の514リクエスト13分から1分弱になる。

    ## 厩舎コメントも同じページ群にある

    馬別の kyusya_comment は本文が課金の壁で伏せられていて諦めていたが、
    **それは課金前に測ったもの**だった。レース単位のこのページなら壁なしで
    全頭ぶん出る（実測で本文70〜80文字）。

    ## 過去の履歴は馬単位のままでよい

    ここで取れるのは当該レース向けの追い切りだけ。それ以前の履歴は馬別ページ
    にあるので、両方を足して初めて「直近21日に何本」が埋まる。
    """
    from keiba.backfill import _merge_workout_file
    from keiba.sources import netkeiba_auth

    # 未ログインでも netkeiba は 200 で案内ページを返す。黙って空を積まない
    netkeiba_auth.login(fetcher, required=True)

    stats = {"races": 0, "workouts": 0, "comments": 0, "failed": 0}
    records: list[dict] = []

    for day in sorted(days):
        path = raw_dir / str(day.year) / f"{day.isoformat()}.jsonl.gz"
        if not path.exists():
            log.warning("%s の出馬表が無い。先に collect --upcoming を走らせること", day)
            continue

        cards = list(read_jsonl(path))
        for card in cards:
            race_id = card.race.race_id
            stats["races"] += 1
            try:
                rows = netkeiba.parse_race_oikiri(
                    fetcher.fetch(OIKIRI_URL.format(race_id=race_id), force=True)
                )
                comments = netkeiba.parse_race_comments(
                    fetcher.fetch(COMMENT_URL.format(race_id=race_id), force=True),
                    race_id,
                )
            except FetchError as exc:
                log.warning("%s の有料データを取れない: %s", race_id, exc)
                stats["failed"] += 1
                continue

            records.extend(w.to_dict() for w in rows)
            stats["workouts"] += len(rows)
            # コメントは出馬表と同じファイルに持つ。rebuild が comments へ
            # 流し込むので、収集経路をここに足すだけで features 側に届く。
            card.comments = comments
            stats["comments"] += len(comments)

        write_jsonl(cards, path)
        log.info("%s: %d レース", day, len(cards))

    if records:
        _merge_workout_file(records, workouts_path)

    # 「収集は成功・中身は空」を黙って通さない。ログインが切れていれば
    # 全レースで0本になるので、そこで気づけるようにする。
    if stats["races"] and not stats["workouts"]:
        raise RuntimeError(
            f"{stats['races']}レース引いて追い切りが1本も取れていない。"
            " 有料プランの状態と Secrets を確かめること"
        )
    log.info(
        "有料データ: %d レース / 追い切り %d本 / 厩舎コメント %d件（失敗 %d）",
        stats["races"], stats["workouts"], stats["comments"], stats["failed"],
    )
    return stats
