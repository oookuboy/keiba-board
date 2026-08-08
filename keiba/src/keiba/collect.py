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
