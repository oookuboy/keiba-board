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

from keiba.models import RaceCard
from keiba.sources import netkeiba
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
