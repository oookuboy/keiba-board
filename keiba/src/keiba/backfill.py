"""過去データの一括収集。

2段構えになっている。

第1段（レース）: 開催日ごとにレース一覧を引き、中央のレースページを全部取る。
  3年で約1万レース。1.5秒ウェイトなので約4時間。年ごとに分けて走らせる。

第2段（血統）: 第1段で集めた出走馬の血統ページを引く。3年で約2.5万頭。
  約10時間かかるので、--limit で刻んで複数ジョブに割る。
  血統は SKILL.md が「最重要」とする要素なので、重いが避けて通れない。

いずれも取得済みは Fetcher のディスクキャッシュと raw/ の存在チェックで
スキップされるため、途中で落ちても再実行すれば続きから進む。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

from keiba.models import RaceCard
from keiba.sources import netkeiba
from keiba.sources.http import Fetcher, FetchError
from keiba.store import Store, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

LIST_URL = "https://db.netkeiba.com/race/list/{stamp}/"
RACE_URL = "https://db.netkeiba.com/race/{race_id}/"
PED_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"


def racing_days(start: date, end: date) -> Iterator[date]:
    """中央開催の候補日。

    中央は基本的に土日開催だが、月曜が祝日の3連休には月曜も開催がある。
    祝日表を持ち込むより、月曜も候補に入れて空振りを許容するほうが安く済む
    （空振りは一覧ページ1回で判明する）。
    """
    day = start
    while day <= end:
        if day.weekday() in (0, 5, 6):  # 月・土・日
            yield day
        day += timedelta(days=1)


def race_ids_for_day(fetcher: Fetcher, day: date) -> list[str]:
    """その日の中央レースID。

    db.netkeiba は開催の無い日をリクエストすると直近の開催日へ寄せた内容を
    返すことがある。レースページ側の日付で後から弾くので、ここでは拾うだけ。
    """
    stamp = day.strftime("%Y%m%d")
    try:
        html = fetcher.fetch(LIST_URL.format(stamp=stamp))
    except FetchError as exc:
        log.warning("一覧を取得できない %s: %s", stamp, exc)
        return []
    return netkeiba.parse_race_list(html, jra_only=True)


def collect_races(
    fetcher: Fetcher, start: date, end: date, out_dir: Path
) -> tuple[int, int]:
    """期間内の中央レースを raw/YYYY/YYYY-MM-DD.jsonl.gz に書き出す。

    戻り値は (書き出したレース数, 失敗数)。
    """
    written = failed = 0
    seen: set[str] = set()

    for day in racing_days(start, end):
        out_path = out_dir / str(day.year) / f"{day.isoformat()}.jsonl.gz"
        if out_path.exists():
            # 既に取れている日は触らない。再実行で最初からやり直さないため。
            seen.update(c.race.race_id for c in read_jsonl(out_path))
            continue

        race_ids = [r for r in race_ids_for_day(fetcher, day) if r not in seen]
        if not race_ids:
            continue

        cards: list[RaceCard] = []
        for race_id in race_ids:
            try:
                html = fetcher.fetch(RACE_URL.format(race_id=race_id))
                card = netkeiba.parse_race_page(html, race_id)
            except (FetchError, ValueError) as exc:
                log.warning("スキップ %s: %s", race_id, exc)
                failed += 1
                continue

            # 一覧が別日へ寄せられていた場合に備え、実際の開催日で確認する
            if card.race.race_date != day:
                continue
            if not card.results:
                # 結果が入っていない＝発走前。バックフィルの対象外
                continue
            cards.append(card)
            seen.add(race_id)

        if cards:
            written += write_jsonl(cards, out_path)
            log.info("%s: %d レース → %s", day, len(cards), out_path.name)

    return written, failed


def collect_pedigrees(
    fetcher: Fetcher,
    store: Store,
    out_path: Path,
    limit: int | None = None,
    offset: int = 0,
) -> int:
    """血統をまだ持っていない馬の血統ページを引く。

    entries に居るのに horses に居ない馬だけを対象にするので、
    途中で止めて再実行しても取り直しにならない。

    対象リストは horse_id の昇順で安定しているため、offset と limit で
    重複なく分割できる。2.5万頭を1ジョブで引くと Actions の6時間上限を
    超えるので、並列ジョブに割るために使う。
    """
    targets = store.horse_ids_without_pedigree()[offset:]
    if limit:
        targets = targets[:limit]
    if not targets:
        log.info("血統の未取得馬なし")
        return 0

    log.info("血統を引く対象: %d頭", len(targets))
    records: list[dict] = []
    for i, horse_id in enumerate(targets, 1):
        try:
            html = fetcher.fetch(PED_URL.format(horse_id=horse_id))
        except FetchError as exc:
            log.warning("血統を取得できない %s: %s", horse_id, exc)
            continue
        ped = netkeiba.parse_pedigree(html)
        records.append({"horse_id": horse_id, **ped})
        if i % 200 == 0:
            log.info("  %d/%d", i, len(targets))

    if records:
        store.upsert_horses(records)
        _merge_pedigree_file(records, out_path)
    return len(records)


def _merge_pedigree_file(records: list[dict], path: Path) -> None:
    """血統は馬単位の追記型ファイルにまとめる。

    出走ごとに持つと raw が肥大するので分離してある。既存分と突き合わせて
    書き直すことで、git の差分が追記だけになるようにする。
    """
    import gzip
    import json

    merged: dict[str, dict] = {}
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    merged[row["horse_id"]] = row
    for row in records:
        merged[row["horse_id"]] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for horse_id in sorted(merged):
            fh.write(json.dumps(merged[horse_id], ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def load_pedigree_file(store: Store, path: Path) -> int:
    """raw の血統ファイルを horses テーブルへ流し込む（rebuild 時に使う）。"""
    if not path.exists():
        return 0
    import gzip
    import json

    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return store.upsert_horses(rows)
