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


def repair_races(
    fetcher: Fetcher, start: date, end: date, out_dir: Path
) -> dict[str, int]:
    """既に取ってある日の、**欠けているレースだけ**を取り直す。

    ## collect_races では直らない

    collect_races は「ファイルがある日は触らない」。途中で止まっても最初から
    やり直さないための作りだが、そのせいで**一度欠けたレースは永久に欠けた
    まま**になる。

    コース表記（「芝右 外1200m」）を読めずに捨てていたレースが、3年分で
    開催の半分に及んでいた。正規表現を直しても、既存の日を触らない限り
    戻ってこない。

    ## 既にあるものは書き換えない

    その日の一覧を取り直し、raw に無い race_id だけを引いて**足す**。既存の
    カードはそのまま残す（後から入れた調教・コメント・払戻を消さないため。
    出馬表の取り直しで一度その形を踏んでいる）。

    ## 失敗を数えて返す

    読めなかったレースを黙って捨てていたのが今回の原因なので、失敗は必ず
    数えて返し、呼ぶ側で見えるようにする。
    """
    stats = {"days": 0, "added": 0, "failed": 0, "still_short": 0}

    for day in racing_days(start, end):
        out_path = out_dir / str(day.year) / f"{day.isoformat()}.jsonl.gz"
        if not out_path.exists():
            continue

        existing = list(read_jsonl(out_path))
        have = {c.race.race_id for c in existing}
        listed = race_ids_for_day(fetcher, day)
        missing = [r for r in listed if r not in have]
        if not missing:
            continue

        stats["days"] += 1
        added: list[RaceCard] = []
        for race_id in missing:
            try:
                html = fetcher.fetch(RACE_URL.format(race_id=race_id))
                card = netkeiba.parse_race_page(html, race_id)
            except (FetchError, ValueError) as exc:
                log.warning("取り直せない %s: %s", race_id, exc)
                stats["failed"] += 1
                continue
            if card.race.race_date != day or not card.results:
                continue
            added.append(card)

        if added:
            merged = sorted(existing + added, key=lambda c: c.race.race_id)
            write_jsonl(merged, out_path)
            stats["added"] += len(added)
            log.info("%s: %d レースを足した（計 %d）", day, len(added), len(merged))

        # 場ごとに12レース揃ったかを見る。揃わない場合は一覧側の取りこぼし
        # なので、ここで数えて見えるようにする。
        by_venue: dict[str, int] = {}
        for c in existing + added:
            by_venue[c.race.venue] = by_venue.get(c.race.venue, 0) + 1
        stats["still_short"] += sum(1 for n in by_venue.values() if n < 12)

    return stats


def needs_refill(card: RaceCard) -> bool:
    """db.netkeiba から埋め直すべき欠けがあるか。

    JRA公式経由（2026-08-08〜）の出馬表は、馬場状態と枠番が全レースで空、
    馬体重も1割ほど欠けていた。着順が片方の開催だけ入っていない日もあった。
    """
    if card.race.going is None:
        return True
    live = [e for e in card.entries if not e.scratched and e.umaban is not None]
    if any(e.waku is None for e in live):
        return True
    if not card.results:
        return True
    return False


def fill_blanks(card: RaceCard, ref: RaceCard) -> int:
    """card の空欄だけを ref（db.netkeiba）の値で埋める。埋めた欄の数を返す。

    **既にある値は書き換えない。** JRA公式の着順・払戻、後から入れた調教や
    コメントの数を消さないため。馬の突き合わせは馬番ではなく馬IDで行う
    （取り違えたら別の馬の枠と馬体重が付く）。
    """
    n = 0
    for attr in ("going", "weather", "grade", "direction"):
        if getattr(card.race, attr) is None and getattr(ref.race, attr) is not None:
            setattr(card.race, attr, getattr(ref.race, attr))
            n += 1

    ref_entries = {e.horse_id: e for e in ref.entries if e.horse_id}
    for e in card.entries:
        r = ref_entries.get(e.horse_id)
        if r is None or (e.umaban is not None and r.umaban != e.umaban):
            continue
        for attr in (
            "umaban", "waku", "body_weight", "body_weight_diff",
            "market_odds", "market_popularity", "sire", "dam", "damsire",
        ):
            if getattr(e, attr, None) is None and getattr(r, attr, None) is not None:
                setattr(e, attr, getattr(r, attr))
                n += 1

    if not card.results and ref.results:
        card.results = list(ref.results)
        n += len(ref.results)
    else:
        by_umaban = {r.umaban: r for r in ref.results}
        for res in card.results:
            r = by_umaban.get(res.umaban)
            if r is None:
                continue
            for attr in ("time_sec", "margin", "last3f", "body_weight", "body_weight_diff"):
                if getattr(res, attr) is None and getattr(r, attr) is not None:
                    setattr(res, attr, getattr(r, attr))
                    n += 1
            if not res.corners and r.corners:
                res.corners = list(r.corners)
                n += 1
    if not card.payouts and ref.payouts:
        card.payouts = list(ref.payouts)
        n += len(ref.payouts)
    return n


def refill_races(
    fetcher: Fetcher, start: date, end: date, out_dir: Path
) -> dict[str, int]:
    """取ってある日の**欠けた欄**を db.netkeiba で埋める。

    repair_races は「レースごと無い」を直す。こちらは「レースはあるが中身が
    欠けている」を直す。JRA公式に切り替えた 2026-08-08 以降、馬場状態と
    枠番が全レースで空、馬体重・着順も一部欠けていた。db.netkeiba は反映が
    遅いだけで、数日後には全部載る。

    埋まらなかったレースは数えて返す（黙って残さない）。
    """
    stats = {"days": 0, "races": 0, "filled": 0, "failed": 0, "still_missing": 0}

    for day in racing_days(start, end):
        out_path = out_dir / str(day.year) / f"{day.isoformat()}.jsonl.gz"
        if not out_path.exists():
            continue
        cards = list(read_jsonl(out_path))
        todo = [c for c in cards if needs_refill(c)]
        if not todo:
            continue

        stats["days"] += 1
        changed = 0
        for card in todo:
            race_id = card.race.race_id
            try:
                html = fetcher.fetch(RACE_URL.format(race_id=race_id))
                ref = netkeiba.parse_race_page(html, race_id)
            except (FetchError, ValueError) as exc:
                log.warning("埋め直せない %s: %s", race_id, exc)
                stats["failed"] += 1
                continue
            n = fill_blanks(card, ref)
            if n:
                changed += 1
                stats["filled"] += n
            if needs_refill(card):
                stats["still_missing"] += 1
        stats["races"] += changed

        if changed:
            write_jsonl(cards, out_path)
            log.info("%s: %d / %d レースの欠けを埋めた", day, changed, len(todo))

    return stats


def collect_pedigrees(
    fetcher: Fetcher,
    store: Store,
    out_path: Path,
    limit: int | None = None,
    offset: int = 0,
    newest_first: bool = False,
) -> int:
    """血統をまだ持っていない馬の血統ページを引く。

    newest_first は週次の運用向け。新馬・2歳馬は horse_id が大きいので、
    後ろから引くと「過去走が無く、血統でしか評価できない馬」が先に埋まる。

    entries に居るのに horses に居ない馬だけを対象にするので、
    途中で止めて再実行しても取り直しにならない。

    対象リストは horse_id の昇順で安定しているため、offset と limit で
    重複なく分割できる。2.5万頭を1ジョブで引くと Actions の6時間上限を
    超えるので、並列ジョブに割るために使う。
    """
    targets = store.horse_ids_without_pedigree()
    if newest_first:
        targets = targets[::-1]
    targets = targets[offset:]
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


TRAINING_URL = "https://db.netkeiba.com/?pid=horse_training&id={horse_id}"

# 調教が空で返る馬はデビュー前などで普通に居る。ただし**全部空**なら、
# それは馬の問題ではなくログインが切れている。この開発では「収集は成功・
# 中身は空」を何度も踏んでいるので、比率で見張って落とす。
EMPTY_RATIO_LIMIT = 0.5
EMPTY_CHECK_AFTER = 40


def collect_workouts(
    fetcher: Fetcher,
    store: Store,
    out_path: Path,
    limit: int | None = None,
    offset: int = 0,
    targets: list[str] | None = None,
    refresh: bool = False,
) -> int:
    """netkeiba 有料プランの調教タイムを馬単位で引く。

    rid を付けなければ1リクエストでその馬の全履歴が返るので、出走ごとでは
    なく馬ごとに引く（実測。約14.9万回が約2.4万回になる）。

    targets を渡さなければ、対象は「まだ一度も引いていない馬」を最終出走日の
    新しい順に並べたもの。途中で止めても「直近◯ヶ月は揃っている」状態になり、
    その期間だけで調教あり／なしを比べられる。

    毎週の運用では呼び出し側が targets を作って渡す。今週の出走馬だけを、
    しかも**一度引いた馬も含めて**引き直す必要があるため（調教は毎週更新され、
    3週間より古いものは特徴量として使われない）。

    refresh を立てるとディスクキャッシュを無視して取り直す。同じ URL でも
    中身が毎週変わるページなので、引き直しのつもりでキャッシュを読んでいたら
    先週の追い切りを今週の状態として使うことになる。
    """
    from keiba.sources import netkeiba_auth

    # 未ログインでも netkeiba は200で案内ページを返す。黙って空を積まないよう、
    # ここは認証情報が無い時点で落とす。
    netkeiba_auth.login(fetcher, required=True)

    if targets is None:
        targets = store.horse_ids_without_workouts()
    targets = targets[offset:]
    if limit:
        targets = targets[:limit]
    if not targets:
        log.info("調教の未取得馬なし")
        return 0

    log.info("調教を引く対象: %d頭", len(targets))
    records: list[dict] = []
    empty = 0
    empty_bytes: list[int] = []
    for i, horse_id in enumerate(targets, 1):
        try:
            html = fetcher.fetch(
                TRAINING_URL.format(horse_id=horse_id), force=refresh
            )
        except FetchError as exc:
            log.warning("調教を取得できない %s: %s", horse_id, exc)
            continue

        rows = netkeiba.parse_horse_training(html, horse_id)
        if not rows:
            empty += 1
            empty_bytes.append(len(html))
        records.extend(w.to_dict() for w in rows)

        if i == EMPTY_CHECK_AFTER and empty / i > EMPTY_RATIO_LIMIT:
            # ここで止めないと、空のまま2.4万頭ぶん走って何時間も無駄にする。
            #
            # 止める理由は正しいが、**理由の言い当ては間違えうる**。2026-09-11
            # はここで「ログインが効いていない」と言って落ちた。実際には同じ
            # run の25秒前にログインは成功していて、レース単位の追い切りページ
            # からは131頭ぶん取れていた。効いていなかったのは資格情報ではなく、
            # このエンドポイントのほう（全頭が 200 で 85バイト）。
            #
            # 推測を書かず、**観測した大きさを書く**。中身の薄いページが返って
            # いるのか、中身はあるのに解析できていないのかが、これで分かれる。
            biggest = max(empty_bytes)
            raise RuntimeError(
                f"最初の{i}頭のうち{empty}頭が空（本文は最大 {biggest} バイト）。"
                " 数百バイトしか返っていないならページ側が変わっている。"
                " 数万バイトあるなら解析が合っていない。"
                " どちらでもなければ有料プランの状態と Secrets を確かめること"
            )
        if i % 200 == 0:
            log.info("  %d/%d（空 %d頭・調教 %d本）", i, len(targets), empty, len(records))

    if records:
        store.upsert_horse_workouts(records)
        _merge_workout_file(records, out_path)
    log.info("調教 %d本 / %d頭（うち空 %d頭）", len(records), len(targets), empty)
    return len(records)


def _merge_workout_file(records: list[dict], path: Path) -> None:
    """調教は馬・日付・コース単位の追記型ファイルにまとめる。

    血統と同じ形。既存分と突き合わせて書き直すので、途中で止めて再実行しても
    重複せず、git の差分も追記だけになる。
    """
    import gzip
    import json

    def key(row: dict) -> tuple[str, str, str]:
        return (row["horse_id"], row["workout_date"], row["course"])

    merged: dict[tuple[str, str, str], dict] = {}
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    merged[key(row)] = row
    for row in records:
        merged[key(row)] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for k in sorted(merged):
            fh.write(json.dumps(merged[k], ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def load_workout_file(store: Store, path: Path) -> int:
    """raw の調教ファイルを horse_workouts へ流し込む（rebuild 時に使う）。"""
    if not path.exists():
        return 0
    import gzip
    import json

    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return store.upsert_horse_workouts(rows)
