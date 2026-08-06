"""JRA公式サイトの遷移と出馬表パーサ。

規約上いちばん安全な経路。`www.jra.go.jp/robots.txt` は `Disallow:` が空で、
全面許可であることを実測で確認している。

そして**発走前の出馬表はここでしか取れない**。db.netkeiba.com は結果の
データベースなので、まだ行われていないレースは race_id ごと存在しない。
2026-08-06（木・出馬表公開後）に実測した結果:

    db.netkeiba.com/race/list/20260808/ → 開催場の名前は出るが race_id は0件

「木曜になれば netkeiba 側にも生える」という読みは外れだった。

## 遷移

URL を組み立てることはできない。cname 末尾2桁（例 `/65`）が何のハッシュか
分からないため。ページから拾って POST で辿る。

    accessD.html  cname=pw01dli00/F3                        開催選択
      └ cname=pw01drl00 + 場(2) + 年(4) + 回(2) + 日(2) + 日付(8) + /xx   出走馬一覧

出走馬一覧は**その開催の全12レースが1ページ**に載っている。3場×2日＝6回の
POST で週末ぶんが揃うので、レース単位で叩く必要はない。

## 木曜時点で埋まらない欄

枠・馬番・馬体重・単勝オッズは空で返る。枠順確定は金曜なので、木曜に
取れるのは「誰が出るか」まで。買い目は馬番で組むので、確定するまでは
賭けられない。`post_positions_confirmed` で明示的に見分ける。

## 馬IDは netkeiba と同一体系

馬名リンクの `CNAME=pw01dud00` に続く10桁が馬ID。2026-08-08 新潟の162頭で
照合したところ、138頭が手元の3年分と一致し、**名前の食い違いはゼロ**だった
（残り24頭は過去走の無い新馬）。よって過去走・血統とそのまま結合できる。
"""

from __future__ import annotations

import logging
import re
from datetime import date

from bs4 import BeautifulSoup, Tag

from keiba.models import VENUES, Entry, Race, RaceCard
from keiba.sources.http import Fetcher, FetchError

log = logging.getLogger(__name__)

BASE = "https://www.jra.go.jp"
DOACTION_RE = re.compile(r"doAction\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)")

# トップページの doAction から採取した実物の入口。
SEEDS = {
    "racecard": ("/JRADB/accessD.html", "pw01dli00/F3"),   # 出馬表
    "results": ("/JRADB/accessS.html", "pw01sli00/AF"),    # レース結果
    "training": ("/JRADB/accessT.html", "pw03trl00/29"),   # 調教
    "info": ("/JRADB/accessI.html", "pw01ide01/4F"),       # 開催お知らせ
}

# どのページにも並ぶ共通メニュー。辿る先から除外しないと
# 「オッズ」「払戻金一覧」「競走馬検索」ばかりを拾ってしまう。
MENU_CNAMES = {cname for _, cname in SEEDS.values()} | {
    "pw15oli00/6D",     # オッズ
    "pw01hli00/03",     # 払戻金
    "pw02uliD19999",    # 競走馬検索
    "pr13rli00/8E",     # レコードタイム表
    "pv156liH1/98",     # 競走馬登録・抹消一覧
    "pr12rnk00/01",     # 払戻金ランキング
}

# 開催リンク: pw01drl + 00 + 場コード(2) + 年(4) + 回(2) + 日(2) + 日付(8)
KAISAI_RE = re.compile(r"pw01drl00(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})")
HORSE_ID_RE = re.compile(r"CNAME=pw01dud00(\d{10})")
JOCKEY_ID_RE = re.compile(r"pw04kmk00(\d+)")
TRAINER_ID_RE = re.compile(r"pw05cmk00(\d+)")

VENUE_CODE = {name: code for code, name in VENUES.items()}

_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_KAISAI_TEXT_RE = re.compile(r"(\d+)回(\D+?)(\d+)日")
_RACE_NO_RE = re.compile(r"(\d+)\s*レース")
_DISTANCE_RE = re.compile(r"([\d,]+)\s*メートル")
# 新潟1000mは「（芝・直）」で、「直線」とは書かない。外回りは「（芝・左 外）」。
_SURFACE_RE = re.compile(r"[（(]\s*(芝|ダート|障害)\s*[・･]?\s*(左|右|直線|直)?\s*(外|内)?")
_POST_TIME_RE = re.compile(r"(\d{1,2})時(\d{1,2})分")
_AGE_RE = re.compile(r"(牡|牝|セン|セ)\s*(\d+)")
_PRIZE_RE = re.compile(r"1着\s*([\d,]+)")


# ------------------------------------------------------------------ 遷移

def open_page(fetcher: Fetcher, action: str, cname: str) -> str | None:
    """cname を POST して1ページ取る。"""
    try:
        return fetcher.fetch(f"{BASE}{action}", method="POST", data={"cname": cname})
    except FetchError as exc:
        log.warning("JRA公式を取得できない %s %s: %s", action, cname, exc)
        return None


def links(html: str, *, skip_menu: bool = True) -> list[tuple[str, str]]:
    """ページ内の遷移先 (action, cname) を返す。共通メニューは既定で除く。"""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for action, cname in DOACTION_RE.findall(html):
        if cname in seen or (skip_menu and cname in MENU_CNAMES):
            continue
        seen.add(cname)
        out.append((action, cname))
    return out


def open_seed(fetcher: Fetcher, name: str) -> str | None:
    """入口ページを開く。name は SEEDS のキー。"""
    action, cname = SEEDS[name]
    return open_page(fetcher, action, cname)


def find_kaisai_links(html: str) -> list[dict]:
    """開催選択ページから、各開催（場×日）へのリンクを拾う。

    cname はそのまま POST に使う。こちらで組み立てないのは末尾2桁が
    何なのか分かっていないため。開催が未公開の時期は空リストになる。
    """
    out: list[dict] = []
    seen: set[str] = set()
    for action, cname in DOACTION_RE.findall(html):
        m = KAISAI_RE.match(cname)
        if not m or cname in seen:
            continue
        seen.add(cname)
        venue_code, year, kai, nichi, stamp = m.groups()
        out.append(
            {
                "action": action,
                "cname": cname,
                "venue_code": venue_code,
                "venue": VENUES.get(venue_code, venue_code),
                "kai": int(kai),
                "nichi": int(nichi),
                "race_date": date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8])),
                "year": int(year),
            }
        )
    return out


# ------------------------------------------------------------------ パーサ

def _text(node: Tag | None) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)) if node else ""


def _race_block_of(table: Tag) -> Tag | None:
    """1レースぶんの見出しと表をまとめている親を返す。

    見出しだけを別途たどると、レースを取り違えたときに気づけない。表を
    含む親から探すことで、番号と表が必ず同じレースのものになる。
    """
    parent = table.parent
    while parent is not None:
        if parent.select_one(".race_number") is not None:
            return parent
        parent = parent.parent
    return None


def _as_int(node: Tag | None) -> int | None:
    raw = _text(node)
    return int(raw) if raw.isdigit() else None


def _person(row: Tag, css: str, pattern: re.Pattern[str]) -> tuple[str | None, str | None]:
    node = row.select_one(f"td.{css}")
    if node is None:
        return None, None
    # 「△ 石神 深道」のような減量記号を落とす
    label = re.sub(r"^[▲△☆★◇]\s*", "", _text(node)) or None
    anchor = node.find("a")
    ident = None
    if isinstance(anchor, Tag):
        m = pattern.search(str(anchor.get("onclick") or ""))
        ident = m.group(1) if m else None
    return label, ident


def _parse_entry_row(row: Tag, race_id: str) -> Entry | None:
    horse_cell = row.select_one("td.horse")
    if horse_cell is None:
        return None
    name = _text(horse_cell)
    if not name:
        return None

    horse_id = ""
    link = horse_cell.find("a")
    if isinstance(link, Tag):
        if m := HORSE_ID_RE.search(str(link.get("href") or "")):
            horse_id = m.group(1)

    sex = age = None
    if m := _AGE_RE.search(_text(row.select_one("td.age"))):
        sex = "セ" if m.group(1).startswith("セ") else m.group(1)
        age = int(m.group(2))

    weight_carried = None
    if m := re.search(r"([\d.]+)", _text(row.select_one("td.weight"))):
        weight_carried = float(m.group(1))

    body_weight = None
    if m := re.search(r"(\d{3})", _text(row.select_one("td.h_weight"))):
        body_weight = int(m.group(1))

    odds = None
    if m := re.search(r"([\d.]+)", _text(row.select_one("td.odds"))):
        odds = float(m.group(1))

    jockey, jockey_id = _person(row, "jockey", JOCKEY_ID_RE)
    trainer, trainer_id = _person(row, "trainer", TRAINER_ID_RE)

    return Entry(
        race_id=race_id,
        # 枠・馬番は金曜の枠順確定まで空。数字が無ければ None のままにする
        umaban=_as_int(row.select_one("td.num")),
        horse_name=name,
        horse_id=horse_id,
        waku=_as_int(row.select_one("td.waku")),
        sex=sex,
        age=age,
        weight_carried=weight_carried,
        jockey=jockey,
        jockey_id=jockey_id,
        trainer=trainer,
        trainer_id=trainer_id,
        body_weight=body_weight,
        market_odds=odds,
    )


def parse_racecard_page(html: str) -> list[RaceCard]:
    """出走馬一覧（1開催＝全12レース）を RaceCard のリストにする。

    馬番が未確定でも落とさずに返す。「まだ枠順が出ていない」ことと
    「パースに失敗した」ことは区別できないと運用で困る。
    """
    soup = BeautifulSoup(html, "lxml")

    # 先頭の h1 は空で、h2 の先頭は「検索ウィンドウ」。位置で決め打ちせず、
    # 日付と開催（n回◯◯m日）の両方を含む見出しを探す。
    heading = ""
    for node in soup.select("h1, h2"):
        text = _text(node)
        if _DATE_RE.search(text) and _KAISAI_TEXT_RE.search(text):
            heading = text
            break
    dm = _DATE_RE.search(heading)
    km = _KAISAI_TEXT_RE.search(heading)
    if not dm or not km:
        raise ValueError(f"開催見出しを解釈できない: {heading[:80]!r}")

    race_date = date(int(dm.group(1)), int(dm.group(2)), int(dm.group(3)))
    kai, venue, nichi = int(km.group(1)), km.group(2).strip(), int(km.group(3))
    venue_code = VENUE_CODE.get(venue)
    if venue_code is None:
        raise ValueError(f"中央の競馬場ではない: {venue!r}")

    cards: list[RaceCard] = []
    for table in soup.select("table.basic"):
        if table.select_one("td.horse") is None:
            continue
        block = _race_block_of(table)
        if block is None:
            continue

        num_img = block.select_one(".race_number img")
        alt = str(num_img.get("alt") or "") if isinstance(num_img, Tag) else ""
        rm = _RACE_NO_RE.search(alt)
        if not rm:
            log.warning("レース番号を取れない（alt=%r）", alt)
            continue
        race_no = int(rm.group(1))

        text = _text(block)
        race_id = f"{race_date.year}{venue_code}{kai:02d}{nichi:02d}{race_no:02d}"

        distance = surface = direction = None
        if m := _DISTANCE_RE.search(text):
            distance = int(m.group(1).replace(",", ""))
        if m := _SURFACE_RE.search(text):
            surface = {"ダート": "ダ", "障害": "障"}.get(m.group(1), m.group(1))
            # netkeiba 側の表記に寄せる（「直」→「直線」）
            direction = {"直": "直線"}.get(m.group(2) or "", m.group(2))
        if distance is None or surface is None:
            log.warning("%s: コース情報を取れない", race_id)
            continue

        post_time = None
        if m := _POST_TIME_RE.search(text):
            post_time = f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
        prize = None
        if m := _PRIZE_RE.search(text):
            prize = int(m.group(1).replace(",", ""))

        entries = [
            e
            for row in table.select("tbody tr")
            if (e := _parse_entry_row(row, race_id)) is not None
        ]
        if not entries:
            continue

        race = Race(
            race_id=race_id,
            race_date=race_date,
            venue=venue,
            venue_code=venue_code,
            kai=kai,
            nichi=nichi,
            race_no=race_no,
            name=_text(block.select_one(".race_name")) or f"{race_no}R",
            surface=surface,
            distance=distance,
            direction=direction,
            race_class=_text(block.select_one(".cell.class")) or None,
            post_time=post_time,
            prize=prize,
            field_size=len(entries),
        )
        cards.append(RaceCard(race=race, entries=entries, results=[], payouts=[]))

    return cards


def post_positions_confirmed(card: RaceCard) -> bool:
    """枠順が確定しているか。

    木曜の出馬表は馬番が空で返る。買い目は馬番で組むので、確定するまでは
    予想を出しても賭けられない。「取れなかった」と混同しないよう分けて扱う。
    """
    return bool(card.entries) and all(e.umaban is not None for e in card.entries)


def collect_racecards(fetcher: Fetcher) -> list[RaceCard]:
    """公開中の全開催ぶんの出馬表を取る。

    開催が未公開なら空リスト。呼び出し側はそれを「まだ出ていない」として
    扱うこと。黙って空の予想を出すより、取れていないと分かるほうがよい。
    """
    html = open_seed(fetcher, "racecard")
    if html is None:
        return []

    kaisai = find_kaisai_links(html)
    if not kaisai:
        log.info("JRA公式に開催リンクなし（出馬表は木曜公開）")
        return []

    cards: list[RaceCard] = []
    for meeting in kaisai:
        page = open_page(fetcher, meeting["action"], meeting["cname"])
        if page is None:
            continue
        try:
            found = parse_racecard_page(page)
        except ValueError as exc:
            log.warning("%s %s の出馬表を解釈できない: %s",
                        meeting["race_date"], meeting["venue"], exc)
            continue
        log.info("%s %s: %d レース", meeting["race_date"], meeting["venue"], len(found))
        cards += found
    return cards
