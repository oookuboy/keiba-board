"""db.netkeiba.com のパーサ。

race.netkeiba.com は JavaScript レンダリングで requests から中身が取れないため、
過去データの収集源は db.netkeiba.com に一本化する。ここはサーバーレンダリングで
結果・払戻・血統がすべて HTML に入っている。

取得できるもの:
    /race/list/YYYYMMDD/   その日のレース一覧（地方も混ざるので場コードで絞る）
    /race/{race_id}/       出走馬・着順・通過順・上り・馬体重・払戻
    /horse/ped/{horse_id}/ 5代血統表（父・母・母父・系統）

出走馬の父はレースページに無く、馬ごとの血統ページを引く必要がある。
3年分で約2.5万頭ぶんのリクエストになるので、バックフィルでは馬IDを集めてから
まとめて引く二段構えにする（backfill.py 側の責務）。
"""

from __future__ import annotations

import logging
import re
from datetime import date

from bs4 import BeautifulSoup, Tag

from keiba.models import VENUES, Entry, Payout, Race, RaceCard, Result

log = logging.getLogger(__name__)

RACE_ID_RE = re.compile(r"/race/(\d{12})")
HORSE_ID_RE = re.compile(r"/horse/(\d+)")
JOCKEY_ID_RE = re.compile(r"/jockey/(?:result/recent/)?(\w+)")
TRAINER_ID_RE = re.compile(r"/trainer/(?:result/recent/)?(\w+)")

# '芝左1200m' / 'ダ右1800m' / '障芝ダ3000m' / '直200m'(ばんえい)
COURSE_RE = re.compile(r"(障芝ダ|障芝|障ダ|芝|ダ|直)\s*([右左内外直]*)\s*(\d+)\s*m")
GOING_RE = re.compile(r"(?:芝|ダート|ダ|障)\s*[:：]\s*(良|稍重|重|不良)")
WEATHER_RE = re.compile(r"天候\s*[:：]\s*(\S+?)(?:\s|/|$)")
POST_RE = re.compile(r"発走\s*[:：]\s*(\d{1,2}:\d{2})")
DATE_RE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日")
MEETING_RE = re.compile(r"(\d+)回(\D+?)(\d+)日目")
GRADE_RE = re.compile(r"\((G[123]|GI{1,3}|J\.?G[123]|L)\)")
SEX_AGE_RE = re.compile(r"([牡牝セせん]+)\s*(\d+)")
BODY_WEIGHT_RE = re.compile(r"(\d+)\s*\(([-+]?\d+)\)")


def _text(node: Tag | None) -> str:
    return node.get_text(" ", strip=True).replace("\xa0", " ") if node else ""


def _first_id(node: Tag | None, pattern: re.Pattern[str]) -> str | None:
    if node is None:
        return None
    link = node if node.name == "a" else node.find("a")
    if not isinstance(link, Tag) or not link.has_attr("href"):
        return None
    m = pattern.search(str(link["href"]))
    return m.group(1) if m else None


def _int(value: str) -> int | None:
    m = re.search(r"-?\d+", value.replace(",", ""))
    return int(m.group()) if m else None


def _float(value: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(m.group()) if m else None


def time_to_sec(value: str) -> float | None:
    """'1:07.8' → 67.8 / '57.4' → 57.4"""
    value = value.strip()
    m = re.match(r"(?:(\d+):)?(\d+(?:\.\d+)?)$", value)
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    return minutes * 60 + float(m.group(2))


def parse_finish_pos(value: str) -> int | None:
    """着順。中止・除外・失格・取消は None（走ったが着順が無い、を区別しない）。"""
    value = value.strip()
    return int(value) if value.isdigit() else None


def parse_body_weight(value: str) -> tuple[int | None, int | None]:
    """'466(-2)' → (466, -2)。'計不' などは (None, None)。"""
    m = BODY_WEIGHT_RE.search(value.replace(" ", ""))
    if m:
        return int(m.group(1)), int(m.group(2))
    only = re.fullmatch(r"\s*(\d+)\s*", value)
    return (int(only.group(1)), None) if only else (None, None)


def parse_corners(value: str) -> list[int]:
    """'8-7' → [8, 7]"""
    return [int(x) for x in re.findall(r"\d+", value)]


# ------------------------------------------------------------------ 一覧

def parse_race_list(html: str, jra_only: bool = True) -> list[str]:
    """レース一覧ページから race_id を拾う。

    地方競馬は毎日開催しているので、既定では場コード 01〜10（中央）に絞る。
    絞らないと盛岡・船橋・高知・帯広ばかりになる。
    """
    ids = sorted(set(RACE_ID_RE.findall(html)))
    return [i for i in ids if not jra_only or i[4:6] in VENUES]


# -------------------------------------------------------------- レース

def _parse_header(soup: BeautifulSoup, race_id: str) -> Race:
    name = _text(soup.select_one("dl.racedata h1"))
    spec = _text(soup.select_one("dl.racedata span"))
    small = _text(soup.select_one("p.smalltxt"))

    course = COURSE_RE.search(spec)
    if not course:
        raise ValueError(f"コース情報を読めない: {spec!r} ({race_id})")
    raw_surface, direction, distance = course.groups()
    surface = "障" if raw_surface.startswith("障") else raw_surface
    if surface == "直":  # ばんえい。中央には出ないが型は保つ
        surface = "ダ"

    d = DATE_RE.search(small)
    if not d:
        raise ValueError(f"日付を読めない: {small!r} ({race_id})")
    race_date = date(int(d.group(1)), int(d.group(2)), int(d.group(3)))

    meeting = MEETING_RE.search(small)
    kai, venue, nichi = (
        (int(meeting.group(1)), meeting.group(2), int(meeting.group(3)))
        if meeting
        else (0, VENUES.get(race_id[4:6], "?"), 0)
    )

    going = GOING_RE.search(spec)
    weather = WEATHER_RE.search(spec)
    post = POST_RE.search(spec)
    grade = GRADE_RE.search(name)

    # smalltxt の末尾に条件が付く: '… 3歳未勝利  [指](馬齢)'
    # smalltxt の末尾: '3歳未勝利  [指](馬齢)' や '2歳未勝利  (混)'。
    # クラス比較（降級・格上帰りの判定）に使うので、条件記号は落として素の条件名にする。
    race_class = None
    if meeting:
        tail = small[meeting.end() :].strip()
        tail = re.split(r"\s*\[", tail)[0]
        race_class = re.sub(r"\s*[(（][^)）]*[)）]", "", tail).strip() or None

    header_no = _text(soup.select_one("div.mainrace_data dl dt"))

    return Race(
        race_id=race_id,
        race_date=race_date,
        venue=venue,
        venue_code=race_id[4:6],
        kai=kai,
        nichi=nichi,
        race_no=_int(header_no) or int(race_id[10:12]),
        name=name,
        surface=surface,
        distance=int(distance),
        grade=grade.group(1) if grade else None,
        race_class=race_class,
        direction=direction or None,
        going=going.group(1) if going else None,
        weather=weather.group(1) if weather else None,
        post_time=post.group(1) if post else None,
    )


# 結果テーブルの列位置。db.netkeiba の 25 列レイアウトに対応する。
# 9〜13 はタイム指数系で、無料では '**' しか返らないので使わない。
COL = {
    "finish": 0,
    "waku": 1,
    "umaban": 2,
    "horse": 3,
    "sex_age": 4,
    "weight_carried": 5,
    "jockey": 6,
    "time": 7,
    "margin": 8,
    "corners": 14,
    "last3f": 15,
    "odds": 16,
    "popularity": 17,
    "body_weight": 18,
    "trainer": 22,
}


def _parse_results_table(table: Tag, race_id: str) -> tuple[list[Entry], list[Result]]:
    entries: list[Entry] = []
    results: list[Result] = []

    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) <= COL["trainer"]:
            continue

        def cell(key: str) -> str:
            return _text(tds[COL[key]])

        umaban = _int(cell("umaban"))
        if umaban is None:
            continue

        sex_age = SEX_AGE_RE.search(cell("sex_age"))
        trainer_raw = cell("trainer")
        affiliation = None
        if "[西]" in trainer_raw:
            affiliation = "栗東"
        elif "[東]" in trainer_raw:
            affiliation = "美浦"
        elif "[地]" in trainer_raw:
            affiliation = "地方"
        elif "[外]" in trainer_raw:
            affiliation = "外国"

        body_weight, body_diff = parse_body_weight(cell("body_weight"))

        entries.append(
            Entry(
                race_id=race_id,
                umaban=umaban,
                horse_name=cell("horse"),
                horse_id=_first_id(tds[COL["horse"]], HORSE_ID_RE) or "",
                waku=_int(cell("waku")),
                sex=sex_age.group(1) if sex_age else None,
                age=int(sex_age.group(2)) if sex_age else None,
                weight_carried=_float(cell("weight_carried")),
                jockey=cell("jockey"),
                jockey_id=_first_id(tds[COL["jockey"]], JOCKEY_ID_RE),
                trainer=re.sub(r"\[.\]\s*", "", trainer_raw),
                trainer_id=_first_id(tds[COL["trainer"]], TRAINER_ID_RE),
                affiliation=affiliation,
                body_weight=body_weight,
                body_weight_diff=body_diff,
                # 記録専用。エンジンからは参照しない
                market_odds=_float(cell("odds")),
                market_popularity=_int(cell("popularity")),
            )
        )
        results.append(
            Result(
                race_id=race_id,
                umaban=umaban,
                finish_pos=parse_finish_pos(cell("finish")),
                time_sec=time_to_sec(cell("time")),
                margin=cell("margin") or None,
                corners=parse_corners(cell("corners")),
                last3f=_float(cell("last3f")),
                body_weight=body_weight,
                body_weight_diff=body_diff,
            )
        )

    return entries, results


def _parse_payouts(soup: BeautifulSoup, race_id: str) -> list[Payout]:
    """払戻テーブル。複勝・ワイドは1セルに複数行入るので <br> で割る。"""
    payouts: list[Payout] = []
    for table in soup.select("table.pay_table_01"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) < 3:
                continue

            def lines(cell: Tag) -> list[str]:
                for br in cell.find_all("br"):
                    br.replace_with("\n")
                return [s.strip() for s in cell.get_text("\n").split("\n") if s.strip()]

            bet_type = _text(cells[0])
            combos = lines(cells[1])
            amounts = lines(cells[2])
            pops = lines(cells[3]) if len(cells) > 3 else []

            for i, combo in enumerate(combos):
                amount = _int(amounts[i]) if i < len(amounts) else None
                if amount is None:
                    continue
                payouts.append(
                    Payout(
                        race_id=race_id,
                        bet_type=bet_type,
                        combination=re.sub(r"\s*[-→]\s*", "-", combo),
                        payout=amount,
                        popularity=_int(pops[i]) if i < len(pops) else None,
                    )
                )
    return payouts


def parse_race_page(html: str, race_id: str) -> RaceCard:
    """/race/{race_id}/ を RaceCard にする。"""
    soup = BeautifulSoup(html, "lxml")
    race = _parse_header(soup, race_id)

    table = soup.select_one("table.race_table_01")
    if table is None:
        raise ValueError(f"結果テーブルが無い: {race_id}")
    entries, results = _parse_results_table(table, race_id)

    race.field_size = len(entries)
    # 着順が1つも無ければ発走前（出馬表のみ）とみなし、結果は捨てる
    if not any(r.finish_pos is not None for r in results):
        results = []

    return RaceCard(
        race=race,
        entries=entries,
        results=results,
        payouts=_parse_payouts(soup, race_id),
    )


# ---------------------------------------------------------------- 血統

def parse_pedigree(html: str) -> dict[str, str | None]:
    """/horse/ped/{horse_id}/ から父・母・母父・父の系統を取る。

    5代血統表は rowspan で組まれていて、32行のうち row[0] の先頭セルが父、
    row[16] の先頭セルが母、その隣が母父になる。
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.blood_table.detail") or soup.select_one(
        "table.blood_table"
    )
    if table is None:
        return {"sire": None, "dam": None, "damsire": None, "sire_line": None}

    rows = table.find_all("tr")
    if not rows:
        return {"sire": None, "dam": None, "damsire": None, "sire_line": None}

    def horse_name(cell: Tag | None) -> str | None:
        if cell is None:
            return None
        link = cell.find("a")
        name = _text(link) if link else _text(cell)
        # 'アルアイン 2014 鹿毛 [ 血統 ][ 産駒 ] Halo系' から名前だけ削り出す
        name = re.split(r"\s*\d{4}\s*", name)[0]
        name = re.sub(r"\s*\[.*", "", name)
        # 外国産馬は 'コンクエストハーラネイト Conquest Harlanate(加)' のように
        # 英語表記が続く。カナ名で始まるならカナ部分だけを採る。
        kana = re.match(r"^([ァ-ヶー・]+)\s+[A-Za-z]", name)
        if kana:
            name = kana.group(1)
        return name.strip() or None

    top = rows[0].find_all("td")
    sire = horse_name(top[0] if top else None)

    line = None
    if top:
        m = re.search(r"([ァ-ヶА-Яa-zA-Z]+系)", _text(top[0]))
        line = m.group(1) if m else None

    dam = damsire = None
    half = len(rows) // 2
    if len(rows) > half:
        bottom = rows[half].find_all("td")
        dam = horse_name(bottom[0] if bottom else None)
        damsire = horse_name(bottom[1] if len(bottom) > 1 else None)

    return {"sire": sire, "dam": dam, "damsire": damsire, "sire_line": line}
