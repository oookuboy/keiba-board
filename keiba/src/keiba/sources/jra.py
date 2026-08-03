"""JRA公式サイトの遷移。

規約上いちばん安全な経路。`www.jra.go.jp/robots.txt` は `Disallow:` が空で、
全面許可であることを実測で確認している。

遷移は素の URL ではなく doAction('/JRADB/accessX.html','pw01...') の POST
方式になっている。入口の cname は固定だが、その先（開催日・レース単位）の
cname は毎回変わるので、辿って採取するしかない。ここまでは実測で動作を確認
済み（accessD/accessS/accessT/accessI すべて 200 で実ページが返る）。

**未完: 出馬表・調教テーブルのパーサ。**
JRA公式は出馬表を木曜に公開するため、週明けに叩くと開催選択ページに開催
リンクが1つも無い（実測で確認）。テーブルの実物を採取できていないので、
パーサは木曜以降の keiba-probe の結果を見てから書く。それまで collect.py は
db.netkeiba を主経路として動く。
"""

from __future__ import annotations

import logging
import re

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


def meeting_links(fetcher: Fetcher, name: str) -> list[tuple[str, str]]:
    """入口から辿れる開催（または個別レース）の遷移先を返す。

    開催が公開されていない時期（週明けなど）は空リストになる。呼び出し側は
    それを「JRA公式からは取れない」として扱い、別経路へ落とすこと。
    """
    html = open_seed(fetcher, name)
    if html is None:
        return []
    found = links(html)
    if not found:
        log.info("JRA公式 %s に開催リンクなし（出馬表は木曜公開）", name)
    return found
