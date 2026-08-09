"""実HTMLフィクスチャ収集。

Claude Code のセッションからは netkeiba / jra.go.jp に到達できない（egress ポリシー）
ため、パーサを書く前に実物のHTMLを GitHub Actions 経由で持ち帰る必要がある。
このモジュールはそのためだけに存在し、定常運用では使わない。

    python -m keiba.probe --date 20260802 --out keiba/tests/fixtures/probe

第1回の探索で分かったこと（このモジュールの設計はこれに基づく）:

- race.netkeiba.com（出馬表・馬柱・結果・調教・談話）は JavaScript レンダリングの
  空シェルで、requests では表のヘッダしか取れない。**収集源として使えない。**
- db.netkeiba.com はサーバーレンダリングで実データが入っている。過去成績・馬・血統はここ。
- www.jra.go.jp もサーバーレンダリング。robots.txt は `Disallow:` が空＝全面許可。
  ただし遷移が doAction('/JRADB/accessX.html','pw01...') の POST 方式なので、
  入口の cname から順に辿らないと目的のページに届かない。
- 中央のレースを拾うには race_id の 5-6 桁目（場コード）で 01〜10 に絞る必要がある。
  絞らないと地方（盛岡35・船橋43・高知54・帯広65）ばかり拾ってしまう。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path

from keiba.models import VENUES
from keiba.sources.http import Fetcher, FetchError

log = logging.getLogger(__name__)

RACE_ID_RE = re.compile(r"/race/(\d{12})")
HORSE_ID_RE = re.compile(r"/horse/(\d{10})")
JRA_DOACTION_RE = re.compile(r"doAction\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)")

JRA_BASE = "https://www.jra.go.jp"

# JRA公式の入口。cname はトップページの doAction から採取した実物。
JRA_SEEDS = [
    ("racecard", "/JRADB/accessD.html", "pw01dli00/F3"),   # 出馬表
    ("results", "/JRADB/accessS.html", "pw01sli00/AF"),    # レース結果
    ("training", "/JRADB/accessT.html", "pw03trl00/29"),   # 調教
    ("info", "/JRADB/accessI.html", "pw01ide01/4F"),       # 開催情報
]

# 競馬新聞ページは race.netkeiba.com で唯一サーバーレンダリングされる。
# 調教タイムが入っているが、無料枠では先頭3頭までしか見えない。
RACE_LEVEL_PAGES = [
    ("news_comment", "https://race.netkeiba.com/race/newspaper.html?race_id={race_id}"),
]

# db_race の結果テーブルが指していた実URL。厩舎コメントと調教はここにある。
# 実物のリンクから採ったもので、推測ではない（結果表の各馬から18本ずつ出ている）。
#
# horse_training を rid 付きと rid 無しの両方で取るのは、**バックフィルの規模が
# ここで決まる**ため。rid が単なるアンカーで1回のリクエストに馬の全調教が載るなら
# 約2.4万頭ぶんで済み、血統バックフィルと同じ規模（約10時間）になる。rid が
# サーバ側で絞り込んでいるなら1出走ごとに1リクエストで約14.9万回（約62時間）。
# 10倍以上違うので、実測してから走らせる。
HORSE_LEVEL_PAGES = [
    ("kyusya_comment", "https://db.netkeiba.com/horse/kyusya_comment.html?id={horse_id}"),
    ("horse_training", "https://db.netkeiba.com/?pid=horse_training&id={horse_id}&rid={race_id}"),
    ("horse_training_all", "https://db.netkeiba.com/?pid=horse_training&id={horse_id}"),
]


# netkeiba がデータ表に付けるクラス。レイアウト用の table と区別するために使う。
DATA_TABLE_CLASSES = ("nk_tb_common", "race_table", "db_table", "tb_common")


def _extract_tables(html: str) -> str:
    """データの表だけを切り出す。

    会員ページをそのままコミットするとヘッダのアカウント名やメールアドレスが
    漏れる。調教タイム・コメントの表そのものに個人情報は無いので、表だけ抜けば
    公開リポジトリに置ける。こうしないとパーサをテストする材料が手元に作れず、
    直すたびに Actions を1往復することになる。

    安全側に倒すため三重にする。

      1. netkeiba がデータ表に付けるクラスを持つ table だけを採る
         （レイアウト用の table にヘッダのアカウント名が入りうるため）
      2. 切り出したものに @ が1つでも残っていたら、伏せずに丸ごと捨てる

    伏せ字に置き換えるのではなく捨てるのは、置換が効いたかどうかを後から
    確かめられないため。公開リポジトリに置くものなので、迷ったら出さない。
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    tables = []
    for table in soup.select("table"):
        classes = " ".join(table.get("class") or [])
        if not any(c in classes for c in DATA_TABLE_CLASSES):
            continue
        if not any(td.get_text(strip=True) for td in table.select("td")):
            continue
        tables.append(str(table))
    if not tables:
        return ""

    body = "\n".join(tables[:4])
    if "@" in body:
        # 想定外のものが混ざっている。競馬のデータ表に @ は出ないはずなので、
        # 出たなら切り出し方が間違っている。直すまで何も置かない。
        log.warning("表の中に @ が混ざっていたので切り出しを見送った")
        return ""
    return (
        "<!-- netkeiba の会員ページからデータ表だけを切り出したもの。\n"
        "     ヘッダ等の個人情報は含まない。パーサのテスト用。 -->\n"
        "<html><body>\n" + body + "\n</body></html>\n"
    )


def _table_structure(html: str) -> list[dict]:
    """表の骨格だけを抜く。

    会員ページの本文をそのままログに出すとアカウント名が漏れる（Actions の
    ログは公開リポジトリでは誰でも読める）。パーサを書くのに要るのは
    クラス名・列見出し・行数なので、そこだけを取る。セルの中身は出さない。
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []
    for table in soup.select("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        headers = [
            re.sub(r"\s+", " ", c.get_text(" ", strip=True))[:20]
            for c in rows[0].find_all(["th", "td"])
        ]
        cells = rows[1].find_all(["th", "td"]) if len(rows) > 1 else []
        out.append(
            {
                "class": table.get("class"),
                "rows": len(rows),
                "headers": headers[:16],
                "cell_classes": [c.get("class") for c in cells][:16],
            }
        )
    return out[:8]


def last_sunday(today: date) -> date:
    offset = (today.weekday() + 1) % 7 or 7
    return today - timedelta(days=offset)


def _safe_name(url: str) -> str:
    name = re.sub(r"^https?://", "", url)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:110]


class Probe:
    def __init__(self, out_dir: Path, fetcher: Fetcher) -> None:
        self.out_dir = out_dir
        self.fetcher = fetcher
        self.manifest: list[dict] = []

    def grab(self, url: str, label: str, **kwargs) -> str | None:
        """1ページ取得して保存。失敗しても manifest に理由を残して続行する。"""
        try:
            html = self.fetcher.fetch(url, **kwargs)
        except FetchError as exc:
            log.warning("MISS %s: %s", label, exc)
            self.manifest.append(
                {"label": label, "url": url, "ok": False, "error": str(exc)}
            )
            return None

        (self.out_dir / f"{label}__{_safe_name(url)}.html").write_text(
            html, encoding="utf-8"
        )
        entry: dict = {
            "label": label,
            "url": url,
            "ok": True,
            "chars": len(html),
            "title": self._title(html),
        }
        if "jra.go.jp" in url:
            entry["doAction"] = [
                {"action": a, "cname": c} for a, c in JRA_DOACTION_RE.findall(html)
            ][:60]
        self.manifest.append(entry)
        log.info("OK   %-28s %6d chars  %s", label, len(html), entry["title"] or "")
        return html

    @staticmethod
    def _title(html: str) -> str | None:
        m = re.search(r"<title>(.*?)</title>", html, re.S)
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else None

    # -------------------------------------------------------------- netkeiba

    def find_jra_race_ids(self, start: date, lookback: int = 21) -> tuple[str, list[str]]:
        """中央のレースがある開催日まで日付を遡る。

        地方は毎日開催しているので、日付をそのまま信じると盛岡や船橋を拾ってしまう。
        場コード 01〜10 に該当する race_id が出るまで戻る。
        """
        for offset in range(lookback + 1):
            day = start - timedelta(days=offset)
            stamp = day.strftime("%Y%m%d")
            html = self.grab(
                f"https://db.netkeiba.com/race/list/{stamp}/", f"db_race_list_{stamp}"
            )
            if not html:
                continue
            ids = sorted(set(RACE_ID_RE.findall(html)))
            jra = [i for i in ids if i[4:6] in VENUES]
            log.info(
                "%s: 全%d件 / 中央%d件 %s",
                stamp,
                len(ids),
                len(jra),
                sorted({VENUES[i[4:6]] for i in jra}),
            )
            if jra:
                return stamp, jra
        return start.strftime("%Y%m%d"), []

    # ------------------------------------------------------------------ JRA

    def walk_jra_racecard(self) -> None:
        """出馬表だけを狙って「開催選択 → レース選択 → 出馬表」を辿る。

        総当たりの BFS だと共通ナビ（オッズ・払戻金・競走馬検索…）を先に拾って
        しまい、本文の開催リンクに到達しない。実際そうなった。

        開催リンクの cname は pw01drl + 場コード + 年 + 回 + 日 + 日付 + 末尾2桁
        という形をしていて、末尾はこちらで計算できない。よって構築せず、
        ページから拾って辿る。
        """
        html = self.grab(
            f"{JRA_BASE}/JRADB/accessD.html",
            "card_L1_kaisai",
            method="POST",
            data={"cname": "pw01dli00/F3"},
        )
        if not html:
            log.error("出馬表の開催選択ページを取得できない")
            return

        kaisai = [
            (a, c) for a, c in JRA_DOACTION_RE.findall(html)
            if c.startswith("pw01drl")
        ]
        log.info("開催リンク %d件: %s", len(kaisai), [c for _, c in kaisai])
        if not kaisai:
            log.error("開催リンク（pw01drl）が見つからない。まだ公開されていない可能性")
            return

        # 1開催ぶん辿ればレース選択と出馬表の形は分かる
        action2, cname2 = kaisai[0]
        race_list = self.grab(
            f"{JRA_BASE}{action2}", "card_L2_racelist",
            method="POST", data={"cname": cname2},
        )
        if not race_list:
            return

        # レース選択ページ側のリンク。共通ナビを除くため、L1 に出ていた
        # cname は落とす（ナビは全ページ共通なので L1 に必ず出ている）
        nav = {c for _, c in JRA_DOACTION_RE.findall(html)}
        races = [
            (a, c) for a, c in JRA_DOACTION_RE.findall(race_list)
            if c not in nav and not c.startswith("pw01drl")
        ]
        log.info("レースリンク %d件: %s", len(races), [c for _, c in races][:14])
        for idx, (action3, cname3) in enumerate(races[:2]):
            self.grab(
                f"{JRA_BASE}{action3}", f"card_L3_race{idx}",
                method="POST", data={"cname": cname3},
            )

    def probe_member_pages(self, race_id: str, horse_ids: list[str]) -> None:
        """会員ログイン後に、有料ページが実際に開くかを確かめる。

        未ログインでは本文が「スーパープレミアムコースからご利用頂けます」に
        差し替わることを実測済み。ログイン後に同じURLで中身が出るなら、
        調教タイムと厩舎コメントが特徴量に使える。

        取れなければ取れないと記録する。ここを曖昧にすると「有料にしたのに
        効いていない」状態に気づけない。
        """
        from keiba.sources import netkeiba_auth

        if netkeiba_auth.credentials() is None:
            log.info("netkeiba の認証情報なし。会員ページの採取は飛ばす")
            return

        try:
            netkeiba_auth.login(self.fetcher)
        except netkeiba_auth.LoginError as exc:
            log.error("ログインできない: %s", exc)
            # 診断はログではなく manifest に残す。Actions のログを丸ごと取りに
            # いくのは重く、必要な数行を探すのに毎回大きな往復が要るため。
            self.manifest.append(
                {
                    "label": "member_login",
                    "url": netkeiba_auth.LOGIN_URL,
                    "ok": False,
                    "error": str(exc),
                    "form": netkeiba_auth.inspect_login_form(self.fetcher),
                }
            )
            return

        self.manifest.append(
            {"label": "member_login", "url": netkeiba_auth.LOGIN_URL, "ok": True}
        )

        # ここが今回の目的。課金した情報が実際に開くかを、壁の有無で判定する
        if horse_ids:
            verdict = netkeiba_auth.check_paid_access(
                self.fetcher, horse_ids[0], race_id
            )
            for name, info in verdict.items():
                log.info(
                    "有料ページ %s: %s（%s）",
                    name,
                    "**開いた**" if not info.get("paywalled", True) else "まだ壁",
                    info.get("title", info.get("error", "")),
                )
            self.manifest.append(
                {"label": "paid_access", "url": "", "ok": True, "verdict": verdict}
            )

        # 会員ページのヘッダにはアカウント名やメールアドレスが載る。
        # このリポジトリは公開なので、HTML をそのままコミットすると漏れる。
        # Actions のログも公開なので、本文を出すのも同じく危険。
        # よって「保存しない・構造だけ出す」で扱う。
        member_dir = self.out_dir / "member"
        member_dir.mkdir(parents=True, exist_ok=True)

        for idx, horse_id in enumerate(horse_ids[:2]):
            for name, tmpl in HORSE_LEVEL_PAGES:
                url = tmpl.format(horse_id=horse_id, race_id=race_id)
                try:
                    html = self.fetcher.fetch(url)
                except FetchError as exc:
                    log.warning("MISS member_%s: %s", name, exc)
                    continue

                walled = netkeiba_auth.is_paywalled(html)
                log.info(
                    "  %s: %s", name,
                    "まだ有料の壁" if walled else "**中身が出た**",
                )
                self.manifest.append(
                    {
                        "label": f"member_{name}_{idx}",
                        "url": url,
                        "ok": True,
                        "paywalled": walled,
                        "structure": _table_structure(html),
                    }
                )
                # 手元でパーサを書くためにファイルには残すが、gitignore 済みの
                # member/ に置いてコミットされないようにする
                (member_dir / f"{name}_{idx}.html").write_text(html, encoding="utf-8")

                # 表だけを切り出したものは pinned/ に置いてコミットする。
                # 個人情報はページのヘッダ側にあり、表の中身は調教タイムと
                # コメントだけなので、これならリポジトリに入れて安全。
                # これが無いとパーサをテストできない（会員ページは手元で
                # 再現できず、毎回 Actions を回すことになる）。
                if idx == 0 and not walled:
                    fragment = _extract_tables(html)
                    if fragment:
                        # pinned/ は out_dir の中ではなく隣。probe は毎回 out_dir の
                        # *.html を消すので、テストが読む固定物を中に置いてはいけない。
                        pinned = self.out_dir.parent / "pinned"
                        pinned.mkdir(parents=True, exist_ok=True)
                        (pinned / f"netkeiba_{name}.html").write_text(
                            fragment, encoding="utf-8"
                        )
                        log.info("  %s の表を pinned/ に切り出した", name)

    def probe_today_results(self, day: date) -> None:
        """当日の着順が db.netkeiba に出ているかを確かめる。

        「反映が遅いのだろう」で済ませず、事実として記録する。出ているなら
        収集側の不具合、出ていないなら JRA公式へ切り替える判断材料になる。
        """
        stamp = day.strftime("%Y%m%d")
        html = self.grab(
            f"https://db.netkeiba.com/race/list/{stamp}/", f"today_list_{stamp}"
        )
        if not html:
            return
        jra = [i for i in set(RACE_ID_RE.findall(html)) if i[4:6] in VENUES]
        log.info("db.netkeiba の %s: 中央 race_id %d件", stamp, len(jra))
        if jra:
            self.grab(
                f"https://db.netkeiba.com/race/{sorted(jra)[0]}/",
                f"today_race_{stamp}",
            )

    def walk_jra_results(self) -> None:
        """レース結果を「開催選択 → レース選択 → 全レース表示」で辿る。

        db.netkeiba は結果データベースだが反映が遅く、開催当日の夜になっても
        着順が1件も出ていなかった（2026-08-08 実測）。JRA公式はレース確定後
        すぐ出るので、こちらから取れれば回顧を当日中に回せる。

        出馬表（accessD）と同じ構造だと当たりを付けているが、確かめる。
        """
        html = self.grab(
            f"{JRA_BASE}/JRADB/accessS.html",
            "result_L1_kaisai",
            method="POST",
            data={"cname": "pw01sli00/AF"},
        )
        if not html:
            log.error("レース結果の開催選択ページを取得できない")
            return

        # 結果の開催リンクは pw01srl。出馬表(pw01drl)とは接頭辞が違う。
        # 直近の開催は pw01srl0…、過去は pw01srl1… と1桁目が変わるので \d で受ける。
        # 総当たりで辿ると共通ナビ（レコードタイム表・WIN5…）に逃げるので、
        # 接頭辞で狙い撃ちする。実際それで2回空振りした。
        kaisai = [
            (a, c) for a, c in JRA_DOACTION_RE.findall(html)
            if re.match(r"pw01srl\d", c)
        ]
        if not kaisai:
            log.error("結果の開催リンク（pw01srl）が見つからない")
            return

        # cname の末尾から日付を取り、最新の開催だけを辿る
        def stamp_of(cname: str) -> str:
            m = re.search(r"(\d{8})/", cname)
            return m.group(1) if m else ""

        latest = max(stamp_of(c) for _, c in kaisai)
        newest = [(a, c) for a, c in kaisai if stamp_of(c) == latest]
        log.info("結果の最新開催 %s: %d件 %s", latest, len(newest), [c for _, c in newest])

        nav = {c for _, c in JRA_DOACTION_RE.findall(html)}
        for idx, (action2, cname2) in enumerate(newest[:1]):
            page = self.grab(
                f"{JRA_BASE}{action2}", f"result_L2_racelist",
                method="POST", data={"cname": cname2},
            )
            if not page:
                continue
            # 出馬表と同じなら「全てのレースを表示」（pw01ses…?）があるはず。
            # 接頭辞が読めないので、ナビを除いた残りを数件辿って形を確かめる。
            deeper = [
                (a, c) for a, c in JRA_DOACTION_RE.findall(page)
                if c not in nav and not re.match(r"pw01srl\d", c)
            ]
            log.info("レース選択ページの遷移先 %d件: %s",
                     len(deeper), [c for _, c in deeper][:10])
            for jdx, (action3, cname3) in enumerate(deeper[:3]):
                self.grab(
                    f"{JRA_BASE}{action3}", f"result_L3_{jdx}",
                    method="POST", data={"cname": cname3},
                )

    def walk_jra(self, max_depth: int = 3, branches: int = 3) -> None:
        """JRA公式の POST 遷移を辿って、実データページの形を採取する。

        入口の cname は固定だが、その先（開催日・レース単位）の cname は
        毎回変わるので、辿って採取するしかない。

        出馬表は「入口 → 開催日選択 → レース選択 → 出馬表」で3階層ある。
        2階層で止めるとレース選択画面までしか届かず、肝心の馬柱に到達しない。
        """
        seed_cnames = {s[2] for s in JRA_SEEDS}

        for label, action, cname in JRA_SEEDS:
            html = self.grab(
                f"{JRA_BASE}{action}",
                f"jra_{label}_L1",
                method="POST",
                data={"cname": cname},
            )
            if not html:
                continue

            seen: set[str] = set(seed_cnames)
            frontier = [(action, cname, html)]

            for depth in range(2, max_depth + 1):
                next_frontier: list[tuple[str, str, str]] = []
                for _, _, parent_html in frontier:
                    picked = 0
                    for action2, cname2 in JRA_DOACTION_RE.findall(parent_html):
                        if cname2 in seen:
                            continue
                        seen.add(cname2)
                        child = self.grab(
                            f"{JRA_BASE}{action2}",
                            f"jra_{label}_L{depth}_{len(next_frontier)}",
                            method="POST",
                            data={"cname": cname2},
                        )
                        picked += 1
                        if child:
                            next_frontier.append((action2, cname2, child))
                        if picked >= branches:
                            break
                if not next_frontier:
                    break
                frontier = next_frontier

    # ----------------------------------------------------------------- main

    def run(self, kaisai_date: str) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # 到達不能と分かったページの残骸が居座らないよう、毎回作り直す
        for stale in self.out_dir.glob("*.html"):
            stale.unlink()

        # 何が許可されているかは自分の目で確認する
        self.grab("https://db.netkeiba.com/robots.txt", "robots_db_netkeiba")
        self.grab("https://race.netkeiba.com/robots.txt", "robots_race_netkeiba")
        self.grab(f"{JRA_BASE}/robots.txt", "robots_jra")

        start = date(int(kaisai_date[:4]), int(kaisai_date[4:6]), int(kaisai_date[6:8]))
        stamp, race_ids = self.find_jra_race_ids(start)
        if not race_ids:
            log.error("中央のレースを1件も見つけられなかった。lookback を延ばすこと。")
        else:
            log.info("中央 %d件 @ %s", len(race_ids), stamp)

        # メイン（最終R寄り）と下級条件（1R）の2本立て。
        # 1件だけだと重賞にしか無い要素を見落とす。
        targets = [race_ids[-1], race_ids[0]] if len(race_ids) > 1 else race_ids
        horse_ids: list[str] = []

        for idx, race_id in enumerate(targets):
            tag = "main" if idx == 0 else "sub"
            html = self.grab(f"https://db.netkeiba.com/race/{race_id}/", f"db_race_{tag}")
            ids = HORSE_ID_RE.findall(html) if html else []
            horse_ids += ids
            for name, tmpl in RACE_LEVEL_PAGES:
                self.grab(tmpl.format(race_id=race_id), f"{name}_{tag}")
            for horse_id in sorted(set(ids))[:1]:
                for name, tmpl in HORSE_LEVEL_PAGES:
                    self.grab(
                        tmpl.format(horse_id=horse_id, race_id=race_id),
                        f"{name}_{tag}",
                    )

        for horse_id in sorted(set(horse_ids))[:2]:
            self.grab(f"https://db.netkeiba.com/horse/{horse_id}/", f"horse_{horse_id}")
            self.grab(f"https://db.netkeiba.com/horse/ped/{horse_id}/", f"ped_{horse_id}")

        # 会員ページは最後に見る。ログインするとキャッシュ名前空間が変わるので、
        # 未ログインで採るぶんを先に済ませておく
        if targets:
            self.probe_member_pages(targets[0], sorted(set(horse_ids)))

        self.probe_today_results(date.today())
        self.probe_future(start)
        self.walk_jra_racecard()
        self.walk_jra_results()
        self.walk_jra()
        self.write_manifest(kaisai_date, stamp, race_ids)

    def write_manifest(self, requested: str, stamp: str, race_ids: list[str]) -> None:
        """採取結果の目録。ワークフローの Summarise がこれを読む。

        run() の最後で必ず書く。以前これが probe_future の末尾に紛れており、
        未来レースが見つかると早期 return で書かれず、見つからないと
        スコープ外の変数を参照して NameError になっていた。
        """
        (self.out_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "requested_date": requested,
                    "kaisai_date": stamp,
                    "jra_race_ids": race_ids,
                    "pages": self.manifest,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        ok = sum(1 for e in self.manifest if e["ok"])
        log.info("完了: %s/%s ページ取得", ok, len(self.manifest))

    def probe_future(self, today: date) -> None:
        """これから行われるレースが db.netkeiba に出るかを確かめる。

        予想には未来の出馬表が要るが、db.netkeiba は過去成績のデータベースなので
        発走前にどこまで見えるかが分かっていない。JRA公式の出馬表は木曜公開なので、
        週明けに走らせたこのプローブでは空振りしうる。空振りも記録として残す。
        """
        for offset in range(1, 8):
            day = today + timedelta(days=offset)
            if day.weekday() not in (5, 6):  # 土日のみ
                continue
            stamp = day.strftime("%Y%m%d")
            html = self.grab(
                f"https://db.netkeiba.com/race/list/{stamp}/", f"future_list_{stamp}"
            )
            if not html:
                continue
            jra = [i for i in set(RACE_ID_RE.findall(html)) if i[4:6] in VENUES]
            log.info("未来 %s: 中央 %d件", stamp, len(jra))
            if jra:
                self.grab(
                    f"https://db.netkeiba.com/race/{sorted(jra)[0]}/",
                    f"future_race_{stamp}",
                )
                return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=last_sunday(date.today()).strftime("%Y%m%d"),
        help="探索の起点 YYYYMMDD（既定: 直近の日曜。ここから中央開催日まで遡る）",
    )
    # 採取先はテストが読む pinned/ とは別にする。probe は毎回 *.html を消して
    # 採り直すので、同じ場所に置くと採取レースが変わるたびにテストが壊れる。
    parser.add_argument(
        "--out", type=Path, default=Path("keiba/tests/fixtures/probe")
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    Probe(args.out, Fetcher(use_cache=False)).run(args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
