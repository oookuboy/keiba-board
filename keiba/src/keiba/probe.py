"""実HTMLフィクスチャ収集。

Claude Code のセッションからは netkeiba / jra.go.jp に到達できない（egress ポリシー）
ため、パーサを書く前に実物のHTMLを GitHub Actions 経由で持ち帰る必要がある。
このモジュールはそのためだけに存在し、定常運用では使わない。

    python -m keiba.probe --date 20260802 --out keiba/tests/fixtures

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
HORSE_LEVEL_PAGES = [
    ("kyusya_comment", "https://db.netkeiba.com/horse/kyusya_comment.html?id={horse_id}"),
    ("horse_training", "https://db.netkeiba.com/?pid=horse_training&id={horse_id}&rid={race_id}"),
]


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

    def walk_jra(self) -> None:
        """JRA公式の POST 遷移を2階層辿って、実データページの形を採取する。

        入口の cname は固定だが、その先（開催日・レース単位）の cname は
        毎回変わるので、辿って採取するしかない。
        """
        for label, action, cname in JRA_SEEDS:
            html = self.grab(
                f"{JRA_BASE}{action}",
                f"jra_{label}_L1",
                method="POST",
                data={"cname": cname},
            )
            if not html:
                continue

            # 1階層目で拾った遷移先のうち、メニュー系ではないものを数件辿る
            nexts = [
                (a, c)
                for a, c in JRA_DOACTION_RE.findall(html)
                if c not in {s[2] for s in JRA_SEEDS}
            ]
            seen: set[str] = set()
            depth2 = 0
            for action2, cname2 in nexts:
                if cname2 in seen:
                    continue
                seen.add(cname2)
                self.grab(
                    f"{JRA_BASE}{action2}",
                    f"jra_{label}_L2_{depth2}",
                    method="POST",
                    data={"cname": cname2},
                )
                depth2 += 1
                if depth2 >= 3:
                    break

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

        self.probe_future(start)
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
    parser.add_argument("--out", type=Path, default=Path("keiba/tests/fixtures"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    Probe(args.out, Fetcher(use_cache=False)).run(args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
