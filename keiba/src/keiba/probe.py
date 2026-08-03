"""実HTMLフィクスチャ収集。

Claude Code のセッションからは netkeiba / jra.go.jp に到達できない（egress ポリシー）
ため、パーサを書く前に実物のHTMLを GitHub Actions 経由で持ち帰る必要がある。
このモジュールはそのためだけに存在し、定常運用では使わない。

    python -m keiba.probe --date 20260802 --out keiba/tests/fixtures

各ページの取得可否・サイズ・文字コードを manifest.json に残すので、
どのURLが生きているかは manifest だけ見れば分かる。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path

from keiba.sources.http import Fetcher, FetchError

log = logging.getLogger(__name__)

# db.netkeiba.com のレース一覧に並ぶ /race/<18桁> へのリンク
RACE_ID_RE = re.compile(r"/race/(\d{12})")
HORSE_ID_RE = re.compile(r"/horse/(\d{10})")
# JRA公式の遷移は doAction('/JRADB/accessX.html','pw01sde...') 形式。
# トークンの実際の形が分からないと POST を組めないので、生のまま採取する。
JRA_DOACTION_RE = re.compile(r"doAction\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)")


def last_sunday(today: date) -> date:
    """直近の日曜（今日が日曜なら1週間前）。結果が確定済みの開催日を選ぶため。"""
    offset = (today.weekday() + 1) % 7 or 7
    return today - timedelta(days=offset)


def _safe_name(url: str) -> str:
    name = re.sub(r"^https?://", "", url)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:120]


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
            log.warning("MISS %s (%s): %s", label, url, exc)
            self.manifest.append(
                {"label": label, "url": url, "ok": False, "error": str(exc)}
            )
            return None

        path = self.out_dir / f"{label}__{_safe_name(url)}.html"
        path.write_text(html, encoding="utf-8")
        entry = {
            "label": label,
            "url": url,
            "ok": True,
            "chars": len(html),
            "file": path.name,
        }
        if "jra.go.jp" in url:
            # トークンの形と、どのリンクがどのページに対応するかを記録
            entry["doAction"] = [
                {"action": a, "cname": c}
                for a, c in JRA_DOACTION_RE.findall(html)[:40]
            ]
        self.manifest.append(entry)
        log.info("OK   %s (%s chars) %s", label, len(html), url)
        return html

    def run(self, kaisai_date: str) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # 0. 規約の確認材料。何が許可されているかは自分の目で見る。
        self.grab("https://www.netkeiba.com/robots.txt", "robots_netkeiba")
        self.grab("https://www.jra.go.jp/robots.txt", "robots_jra")

        # 1. netkeiba: 開催日のレース一覧 → race_id を採取
        race_ids: list[str] = []
        listing = self.grab(
            f"https://db.netkeiba.com/race/list/{kaisai_date}/", "db_race_list"
        )
        if listing:
            race_ids = sorted(set(RACE_ID_RE.findall(listing)))
        self.grab(
            f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_date}",
            "race_list_top",
        )

        if not race_ids:
            log.error("race_id を1つも抽出できなかった。日付かURL構造を疑うこと。")
        else:
            log.info("抽出した race_id: %s件 %s", len(race_ids), race_ids[:5])

        # 2. 代表レース2件（メインと下級条件）で全ページ種を採取。
        #    1件だけだと「重賞にしか無い要素」を見落とす。
        targets = [race_ids[-1], race_ids[0]] if len(race_ids) > 1 else race_ids
        horse_ids: list[str] = []

        for idx, race_id in enumerate(targets):
            tag = "main" if idx == 0 else "sub"
            self.grab(f"https://db.netkeiba.com/race/{race_id}/", f"db_race_{tag}")
            self.grab(
                f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}",
                f"shutuba_{tag}",
            )
            past = self.grab(
                f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}",
                f"shutuba_past_{tag}",
            )
            self.grab(
                f"https://race.netkeiba.com/race/result.html?race_id={race_id}",
                f"result_{tag}",
            )
            self.grab(
                f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}",
                f"oikiri_{tag}",
            )
            # 調教師コメント。専用ページが生きているかここで確認する。
            self.grab(
                f"https://race.netkeiba.com/race/danwa.html?race_id={race_id}",
                f"danwa_{tag}",
            )
            if past:
                horse_ids += HORSE_ID_RE.findall(past)

        # 3. 馬個別ページ（血統・全成績）
        for horse_id in sorted(set(horse_ids))[:2]:
            self.grab(f"https://db.netkeiba.com/horse/{horse_id}/", f"horse_{horse_id}")
            self.grab(
                f"https://db.netkeiba.com/horse/ped/{horse_id}/", f"ped_{horse_id}"
            )

        # 4. JRA公式。POSTトークン方式なので、まず入口の doAction を採取する。
        self.grab("https://www.jra.go.jp/keiba/", "jra_keiba_top")
        self.grab("https://www.jra.go.jp/keiba/thisweek/", "jra_thisweek")
        self.grab("https://www.jra.go.jp/datafile/seiseki/", "jra_seiseki_index")

        (self.out_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "kaisai_date": kaisai_date,
                    "race_ids": race_ids,
                    "pages": self.manifest,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        ok = sum(1 for e in self.manifest if e["ok"])
        log.info("完了: %s/%s ページ取得", ok, len(self.manifest))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=last_sunday(date.today()).strftime("%Y%m%d"),
        help="開催日 YYYYMMDD（既定: 直近の日曜）",
    )
    parser.add_argument("--out", type=Path, default=Path("keiba/tests/fixtures"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # フィクスチャは常に新鮮なものを取る
    Probe(args.out, Fetcher(use_cache=False)).run(args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
