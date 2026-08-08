"""共通HTTPフェッチャ。

外部サイトへのアクセスは必ずこのモジュールを経由させる。レート制限とキャッシュを
呼び出し側の善意ではなく実装で強制するのが目的。

- ホスト単位のレート制限（既定1.5秒）
- ディスクキャッシュ。同じURLは二度取りに行かない
- 文字コード自動判定（db.netkeiba.com は EUC-JP、race.netkeiba.com と jra.go.jp は UTF-8）
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".cache/keiba")

# 明示しておく。匿名の大量アクセスにしない。
USER_AGENT = (
    "keiba-board/0.1 (personal race-prediction research; "
    "+https://github.com/oookuboy/keiba-board)"
)

# meta タグが無い/壊れている場合のホスト別フォールバック
HOST_ENCODING = {
    "db.netkeiba.com": "euc-jp",
    "race.netkeiba.com": "utf-8",
    "www.netkeiba.com": "euc-jp",
    "www.jra.go.jp": "utf-8",
}

# ホスト別の最小アクセス間隔（秒）
HOST_MIN_INTERVAL = {
    "db.netkeiba.com": 1.5,
    "race.netkeiba.com": 1.5,
    "www.netkeiba.com": 1.5,
    "www.jra.go.jp": 1.0,
}
DEFAULT_MIN_INTERVAL = 1.5

_META_CHARSET = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""", re.IGNORECASE
)


class FetchError(RuntimeError):
    """リトライを尽くしても取得できなかった。"""


@dataclass
class Fetcher:
    cache_dir: Path = DEFAULT_CACHE_DIR
    use_cache: bool = True
    max_retries: int = 4
    timeout: float = 30.0
    session: requests.Session = field(default_factory=requests.Session)
    _last_request: dict[str, float] = field(default_factory=dict, init=False)
    stats: dict[str, int] = field(
        default_factory=lambda: {"hit": 0, "miss": 0, "error": 0}, init=False
    )

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Language": "ja,en;q=0.8",
            }
        )

    # ------------------------------------------------------------------ cache

    def _cache_path(self, url: str, method: str, body: str) -> Path:
        key = hashlib.sha256(f"{method} {url} {body}".encode()).hexdigest()
        host = urlparse(url).netloc or "unknown"
        # 2階層に切っておかないと1ディレクトリに数万ファイルが並ぶ
        return self.cache_dir / host / key[:2] / f"{key}.html"

    # ------------------------------------------------------------- rate limit

    def _throttle(self, host: str) -> None:
        interval = HOST_MIN_INTERVAL.get(host, DEFAULT_MIN_INTERVAL)
        last = self._last_request.get(host)
        if last is not None:
            wait = interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        self._last_request[host] = time.monotonic()

    # ---------------------------------------------------------------- decode

    @staticmethod
    def decode(raw: bytes, url: str) -> str:
        """バイト列を文字列へ。meta charset を最優先し、ホスト別既定に落とす。"""
        match = _META_CHARSET.search(raw[:4096])
        candidates: list[str] = []
        if match:
            candidates.append(match.group(1).decode("ascii", "ignore"))
        host = urlparse(url).netloc
        if host in HOST_ENCODING:
            candidates.append(HOST_ENCODING[host])
        candidates += ["utf-8", "euc-jp", "cp932"]

        for enc in candidates:
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        # ここまで来たら壊れたページ。捨てずに読めるところだけ拾う。
        return raw.decode("utf-8", errors="replace")

    # ----------------------------------------------------------------- fetch

    def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        data: dict[str, str] | None = None,
        force: bool = False,
    ) -> str:
        """URL を取得して文字列で返す。キャッシュがあればネットワークに出ない。"""
        body = "&".join(f"{k}={v}" for k, v in sorted((data or {}).items()))
        path = self._cache_path(url, method, body)

        if self.use_cache and not force and path.exists():
            self.stats["hit"] += 1
            return self.decode(path.read_bytes(), url)

        raw = self._request(url, method, data)
        self.stats["miss"] += 1

        if self.use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        return self.decode(raw, url)

    def _request(self, url: str, method: str, data: dict[str, str] | None) -> bytes:
        host = urlparse(url).netloc
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            self._throttle(host)
            try:
                resp = self.session.request(
                    method, url, data=data, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                if resp.status_code == 200:
                    return resp.content
                # 404 はリトライしても無駄。存在しないレースIDを掃くとき普通に起きる。
                if resp.status_code == 404:
                    raise FetchError(f"404 Not Found: {url}")
                last_error = FetchError(f"HTTP {resp.status_code}: {url}")

            backoff = 2**attempt
            log.warning(
                "fetch failed (%s/%s) %s: %s — retrying in %ss",
                attempt + 1,
                self.max_retries,
                url,
                last_error,
                backoff,
            )
            time.sleep(backoff)

        self.stats["error"] += 1
        raise FetchError(f"giving up after {self.max_retries} attempts: {url}") from last_error
