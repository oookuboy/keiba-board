"""netkeiba ログインの検証。

netkeiba は未ログインでも有料ページを 200 で返し、本文だけを
「スーパープレミアムコースからご利用頂けます」に差し替える。つまり
**ログインに失敗しても取得は成功する**。

この開発では「収集は成功・中身は空」を何度も踏んでいる（出馬表0件、
着順0件、予想0件）。同じ形を認証でも作らないよう、ログイン失敗が
例外になることをここで固定する。
"""

from __future__ import annotations

import pytest

from keiba.sources.http import Fetcher
from keiba.sources.netkeiba_auth import (
    LoginError,
    credentials,
    is_paywalled,
    login,
)


class FakeFetcher(Fetcher):
    """ネットワークに出ない Fetcher。返す HTML を差し替えられる。"""

    def __init__(self, verify_html: str) -> None:
        super().__init__(use_cache=False)
        self.verify_html = verify_html
        self.calls: list[tuple[str, str]] = []

    def fetch(self, url, *, method="GET", data=None, force=False):  # type: ignore[override]
        self.calls.append((method, url))
        if "action=auth" in url:
            return "<html>ok</html>"
        return self.verify_html


def test_no_credentials_runs_anonymously(monkeypatch) -> None:
    """認証情報が無ければ、落とさず無料範囲で動く。"""
    monkeypatch.delenv("NETKEIBA_EMAIL", raising=False)
    monkeypatch.delenv("NETKEIBA_PASSWORD", raising=False)
    assert credentials() is None
    assert login(FakeFetcher("")) is False


def test_missing_credentials_can_be_made_fatal(monkeypatch) -> None:
    """有料データが前提の処理では、認証情報が無い時点で落とせること。"""
    monkeypatch.delenv("NETKEIBA_EMAIL", raising=False)
    monkeypatch.delenv("NETKEIBA_PASSWORD", raising=False)
    with pytest.raises(LoginError):
        login(FakeFetcher(""), required=True)


def test_failed_login_raises_instead_of_continuing(monkeypatch) -> None:
    """ログインできていないのに続行しない。

    ここが通ってしまうと、伏せ字のページを延々と取り込んで
    「調教タイム0件」のデータが静かに積み上がる。
    """
    monkeypatch.setenv("NETKEIBA_EMAIL", "someone@example.com")
    monkeypatch.setenv("NETKEIBA_PASSWORD", "wrong")
    fetcher = FakeFetcher("<html>ログインしてください</html>")

    with pytest.raises(LoginError):
        login(fetcher)
    # 失敗したら名前空間を戻す。会員用キャッシュに未ログインの中身を残さない
    assert fetcher.cache_namespace == ""


def test_successful_login_switches_cache_namespace(monkeypatch) -> None:
    """成功したらキャッシュを分ける。

    分けないと、ログイン前に取った伏せ字のページがキャッシュから返り続け、
    ログインした意味が無くなる。
    """
    monkeypatch.setenv("NETKEIBA_EMAIL", "someone@example.com")
    monkeypatch.setenv("NETKEIBA_PASSWORD", "right")
    fetcher = FakeFetcher("<html>マイページ ログアウト</html>")

    assert login(fetcher) is True
    assert fetcher.cache_namespace == "auth"


def test_password_is_never_put_in_the_error(monkeypatch) -> None:
    """例外メッセージにパスワードを載せない。ログに残る。"""
    monkeypatch.setenv("NETKEIBA_EMAIL", "someone@example.com")
    monkeypatch.setenv("NETKEIBA_PASSWORD", "hunter2-secret")
    with pytest.raises(LoginError) as exc:
        login(FakeFetcher("<html>会員登録</html>"))
    assert "hunter2-secret" not in str(exc.value)


def test_paywall_is_detected() -> None:
    """有料の壁を文言で見分けられること。収集時にも使う。"""
    assert is_paywalled("※スーパープレミアムコースからご利用頂けます")
    assert not is_paywalled("<table><tr><td>栗東CW 82.4</td></tr></table>")


def test_member_fixtures_are_not_committable() -> None:
    """会員ページのHTMLが公開リポジトリに入らないこと。

    ログイン後のページはヘッダにアカウント名やメールアドレスを含む。
    このリポジトリは公開なので、コミットすると漏れる。gitignore で塞ぐ。
    """
    import pathlib
    import subprocess

    repo = pathlib.Path(__file__).parents[2]
    target = "keiba/tests/fixtures/member/kyusya_comment_0.html"
    result = subprocess.run(
        ["git", "check-ignore", target],
        cwd=repo, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"{target} が gitignore されていない。会員ページを公開リポジトリへ"
        " コミットするとアカウント情報が漏れる"
    )


def test_structure_summary_hides_cell_contents() -> None:
    """構造の要約に本文を含めないこと。

    Actions のログは公開リポジトリでは誰でも読める。表のクラス名と見出しは
    パーサを書くのに要るが、セルの中身を出す必要は無い。
    """
    from keiba.probe import _table_structure

    html = """
    <html><body>
      <div class="user">oookuboy@example.com でログイン中</div>
      <table class="race_table_01">
        <tr><th>日付</th><th>コース</th><th>タイム</th></tr>
        <tr><td class="d">2026/08/05</td><td class="c">栗東CW</td><td>82.4</td></tr>
      </table>
    </body></html>
    """
    dumped = str(_table_structure(html))
    assert "race_table_01" in dumped, "クラス名は要る"
    assert "日付" in dumped, "列見出しは要る"
    assert "oookuboy@example.com" not in dumped
    assert "栗東CW" not in dumped, "セルの中身を出さない"
    assert "82.4" not in dumped


def test_cache_namespace_changes_the_key() -> None:
    """名前空間が違えばキャッシュのパスも違うこと。"""
    anon = Fetcher(use_cache=False)
    auth = Fetcher(use_cache=False, cache_namespace="auth")
    url = "https://db.netkeiba.com/?pid=horse_training&id=1"
    assert anon._cache_path(url, "GET", "") != auth._cache_path(url, "GET", "")
