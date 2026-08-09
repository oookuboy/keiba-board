"""netkeiba への会員ログイン。

有料プラン限定の調教タイム・厩舎コメントを取るために使う。認証情報は
**環境変数からのみ**読む。コードにもワークフローにも書かない。

    NETKEIBA_EMAIL
    NETKEIBA_PASSWORD

GitHub Actions では Secrets から env に注入する。リポジトリが公開でも
Secrets は third party から見えず、ログ上も自動でマスクされる。

## ログイン失敗を握りつぶさない

このモジュールで一番大事なのはここ。netkeiba は未ログインでも有料ページを
200 で返し、本文だけを「スーパープレミアムコースからご利用頂けます」に
差し替える。つまり**ログインに失敗しても取得は成功する**。

そのまま流すと「調教タイム0件」のデータが静かに積み上がり、モデルは
特徴量が欠けたまま学習してしまう。これは今回の開発で何度も踏んだ形
（収集は成功・中身は空）なので、ログインは必ず検証し、失敗したら例外で
落とす。黙って未ログインのまま続行しない。

## 規約について

netkeiba の規約は自動アクセスを制限しているのが通例で、アカウント停止の
リスクがある。利用者がそれを承知のうえで自分の課金アカウントを使う前提の
実装であり、共有・再配布を意図しない。アクセス間隔は sources/http.py の
1.5秒ウェイトをそのまま通す。
"""

from __future__ import annotations

import logging
import os

from keiba.sources.http import FetchError, Fetcher

log = logging.getLogger(__name__)

LOGIN_PAGE_URL = "https://regist.netkeiba.com/account/?pid=login"
LOGIN_URL = "https://regist.netkeiba.com/account/?pid=login&action=auth"
# ログイン済みかどうかを確かめるためのページ。会員でなければ本文が伏せられる
VERIFY_URL = "https://regist.netkeiba.com/account/?pid=my_account"

EMAIL_ENV = "NETKEIBA_EMAIL"
PASSWORD_ENV = "NETKEIBA_PASSWORD"

# 未ログイン／非会員のときに出る文言。どれか出たら「取れていない」と判断する
PAYWALL_MARKERS = (
    "スーパープレミアム",
    "プレミアムコース",
    "有料会員",
    "ログインしてください",
    "会員登録",
)
# ログイン済みのときだけ出る手がかり
LOGGED_IN_MARKERS = ("ログアウト", "マイページ", "会員情報")


class LoginError(RuntimeError):
    """ログインできなかった。未ログインのまま処理を続けさせないための例外。"""


def credentials() -> tuple[str, str] | None:
    """環境変数から認証情報を読む。無ければ None（＝未ログインで動かす）。"""
    email = os.environ.get(EMAIL_ENV, "").strip()
    password = os.environ.get(PASSWORD_ENV, "")
    if not email or not password:
        return None
    return email, password


def is_paywalled(html: str) -> bool:
    """有料の壁に当たっているか。ログイン検証と収集時の両方で使う。"""
    return any(m in html for m in PAYWALL_MARKERS)


def login(fetcher: Fetcher, *, required: bool = False) -> bool:
    """netkeiba にログインする。

    成功したら fetcher のキャッシュ名前空間を "auth" に切り替えて True を返す。
    認証情報が無ければ False（未ログインで続行）。ログインを試みて失敗した
    場合は LoginError。required=True なら認証情報が無い時点でも落とす。

    パスワードはログに出さない。例外メッセージにも入れない。
    """
    creds = credentials()
    if creds is None:
        # 「Secretsに入れたのに届いていない」と「値が間違っている」は別問題。
        # 名前の綴り違いは実際よくあるので、どちらが欠けたかを出す（値は出さない）。
        detail = (
            f"{EMAIL_ENV}={'あり' if os.environ.get(EMAIL_ENV) else 'なし'} "
            f"{PASSWORD_ENV}={'あり' if os.environ.get(PASSWORD_ENV) else 'なし'}"
        )
        if required:
            raise LoginError(
                f"認証情報が env に届いていない（{detail}）。"
                " Secrets の名前の綴りと、ワークフローの env: 渡しを確認すること"
            )
        log.info("netkeiba の認証情報なし（%s）。無料範囲で動かす", detail)
        return False

    email, password = creds
    log.info("netkeiba にログインする（%s）", _mask(email))

    try:
        # ログインは毎回ネットワークに出す。キャッシュに載せない
        response = fetcher.fetch(
            LOGIN_URL,
            method="POST",
            data={
                "login_id": email,
                "pswd": password,
                "pid": "login",
                "action": "auth",
            },
            force=True,
        )
    except FetchError as exc:
        raise LoginError(f"ログイン要求に失敗した: {exc}") from exc

    # POST が 200 でも失敗していることがある。必ず別ページで確かめる
    fetcher.cache_namespace = "auth"
    try:
        page = fetcher.fetch(VERIFY_URL, force=True)
    except FetchError as exc:
        fetcher.cache_namespace = ""
        raise LoginError(f"ログイン確認ページを取得できない: {exc}") from exc

    if not any(m in page for m in LOGGED_IN_MARKERS):
        fetcher.cache_namespace = ""
        # 何が起きたかを、認証情報を出さずに分かる形で残す。
        # 当てずっぽうで直すより、手がかりを持って直すほうが早い。
        log.error("ログイン診断: %s", diagnose(response, page))
        log.error("ログインフォーム: %s", inspect_login_form(fetcher))
        raise LoginError(
            "ログインしたつもりで未ログインのまま。"
            " 上の診断を見て、投稿先・フィールド名・判定文言のどれが違うかを絞ること"
            "（認証情報はログに出していない）"
        )

    log.info("netkeiba にログインした")
    return True


# ログイン後のページに出そうな語の候補。どれが実際に出るか分からないので、
# 判定に使う LOGGED_IN_MARKERS とは別に、診断用として広めに見る。
_PROBE_WORDS = (
    "ログアウト", "マイページ", "会員情報", "ログイン", "パスワード",
    "メールアドレス", "登録", "エラー", "正しく", "一致しません",
    "プレミアム", "退会", "ようこそ",
)


def inspect_login_form(fetcher: Fetcher) -> dict:
    """ログイン画面のフォーム構造を読む。

    投稿先とフィールド名を推測で書いたら認証されなかったので、実物から取る。
    フォームの action と input の name は仕様であって秘密ではないため、
    ログに出しても問題ない（value は出さない）。
    """
    from bs4 import BeautifulSoup, Tag

    try:
        html = fetcher.fetch(LOGIN_PAGE_URL, force=True)
    except FetchError as exc:
        return {"error": str(exc)}

    soup = BeautifulSoup(html, "lxml")
    forms = []
    for form in soup.select("form"):
        inputs = []
        for tag in form.select("input, select"):
            if not isinstance(tag, Tag):
                continue
            name = tag.get("name")
            if name:
                inputs.append({"name": name, "type": tag.get("type") or tag.name})
        if inputs:
            forms.append(
                {
                    "action": form.get("action"),
                    "method": (form.get("method") or "get").lower(),
                    "inputs": inputs[:12],
                }
            )
    return {"title": _title(html), "forms": forms[:4]}


def _title(html: str) -> str:
    import re

    m = re.search(r"<title>(.*?)</title>", html, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip()[:60] if m else "(なし)"


def diagnose(login_response: str, verify_page: str) -> dict:
    """ログインが通らない原因を絞るための手がかり。

    認証情報は一切含めない。ページのタイトルと、特定の語が出たかどうかの
    真偽値だけを返す。Actions のログは公開リポジトリでは誰でも読めるので、
    本文をそのまま出してはいけない。
    """
    import re

    def title(html: str) -> str:
        m = re.search(r"<title>(.*?)</title>", html, re.S)
        return re.sub(r"\s+", " ", m.group(1)).strip()[:60] if m else "(なし)"

    return {
        "login_title": title(login_response),
        "login_len": len(login_response),
        "verify_title": title(verify_page),
        "verify_len": len(verify_page),
        "login_words": [w for w in _PROBE_WORDS if w in login_response],
        "verify_words": [w for w in _PROBE_WORDS if w in verify_page],
    }


def _mask(email: str) -> str:
    """ログ用。アカウントの取り違えは分かるが、値は残さない。"""
    name, _, domain = email.partition("@")
    head = name[:2] if len(name) > 2 else name[:1]
    return f"{head}***@{domain}" if domain else f"{head}***"
