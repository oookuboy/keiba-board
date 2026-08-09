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
        if required:
            raise LoginError(
                f"{EMAIL_ENV} / {PASSWORD_ENV} が設定されていない。"
                " GitHub Actions では Secrets から env に渡すこと"
            )
        log.info("netkeiba の認証情報なし。無料範囲で動かす")
        return False

    email, password = creds
    log.info("netkeiba にログインする（%s）", _mask(email))

    try:
        # ログインは毎回ネットワークに出す。キャッシュに載せない
        fetcher.fetch(
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
        raise LoginError(
            "ログインしたつもりで未ログインのまま。"
            " メールアドレスかパスワードが違う可能性がある"
            "（値はログに出していない）"
        )

    log.info("netkeiba にログインした")
    return True


def _mask(email: str) -> str:
    """ログ用。アカウントの取り違えは分かるが、値は残さない。"""
    name, _, domain = email.partition("@")
    head = name[:2] if len(name) > 2 else name[:1]
    return f"{head}***@{domain}" if domain else f"{head}***"
