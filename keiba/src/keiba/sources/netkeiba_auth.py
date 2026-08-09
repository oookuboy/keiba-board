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
# 投稿先はクエリ付きのURLではなく**ルート**。pid と action は本文に入れる。
# 実物のフォームを読んで確定させた（推測で ?pid=login&action=auth に投げて
# いたときは、ログイン画面がそのまま返ってきて認証されなかった）。
#   action="https://regist.netkeiba.com/" method=post
#   inputs: pid / action / rtn_url / login_id / pswd
LOGIN_URL = "https://regist.netkeiba.com/"
# ログイン後の戻り先。フォームが hidden で持っているので同じ値を送る
LOGIN_RETURN_URL = "https://www.netkeiba.com/"
# ログイン済みかどうかを確かめるページ。
# ?pid=my_account は空で返ったので、会員トップを見る
VERIFY_URL = "https://regist.netkeiba.com/account/"

EMAIL_ENV = "NETKEIBA_EMAIL"
PASSWORD_ENV = "NETKEIBA_PASSWORD"

# 「そのページが壁そのもの」であることを示す文言だけを置く。
# 「プレミアム」単体のような広い語を入れてはいけない。会員ページにも課金導線が
# 残っているため、実際に調教タイムの表が出ているのに壁だと誤判定していた。
WALL_PAGE_MARKERS = (
    "プレミアムサービス案内",          # 未ログインで有料ページを開くとここへ飛ぶ
    "からご利用頂けます",              # 本文だけ伏せ字に差し替えられた形
    "からご利用いただけます",
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
    """有料の壁に当たっているか。

    文言だけで判定すると誤判定する。会員ページにも「プレミアム」という語が
    導線として残っており、実際に表が出ているのに壁だと判定していた。

    見るべきは「データの表が実際にあるか」。ただし案内ページ自体が料金比較の
    大きな表を持っている（実測: 97行）ので、表の有無だけでも判定できない。
    そこで二段にする。

      1. 案内ページそのもの／伏せ字の文言が出ていたら壁
      2. そうでなければ、データの入った表があるかどうか
    """
    if any(m in html for m in WALL_PAGE_MARKERS):
        return True
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for table in soup.select("table"):
        # 見出しだけの空表は数えない。中身のある td があるものだけ
        if any(td.get_text(strip=True) for td in table.select("td")):
            return False
    return True


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
                "pid": "login",
                "action": "auth",
                "rtn_url": LOGIN_RETURN_URL,
                "login_id": email,
                "pswd": password,
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


def check_paid_access(fetcher: Fetcher, horse_id: str, race_id: str) -> dict:
    """有料ページが実際に開くかを確かめる。

    これが今回の目的そのもの。課金しても取れないなら、モデルに足す前に
    そう分かる必要がある。「取れたつもりで空」が一番損なので、壁の有無を
    はっきり返す。本文はログに出さない（Actions のログは公開）。
    """
    from keiba.probe import HORSE_LEVEL_PAGES, _table_structure

    out: dict[str, dict] = {}
    for name, tmpl in HORSE_LEVEL_PAGES:
        url = tmpl.format(horse_id=horse_id, race_id=race_id)
        try:
            html = fetcher.fetch(url, force=True)
        except FetchError as exc:
            out[name] = {"error": str(exc)}
            continue
        out[name] = {
            "title": _title(html),
            "len": len(html),
            "paywalled": is_paywalled(html),
            "tables": _table_structure(html),
        }
    return out


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
