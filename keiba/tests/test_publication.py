"""公開範囲の検証。

GitHub Pages はリポジトリのルートから公開すると、**中の全ファイル**を配信する。
ボードが keiba/data/index.json を取れていたのがその証拠で、同じ理屈で
keiba/raw/ の中身も URL を叩けば誰でも落とせる状態だった。

リポジトリを非公開にしても、Pages が配信しているものは公開のままになる。
つまり有料データをコミットする前に、Pages の配信範囲を docs/ に閉じ込めて
おかないと、非公開にした意味が無くなる。

ここはブラウザからしか確かめられない部分なので、せめて構造のほうを固定する。
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).parents[2]
DOCS = REPO / "docs"

# Pages が配信してよいもの。ボードとその表示データだけ。
ALLOWED_SUFFIXES = {".html", ".json", ".css", ".js", ".svg", ".ico", ".txt"}


def test_the_board_lives_under_docs() -> None:
    """Pages の配信元は docs/ に閉じる。

    Settings → Pages → Source を main / docs にしておくこと。ルート配信の
    ままだと keiba/raw/ の中身まで URL で取れる。
    """
    assert (DOCS / "index.html").exists(), "ボードが docs/ に無い"
    assert not (REPO / "index.html").exists(), (
        "ルートに index.html が残っている。Pages をルート配信に戻す誘惑に"
        " なるので置かない"
    )


def test_nothing_but_the_board_is_published() -> None:
    """docs/ に配信してよくないものを置かない。

    有料の調教データをうっかりここへ置くと、リポジトリが非公開でも
    URL を知っていれば誰でも落とせる。
    """
    unexpected = [
        str(p.relative_to(REPO))
        for p in DOCS.rglob("*")
        if p.is_file() and p.suffix.lower() not in ALLOWED_SUFFIXES
    ]
    assert not unexpected, f"docs/ に想定外のファイルがある: {unexpected[:5]}"


def test_paid_data_is_not_under_docs() -> None:
    """調教データの置き場所が docs/ の外であること。"""
    from keiba.cli import WORKOUT_PATH

    assert "docs" not in WORKOUT_PATH.parts, (
        f"{WORKOUT_PATH} は Pages から配信される場所にある。"
        " 有料データをここへ置いてはいけない"
    )


def test_board_only_fetches_relative_to_itself() -> None:
    """ボードが docs/ の外を取りに行っていないこと。

    パスを直し忘れると、非公開にした瞬間にボードが空になる。
    """
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert "keiba/data" not in html, (
        "ボードがまだ keiba/data を見ている。docs/ 配信では届かない"
    )
    assert 'fetch("data/index.json"' in html
