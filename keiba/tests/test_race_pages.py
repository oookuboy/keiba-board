"""レース単位の追い切り・厩舎コメントのパーサ。

## なぜこの2つを別経路で取るのか

馬別の調教ページ（db.netkeiba の ?pid=horse_training）は、**既に走った
レースに紐づく調教しか持たない**。2026-08-28 の収集で、今週の出走馬514頭を
引いて 10,981本は取れたのに、直近21日の調教を持つ馬は93頭（18.1%）だった。
最新が11日前・25日前・39日前で、どれも「その馬の前走の直前」。

学習と本番でズレる。過去のレースは当該レースの追い切りが入っているのに、
本番はこれから走るので同じ列が空になる。gain 上位10個のうち3つが調教なので
影響は小さくない。

厩舎コメントのほうは、課金前に「スーパープレミアム限定」と測ったまま
放置していた。課金後に測り直したら壁なしで全頭ぶん出た。

## HTML の形は実ページで測った

推測で書いて2回外したので、Actions から実ページの構造だけを測った
（セル数の分布・行頭が日付か・馬リンクを持つ行数。中身は出していない）。
ここのフィクスチャはその実測に合わせてある。

    追い切り  馬の行    5セル   枠 / 馬番 / 印 / 馬名 / …
              調教の行 10セル   日付 / コース / 馬場 / 乗り役 / タイム /
                                位置 / 脚色 / 評価 / ランク / 映像
    コメント             5セル   枠 / 馬番 / 馬名 / コメント / 評価

見出しは13列あるのにデータ行は5列と10列。**見出しの位置から列を引くと
どちらの行もずれて0本になる**（最初にそれをやって実ページで0本だった）。
"""

from __future__ import annotations

from datetime import date

from keiba.sources.netkeiba import parse_race_comments, parse_race_oikiri

OIKIRI = """
<table class="race_table_01 nk_tb_common OikiriTable Stable_Time">
  <tr><th>枠</th><th>馬 番</th><th>印</th><th>馬名</th><th>日付</th>
      <th>コース</th><th>馬場</th><th>乗り役</th><th>調教タイム ラップ表示</th>
      <th>位置</th><th>脚色</th><th>評価</th><th>映像</th></tr>

  <tr><td>1</td><td>1</td><td>◎</td>
      <td><a href="https://db.netkeiba.com/horse/2021104123/">ウマイチ</a></td>
      <td>栗東</td></tr>
  <tr><td>2026/08/26</td><td>栗CW</td><td>良</td><td>助手</td>
      <td>82.1 66.5 51.8 38.0 11.9</td><td>(2)</td><td>一杯</td>
      <td>動き軽快</td><td>A</td><td></td></tr>

  <tr><td>2</td><td>2</td><td>○</td>
      <td><a href="https://db.netkeiba.com/horse/2020101999/">ウマニ</a></td>
      <td>美浦</td></tr>
  <!-- 併せ馬の相手へのリンクが調教の行に混ざる。ここで馬を切り替えてはいけない -->
  <tr><td>2026/08/27</td><td>南W</td><td>稍</td><td>騎手</td>
      <td>68.0 53.2 39.1 12.4</td><td>(1)</td><td>強め</td>
      <td>反応平凡</td><td>B</td>
      <td><a href="https://db.netkeiba.com/horse/2019100001/">アイテ</a></td></tr>
</table>
"""

COMMENTS = """
<table class="Stable_Comment Comment_Table_Show_All">
  <tr><th>枠</th><th>馬 番</th><th>馬名</th><th>コメント</th><th>評価</th></tr>
  <tr><td>1</td><td>1</td>
      <td><a href="https://db.netkeiba.com/horse/2021104123/">ウマイチ</a></td>
      <td>steady。前走から変わり身があり、状態は上向き</td>
      <td><img alt="A"></td></tr>
  <tr><td>2</td><td>2</td>
      <td><a href="https://db.netkeiba.com/horse/2020101999/">ウマニ</a></td>
      <td>今回は距離が長い。仕上がりは悪くない</td>
      <td><img alt="B"></td></tr>
</table>
"""


# --- 追い切り -----------------------------------------------------------


def by_horse(rows: list) -> dict:
    return {r.horse_id: r for r in rows}


def test_oikiri_reads_every_workout() -> None:
    rows = parse_race_oikiri(OIKIRI)
    assert len(rows) == 2

    first = by_horse(rows)["2021104123"]
    assert first.horse_id == "2021104123"
    assert first.workout_date == date(2026, 8, 26)
    assert first.course == "栗CW"
    assert first.going == "良"
    assert first.rider == "助手"
    assert first.times == [82.1, 66.5, 51.8, 38.0, 11.9]
    assert first.leg == "一杯"
    assert first.evaluation == "動き軽快"
    assert first.rank == "A"


def test_坂路は先頭の区間が欠ける() -> None:
    """坂路は800mしか走らないので、6F・5F が None になる。

    詰めてしまうと坂路の4Fタイムとコースの6Fタイムを取り違えるので、
    位置を保ったまま None で埋める（馬別ページのパーサと同じ約束）。
    """
    rows = parse_race_oikiri(OIKIRI)
    assert by_horse(rows)["2020101999"].times == [68.0, 53.2, 39.1, 12.4, None]


def test_併せ馬の相手に追い切りを紐づけない() -> None:
    """調教の行にあるリンクで馬を切り替えないこと。

    実ページでは22行中15行がリンクを持っていたのに、馬の行は11行しか
    なかった。差は併せ馬の相手へのリンク。ここで切り替えると、追い切りが
    相手の馬に付く。
    """
    rows = parse_race_oikiri(OIKIRI)
    assert {r.horse_id for r in rows} == {"2021104123", "2020101999"}
    assert "2019100001" not in {r.horse_id for r in rows}


def test_行の形が変わっても見出しでは引かない() -> None:
    """列が1つ増えても落ちないこと。

    見出しは13列なのにデータ行は5列と10列で、そもそも桁が合っていない。
    行頭が日付かどうかで見分けているので、末尾に列が増えても効かない。
    """
    grown = OIKIRI.replace("<td>A</td><td></td></tr>", "<td>A</td><td></td><td>新</td></tr>")
    assert len(parse_race_oikiri(grown)) == 2


def test_oikiri_ignores_unrelated_tables() -> None:
    assert parse_race_oikiri("<table><tr><th>枠</th></tr></table>") == []
    assert parse_race_oikiri("") == []


# --- 厩舎コメント -------------------------------------------------------


def test_comments_read_every_runner() -> None:
    rows = parse_race_comments(COMMENTS, "202604030408")
    assert len(rows) == 2
    assert rows[0].race_id == "202604030408"
    assert rows[0].umaban == 1
    assert "変わり身" in rows[0].body
    assert rows[0].source == "netkeiba"


def test_comment_keywords_reach_the_feature() -> None:
    """本文が features.condition_changes のキーワードに当たること。

    コメント欄はテーブルもコードも前から置いてあったのに、収集経路が無く
    ずっと空だった。読む側と書く側がつながっていることをここで固定する。
    """
    import pathlib

    import yaml

    weights = yaml.safe_load(
        (pathlib.Path(__file__).parents[1] / "config/weights.yml").read_text()
    )
    keywords = weights["condition_change"]["positive_keywords"]
    body = parse_race_comments(COMMENTS, "x")[0].body
    assert [k for k in keywords if k in body], "前向きな語を1つも拾えていない"


def test_comments_ignore_rows_without_a_number() -> None:
    broken = COMMENTS.replace("<td>1</td><td>1</td>", "<td>1</td><td>—</td>", 1)
    assert [r.umaban for r in parse_race_comments(broken, "x")] == [2]


# --- コメントが予想まで届くか -------------------------------------------


def _store_with_comment(tmp_path, body: str):
    """本文を**収集の経路**に通してから入れる。

    本文は保存しない設計なので、テストで body 列へ直接入れると実際と違う
    ものを測る。collect._score_comments を通し、DB に入るのと同じ形
    （数と当たった語）で置く。
    """
    import json

    from keiba.collect import _score_comments
    from keiba.models import TrainerComment
    from keiba.store import Store

    scored = _score_comments([TrainerComment(race_id="R", umaban=1, body=body)])[0]
    assert not scored.body, "収集の出口で本文が落ちていない"

    store = Store(tmp_path / "t.db")
    store.conn.execute(
        "INSERT INTO comments (race_id, umaban, positive, negative, hits, length)"
        " VALUES ('R', 1, ?, ?, ?, ?)",
        (
            scored.positive,
            scored.negative,
            json.dumps(scored.hits, ensure_ascii=False),
            scored.length,
        ),
    )
    store.conn.commit()
    return store


def _entry_and_race():
    import datetime as dt

    from keiba.models import Entry, Race

    return (
        Entry(race_id="R", umaban=1, horse_name="ウマ", horse_id="1"),
        Race(
            race_id="R", race_date=dt.date(2026, 9, 12), venue="中山", venue_code="06",
            kai=4, nichi=3, race_no=5, name="メイクデビュー中山",
            surface="芝", distance=1600,
        ),
    )


def _cfg():
    import pathlib

    import yaml

    return yaml.safe_load(
        (pathlib.Path(__file__).parents[1] / "config/weights.yml").read_text()
    )["condition_change"]


def test_過去走の無い馬でもコメントを読む(tmp_path) -> None:
    """新馬戦のコメントがまるごと落ちていた。

    厩舎コメントの判定は condition_changes の**末尾**にあり、その手前に
    `if not past_runs: return` があった。新馬は定義上つねに過去走が無いので、
    1件も読まれない。しかも netkeiba がいちばんコメントを出すのが新馬戦で、
    2026-09-12 でいえばコメントのある10レースのうち4つがメイクデビュー。
    有料データのいちばん厚いところが落ちていた。
    """
    from keiba.features import condition_changes

    entry, race = _entry_and_race()
    store = _store_with_comment(tmp_path, "追うごとに動きは良化。気性も素直で初戦から。")
    score, notes = condition_changes(
        entry, race, [], {}, store, _cfg(), {}, {},
    )
    assert score > 0, "新馬のコメントが読まれていない"
    assert any("厩舎コメント" in n for n in notes)


def test_後ろ向きな語のほうが多ければ減点する(tmp_path) -> None:
    """前向きな語だけ数えると、コメントは加点しかしない道具になる。

    厩舎の話は半分が留保で、実際「動きは水準以上」と「使ってからでは」が
    同じ文に並ぶ。差し引きで向きを決める。
    """
    from keiba.features import trainer_comment_bonus

    entry, race = _entry_and_race()
    store = _store_with_comment(
        tmp_path, "乗り込んでいるが時計が詰まってこない。実戦を使いつつ力をつけてから。"
    )
    bonus, note = trainer_comment_bonus(entry, race, store, _cfg())
    assert bonus < 0
    assert note and "後ろ向き" in note


def test_前向きと後ろ向きが同数なら何も言っていない扱い(tmp_path) -> None:
    from keiba.features import trainer_comment_bonus

    entry, race = _entry_and_race()
    store = _store_with_comment(tmp_path, "動きは水準以上。ただ使ってからでは。")
    assert trainer_comment_bonus(entry, race, store, _cfg()) == (0.0, None)


def test_向きが反転する語を表に入れない() -> None:
    """「悪くない」「良くなって」は含むかどうかで見ると向きが逆に出る。

    素朴な部分一致で読んでいるので、打ち消しの効いた語を表に入れると
    **逆の向きに加点する**。入れないことを明示しておく。
    """
    cfg = _cfg()
    for word in ("悪くな", "良くな"):
        assert word not in cfg["negative_keywords"]
        assert word not in cfg["positive_keywords"]


def test_同じ場所を二度数えない(tmp_path) -> None:
    """語の表に重なりがある。

    「動きは水準以上」は `動きは水準` と `水準以上` の両方に当たる。素直に
    数えると前向き2になり、重なりの多い言い回しほど強く出るという歪みが
    入る。実文を通して初めて出た。
    """
    from keiba.features import _keyword_sides

    good, bad = _keyword_sides(
        "動きは水準以上。ただ使ってからでは。",
        ["動きは水準", "水準以上"],
        ["使ってから"],
    )
    assert len(good) == 1, f"同じ場所を二度数えている: {good}"
    assert len(bad) == 1
