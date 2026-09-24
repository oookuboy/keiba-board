"""ボードに出す予想が、実際に学習モデルで採点されていることを確かめる。

## なぜこのテストが要るか

`backtest` は `--no-model` を付けない限り学習モデルのスコアで採点していた。
一方 `predict`（ボードに出るほう）は `weights.yml` の手置きスコアだけで
動いていた。**測っているものと出しているものが別だった**。

しかもこれは静かに壊れる形をしている。手置きでも予想は普通に出るので、
ログにも成果物にも異常が現れない。タイム指数を足して AUC が上がっても、
調教を測っても、ボードには一度も届いていなかった。

型でも例外でも捕まらないので、テストで押さえる。
"""

from __future__ import annotations

from datetime import date

import pytest
import yaml
import pathlib

from keiba import predict
from keiba.models import Entry, Race, RaceCard
from keiba.store import Store

WEIGHTS = yaml.safe_load(
    (pathlib.Path(__file__).parents[1] / "config/weights.yml").read_text()
)
TODAY = date(2026, 8, 1)


@pytest.fixture
def store(tmp_path) -> Store:
    with Store(tmp_path / "t.db") as s:
        yield s


def make_card(confirmed: bool = True) -> RaceCard:
    race = Race(
        race_id="202605021211", race_date=TODAY, venue="東京", venue_code="05",
        kai=2, nichi=11, race_no=11, name="テストS", surface="芝", distance=1600,
        going="良", race_class="3勝クラス", field_size=8,
    )
    entries = [
        Entry(
            race_id=race.race_id,
            umaban=i + 1 if confirmed else None,
            horse_name=f"馬{i + 1}",
            horse_id=f"20221000{i:02d}",
            jockey_id="00001",
            market_popularity=i + 1,
            market_odds=float(i + 1) * 2.5,
        )
        for i in range(8)
    ]
    return RaceCard(race=race, entries=entries, results=[])


def test_model_scores_decide_the_order(store: Store) -> None:
    """モデルのスコアを渡したら、その順に印が付くこと。

    手置きの重みは能力順位が市場の半分しか当たらなかった。順位付けをモデルへ
    移したのはそのためで、ここが繋がっていなければ移した意味が無い。
    """
    card = make_card()
    # 8番を一番高く、1番を一番低く。手置きの重みでは出ない並びにする。
    scores = {i: 0.05 * i for i in range(1, 9)}

    payload = predict.predict_card(card, store, WEIGHTS, {}, scores)
    assert payload is not None

    order = [h["umaban"] for h in payload["horses"]]
    assert order == [8, 7, 6, 5, 4, 3, 2, 1], "モデルのスコア順になっていない"
    # 確率をそのまま 0〜100 に写している。床は当てない（スケールが違う）。
    assert payload["horses"][0]["score"] == pytest.approx(40.0)


def test_without_model_the_hand_weights_are_used(store: Store) -> None:
    """モデルを渡さなければ手置きの重みで動くこと（後方互換）。"""
    scores = {i: 0.05 * i for i in range(1, 9)}
    with_model = predict.predict_card(make_card(), store, WEIGHTS, {}, scores)
    without = predict.predict_card(make_card(), store, WEIGHTS, {}, None)
    assert with_model is not None and without is not None

    # 同じ出馬表で採点が一致するなら、渡したスコアが無視されている。
    assert {h["umaban"]: h["score"] for h in without["horses"]} != {
        h["umaban"]: h["score"] for h in with_model["horses"]
    }


def test_provisional_cards_take_model_scores(store: Store) -> None:
    """枠順未確定でもモデルのスコアで印が付くこと。

    以前はここでスコアを捨てていて、木曜の印は手置きの重みだけで付いていた。
    DB 側（store.pseudo_numbered）と予想側で同じ仮番号を振るので、
    突き合わせても別の馬のスコアにはならない。出力に馬番は載せない。
    """
    card = make_card(confirmed=False)
    scores = {i: 0.05 * i for i in range(1, 9)}

    payload = predict.predict_card(card, store, WEIGHTS, {}, scores)
    assert payload is not None
    assert payload["post_positions_confirmed"] is False
    assert all(h["umaban"] is None for h in payload["horses"])
    assert not payload["bets"], "仮番号で券を組んではいけない"
    # 仮番号 8 = 表示順 8 頭目の馬に 0.40 が付く
    top = payload["horses"][0]
    assert top["name"] == "馬8"
    assert top["score"] == pytest.approx(40.0)


def test_unconfirmed_card_enters_db_with_the_same_pseudo_numbers(store: Store) -> None:
    """枠順確定前の出走馬が、予想側と同じ仮番号で DB に入ること。

    入らないと、その日の行が特徴量の表に1行も出ず、モデルが使われない。
    """
    from keiba.store import pseudo_numbered

    card = make_card(confirmed=False)
    store.save_cards([card])
    rows = store.conn.execute(
        "SELECT umaban, horse_name, waku FROM entries WHERE race_id = ? ORDER BY umaban",
        (card.race.race_id,),
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [(i, f"馬{i}") for i in range(1, 9)]
    assert all(r[2] is None for r in rows), "枠は未確定のまま残すこと"
    assert [e.umaban for e in pseudo_numbered(card)] == list(range(1, 9))


def test_confirmed_card_replaces_pseudo_rows(store: Store) -> None:
    """枠順確定後に、仮番号の行が同じ馬の別番号として残らないこと。"""
    from dataclasses import replace

    store.save_cards([make_card(confirmed=False)])
    confirmed = make_card(confirmed=True)
    # 確定した馬番は表示順と逆
    confirmed.entries = [replace(e, umaban=9 - e.umaban) for e in confirmed.entries]
    store.save_cards([confirmed])
    rows = store.conn.execute(
        "SELECT horse_name, umaban FROM entries WHERE race_id = ?",
        (confirmed.race.race_id,),
    ).fetchall()
    assert len(rows) == 8, "仮番号の行が残って同じ馬が2頭いる"
    assert dict(rows)["馬1"] == 8


def test_unknown_draw_is_blanked_for_the_model() -> None:
    """枠が未確定の行は、枠由来の列を欠損にしてモデルへ渡すこと。

    仮番号は五十音順でしかない。そのまま渡すと「1番の馬」として採点される。
    """
    import pandas as pd

    df = pd.DataFrame({
        "waku": [None, 3.0], "umaban": [1, 5],
        "draw_ratio": [0.1, 0.5], "pc_draw_x_front": [0.1, 0.2],
    })
    out = predict._blank_unknown_draw(df)
    assert out.loc[0, ["umaban", "draw_ratio", "pc_draw_x_front"]].isna().all()
    assert out.loc[1, "umaban"] == 5 and out.loc[1, "draw_ratio"] == 0.5
    assert df.loc[0, "umaban"] == 1, "元の表（スコアの突き合わせに使う馬番）を壊さない"


def test_predict_day_records_what_scored_it(store: Store) -> None:
    """何で採点したかが成果物に残ること。

    残っていないと、回顧のときに手置きの成績とモデルの成績が混ざる。
    """
    payload = predict.predict_day(store, WEIGHTS, {}, TODAY)
    assert payload["scored_by"] == "weights"


def test_an_empty_day_never_overwrites_a_good_prediction(tmp_path) -> None:
    """収集に失敗した日が、前回の予想を空ファイルで潰さないこと。

    書き出しが先で判定が後だったため、レースを1件も取れなかったときに
    空の payload をそのまま上書きしていた。当日朝の再生成が一度でも空振り
    すれば、前夜に出した買い目がボードから消える形だった（手元で踏んだ）。

    しかも exit 0 で返していたので、ワークフローも成功扱いになる。

    index.json も同じ経路（write_day）で書き換わる。実際に races 24 → 0 に
    潰れて、ボードの一覧から当日が消えた。日別ファイルだけ見ていると
    見落とすので、両方を確かめる。
    """
    import shlex

    from keiba.cli import build_parser, cmd_predict

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    good = data_dir / "2026-08-16.json"
    good.write_text('{"date": "2026-08-16", "races": ["前回の予想"]}', encoding="utf-8")
    index = data_dir / "index.json"
    index.write_text('{"days":[{"date":"2026-08-16","races":24}]}', encoding="utf-8")

    args = build_parser().parse_args(
        shlex.split(
            f"--db {tmp_path / 'empty.db'} --raw-dir {tmp_path / 'raw'}"
            f" --data-dir {data_dir} --config-dir keiba/config"
            " predict --date 2026-08-16"
        )
    )
    # 空のDBなので1レースも出てこない
    assert cmd_predict(args) == 1, "空の日を成功として返している"
    assert "前回の予想" in good.read_text(encoding="utf-8"), "前回の予想を潰した"
    assert '"races":24' in index.read_text(encoding="utf-8"), "index.json を潰した"


# --- 市場と比べる物差し -------------------------------------------------


def test_学習のたびに市場と比べる() -> None:
    """AUC だけ見て「学習できている」と判断していた。

    AUC は全レースの全頭を混ぜて並べた指標で、**レースの中で誰が上か**を
    測っていない。実測では AUC 0.75 のまま、◎ の3着内率 49.4% に対し市場の
    1番人気が 61.8%。負けているのに気づけない物差しを使っていた。

    モデルの1位と1番人気を、同じレースの上で直接比べる。
    """
    import numpy as np
    import pandas as pd

    from keiba import ml

    # 3レース。モデルは毎回1番人気とは違う馬を1位にする
    df = pd.DataFrame({
        "race_id": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
        "market_popularity": [1, 2, 3] * 3,
        "finish_pos": [1, 2, 5,   4, 1, 2,   1, 3, 6],
        ml.TARGET: [1, 1, 0,   0, 1, 1,   1, 1, 0],
    })
    # 各レースの2番人気を1位に置くスコア
    scores = np.array([0.1, 0.9, 0.2] * 3)
    s = ml.vs_market(df, scores)

    assert s["races"] == 3
    # モデル1位（2番人気）の勝率は 1/3、3着内は 3/3
    assert s["model_win"] == pytest.approx(1 / 3)
    assert s["model_p3"] == pytest.approx(1.0)
    # 1番人気の勝率は 2/3、3着内は 2/3
    assert s["market_win"] == pytest.approx(2 / 3)
    assert s["market_p3"] == pytest.approx(2 / 3)
    assert "モデルが上" in ml.format_vs_market(df, scores)


def test_ランキング学習を選べる() -> None:
    """レースの中の順位を学習する経路があること。

    既定の binary は1頭ずつ独立に「3着以内か」を当てる。同じレースの14頭を
    正しく並べる、という本来の目的とずれている。
    """
    import inspect

    from keiba import ml

    assert ml.RANK_PARAMS["objective"] == "lambdarank"
    assert "objective" in inspect.signature(ml.train).parameters


def test_ランキング学習はレース単位で並べ直す() -> None:
    """group と行の並びが一致していないと、別レースの馬を同じレースとして学ぶ。

    lambdarank でいちばん壊しやすい所なので、並べ直しが残っていることを見る。
    """
    import inspect

    from keiba import ml

    source = inspect.getsource(ml.train)
    assert 'sort_values(["race_date", "race_id"])' in source, (
        "group を渡す前に並べ直していない。別レースの馬が同じ group に入る"
    )
