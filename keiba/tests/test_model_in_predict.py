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


def test_provisional_cards_do_not_take_model_scores(store: Store) -> None:
    """枠順未確定のときはモデルのスコアを使わないこと。

    馬番が空の出馬表には内部用の仮番号を振っている。モデル側の馬番は
    **確定した本物の馬番**なので、突き合わせると別の馬のスコアが付く。
    静かに間違った印が出るので、ここで断つ。
    """
    card = make_card(confirmed=False)
    scores = {i: 0.05 * i for i in range(1, 9)}

    payload = predict.predict_card(card, store, WEIGHTS, {}, scores)
    assert payload is not None
    assert payload["post_positions_confirmed"] is False
    assert all(h["umaban"] is None for h in payload["horses"])
    # モデルのスコアを当てていたら 40.0 が最大になるはず
    assert max(h["score"] for h in payload["horses"]) != pytest.approx(40.0)


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
