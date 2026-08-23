"""買い目の組み方の検証。

## 何を守るテストか

モデルが出しているのは「その馬が3着以内に入る確率」で、印はその上位N頭。
つまり印を打った馬は**どれも3着以内に来そうな馬**であって、1位だけが特別
なわけではない。

にもかかわらず、従来の買い目は◎を軸に固定して流していた。全ての点に◎が
入るので、◎が3着以内を外すと、残りの印が1〜3着を独占していても0点になる。

2026-08-16 札幌1R が実例。◎13 を軸にした8点を買い、決着は ○3・☆14・▲ で、
印は当てているのに買い目に無かった。

「印を打った馬は必ず1点以上の買い目に含める」（教訓10）は満たしていても、
**印どうしの組み合わせが買えていない**なら同じ取りこぼしが起きる。そこを
ここで押さえる。
"""

from __future__ import annotations

import copy
import pathlib

import pytest
import yaml

from keiba import betting
from keiba.confidence import Confidence
from keiba.engine import ScoredHorse

WEIGHTS = yaml.safe_load(
    (pathlib.Path(__file__).parents[1] / "config/weights.yml").read_text()
)

# ◎ から順に印を打った5頭。スコア差はわずかで、どれも「来そうな馬」。
MARKS = ["◎", "○", "▲", "☆", "▲"]
SCORES = [62.1, 57.9, 56.3, 54.1, 53.4]


def horses() -> list[ScoredHorse]:
    return [
        ScoredHorse(
            umaban=umaban,
            horse_id=f"2022{umaban:06d}",
            horse_name=f"馬{umaban}",
            score=score,
            style="先行",
            reasons=[],
            mark=mark,
            market_popularity=i + 1,
            market_odds=(i + 1) * 2.0,
        )
        for i, (umaban, mark, score) in enumerate(
            zip([13, 3, 5, 14, 10], MARKS, SCORES, strict=True)
        )
    ]


def weights_with(shape: str) -> dict:
    cfg = copy.deepcopy(WEIGHTS)
    cfg["betting"]["trio_shape"] = shape
    return cfg


def grade() -> Confidence:
    return Confidence(
        grade="◎", popularity_sum=11, separation=8.7, expected_odds=80.0, reason="測定用"
    )


def combos(tickets) -> set[frozenset[int]]:
    return {
        frozenset(t.umabans) for t in tickets if t.bet_type == betting.TRIO
    }


def test_axis_flow_misses_when_the_favourite_of_the_model_fails() -> None:
    """◎軸流しは、◎を外した決着を1点も持っていない。

    これは「不運」ではなく組み方の帰結なので、事実として固定しておく。
    """
    tickets = betting.build(horses(), grade(), weights_with("axis"))
    assert tickets

    trio = combos(tickets)
    assert all(13 in c for c in trio), "◎軸流しなのに◎の無い点がある"
    # ○3・☆14・▲5 の決着（◎を含まない）
    assert frozenset({3, 14, 5}) not in trio


def test_box_covers_any_three_of_the_marked_horses() -> None:
    """ボックスは、印5頭のどの3頭で決まっても当たる。"""
    tickets = betting.build(horses(), grade(), weights_with("box"))
    trio = combos(tickets)

    import itertools

    for combo in itertools.combinations([13, 3, 5, 14, 10], 3):
        assert frozenset(combo) in trio, f"{combo} が買えていない"


def test_box_costs_only_a_little_more() -> None:
    """点数が跳ね上がらないこと。

    C(5,3)=10点。◎軸流し（C(4,2)=6点＋押さえ）から数点増えるだけで、
    レース上限にも収まる。ここが何十点にもなるなら採用できない。
    """
    axis = betting.build(horses(), grade(), weights_with("axis"))
    box = betting.build(horses(), grade(), weights_with("box"))

    axis_trio = [t for t in axis if t.bet_type == betting.TRIO]
    box_trio = [t for t in box if t.bet_type == betting.TRIO]
    assert len(box_trio) == 10
    assert len(box_trio) - len(axis_trio) <= 4

    cap = WEIGHTS["betting"]["stake"]["◎"]["race_cap"]
    assert sum(t.amount for t in box) <= cap


def test_every_marked_horse_is_still_covered() -> None:
    """教訓10 の必須ルールは組み方を変えても守られること。"""
    for shape in ("axis", "box"):
        tickets = betting.build(horses(), grade(), weights_with(shape))
        covered: set[int] = set()
        for ticket in tickets:
            covered |= ticket.umabans
        for horse in horses():
            assert horse.umaban in covered, f"{shape}: 印馬 {horse.umaban} が買い目に無い"


def test_skipped_races_buy_nothing_in_either_shape() -> None:
    """× のレースは組み方によらず1点も買わない。"""
    skip = Confidence(
        grade="×", popularity_sum=6, separation=9.0, expected_odds=12.0, reason="堅い"
    )
    for shape in ("axis", "box"):
        assert betting.build(horses(), skip, weights_with(shape)) == []


@pytest.mark.parametrize("shape", ["axis", "box"])
def test_no_duplicate_combinations(shape: str) -> None:
    """同じ組を二重に買わないこと。"""
    tickets = betting.build(horses(), grade(), weights_with(shape))
    keys = [(t.bet_type, t.combination) for t in tickets]
    assert len(keys) == len(set(keys))


def test_shape_can_differ_by_confidence() -> None:
    """自信度ごとに組み方を変えられること。

    勝負どころ（◎）は軸を立てて厚く、自信の落ちるところは広く、という
    持ち方を設定で表せるようにしてある。
    """
    cfg = copy.deepcopy(WEIGHTS)
    cfg["betting"]["trio_shape"] = {"◎": "axis", "○": "box", "△": "box"}

    honmei = betting.build(horses(), grade(), cfg)
    assert all(13 in c for c in combos(honmei)), "◎ が axis になっていない"

    taikou = Confidence(
        grade="○", popularity_sum=11, separation=2.0, expected_odds=80.0, reason=""
    )
    assert frozenset({3, 14, 5}) in combos(betting.build(horses(), taikou, cfg)), (
        "○ が box になっていない"
    )


def test_an_unlisted_confidence_falls_back_to_axis() -> None:
    """書き忘れた自信度は axis に落ちること。

    既定を box にすると、書き忘れただけで点数と金額が黙って増える。
    増えるほうを既定にしない。
    """
    cfg = copy.deepcopy(WEIGHTS)
    cfg["betting"]["trio_shape"] = {"○": "box"}
    assert all(13 in c for c in combos(betting.build(horses(), grade(), cfg)))


def test_the_longshot_grade_now_gets_a_real_box() -> None:
    """△ でも本物のボックスになること。

    以前はここが「axis と box が完全一致する」ことを確かめるテストだった。
    longshot_max_points（☆を含む点を優先して3点残す）がボックスより後に
    効いて元に戻していたため。◎軸をやめたのに穴枠だけ☆軸だった。

    2026-08-22 中京4R でその形が出た。印5頭のうち ◎9・▲12・▲3 が1〜3着
    （三連複 28,630円・その日の最高配当）なのに、残った3点は 8-9-18 /
    3-8-9 / 8-9-12 で全て ☆8 経由。☆8 は4着以下だった。

    絞り込みは box_marks（箱に入れる頭数）へ移した。
    """
    longshot = Confidence(
        grade="△", popularity_sum=16, separation=2.0, expected_odds=200.0, reason=""
    )
    tickets = betting.build(horses(), longshot, weights_with("box"))
    trio = combos(tickets)

    import itertools

    for combo in itertools.combinations([13, 3, 5, 14, 10], 3):
        assert frozenset(combo) in trio, f"△ で {combo} が買えていない"
    assert sum(t.amount for t in tickets) <= WEIGHTS["betting"]["stake"]["△"]["race_cap"]


def test_narrowing_shrinks_the_box_instead_of_planting_an_axis() -> None:
    """絞り込みは頭数を減らす形であること。軸を立てる形ではない。

    「自信があるから絞る」を◎軸流しで表していたが、あれは絞り込みではなく
    「この1頭は必ず来る」という別の賭けだった。◎が3着以内を外せば、他の印が
    1〜3着を独占していても0点になる。

    頭数を減らせば点数は落ちるが、残った馬どうしの組は全部買えている。
    """
    cfg = copy.deepcopy(WEIGHTS)
    cfg["betting"]["box_marks"] = {"◎": 4}

    tickets = betting.build(horses(), grade(), cfg)
    trio = combos(tickets)
    assert len(trio) == 4, f"C(4,3)=4点のはずが {len(trio)}点"

    import itertools

    # 上位4頭（13/3/5/14）の総当たり。5頭目の 10 は買わないと決めた馬
    for combo in itertools.combinations([13, 3, 5, 14], 3):
        assert frozenset(combo) in trio
    assert not any(10 in c for c in trio), "箱から外した馬が買い目に残っている"

    # 軸は立っていない。◎13 を含まない点が必ずある
    assert any(13 not in c for c in trio), "絞った結果◎軸になっている"
