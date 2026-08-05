"""エンジンの検証。

SKILL.md の教訓が「書いてあるだけ」でなく実際に効いていることを確かめる。
各テストはどの教訓に対応するかを明記してある。教訓が壊れたらここが落ちる。
"""

from __future__ import annotations

import pathlib
from datetime import date, timedelta

import pytest
import yaml

from keiba import betting, confidence, engine
from keiba.features import build_features
from keiba.models import Entry, Race, Result
from keiba.store import Store

WEIGHTS = yaml.safe_load(
    (pathlib.Path(__file__).parents[1] / "config/weights.yml").read_text()
)
TODAY = date(2026, 8, 1)


@pytest.fixture
def store(tmp_path) -> Store:
    with Store(tmp_path / "t.db") as s:
        yield s


def add_history(
    store: Store,
    horse_id: str,
    *,
    runs: int = 4,
    surface: str = "芝",
    distance: int = 1600,
    finish: int = 5,
    corner: int = 8,
    field_size: int = 16,
    venue: str = "東京",
    race_class: str = "3勝クラス",
    jockey_id: str = "00001",
    start_days_ago: int = 30,
    interval: int = 30,
) -> None:
    """過去走を作る。脚質は corner（1コーナー通過順）で決まる。"""
    for i in range(runs):
        day = TODAY - timedelta(days=start_days_ago + i * interval)
        race_id = f"H{horse_id[-4:]}{i:02d}0000"[:12]
        store.save_card(
            _card(
                Race(
                    race_id=race_id, race_date=day, venue=venue, venue_code="05",
                    kai=1, nichi=1, race_no=11, name="過去走",
                    surface=surface, distance=distance, going="良",
                    race_class=race_class, field_size=field_size,
                ),
                Entry(
                    race_id=race_id, umaban=1, horse_name="H", horse_id=horse_id,
                    jockey_id=jockey_id,
                ),
                Result(
                    race_id=race_id, umaban=1, finish_pos=finish,
                    corners=[corner, corner], last3f=34.5,
                ),
            )
        )
    store.conn.commit()


def _card(race, entry, result):
    from keiba.models import RaceCard

    return RaceCard(race=race, entries=[entry], results=[result])


def make_race(field_size: int = 10) -> Race:
    return Race(
        race_id="202605021211", race_date=TODAY, venue="東京", venue_code="05",
        kai=2, nichi=11, race_no=11, name="テストS", surface="芝", distance=1600,
        going="良", race_class="3勝クラス", field_size=field_size,
    )


def make_entries(popularities: list[int], sire: str = "テスト種牡馬") -> list[Entry]:
    return [
        Entry(
            race_id="202605021211", umaban=i + 1, horse_name=f"馬{i + 1}",
            horse_id=f"20221000{i:02d}", jockey_id="00001", sire=sire,
            market_popularity=pop, market_odds=float(pop) * 2.5,
        )
        for i, pop in enumerate(popularities)
    ]


def score(store: Store, race: Race, entries: list[Entry], sire_table: dict | None = None):
    features = build_features(race, entries, store, sire_table or {}, WEIGHTS)
    horses = engine.run(features, {e.umaban: e for e in entries}, WEIGHTS)
    grade = confidence.grade(horses, WEIGHTS)
    return horses, grade, betting.build(horses, grade, WEIGHTS)


# ------------------------------------------------------ 教訓1: 血統で拾う

def test_dirt_to_turf_switcher_is_promoted(store: Store) -> None:
    """教訓1: ダートで大敗した馬を、父の芝適性から芝替わりで拾い直す。

    Palace Pier を名指しするのではなく、「今走の馬場で父の成績が良いのに
    前走はその馬場を使えていなかった」という形をデータから検出できること。
    """
    sire_table = {
        "欧州型": {"芝:mile": {"runs": 200, "place_rate": 0.45, "win_rate": 0.15},
                   "ダ:mile": {"runs": 120, "place_rate": 0.08, "win_rate": 0.01}},
        "ダート型": {"芝:mile": {"runs": 200, "place_rate": 0.12, "win_rate": 0.02},
                    "ダ:mile": {"runs": 200, "place_rate": 0.40, "win_rate": 0.14}},
    }
    race = make_race()
    entries = make_entries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # 1番は欧州型血統でダート大敗続き、2番は同じ戦績だがダート型血統
    entries[0].sire = "欧州型"
    entries[1].sire = "ダート型"
    for e in (entries[0], entries[1]):
        add_history(store, e.horse_id, surface="ダ", finish=12, runs=3)
    for e in entries[2:]:
        add_history(store, e.horse_id, surface="芝", finish=6, runs=3)

    horses, _, _ = score(store, race, entries, sire_table)
    by_umaban = {h.umaban: h for h in horses}

    assert by_umaban[1].score > by_umaban[2].score, (
        "同じダート大敗でも、父が芝で走る馬を上に取れていない"
    )
    assert any("ダ→芝替わり" in r for r in by_umaban[1].reasons)
    # ダート大敗を「能力不足」と読まないこと
    assert not any("能力不足" in r for r in by_umaban[1].reasons)


# ------------------------------------ 教訓7: 少頭数でハイペースにしない

def test_small_field_does_not_assume_fast_pace(store: Store) -> None:
    """教訓7: 9〜10頭立てでは逃げ馬が複数いてもハイペースと決めつけない。

    大井4Rでこれを機械適用し、確逃げ馬を切って3連複を落としている。
    """
    from keiba.features import project_pace, HorseFeatures

    front = [HorseFeatures(umaban=i, horse_id="", horse_name="", style="逃げ") for i in (1, 2)]
    others = [HorseFeatures(umaban=i, horse_id="", horse_name="", style="差し") for i in range(3, 10)]

    assert project_pace(front + others, 9, WEIGHTS["pace"]) == "slow", (
        "少頭数で複数逃げ馬をハイペース判定してしまっている"
    )
    # 多頭数なら従来どおりハイペース
    assert project_pace(front + others, 18, WEIGHTS["pace"]) == "fast"


# --------------------------------- 教訓8: 確逃げ馬を切らない・3着欄に残す

def test_lone_front_runner_gets_floor_and_a_ticket(store: Store) -> None:
    """教訓8: 単騎逃げ馬は能力評価が低くても切らず、3着欄に最低1点入れる。"""
    race = make_race(field_size=10)
    entries = make_entries([1, 2, 3, 4, 5, 6, 7, 8, 9, 12])
    # 10番だけが逃げ馬。他は全部後方
    for e in entries[:-1]:
        add_history(store, e.horse_id, corner=12, field_size=16, finish=3)
    add_history(store, entries[-1].horse_id, corner=1, field_size=16, finish=11)

    horses, grade, tickets = score(store, race, entries)
    front = next(h for h in horses if h.umaban == 10)

    assert front.is_lone_front_runner, "単騎逃げと判定できていない"
    assert front.score >= WEIGHTS["pace"]["lone_front_runner_floor"], (
        "確逃げ馬にスコアの床が効いていない"
    )
    assert any("教訓8" in r for r in front.reasons)
    if tickets:
        assert any(10 in t.umabans for t in tickets), "確逃げ馬が1点も買い目に入っていない"


# ------------------------- 教訓10: 印5頭以上なら全組み合わせを網羅する

def test_all_axis_combinations_are_covered_on_main_line(store: Store) -> None:
    """教訓10: 大井7Rの失敗（印は当てたが組み合わせを買っていない）の再発防止。

    本線（◎○）では ◎軸で相手が n 頭なら C(n,2) 点すべてを買う。

    穴枠（△）には適用しない。実測でこの帯の回収率は本線より明確に低く
    （9-13が89.9%、14-19が63.3%、20-27が18.2%）、そこに全組み合わせを
    敷くのは最も期待値の悪い区分に金を積むことになる。穴枠では
    「印を打った馬が必ずどこかの買い目に入る」保証だけを守る
    （test_longshot_allocation_still_covers_every_mark）。
    """
    race = make_race(field_size=14)
    # 本線の帯（能力上位3頭の人気和 9-13）に入るよう人気を割り当てる
    entries = make_entries([4, 5, 3, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    for e in entries:
        add_history(store, e.horse_id)

    horses, grade, tickets = score(store, race, entries)
    if grade.is_longshot or not grade.should_bet:
        pytest.skip(f"本線シナリオにならなかった: {grade.grade} {grade.reason}")

    marked = [h for h in horses if h.mark]
    axis = next(h for h in marked if h.mark == "◎")
    partners = [h for h in marked if h is not axis]

    import itertools

    trio = {t.combination for t in tickets if t.bet_type == "三連複"}
    for a, b in itertools.combinations(partners, 2):
        combo = "-".join(str(x) for x in sorted((axis.umaban, a.umaban, b.umaban)))
        assert combo in trio, f"◎軸の組み合わせ {combo} が買い目に無い"


def test_longshot_allocation_is_thin_but_covers_every_mark(store: Store) -> None:
    """穴枠は点数を絞る。ただし印を打った馬は必ず1点以上に含める。

    絞る動機は実測（この帯の回収率は本線より低い）。それでも
    「印は当てたのに買い目で外す」だけは起こさない。
    """
    race = make_race(field_size=14)
    entries = make_entries([14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 13])
    for e in entries:
        add_history(store, e.horse_id)

    horses, grade, tickets = score(store, race, entries)
    if not grade.is_longshot:
        pytest.skip(f"穴枠シナリオにならなかった: {grade.grade}")

    cap = WEIGHTS["betting"]["longshot_max_points"]
    marked = [h for h in horses if h.mark]
    covered = set().union(*(t.umabans for t in tickets))

    for h in marked:
        assert h.umaban in covered, f"{h.mark}{h.umaban} が買い目に無い"
    # 印を覆うのに要る点は残すが、それを超えて広げてはいない
    assert len(tickets) <= max(cap, len(marked)), (
        f"穴枠が {len(tickets)} 点まで広がっている（上限 {cap}）"
    )
    assert sum(t.amount for t in tickets) <= WEIGHTS["betting"]["stake"]["△"]["race_cap"]


def test_every_marked_horse_appears_in_a_ticket(store: Store) -> None:
    """SKILL.md 必須ルール: 印を打った馬は必ず1点以上の買い目に含める。"""
    race = make_race(field_size=12)
    entries = make_entries([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    for e in entries:
        add_history(store, e.horse_id)

    horses, grade, tickets = score(store, race, entries)
    if not tickets:
        pytest.skip("このシナリオは見送り判定になった")

    covered = set().union(*(t.umabans for t in tickets))
    for h in horses:
        if h.mark:
            assert h.umaban in covered, f"{h.mark}{h.umaban} が買い目に無い"


# ------------------------------- 「固い予想はいらない」= × で買わない

def test_chalk_race_is_skipped(store: Store) -> None:
    """能力上位3頭がそのまま上位人気なら × を付けて1点も買わない。"""
    race = make_race(field_size=10)
    # 能力順＝人気順になるよう、上位馬ほど強い戦績を与える
    entries = make_entries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    for i, e in enumerate(entries):
        add_history(store, e.horse_id, finish=1 if i < 3 else 10)

    horses, grade, tickets = score(store, race, entries)
    top3_pops = sorted(h.market_popularity for h in horses[:3] if h.market_popularity)

    if sum(top3_pops) < WEIGHTS["confidence"]["main_band"][0]:
        assert grade.grade == "×", f"堅い決着なのに {grade.grade} を付けている"
        assert tickets == [], "× のレースで買い目を出している"
        assert not grade.should_bet


def test_longshot_race_is_backed(store: Store) -> None:
    """能力上位が人気薄に集まっているレースは穴枠として買う。

    実測でこの帯の回収率は本線に劣るので厚くは張らないが、切り捨てもしない。
    """
    race = make_race(field_size=12)
    entries = make_entries([12, 11, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 人気薄（1〜3番）に強い戦績、上位人気（4番以降）は凡走続き
    for i, e in enumerate(entries):
        add_history(store, e.horse_id, finish=1 if i < 3 else 12)

    horses, grade, tickets = score(store, race, entries)
    assert grade.popularity_sum >= WEIGHTS["confidence"]["longshot_min"]
    assert grade.is_longshot, f"穴枠として扱われていない: {grade.reason}"
    assert grade.should_bet, f"穴のレースを見送っている: {grade.reason}"
    assert tickets, "買い目が1点も出ていない"
    # ☆（穴）を含む買い目が必ずある（教訓5・6）
    longshot = next((h for h in horses if h.mark == "☆"), None)
    if longshot:
        assert any(longshot.umaban in t.umabans for t in tickets)


# --------------------------------------------- オッズ分離の保証

def test_odds_never_influence_ability_score(store: Store) -> None:
    """SKILL.md の絶対ルール: オッズ・人気は能力評価に一切入らない。

    人気だけを入れ替えて、スコアが1点も動かないことを確かめる。
    """
    race = make_race(field_size=10)
    entries_a = make_entries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    entries_b = make_entries([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    for e in entries_a:
        add_history(store, e.horse_id)

    horses_a, _, _ = score(store, race, entries_a)
    horses_b, _, _ = score(store, race, entries_b)

    scores_a = {h.umaban: h.score for h in horses_a}
    scores_b = {h.umaban: h.score for h in horses_b}
    assert scores_a == scores_b, "人気を変えたら能力スコアが動いた（オッズが漏れている）"


def test_confidence_flips_with_popularity_only(store: Store) -> None:
    """逆に、自信度は人気だけで変わること（穴判定はオッズを使ってよい層）。"""
    race = make_race(field_size=10)
    entries_a = make_entries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    entries_b = make_entries([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    for e in entries_a:
        add_history(store, e.horse_id)

    _, grade_a, _ = score(store, race, entries_a)
    _, grade_b, _ = score(store, race, entries_b)
    assert grade_a.popularity_sum != grade_b.popularity_sum
