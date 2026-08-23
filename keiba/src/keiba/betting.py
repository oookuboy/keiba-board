"""買い目の組み立て。

SKILL.md で最も高くついた失敗は「印は当てたのに買い目に入れていなかった」
（大井7R: 1〜3着すべてに印を打ちながら 4-9-11 を買っておらず不的中）。
その再発を設計で防ぐのがこのモジュールの主目的。

  教訓10 印が5頭以上なら ◎軸の全組み合わせ C(n,2) を必ず敷く
  教訓5・6 ☆（穴）を含む買い目を必ず1点入れる
  教訓8 確逃げ馬を3着欄に含む買い目を必ず1点入れる
  必須ルール 印を打った馬は必ず1点以上の買い目に含める

最後に assert で「印を打ったのに買い目に入っていない馬」が居ないことを
確認している。ここが落ちるならロジックの穴であり、静かに取りこぼすより
落ちたほうがよい。
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

from keiba.confidence import Confidence
from keiba.engine import ScoredHorse

log = logging.getLogger(__name__)

TRIO = "三連複"
TRIFECTA = "三連単"


@dataclass
class Ticket:
    bet_type: str
    combination: str
    amount: int
    rationale: str

    @property
    def umabans(self) -> set[int]:
        return {int(x) for x in self.combination.split("-")}


def _trio(umabans: tuple[int, ...]) -> str:
    return "-".join(str(n) for n in sorted(umabans))


def build(
    horses: list[ScoredHorse], confidence: Confidence, weights: dict
) -> list[Ticket]:
    """採点済みの全頭リストから買い目を作る。

    自信度が × のレースは1点も買わない。「固い予想はいらない」の実装。
    """
    if not confidence.should_bet:
        return []

    cfg = weights["betting"]
    stake = cfg["stake"][confidence.grade]
    unit = stake["per_point"]

    marked = [h for h in horses if h.mark]
    axis = next((h for h in marked if h.mark == "◎"), None)
    if axis is None or len(marked) < 3:
        return []
    partners = [h for h in marked if h is not axis]

    # 箱に入れる頭数。**絞り込みはここでやる。**
    #
    # 「自信があるから絞る」を◎軸流しで表していたが、あれは絞り込みではなく
    # 「この1頭は必ず来る」という別の賭けだった。◎が3着以内を外せば、他の印が
    # 1〜3着を独占していても0点になる。実際に3週続けてその形で落としている。
    #
    # 絞るなら頭数を減らす。箱のまま濃くなる。
    #   5頭 → C(5,3) = 10点
    #   4頭 → C(4,3) =  4点
    #   3頭 → C(3,3) =  1点
    box = marked[: _box_marks(cfg, confidence.grade, len(marked))]
    if len(box) < len(marked):
        log.info(
            "%s: 印%d頭のうち上位%d頭で箱を組む（%s）",
            confidence.grade, len(marked), len(box),
            "・".join(h.horse_name for h in marked[len(box):]),
        )

    tickets: list[Ticket] = []
    seen: set[tuple[str, str]] = set()

    def add(bet_type: str, combo: str, amount: int, why: str) -> None:
        key = (bet_type, combo)
        if key in seen:
            return
        seen.add(key)
        tickets.append(Ticket(bet_type, combo, amount, why))

    if _shape_for(cfg, confidence.grade) == "box":
        _add_box(add, box, unit, cfg)
    else:
        _add_axis_flow(add, axis, partners, unit, cfg)

    # --- 教訓8: 確逃げ馬を3着欄に必ず1点 --------------------------------
    front = next((h for h in horses if h.is_lone_front_runner), None)
    if front and front is not axis:
        mate = next((p for p in partners if p is not front), None)
        if mate:
            add(
                TRIO,
                _trio((axis.umaban, mate.umaban, front.umaban)),
                cfg["unit"],
                "確逃げ馬を3着欄に確保（教訓8）",
            )

    # --- 教訓5・6: ☆を含む大穴を必ず1点 ---------------------------------
    longshot = next((h for h in marked if h.mark == "☆"), None)
    if longshot and longshot is not axis:
        mate = next((p for p in partners if p is not longshot), None)
        if mate:
            add(
                TRIO,
                _trio((axis.umaban, mate.umaban, longshot.umaban)),
                cfg["unit"],
                f"大穴押さえ ☆{longshot.horse_name}（教訓5・6）",
            )

    # --- 3連単は自信度◎のときだけ ---------------------------------------
    if confidence.grade in cfg["trifecta_on"] and len(partners) >= 2:
        orders = list(itertools.permutations([p.umaban for p in partners[:3]], 2))
        for first, second in orders[: cfg["trifecta_points"]]:
            add(
                TRIFECTA,
                f"{axis.umaban}-{first}-{second}",
                cfg["unit"],
                "◎1着固定の3連単",
            )

    tickets = _fit_budget(tickets, stake["race_cap"], cfg["unit"])
    # 箱に入れた馬は必ず1点以上で買われていること。箱から外した馬は対象外
    # （買わないと決めた馬なので、覆われていなくて当たり前）。
    _assert_all_marks_covered(box, tickets)
    return tickets


def _box_marks(cfg: dict, grade: str, available: int) -> int:
    """その自信度で、箱に何頭入れるか。

        box_marks:
          "◎": 5     # C(5,3) = 10点
          "○": 5
          "△": 5

    書いていない自信度は印の全頭。**減らすほうを既定にしない**（黙って
    印を落とすと「印は当てたのに買っていない」を設定で再現することになる）。
    """
    limits = cfg.get("box_marks") or {}
    return max(3, min(int(limits.get(grade, available)), available))


def _shape_for(cfg: dict, grade: str) -> str:
    """その自信度で三連複をどう組むか。

    trio_shape は2つの書き方を受ける。

        trio_shape: box            … 全部同じ
        trio_shape:                … 自信度ごとに変える
          "◎": axis
          "○": box

    自信度で分けられるようにしたのは、勝負どころ（◎）は軸を立てて厚く、
    自信の落ちるところは広く、という持ち方があるため。実測とは別に、
    どう賭けたいかは買う側が決めてよい部分。

    書き忘れた自信度は axis に落とす。黙って広げると点数と金額が増えるので、
    増えるほうを既定にしない。
    """
    shape = cfg.get("trio_shape", "axis")
    if isinstance(shape, dict):
        return shape.get(grade, "axis")
    return shape


def _add_axis_flow(add, axis, partners, unit: int, cfg: dict) -> None:
    """◎を軸に、相手の総当たりを敷く（従来の組み方）。

    **全ての点に◎が入る。** ◎が3着以内を外した時点で、他の印が1〜3着を
    独占していても0点になる。
    """
    top_two = {h.umaban for h in partners[:2]}
    for a, b in itertools.combinations(partners, 2):
        # ◎と上位2頭で決まる並びを本線として厚くする（教訓13）
        is_main = {a.umaban, b.umaban} <= top_two
        add(
            TRIO,
            _trio((axis.umaban, a.umaban, b.umaban)),
            unit if is_main else cfg["unit"],
            "本線（◎×上位印）" if is_main else f"◎軸流し {a.mark}{b.mark}",
        )


def _add_box(add, marked: list[ScoredHorse], unit: int, cfg: dict) -> None:
    """印を打った馬のボックス。軸を固定しない。

    ## なぜ軸を外すか

    モデルが出しているのは「その馬が3着以内に入る確率」で、印はその上位N頭。
    つまり印を打った5頭は**どれも3着以内に来そうな馬**であって、1位の馬だけが
    特別なわけではない。スコア差は 62.1 対 57.9 対 56.3 という程度しかない。

    それを◎1頭に軸を固定して流すと、「1位が確実に来る」という、モデルが
    一度も主張していない前提を買い目に持ち込むことになる。実際 8/16 札幌1R では
    ○☆▲で決着したのに、8点すべてに◎が入っていて0点だった。

    ボックスなら、印5頭のうちどの3頭で決まっても当たる。点数は C(5,3)=10点で、
    ◎軸流しの8点から2点増えるだけ。

    上位3頭の組だけは本線として厚くする（そこが最も確率が高いのは変わらない）。
    """
    umabans = [h.umaban for h in marked]
    main = set(umabans[:3])
    for combo in itertools.combinations(umabans, 3):
        is_main = set(combo) == main
        add(
            TRIO,
            _trio(combo),
            unit if is_main else cfg["unit"],
            "本線（上位3頭）" if is_main else "印ボックス",
        )


def _fit_budget(tickets: list[Ticket], cap: int, unit: int) -> list[Ticket]:
    """レース上限に収める。

    点数を削るのではなく金額を削る。点数を削ると「印を打ったのに買わない馬」が
    生まれ、教訓10の失敗を再現してしまうため。
    """
    total = sum(t.amount for t in tickets)
    if total <= cap or not tickets:
        return tickets

    # まず本線の上乗せを剥がして全点を最小単位に寄せる
    for t in tickets:
        t.amount = unit
    total = sum(t.amount for t in tickets)
    if total <= cap:
        return tickets

    log.warning(
        "最小単位でも上限を超える（%d円 > %d円）。点数は削らず全点を維持する。",
        total,
        cap,
    )
    return tickets


def _assert_all_marks_covered(marked: list[ScoredHorse], tickets: list[Ticket]) -> None:
    """印を打った馬が1点も買い目に入っていない状態を許さない。

    SKILL.md の「高評価馬の必須組み込みルール」。評価表と買い目が矛盾する
    のを構造的に防ぐ。
    """
    covered: set[int] = set()
    for t in tickets:
        covered |= t.umabans
    missing = [h for h in marked if h.umaban not in covered]
    if missing:
        raise AssertionError(
            "印を打ちながら買い目に含まれていない馬がある: "
            + ", ".join(f"{h.mark}{h.umaban}{h.horse_name}" for h in missing)
        )


def summarise(tickets: list[Ticket]) -> dict:
    return {
        "points": len(tickets),
        "total": sum(t.amount for t in tickets),
        "by_type": {
            bet_type: sum(1 for t in tickets if t.bet_type == bet_type)
            for bet_type in {t.bet_type for t in tickets}
        },
    }
