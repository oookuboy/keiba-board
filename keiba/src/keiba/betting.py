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
    # 穴枠は実測回収率が本線より明確に低い。組み合わせを広げず少点数に絞る
    max_points = cfg["longshot_max_points"] if confidence.is_longshot else None

    marked = [h for h in horses if h.mark]
    axis = next((h for h in marked if h.mark == "◎"), None)
    if axis is None or len(marked) < 3:
        return []
    partners = [h for h in marked if h is not axis]

    tickets: list[Ticket] = []
    seen: set[tuple[str, str]] = set()

    def add(bet_type: str, combo: str, amount: int, why: str) -> None:
        key = (bet_type, combo)
        if key in seen:
            return
        seen.add(key)
        tickets.append(Ticket(bet_type, combo, amount, why))

    # --- 本線: ◎軸の総当たり（教訓10） ---------------------------------
    # 相手が5頭以上でも組み合わせを選別しない。C(5,2)=10点を100円ずつ
    # 敷いても予算内に収まるので、選別して取りこぼすほうが高くつく。
    top_two = {h.umaban for h in partners[:2]}
    for a, b in itertools.combinations(partners, 2):
        combo = _trio((axis.umaban, a.umaban, b.umaban))
        # ◎と上位2頭で決まる並びを本線として厚くする（教訓13）
        is_main = {a.umaban, b.umaban} <= top_two
        add(
            TRIO,
            combo,
            unit if is_main else cfg["unit"],
            "本線（◎×上位印）" if is_main else f"◎軸流し {a.mark}{b.mark}",
        )

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

    if max_points is not None and len(tickets) > max_points:
        # 穴枠を絞るときは ◎ と ☆ を含む点を優先して残す。
        # 印を打った馬が全部落ちないよう、covered を見ながら削る。
        tickets = _trim_longshot(tickets, marked, axis, longshot, max_points)

    tickets = _fit_budget(tickets, stake["race_cap"], cfg["unit"])
    _assert_all_marks_covered(marked, tickets)
    return tickets


def _trim_longshot(
    tickets: list[Ticket],
    marked: list[ScoredHorse],
    axis: ScoredHorse,
    longshot: ScoredHorse | None,
    max_points: int,
) -> list[Ticket]:
    """穴枠の点数を絞る。

    ただし「印を打ったのに買い目に無い馬」を作らない（教訓10 で最も高くついた
    失敗）。全印を覆うのに必要な点は max_points を超えても残す。
    """
    def priority(t: Ticket) -> tuple[int, int]:
        # ☆ を含む点を先に残し、次に本線、最後に流し
        has_longshot = longshot is not None and longshot.umaban in t.umabans
        return (0 if has_longshot else 1, 0 if "本線" in t.rationale else 1)

    ordered = sorted(tickets, key=priority)
    kept: list[Ticket] = []
    covered: set[int] = set()
    needed = {h.umaban for h in marked}

    for ticket in ordered:
        if len(kept) < max_points or not needed <= covered:
            kept.append(ticket)
            covered |= ticket.umabans
        if len(kept) >= max_points and needed <= covered:
            break
    return kept


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
