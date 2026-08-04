"""過去データに対してエンジンを回し、的中率と回収率を出す。

先読みバイアスを避けるのが最大の要件。予想時点で知りえない情報を1つでも
混ぜると回収率は簡単に嘘をつくので、次の3つを守っている。

1. 種牡馬適性表は「バックテスト開始日より前」のレースだけで作る
2. 騎手成績・コンビ成績・過去走はレース当日より前だけを見る（as_of）
3. オッズと人気は、そのレースの確定値をそのまま使う。これは予想時点でも
   （締切直前には）観測できる値なので先読みではない

払戻は payouts テーブルの実配当を使う。100円あたりの金額なので、
賭け金 ÷ 100 を掛けて戻りを出す。
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from keiba import betting, confidence, engine
from keiba.features import build_features
from keiba.models import RaceCard
from keiba.store import Store

log = logging.getLogger(__name__)


@dataclass
class RaceOutcome:
    race_id: str
    race_date: date
    grade: str
    spent: int
    returned: int
    hit: bool
    hit_types: list[str] = field(default_factory=list)
    top3_popularity: int = 0
    payout: int = 0


@dataclass
class BacktestResult:
    outcomes: list[RaceOutcome] = field(default_factory=list)
    skipped: int = 0          # × で見送ったレース
    no_bet: int = 0           # 買い目を組めなかったレース
    errors: int = 0

    @property
    def bet_races(self) -> int:
        return len(self.outcomes)

    @property
    def spent(self) -> int:
        return sum(o.spent for o in self.outcomes)

    @property
    def returned(self) -> int:
        return sum(o.returned for o in self.outcomes)

    @property
    def hits(self) -> int:
        return sum(1 for o in self.outcomes if o.hit)

    @property
    def roi(self) -> float:
        """回収率。100%未満が普通であり、超えたからといって将来を保証しない。"""
        return (self.returned / self.spent * 100) if self.spent else 0.0

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.bet_races * 100) if self.bet_races else 0.0

    def report(self) -> str:
        by_grade = Counter(o.grade for o in self.outcomes)
        lines = [
            "=" * 62,
            "バックテスト結果",
            "=" * 62,
            f"  対象レース      {self.bet_races + self.skipped + self.no_bet:,} 件",
            f"    買った        {self.bet_races:,} 件  ({dict(by_grade)})",
            f"    × で見送り    {self.skipped:,} 件",
            f"    買い目組めず  {self.no_bet:,} 件",
            f"    エラー        {self.errors:,} 件",
            "-" * 62,
            f"  的中           {self.hits:,} 件  （的中率 {self.hit_rate:.1f}%）",
            f"  投資           {self.spent:,} 円",
            f"  払戻           {self.returned:,} 円",
            f"  収支           {self.returned - self.spent:+,} 円",
            f"  回収率         {self.roi:.1f}%",
        ]

        hits = [o for o in self.outcomes if o.hit]
        if hits:
            best = max(hits, key=lambda o: o.payout)
            lines += [
                "-" * 62,
                f"  最高配当       {best.payout:,} 円  ({best.race_id} "
                f"{best.race_date} 上位3頭人気和{best.top3_popularity})",
                f"  的中の平均人気和 {sum(o.top3_popularity for o in hits) / len(hits):.1f}",
            ]

        # 自信度ごとの内訳。◎ が本当に穴を取れているかはここで見る
        lines.append("-" * 62)
        for g in ("◎", "○", "△"):
            group = [o for o in self.outcomes if o.grade == g]
            if not group:
                continue
            spent = sum(o.spent for o in group)
            got = sum(o.returned for o in group)
            hit = sum(1 for o in group if o.hit)
            lines.append(
                f"  {g}  {len(group):5,}件  的中{hit:4,}件 "
                f"({hit / len(group) * 100:5.1f}%)  回収率 "
                f"{(got / spent * 100) if spent else 0:6.1f}%"
            )
        lines.append("=" * 62)
        return "\n".join(lines)


def _actual_top3(card: RaceCard) -> list[int] | None:
    """着順1〜3位の馬番。同着や中止で3頭揃わなければ None。"""
    placed = [r for r in card.results if r.finish_pos in (1, 2, 3)]
    if len(placed) != 3:
        return None
    return [r.umaban for r in sorted(placed, key=lambda r: r.finish_pos or 9)]


def _settle(
    tickets: list[betting.Ticket], card: RaceCard, top3: list[int]
) -> tuple[int, int, list[str], int]:
    """買い目を実結果と突き合わせて (投資, 払戻, 的中券種, 最高配当) を返す。"""
    payouts = {(p.bet_type, p.combination): p.payout for p in card.payouts}
    spent = sum(t.amount for t in tickets)
    returned = 0
    hit_types: list[str] = []
    best = 0

    trio_key = "-".join(str(n) for n in sorted(top3))
    trifecta_key = "-".join(str(n) for n in top3)

    for t in tickets:
        won = (t.bet_type == betting.TRIO and t.combination == trio_key) or (
            t.bet_type == betting.TRIFECTA and t.combination == trifecta_key
        )
        if not won:
            continue
        payout = payouts.get((t.bet_type, t.combination))
        if payout is None:
            # 的中しているのに払戻が無いのはデータ欠損。黙って0円にしない
            log.warning("払戻データ欠損: %s %s %s", card.race.race_id, t.bet_type, t.combination)
            continue
        returned += payout * t.amount // 100
        hit_types.append(t.bet_type)
        best = max(best, payout)

    return spent, returned, hit_types, best


def run(
    store: Store,
    weights: dict,
    start: date,
    end: date,
    sire_table: dict | None = None,
    ml_scores: dict[str, dict[int, float]] | None = None,
) -> BacktestResult:
    """期間内の全レースを予想し、実結果と突き合わせる。"""
    if sire_table is None:
        # 開始日より前のデータだけで適性表を作る。ここを全期間で作ると
        # 「未来の産駒成績で過去を予想する」ことになり回収率が嘘になる。
        log.info("種牡馬適性表を %s より前のデータで構築", start)
        sire_table = store.sire_aptitude(
            min_runs=weights["pedigree"]["min_sire_runs"], before=start
        )
        log.info("  → %d頭ぶん", len(sire_table))

    result = BacktestResult()
    race_ids = [
        r[0]
        for r in store.conn.execute(
            "SELECT race_id FROM races WHERE race_date >= ? AND race_date <= ?"
            " ORDER BY race_date, race_id",
            (start.isoformat(), end.isoformat()),
        )
    ]
    log.info("対象 %d レース", len(race_ids))

    for i, race_id in enumerate(race_ids, 1):
        if i % 500 == 0:
            log.info("  %d/%d  回収率 %.1f%%", i, len(race_ids), result.roi)

        card = store.load_card(race_id)
        if card is None or not card.results:
            continue
        top3 = _actual_top3(card)
        if top3 is None:
            continue

        try:
            entries = card.live_entries()
            features = build_features(
                card.race, entries, store, sire_table, weights,
                as_of=card.race.race_date,
            )
            horses = engine.run(
                features,
                {e.umaban: e for e in entries},
                weights,
                ml_scores.get(race_id) if ml_scores else None,
            )
            grade = confidence.grade(horses, weights)
            tickets = betting.build(horses, grade, weights)
        except (AssertionError, KeyError, ValueError) as exc:
            log.warning("%s の予想に失敗: %s", race_id, exc)
            result.errors += 1
            continue

        if not grade.should_bet:
            result.skipped += 1
            continue
        if not tickets:
            result.no_bet += 1
            continue

        spent, returned, hit_types, best = _settle(tickets, card, top3)
        pops = [
            e.market_popularity
            for e in entries
            if e.umaban in top3 and e.market_popularity
        ]
        result.outcomes.append(
            RaceOutcome(
                race_id=race_id,
                race_date=card.race.race_date,
                grade=grade.grade,
                spent=spent,
                returned=returned,
                hit=bool(hit_types),
                hit_types=hit_types,
                top3_popularity=sum(pops),
                payout=best,
            )
        )

    return result
