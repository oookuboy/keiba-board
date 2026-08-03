"""データストア。

正は keiba/raw/YYYY/*.jsonl.gz（テキストなので git の差分が効く）。SQLite は
そこから毎回再構築する使い捨てのインデックスで、コミットしない。バイナリDBを
毎日コミットするとリポジトリが肥大するのを避けるため。

エンジンが必要とする集計（種牡馬適性・騎手成績・馬の過去走）はすべてここに
問い合わせる。features.py が生SQLを書かなくて済むようにするのが役割。
"""

from __future__ import annotations

import gzip
import json
import logging
import sqlite3
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path

from keiba.models import (
    Entry,
    PastRun,
    Payout,
    Race,
    RaceCard,
    Result,
    TrainerComment,
    Workout,
)

log = logging.getLogger(__name__)

# 種牡馬適性を集計する距離帯。細かく切るとサンプルが枯れる。
DISTANCE_BANDS = (
    ("sprint", 0, 1400),
    ("mile", 1401, 1800),
    ("middle", 1801, 2200),
    ("long", 2201, 9999),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id    TEXT PRIMARY KEY,
    race_date  TEXT NOT NULL,
    venue      TEXT NOT NULL,
    venue_code TEXT NOT NULL,
    kai        INTEGER, nichi INTEGER, race_no INTEGER,
    name       TEXT, surface TEXT, distance INTEGER,
    grade      TEXT, race_class TEXT, direction TEXT,
    going      TEXT, weather TEXT, field_size INTEGER,
    post_time  TEXT, prize INTEGER
);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date);

CREATE TABLE IF NOT EXISTS entries (
    race_id TEXT NOT NULL, umaban INTEGER NOT NULL,
    horse_name TEXT, horse_id TEXT, waku INTEGER,
    sex TEXT, age INTEGER, weight_carried REAL,
    jockey TEXT, jockey_id TEXT, trainer TEXT, trainer_id TEXT,
    affiliation TEXT, body_weight INTEGER, body_weight_diff INTEGER,
    sire TEXT, dam TEXT, damsire TEXT, scratched INTEGER DEFAULT 0,
    market_odds REAL, market_popularity INTEGER,
    PRIMARY KEY (race_id, umaban)
);
CREATE INDEX IF NOT EXISTS idx_entries_horse  ON entries(horse_id);
CREATE INDEX IF NOT EXISTS idx_entries_jockey ON entries(jockey_id);
CREATE INDEX IF NOT EXISTS idx_entries_sire   ON entries(sire);

CREATE TABLE IF NOT EXISTS results (
    race_id TEXT NOT NULL, umaban INTEGER NOT NULL,
    finish_pos INTEGER, time_sec REAL, margin TEXT,
    corners TEXT, last3f REAL,
    body_weight INTEGER, body_weight_diff INTEGER,
    PRIMARY KEY (race_id, umaban)
);

CREATE TABLE IF NOT EXISTS payouts (
    race_id TEXT NOT NULL, bet_type TEXT NOT NULL, combination TEXT NOT NULL,
    payout INTEGER, popularity INTEGER,
    PRIMARY KEY (race_id, bet_type, combination)
);

CREATE TABLE IF NOT EXISTS workouts (
    race_id TEXT NOT NULL, umaban INTEGER NOT NULL,
    workout_date TEXT, course TEXT, times TEXT,
    position TEXT, evaluation TEXT, rank TEXT,
    PRIMARY KEY (race_id, umaban)
);

CREATE TABLE IF NOT EXISTS comments (
    race_id TEXT NOT NULL, umaban INTEGER NOT NULL,
    body TEXT, source TEXT, fetched_at TEXT,
    PRIMARY KEY (race_id, umaban)
);

-- backfill 期間外・地方・海外の走り。races×results で復元できない分の補完。
CREATE TABLE IF NOT EXISTS past_runs (
    horse_id TEXT NOT NULL, run_date TEXT NOT NULL,
    venue TEXT, race_name TEXT, surface TEXT, distance INTEGER,
    finish_pos INTEGER, race_id TEXT, grade TEXT, going TEXT,
    field_size INTEGER, umaban INTEGER, jockey TEXT, weight_carried REAL,
    time_sec REAL, margin TEXT, corners TEXT, last3f REAL,
    body_weight INTEGER, market_popularity INTEGER,
    PRIMARY KEY (horse_id, run_date, race_name)
);
CREATE INDEX IF NOT EXISTS idx_past_horse ON past_runs(horse_id);

CREATE TABLE IF NOT EXISTS predictions (
    race_id TEXT PRIMARY KEY,
    generated_at TEXT, engine_version TEXT,
    confidence TEXT,          -- ◎ ○ △ ×
    payload TEXT              -- 全頭のスコア・印・根拠（JSON）
);

CREATE TABLE IF NOT EXISTS bets (
    race_id TEXT NOT NULL, bet_type TEXT NOT NULL, combination TEXT NOT NULL,
    amount INTEGER, rationale TEXT,
    PRIMARY KEY (race_id, bet_type, combination)
);

CREATE TABLE IF NOT EXISTS reviews (
    race_id TEXT PRIMARY KEY,
    hit INTEGER, spent INTEGER, returned INTEGER,
    lessons_worked TEXT, lessons_failed TEXT, note TEXT
);
"""


def band_of(distance: int) -> str:
    for name, lo, hi in DISTANCE_BANDS:
        if lo <= distance <= hi:
            return name
    return "long"


class Store:
    def __init__(self, path: Path | str = "keiba/keiba.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        # 数万行を流し込むので、耐障害性より速度を取る（壊れたら作り直せばよい）
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- 書き込み

    def _upsert(self, table: str, rows: Iterable[dict]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        cols = list(rows[0])
        sql = (
            f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})"
        )
        self.conn.executemany(sql, [[r[c] for c in cols] for r in rows])
        return len(rows)

    def save_card(self, card: RaceCard) -> None:
        self._upsert("races", [card.race.to_dict()])
        self._upsert("entries", [e.to_dict() for e in card.entries])
        self._upsert("results", [r.to_dict() for r in card.results])
        self._upsert("payouts", [p.to_dict() for p in card.payouts])
        self._upsert("workouts", [w.to_dict() for w in card.workouts])
        self._upsert("comments", [c.to_dict() for c in card.comments])
        self._upsert("past_runs", [p.to_dict() for p in card.past_runs])

    def save_cards(self, cards: Iterable[RaceCard]) -> int:
        count = 0
        for card in cards:
            self.save_card(card)
            count += 1
        self.conn.commit()
        return count

    # --------------------------------------------------------------- 読み出し

    def race_ids(self, since: date | None = None) -> list[str]:
        sql = "SELECT race_id FROM races"
        args: list = []
        if since:
            sql += " WHERE race_date >= ?"
            args.append(since.isoformat())
        sql += " ORDER BY race_date, race_id"
        return [r[0] for r in self.conn.execute(sql, args)]

    def load_card(self, race_id: str) -> RaceCard | None:
        row = self.conn.execute(
            "SELECT * FROM races WHERE race_id = ?", (race_id,)
        ).fetchone()
        if row is None:
            return None

        def rows(table: str) -> list[dict]:
            return [
                dict(r)
                for r in self.conn.execute(
                    f"SELECT * FROM {table} WHERE race_id = ?", (race_id,)
                )
            ]

        return RaceCard(
            race=Race.from_dict(dict(row)),
            entries=[Entry.from_dict(d) for d in rows("entries")],
            results=[Result.from_dict(d) for d in rows("results")],
            payouts=[Payout.from_dict(d) for d in rows("payouts")],
            workouts=[Workout.from_dict(d) for d in rows("workouts")],
            comments=[TrainerComment.from_dict(d) for d in rows("comments")],
        )

    def past_runs_for(
        self, horse_id: str, before: date, limit: int = 12
    ) -> list[PastRun]:
        """指定日より前の走りを新しい順で返す。

        backfill 済みのレース（races×results）と、馬柱由来の補完（past_runs）を
        束ねる。重複は日付＋レース名で寄せる。
        """
        merged: dict[tuple[str, str], PastRun] = {}

        sql = """
        SELECT r.race_date, r.venue, r.name AS race_name, r.surface, r.distance,
               r.grade, r.going, r.field_size, r.race_id,
               e.umaban, e.jockey, e.weight_carried, e.market_popularity,
               res.finish_pos, res.time_sec, res.margin, res.corners,
               res.last3f, res.body_weight
        FROM entries e
        JOIN races r   ON r.race_id = e.race_id
        LEFT JOIN results res ON res.race_id = e.race_id AND res.umaban = e.umaban
        WHERE e.horse_id = ? AND r.race_date < ? AND e.scratched = 0
        ORDER BY r.race_date DESC LIMIT ?
        """
        for row in self.conn.execute(sql, (horse_id, before.isoformat(), limit)):
            d = dict(row)
            d["horse_id"] = horse_id
            d["run_date"] = d.pop("race_date")
            merged[(d["run_date"], d["race_name"] or "")] = PastRun.from_dict(d)

        for row in self.conn.execute(
            "SELECT * FROM past_runs WHERE horse_id = ? AND run_date < ?"
            " ORDER BY run_date DESC LIMIT ?",
            (horse_id, before.isoformat(), limit),
        ):
            d = dict(row)
            merged.setdefault(
                (d["run_date"], d["race_name"] or ""), PastRun.from_dict(d)
            )

        return sorted(merged.values(), key=lambda p: p.run_date, reverse=True)[:limit]

    # ------------------------------------------------------------ 集計クエリ

    def sire_aptitude(self, min_runs: int = 30) -> dict[str, dict[str, dict]]:
        """父ごとの 芝/ダ × 距離帯 の複勝率。

        SKILL.md の教訓1（ダート→芝替わりを血統から拾う）を計算可能にする土台。
        Palace Pier を名指しでハードコードする代わりに、同じ性質の父をデータから
        全て拾えるようにする。
        """
        sql = """
        SELECT e.sire, r.surface, r.distance,
               COUNT(*) AS runs,
               SUM(CASE WHEN res.finish_pos <= 3 THEN 1 ELSE 0 END) AS placed,
               SUM(CASE WHEN res.finish_pos =  1 THEN 1 ELSE 0 END) AS won
        FROM entries e
        JOIN races r    ON r.race_id = e.race_id
        JOIN results res ON res.race_id = e.race_id AND res.umaban = e.umaban
        WHERE e.sire IS NOT NULL AND e.sire <> '' AND res.finish_pos IS NOT NULL
        GROUP BY e.sire, r.surface, r.distance
        """
        acc: dict[str, dict[str, dict[str, int]]] = {}
        for row in self.conn.execute(sql):
            key = f"{row['surface']}:{band_of(row['distance'])}"
            bucket = acc.setdefault(row["sire"], {}).setdefault(
                key, {"runs": 0, "placed": 0, "won": 0}
            )
            bucket["runs"] += row["runs"]
            bucket["placed"] += row["placed"]
            bucket["won"] += row["won"]

        table: dict[str, dict[str, dict]] = {}
        for sire, buckets in acc.items():
            total = sum(b["runs"] for b in buckets.values())
            if total < min_runs:
                continue
            table[sire] = {
                key: {
                    "runs": b["runs"],
                    "place_rate": round(b["placed"] / b["runs"], 4),
                    "win_rate": round(b["won"] / b["runs"], 4),
                }
                for key, b in buckets.items()
                if b["runs"] > 0
            }
        return table

    def jockey_record(
        self, jockey_id: str, venue: str | None = None, since: date | None = None
    ) -> tuple[int, float, float]:
        """(騎乗数, 勝率, 複勝率)。サンプルが薄いときは呼び出し側で重みを落とす。"""
        sql = """
        SELECT COUNT(*) n,
               SUM(CASE WHEN res.finish_pos = 1 THEN 1 ELSE 0 END) w,
               SUM(CASE WHEN res.finish_pos <= 3 THEN 1 ELSE 0 END) p
        FROM entries e
        JOIN races r     ON r.race_id = e.race_id
        JOIN results res ON res.race_id = e.race_id AND res.umaban = e.umaban
        WHERE e.jockey_id = ? AND res.finish_pos IS NOT NULL
        """
        args: list = [jockey_id]
        if venue:
            sql += " AND r.venue = ?"
            args.append(venue)
        if since:
            sql += " AND r.race_date >= ?"
            args.append(since.isoformat())
        row = self.conn.execute(sql, args).fetchone()
        n = row["n"] or 0
        if not n:
            return 0, 0.0, 0.0
        return n, (row["w"] or 0) / n, (row["p"] or 0) / n

    def horse_jockey_record(self, horse_id: str, jockey_id: str) -> tuple[int, float]:
        """馬と騎手のコンビ成績 (騎乗数, 複勝率)。"""
        row = self.conn.execute(
            """
            SELECT COUNT(*) n, SUM(CASE WHEN res.finish_pos <= 3 THEN 1 ELSE 0 END) p
            FROM entries e
            JOIN results res ON res.race_id = e.race_id AND res.umaban = e.umaban
            WHERE e.horse_id = ? AND e.jockey_id = ? AND res.finish_pos IS NOT NULL
            """,
            (horse_id, jockey_id),
        ).fetchone()
        n = row["n"] or 0
        return (n, (row["p"] or 0) / n) if n else (0, 0.0)

    def counts(self) -> dict[str, int]:
        tables = ["races", "entries", "results", "payouts", "past_runs", "comments"]
        return {
            t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in tables
        }


# ------------------------------------------------------------------- JSONL

def write_jsonl(cards: Iterable[RaceCard], path: Path) -> int:
    """1行1レースの gzip JSONL。git 差分が効くよう安定した順序で書く。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for card in sorted(cards, key=lambda c: c.race.race_id):
            fh.write(json.dumps(card.to_dict(), ensure_ascii=False, sort_keys=True))
            fh.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[RaceCard]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield RaceCard.from_dict(json.loads(line))


def rebuild(store: Store, raw_dir: Path) -> int:
    """raw/*.jsonl.gz を全部読んで SQLite を作り直す。"""
    total = 0
    for path in sorted(raw_dir.rglob("*.jsonl.gz")):
        n = store.save_cards(read_jsonl(path))
        total += n
        log.info("%s: %d レース", path.name, n)
    return total
