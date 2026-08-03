"""データモデル。

スクレイパの出力とエンジンの入力をこの型で固定する。HTMLの構造が変わっても
ここが変わらなければ下流は無傷で済む。

重要な約束: Entry.odds と Entry.popularity は「記録専用」であり、
features.py / engine.py からは参照しない。オッズを能力評価に混ぜないという
SKILL.md の絶対ルールを型のレベルで意識させるため、両フィールドには
market_ プレフィックスを付けてある。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from typing import Any, Self

# 競馬場コード（netkeiba race_id の 5-6 桁目）
VENUES = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}

# 脚質。過去走のコーナー通過順から推定する
RUNNING_STYLES = ("逃げ", "先行", "差し", "追込")


class _Row:
    """dataclass と dict / SQLite 行の相互変換。"""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if isinstance(value, date):
                value = value.isoformat()
            elif isinstance(value, list):
                value = json.dumps(value, ensure_ascii=False)
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Self:
        kwargs: dict[str, Any] = {}
        known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
        for name, f in known.items():
            if name not in row:
                continue
            value = row[name]
            if value is None:
                kwargs[name] = None
                continue
            if "date" in name and isinstance(value, str):
                kwargs[name] = date.fromisoformat(value)
            elif "list" in str(f.type) and isinstance(value, str):
                kwargs[name] = json.loads(value)
            else:
                kwargs[name] = value
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass
class Race(_Row):
    race_id: str            # netkeiba 形式 202605021211
    race_date: date
    venue: str              # 東京
    venue_code: str         # 05
    kai: int                # 回次
    nichi: int              # 日次
    race_no: int
    name: str
    surface: str            # 芝 / ダ / 障
    distance: int
    grade: str | None = None       # G1 G2 G3 L OP
    race_class: str | None = None  # 3勝クラス 未勝利 新馬 …
    direction: str | None = None   # 右 / 左 / 直線
    going: str | None = None       # 良 / 稍重 / 重 / 不良
    weather: str | None = None
    field_size: int | None = None
    post_time: str | None = None
    prize: int | None = None       # 1着賞金（万円）


@dataclass
class Entry(_Row):
    race_id: str
    umaban: int
    horse_name: str
    horse_id: str
    waku: int | None = None
    sex: str | None = None          # 牡 牝 セ
    age: int | None = None
    weight_carried: float | None = None
    jockey: str | None = None
    jockey_id: str | None = None
    trainer: str | None = None
    trainer_id: str | None = None
    affiliation: str | None = None  # 美浦 / 栗東 / 地方 / 外国
    body_weight: int | None = None
    body_weight_diff: int | None = None
    sire: str | None = None
    dam: str | None = None
    damsire: str | None = None
    scratched: bool = False

    # --- 以下は記録専用。エンジンからは絶対に参照しない ---
    market_odds: float | None = None
    market_popularity: int | None = None


@dataclass
class Result(_Row):
    race_id: str
    umaban: int
    finish_pos: int | None          # None は中止・除外・失格
    time_sec: float | None = None
    margin: str | None = None
    corners: list[int] = field(default_factory=list)  # 通過順
    last3f: float | None = None
    body_weight: int | None = None
    body_weight_diff: int | None = None


@dataclass
class Payout(_Row):
    race_id: str
    bet_type: str       # 単勝 複勝 馬連 ワイド 馬単 三連複 三連単
    combination: str    # "1-5-16"
    payout: int         # 100円あたり
    popularity: int | None = None


@dataclass
class Workout(_Row):
    """最終追い切り。"""

    race_id: str
    umaban: int
    workout_date: date | None = None
    course: str | None = None       # 美浦W 栗東CW 坂路 …
    times: list[float] = field(default_factory=list)  # 6F-5F-4F-3F-1F
    position: str | None = None     # 併せ馬の着順（先着/同入/遅れ）
    evaluation: str | None = None   # 一杯 強め 馬なり
    rank: str | None = None         # netkeiba の追い切り評価 S A B C


@dataclass
class TrainerComment(_Row):
    race_id: str
    umaban: int
    body: str
    source: str | None = None
    fetched_at: str | None = None


@dataclass
class PastRun(_Row):
    """馬柱の1行。backfill 期間外・地方・海外の走りを埋めるために使う。

    backfill 期間内のレースは races×results から復元できるので、これは補完用。
    """

    horse_id: str
    run_date: date
    venue: str
    race_name: str
    surface: str
    distance: int
    finish_pos: int | None
    race_id: str | None = None
    grade: str | None = None
    going: str | None = None
    field_size: int | None = None
    umaban: int | None = None
    jockey: str | None = None
    weight_carried: float | None = None
    time_sec: float | None = None
    margin: str | None = None
    corners: list[int] = field(default_factory=list)
    last3f: float | None = None
    body_weight: int | None = None
    market_popularity: int | None = None


@dataclass
class RaceCard:
    """1レース分の予想入力一式。スクレイパの最終出力であり、エンジンの唯一の入力。"""

    race: Race
    entries: list[Entry] = field(default_factory=list)
    workouts: list[Workout] = field(default_factory=list)
    comments: list[TrainerComment] = field(default_factory=list)
    past_runs: list[PastRun] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)
    payouts: list[Payout] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "race": self.race.to_dict(),
            "entries": [e.to_dict() for e in self.entries],
            "workouts": [w.to_dict() for w in self.workouts],
            "comments": [c.to_dict() for c in self.comments],
            "past_runs": [p.to_dict() for p in self.past_runs],
            "results": [r.to_dict() for r in self.results],
            "payouts": [p.to_dict() for p in self.payouts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RaceCard:
        return cls(
            race=Race.from_dict(data["race"]),
            entries=[Entry.from_dict(d) for d in data.get("entries", [])],
            workouts=[Workout.from_dict(d) for d in data.get("workouts", [])],
            comments=[TrainerComment.from_dict(d) for d in data.get("comments", [])],
            past_runs=[PastRun.from_dict(d) for d in data.get("past_runs", [])],
            results=[Result.from_dict(d) for d in data.get("results", [])],
            payouts=[Payout.from_dict(d) for d in data.get("payouts", [])],
        )

    def live_entries(self) -> list[Entry]:
        """取消・除外を除いた実際の出走馬。"""
        return [e for e in self.entries if not e.scratched]


def parse_race_id(race_id: str) -> tuple[int, str, int, int, int]:
    """race_id を (年, 場コード, 回次, 日次, R) に分解する。"""
    if len(race_id) != 12 or not race_id.isdigit():
        raise ValueError(f"race_id の形式が不正: {race_id!r}")
    return (
        int(race_id[0:4]),
        race_id[4:6],
        int(race_id[6:8]),
        int(race_id[8:10]),
        int(race_id[10:12]),
    )


def dataclass_to_json(obj: Any) -> str:
    return json.dumps(asdict(obj), ensure_ascii=False, default=str)
