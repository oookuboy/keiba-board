# 中央競馬 自動予想システム

中央競馬（JRA）の全レースを記録し、予想する。`/root/.claude/skills/horse-racing-prediction/SKILL.md`
に蓄積された手動プレイブックの14教訓をコードにしたもの。

**狙いは穴・大穴。堅く収まると判断したレースは買わない。**

## 設計の核: オッズの扱い

オッズを使ってよい場所を2層に厳密に分けてある。

| 層 | モジュール | オッズ・人気 |
|---|---|---|
| 能力評価 | `features.py` / `engine.py` | **一切使わない**。特徴量に入れない |
| レース選別・配分 | `confidence.py` / `betting.py` | 妙味フィルタとしてのみ使う |

`Entry` のオッズ関連は `market_odds` / `market_popularity` と接頭辞で隔離してあり、
「人気だけを入れ替えても能力スコアが1点も動かない」ことを
`tests/test_engine.py::test_odds_never_influence_ability_score` で保証している。

これにより「純粋な能力・適性で全頭フラットに評価する」という SKILL.md の絶対
ルールを守ったまま、「固い予想はいらない」を機械的に実現している。

## 自信度 ◎○△×

レース単位。能力評価の上位3頭の**人気順位の和**が大きいほど、世間の評価と
ズレている＝穴。

| 判定 | 条件 | 意味 |
|---|---|---|
| `◎` | 人気和 ≥ 20 かつスコア分離が明確 | 穴で勝負。買い目を厚く |
| `○` | 人気和 ≥ 14 | 妙味あり。標準配分 |
| `△` | 人気和 ≥ 9 | 一応買える。薄く |
| `×` | 人気和 < 9 | **堅い決着。買い目を出さない** |

想定3連複配当が閾値（既定50倍）未満のレースも `×` へ降格する。
全レースを予想・記録するが、買うのは `◎○△` だけ。

## 教訓の実装箇所

| 教訓 | 実装 |
|---|---|
| 1 血統で馬場替わりを拾う | `features.surface_switch_bonus` — 種牡馬適性表から「今走の馬場で父は走るのに前走はその馬場を使えていなかった」形を検出。Palace Pier を名指しせず同種の父を全部拾う |
| 2 展開読みを核心に | `features.project_pace` — 全頭そろってから評価 |
| 5・6 穴を必ず残す | `betting.build` — ☆ を含む買い目を必ず1点 |
| 7 少頭数でハイペースにしない | `features.project_pace` — 10頭以下は先行有利が基本 |
| 8 確逃げ馬を切らない | `engine.apply_floors` のスコア床 ＋ 3着欄に必ず1点 |
| 10 印5頭以上なら全組み合わせ | `betting.build` — ◎軸の C(n,2) を必ず敷く。予算超過時は点数でなく金額を削る |
| 11 実績馬を休み明けで切らない | `engine.apply_floors` のスコア床 |
| 12 降級・格上帰りを上位に | `features.form_score` |
| 必須ルール | `betting._assert_all_marks_covered` — 印を打った馬が買い目に無ければ落とす |

## データ源

| 用途 | 経路 | 状態 |
|---|---|---|
| 過去成績・払戻 | `db.netkeiba.com/race/{race_id}/` | 検証済み。サーバーレンダリング |
| 血統（父・母父） | `db.netkeiba.com/horse/ped/{horse_id}/` | 検証済み |
| 出馬表・調教（ライブ） | JRA公式 `accessD/accessT` | **未完**。出馬表は木曜公開のため実物を採取できていない |
| 調教師コメント | — | **取得不能**。netkeiba は有料プラン限定 |

`race.netkeiba.com`（出馬表・馬柱・結果・調教）は JavaScript レンダリングの
空シェルで、`requests` では表のヘッダしか返らない。収集源として使えない。

`www.jra.go.jp/robots.txt` は `Disallow:` が空で全面許可。netkeiba は
robots.txt を公開していない（404）。いずれに対しても 1.5秒ウェイトと
ディスクキャッシュを `sources/http.py` で実装レベルで強制している。

## 使い方

```bash
pip install -e ".[dev]"

# 過去データ（GitHub Actions の keiba-backfill から実行する）
python -m keiba.cli backfill-races --from 2023-01-01 --to 2023-12-31
python -m keiba.cli backfill-pedigree --offset 0 --limit 7000
python -m keiba.cli build            # raw → SQLite → sire_aptitude.json

# 日常運用（keiba-weekend が cron で叩く）
python -m keiba.cli collect  --date 2026-08-08 --days-ahead 3
python -m keiba.cli predict  --date 2026-08-08
python -m keiba.cli review   --date 2026-08-08

# 検証
python -m pytest keiba/tests -q
python -m keiba.cli backtest --from 2026-01-01 --to 2026-07-31
```

## ファイル構成

```
keiba/
  board.html            閲覧UI（単一ファイル・依存なし）
  data/                 ボードが読む JSON（predict/review が生成）
  raw/YYYY/*.jsonl.gz   生データの正。SQLite はここから毎回作り直す使い捨て
  raw/pedigree.jsonl.gz 血統（馬単位）
  config/weights.yml    全重み。チューニングはここだけを触る
  config/sire_aptitude.json  種牡馬適性表（build が生成）
  LESSONS.md            review が追記する実戦ログ
  src/keiba/            実装
  tests/fixtures/       probe が採取した実HTML
```

## 注意

- 競馬は不確実性を伴う賭博であり、このシステムは的中を保証しない。バックテストの
  回収率が100%を超えても将来の収益を意味しない
- Actions の cron は数十分遅れて発火する。馬体重は発走約50分前確定なので、
  発走直前の情報を当て込む運用は成立しない。前日夜に予想を確定し、当日朝に
  馬場状態と取消だけを反映して作り直す
- `weights.yml` の初期値は SKILL.md の記述に基づく手動設定であり、データ駆動では
  ない。バックテストで調整すること
