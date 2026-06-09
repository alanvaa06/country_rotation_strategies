# TODO — Country Rotation Strategy Program

## Phase 0 — Housekeeping
- [in_progress] Checkpoint commits (legacy deletions / code changes / docs) on `dev`
- [pending] Create public GitHub repo `country_rotation_strategies`, push `dev` + `main`
- [done] Bootstrap context docs (todo, memory, sesion-log)

## Phase 1 — Literature Research
- [pending] /deep-research: country rotation strategies (academic + practitioner evidence)
- [pending] Document findings in `docs/research/country_rotation_literature.md`

## Phase 2 — Thesis Validation + Refactor
- [pending] Brainstorm + spec: package architecture (ingestion / processing / exploration)
- [pending] Look-ahead & data-integrity audit of current pipeline
- [in_progress] Refactor to `country_rotation/` package preserving current features (T6 done: factors/transforms.py)
- [pending] Engine unit tests (pytest): turnover, TC, selection, weights, alignment

## Phase 3 — Feature Selection
- [pending] Propose feature selection process (IC-based, redundancy-aware, OOS-honest)
- [pending] Self-verify: walk-forward IC stability, multiple-testing adjustment

## Phase 4 — Signal Engineering
- [pending] Compose final signal from building blocks (Valuation/Quality/Profitability/Momentum)
- [pending] Signal decomposition + attribution to blocks

## Phase 5 — Backtest Engine Enhancement
- [pending] Benchmark management (cap-weight / equal-weight nulls)
- [pending] Relative (active) strategy support: TE, IR, active weights
- [pending] Validation harness: DSR, PSR, walk-forward, MC null

## Phase 6 — Visualization Report
- [pending] Performance abs/rel, risk abs/rel, IC analysis, score building-block decomposition

## Phase 7 — Strategy Selection
- [pending] Explore dataset, select profitable strategy per EM / DM / World
- [pending] Evidence-based verdict with statistical certification
