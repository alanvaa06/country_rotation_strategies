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
- [done] Refactor to `country_rotation/` package preserving current features — Plan A complete (T1-T13: config, ingestion, processing, integrity, transforms, catalog, composite, metrics, ic, engine, parity, redundancy/benchmarks, CLI scripts)
- [done] Engine unit tests (pytest): 90/90 green including end-to-end CLI script smoke tests
- [done] Final-review fixes: CLI wiring, strict metric weights, cleanup queue, script smoke tests

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
- [done] Performance abs/rel, risk abs/rel, IC analysis, score building-block decomposition

## Phase 7 — Strategy Selection
- [in_progress] Explore dataset, select profitable strategy per EM / DM / World — research driver (scripts/research_run.py) done; World @63d screen kept 0/45 (FDR-binding); DM/EM runs + screening-policy decision pending
- [pending] Evidence-based verdict with statistical certification
- [pending] DECISION NEEDED: BH-FDR family definition (45-factor family kills everything at q=0.10; per-category families / fdr_q / horizon are the levers)
