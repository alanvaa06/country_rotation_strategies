# TODO — Country Rotation Strategy Program

## Phase 0 — Housekeeping
- [done] Checkpoint commits on `dev`; public repo github.com/alanvaa06/country_rotation_strategies; dev+main pushed
- [done] Bootstrap context docs (todo, memory, lessons, results, sesion-log, PRD)

## Phase 1 — Literature Research
- [done] Deep-research: 23 sources, 25 claims adversarially verified → docs/research/country_rotation_literature.md

## Phase 2 — Thesis Validation + Refactor
- [done] Design spec (docs/superpowers/specs/2026-06-09) + Plan A executed (T1-T13): package, leak fixes, catalog, engine parity-locked, CLI scripts, 90+ tests
- [done] Look-ahead audit: bfill leak fixed (ffill-only), perturbation guards, publication-lag config

## Phase 3 — Feature Selection
- [done] OOS-honest screening: per-period IC t-stats, BH-FDR q=0.10, HLZ weak labels, lockbox, min_mean_ic gate (Plan B B3 + power fix)

## Phase 4 — Signal Engineering
- [done] Composite from building blocks with rebased contributions; literature-prior tracks (full 4-category + 50/50 V+M)

## Phase 5 — Backtest Engine Enhancement
- [done] Active mode, equal-weight null, validation harness (DSR/PSR/WF/MC/bootstrap/NW) — Plan B B1/B2/B4

## Phase 6 — Visualization Report
- [done] HTML report: performance abs/rel, risk abs/rel, IC, building-block decomposition, scorecard (Plan C)

## Phase 7 — Strategy Selection
- [done] 8 pre-registered runs World/DM/EM × screen/prior(/vm) → docs/research/segment_verdicts_2026-06-09.md
- [done] Verdict: NO certification at pre-registered gates; DM 50/50 V+M nearest (t 1.96, MC p 0.059, DSR 0.91); EM clear negative; honest-negative documented

## Next (future sessions)
- [blocked] Extend price/fundamental history pre-2010 — NO vendor access to earlier periods (confirmed 2026-06-09). Revisit only if MSCI/index proxies become available.
- [done] Pre-registered robustness: DM prior-vm @21d monthly — monthly DEGRADES vs quarterly; ruled out (docs/research/segment_verdicts_2026-06-09.md Addendum §1)
- [done] Quarterly re-run protocol set up + determinism verified bit-for-bit — runbook docs/research/RUNBOOK_quarterly_recert.md; run each quarter after Inputs/ refresh
- [done] Cap-weighted mandate benchmark (--bmk-source index) + blended configs (--mode blend 0.3/0.5) — tested + self-verified; selection skill MC-significant, IC significant (World rel @63 t 2.29), IR vs cap negative (construction tilt)
- [done] PRE-REGISTERED NEXT: benchmark-aware construction — cap-index base weights ± composite-score tilts under a TE budget; re-run DM/World vs cap index; expect realized IR to reflect the MC-significant selection skill
  - [done] Cap_Tilt engine branch (Engine(base_weights=...), active_share budget, long-only clip, sum-1 conservation) + base_weights threaded through parameter_sweep/walk_forward/monte_carlo_null/compute_validation/build_report; 7 new tests, suite 132 green incl. parity
  - [done] Per-segment Cap_Tilt runs (DM/World/EM vs cap index) via research_run.py --construction cap_tilt; verdict JSONs in outputs/research/
- [done] Multi-strategy toggle dashboard — reporting/dashboard.py (build_strategy_pane + render_dashboard, segment tabs World/DM/EM + strategy toggle, default Cap_Tilt) + scripts/build_dashboard.py (9 panes, one self-contained HTML); 6 TDD tests
- [pending] Consider deleting legacy root scripts (parity lock would need frozen fixtures first — see lessons)
