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
- [done] Engine 'top_n_level' selection (canonical AMP-2013 rank-of-level, all weighting branches incl. Cap_Tilt) + pre-registered 6-spec signal tournament (scripts/spec_tournament.py: screen-window IC + BH-FDR + one-shot lockbox); winners World/DM = S5_amp_ey_change (q .019/.093), EM = S1 (not significant); 145 tests green
- [done] Winner scorecard runs + verdict — research_run.py `--signal amp_ey` (S5 tournament winner, requires --track prior, tag `_S5`); 4 full runs World/DM x cap_tilt/eqw-active @63d vs cap index: ALL FAIL (0/3 checks) — S5 improves composite IC rel (World +0.0705 t 2.56; DM +0.0576 t 1.63) but realized IR is negative everywhere (World cap_tilt -0.26, DM cap_tilt -0.24); old vm blend remains the better traded book; no certification
- [done] ACWI sole-benchmark evaluation — research_run.py `--bmk-index World` (verified ACWI-equivalent: 0.887 DM + 0.111 EM, residual TE 0.14%/yr; overrides --bmk-source, tag `_vsWorld`) + 6 full runs World/DM/EM x cap_tilt/eqw-active vm @63 vs ACWI: best = DM cap_tilt (IR +0.30, MC p .040, param_stable PASS, TE 2.1%, beta 0.98); EM flips negative vs ACWI; no certification; 154 tests green
- [done] ACWI dashboard section + toggle persistence — "vs ACWI (sole benchmark)" collapsible section (evidence grade + chips + stat cards from the _vsWorld verdicts) + openSecs section-state persistence across pane switches; 12 dashboard tests
- [done] Production pipeline — configs/production.json registry (EM_captilt_vsEM, DM_captilt_vsACWI) + scripts/production_run.py: one reproducible periodic command producing per-strategy allocations.csv / allocations_latest.json / metrics.json (timestamp-free) / signal_latest.json / signal_history(.monthly).csv / contributions_latest.csv + run manifest.json under outputs/production/run_{data_end}/; 9 TDD tests (synthetic end-to-end, byte-identical determinism, monkeypatched main()); real-data smoke @2025-11-14 green; 166 tests
- [done] ACWI winner forensics + verdict doc — overfit_forensics.py parameterized (--segment/--bmk-index/--render-only; + stationary-bootstrap p(IR<=0) n=2000 + alpha decomposition); DM cap_tilt vs ACWI: not parameter-overfit (1/7 signatures), DSR fail = power artifact (needs IR .47/45y), BUT active mean is 100% the structural DM-vs-ACWI spread (selection leg IR -0.04); bootstrap p .054, family-wise MC p .26 — report outputs/research/overfit_forensics_DM_vsWorld.md
- [done] Production dashboard — reporting/signal_viz.py (legacy 'Normalized Equity Ranking' 2-row replica + allocation-history area + JS signal-history payload; 8 TDD tests) + reporting/production_dashboard.py + scripts/build_production_dashboard.py (strategy tabs, benchmark-identity badges, 11 stat cards, default-open Latest Signal + top-10 table, chip-toggled SVG evolution chart, allocations + metrics-detail sections, openSecs persistence; 8 TDD tests); real build 0.54 MB / 4 figures, self-verify 14/14 PASS + visual QA on decoded PNGs; 182 tests green
- [done] TCA stage 1 — Engine per-country cost_bps (vector trade pricing, flat-path byte parity) + configs/costs.json tiered cost model + backtest/tca.py (CostModel/load_cost_model, turnover_analysis, cost_decomposition layered IRs, breakeven_cost); 14 TDD tests, 201 green incl. parity
- [done] TCA stage 2 — cost_bps threaded through protocols/scorecard/report; research_run.py --costs (net-of-cost re-runs EM/DM cap_tilt, `_tca` verdicts with cost_model/tca/stats_net blocks); production_run.py always costed (tca.json + turnover.csv + metrics net_of_fee); 'Transaction Costs & Turnover' sections in BOTH dashboards (3 new signal_viz figs + layer table + breakeven chip); 213 tests green
- [pending] Consider deleting legacy root scripts (parity lock would need frozen fixtures first — see lessons)

## Production-Readiness Goal (2026-06-10)
- [done] A1. Test suite green baseline — 213 passed, 0 failed (2026-06-10)
- [done] A2. Strategies re-verified: full production_run re-executed; verdict JSONs match docs; byte-determinism BUG found (1-ULP numpy reduction jitter in DM turnover_ann/breakeven_bps) and FIXED (fsum + 1e-10 quantization) — proof: 20/20 artifacts hash-identical across consecutive full runs
- [done] A3. Overfitting forensics re-verified against artifacts via adversarial fact-check agent (power decompositions, alpha decomposition, signatures — all confirmed)
- [done] A4. Thesis claims fact-checked: 18 load-bearing numbers vs primary artifacts, 15 confirmed exactly, 3 prompt-side conflations corrected in drafts
- [done] B1. scripts/pipeline.py — quarterly orchestrator (recert|production|dashboards|quarterly, --dry-run), registry-driven, 14 tests
- [done] B2. README updated: production-pipeline section, scripts table, 227 test count
- [done] B3. Legacy root scripts: KEEP, parity-locked (README documents; deletion requires frozen fixtures first — unchanged decision)
- [done] C1. Academic paper -> docs/research/paper_country_rotation_2026.md (abstract through appendices, all numbers artifact-verified)
- [done] D1. EM pitch script -> docs/pitch/EM_captilt_pitch_script.md (data exploration -> signal -> construction -> results -> forensics -> ask + Q&A)
- [done] D2. DM pitch script -> docs/pitch/DM_vsACWI_pitch_script.md (built around the alpha-decomposition finding; composition-bet positioning + Q&A)
- [done] RUNBOOK_quarterly_recert.md rewritten to current registry + pipeline.py + kill-switch monitors

## Dashboard UX overhaul (2026-06-09)
- [done] 1. Evidence-grade verdict banner (tier function + per-gate chips, replaces FAIL-wall)
- [done] 2. Collapsible section sub-headers (default all collapsed, button toggles)
- [done] 3. Identifier buttons — index constituents module (S&P 500 / Nasdaq-100 / Russell 1000 from Wikipedia, display-only)
- [done] 4. Viewport-fit shell (100vh, overflow hidden, fixed header + scrollable pane area, denser stat cards)
- [done] 5. Rebuild real dashboard (9 panes, 7.87 MB, EM cap_tilt = power-limited) + 155 tests green + pushed dev
