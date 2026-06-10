# Segment Strategy Verdicts — 2026-06-09

**Protocol:** 8 pre-registered runs on real vendor data (34 countries, 2010–2025 daily). Two tracks — `screen` (walk-forward BH-FDR factor screening @21d) and `prior` (literature-fixed factor sets @63d, no data-driven selection; OOS protection from scorecard) — plus one labeled secondary variant (`prior vm`: 50/50 Value+Momentum, AMP 2013 primary construction). Engine: long-only top-5 relative selection, equal weight, active mode vs equal-weight universe, 2bps TC. Certification gates (pre-registered, docs/references/validation_formulas.md): DSR≥0.95, PSR≥0.95, MC p≤0.05, WFE≥0.5, bootstrap CI>0, Sharpe t≥2, stability gates.

## Results

| Run | Factors | t | PSR | DSR | MC p | WFE | OOS folds + | Boot CI low | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| World screen @21 | 0/45 survive FDR | — | — | — | — | — | — | — | **No factors** |
| DM screen @21 | 0/45 | — | — | — | — | — | — | — | **No factors** |
| EM screen @21 | EV_EBIT only | 1.32 | 0.90 | 0.79 | 0.089 | 0.91 | 1.00 | −0.02 | **FAIL** |
| World prior @63 | 10 (4 cat) | 1.79 | 0.96 | 0.85 | 0.089 | 0.65 | 0.60 | +0.06 | **FAIL** |
| DM prior @63 | 10 (4 cat) | 1.84 | 0.96 | 0.90 | 0.119 | 1.92 | 0.80 | +0.06 | **FAIL** |
| EM prior @63 | 10 (4 cat) | 0.90 | 0.81 | 0.71 | 0.653 | 0.71 | 0.80 | −0.06 | **FAIL (clear)** |
| World prior vm @63 | 6 (V+M 50/50) | 1.92 | 0.96 | 0.87 | 0.059 | 2.13 | 1.00 | +0.10 | **FAIL (near)** |
| DM prior vm @63 | 6 (V+M 50/50) | 1.96 | 0.97 | 0.91 | 0.059 | 4.04 | 1.00 | +0.10 | **FAIL (nearest)** |

(t/PSR/DSR on period-return equity; t is sampling-grid invariant ≈ SR_ann·√years. MC p with add-one smoothing, n=100. Sharpe_ann column of raw JSONs uses the per-period √252 ranking convention — not comparable across periodicities; omitted here.)

## Interpretation

1. **No strategy certifies** at the pre-registered institutional thresholds on 15.5 years of data. No threshold was relaxed post-hoc.
2. **The evidence ladder matches the literature exactly:** 50/50 Value+Momentum > 4-category blend > screened single factors; DM ≥ World > EM (EM noisiest, MC p 0.65 on the 4-category prior — indistinguishable from random selection). This cross-validation of the verified literature ranking is itself meaningful evidence the pipeline measures real structure.
3. **DM 50/50 V+M is the standout candidate:** beats random-selection null at p = 0.059, 100% of walk-forward OOS folds positive, WFE 4.0, bootstrap Sharpe CI strictly positive [0.10, 0.43], PSR 0.97, passes every stability gate. It fails certification on Sharpe t = 1.96 (needs 2.0), MC p = 0.059 (needs 0.05) and DSR 0.91 (needs 0.95).
4. **Why it can't certify yet:** with a true annualized active Sharpe ≈ 0.5, the expected t after 15.5y is ≈ 1.97 — the sample is simply too short for a Sharpe-0.5 strategy to clear t≥2/DSR≥0.95 reliably. This is a power limit of the data span, not a flaw of the strategy or the harness.
5. **Honest negatives:** EM (all tracks) and pure screened tracks (DM/World) are negatives. Factor-level country signals are individually too weak (IC ≈ 0.01–0.06) to survive FDR across a 45-factor family — consistent with the literature's IC ≈ 0.05 norm and "breadth, not strength" maxim.

## Recommended path forward
1. **Extend history** (vendor data pre-2010 or MSCI index-level proxies back to 1990s) — the single highest-value action; doubles power.
2. Run DM vm with monthly periodicity (more rebalances, same horizon) as a pre-registered robustness check.
3. Paper-trade the DM 50/50 V+M composite (top-5, quarterly, 2bps) while evidence accrues; re-run `python scripts/research_run.py --segment DM --track prior --prior-set vm` quarterly — the platform re-certifies automatically.
4. Keep EM out of scope until DM/World certify.

## Reproducibility
- `python scripts/research_run.py --segment {World|DM|EM} --track {screen|prior} [--prior-set vm] --periodicity {21|63}`
- Artifacts per run (gitignored, local): `outputs/research/verdict_*.json`, `report_*.html`, `screening_*.xlsx`.
- Code state: branch `dev` @ this commit; 120 tests green.
