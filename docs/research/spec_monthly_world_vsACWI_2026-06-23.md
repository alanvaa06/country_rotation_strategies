# Pre-registration — Whole-family country rotation vs ACWI (all segments)

- **Date committed:** 2026-06-23 (revised same day BEFORE any full evidence
  run, after scope was widened from World-only to all segments × signals)
- **Author:** quant dev (country_rotation)
- **Status:** PRE-REGISTERED — grid, gates, prediction, and trial count fixed
  below BEFORE running the 36-cell evidence sweep. Editing after seeing
  results = spec burn. (A single `--quick` wiring smoke on World/vm/cap_tilt
  /@21 was run to prove the CLI path; it is NOT evidence — full runs overwrite
  it.)

## 1. Goal / fixed constraints

Backtest **all strategies across all segments** with **ACWI as the sole
benchmark**, exploring **different signals**, primary focus **monthly
rebalancing**.

| Constraint | Value | Mechanism |
|---|---|---|
| Segments | **World (34c), DM (~23c), EM (~11c)** — every segment's book | `--segment {World,DM,EM}` (Type==Country) |
| Benchmark | **ACWI for ALL segments** (vendor `World` col = 0.887·DM + 0.111·EM, resid TE 0.14%/yr) | `--bmk-index World` (overrides `--bmk-source`) |
| Primary cadence | **Monthly = 21 trading days** | `--periodicity 21` |
| Control cadence | Quarterly = 63 trading days | `--periodicity 63` |
| Basis | **active** (excess over ACWI = selection alpha, beta-neutral) | `--basis active` |
| Costs | per-country one-way spread+commission + ETF expense + mgmt scenarios | `--costs configs/costs.json` |
| Sample | 2010-01-01 → data end 2025-11-14 (vendor max) | default.json |

DM and EM books are *deliberately* measured against the GLOBAL benchmark
(ACWI), not their own segment index — the question is whether any
segment-selection book delivers global-relative alpha.

## 2. Prior (declared before running)

**Prediction A (cadence): monthly @21 degrades vs quarterly @63** on every
certification axis (slow country signals + TC drag — see [D4]; smoke already
shows World vm cap_tilt flips IR +0.30 @63 → −0.30 @21, turnover 1.0× →
2.09×).

**Prediction B (benchmark): EM and World books post negative IR vs ACWI**;
DM cap_tilt is the only previously-positive book vs ACWI ([D12]) but its
edge is a structural DM−ACWI composition spread, not selection ([D13]).

**Falsification:** any cell that (a) beats its own @63 control on net IR AND
(b) certifies (DSR≥0.95, MC p≤0.05, bootstrap IR CI low >0, t≥2) refutes
Prediction A and becomes a candidate.

## 3. Grid — 3 segments × 3 signals × 2 constructions × 2 cadences = 36 cells

Shared flags: `--track prior --bmk-index World --mode active --basis active
--costs configs/costs.json`.

| Axis | Values | Flags |
|---|---|---|
| Segment | World / DM / EM | `--segment SEG` |
| Signal | **vm** (50/50 V+M, AMP-2013) | `--prior-set vm --signal blend` |
|        | **full** (4-category: Mom+Val+Quality+Leverage) | `--prior-set full --signal blend` |
|        | **amp_ey** (S5 winner: 0.5·rank(Mom_12_1)+0.5·rank(EY_TTM), change-traded) | `--signal amp_ey` |
| Construction | **eqw** (equal-weight top-N active sleeve) | `--construction eqw` |
|              | **cap_tilt** (cap base ± 0.30 active-share tilt) | `--construction cap_tilt` |
| Cadence | monthly / quarterly | `--periodicity 21` / `63` |

All factor sets fixed ex-ante (prior track / pre-registered tournament) →
**no in-sample screening; lockbox not consumed.**

## 4. Certification gates (existing scorecard, unchanged)

Certified only if all three aggregate checks PASS:
- `no_overfitting`: DSR ≥ 0.95, WFE ≥ 0.5, MC p ≤ 0.05
- `param_stable`: stability frac_positive ≥ 0.7
- `statistically_significant`: Sharpe-t ≥ 2.0, PSR ≥ 0.95, bootstrap IR CI
  low > 0, NW-t-vs-eqw ≥ 2.0

**Deployable bar (net):** `net_mgmt_50bps` IR > 0; breakeven one-way bps
reported.

## 5. Biases controlled

| Bias | Control |
|---|---|
| Look-ahead | ffill-only (no bfill), publication lag, engine perturbation-tested, anchored/expanding folds, IC pairs signal_t with fwd return t→t+1 |
| Data-snooping / selection | track=prior — factor sets fixed ex-ante; amp_ey pair fixed by pre-registered tournament; no in-sample screening |
| **Multiple testing** | DSR penalizes trial count. Ledger ~204 pre-exercise. **This grid = 36 cells; 30 net-new trials** (6 vm@63-vsWorld pre-exist in [D12]) → ledger ~234. Significance read against a **family-wise lens (36-cell Bonferroni: per-cell α' = 0.05/36 ≈ 0.0014; or BH-FDR across the 36 net IRs)**. No single-cell cherry-pick without this penalty. |
| Survivorship | fixed universe membership; vendor index series carries own survivorship — acknowledged, not correctable |
| Transaction cost / turnover | costed everywhere (validation, WF, MC null like-for-like); deployable bar net of 50bps mgmt; breakeven reported |

## 6. Verification plan (after runs)

1. Reconstruct headline stats from artifacts (parity < 1e-8 where applicable).
2. Per cell: monthly-vs-quarterly net-IR delta; sign vs Prediction A.
3. Cross-segment: World/DM/EM net IR vs ACWI; sign vs Prediction B.
4. Adversarial review: any surviving cell read against family-wise + net-of-cost lens.
5. Results → results.md; decisions → memory.md; ledger update; session-log.

## 7. Expected outcome (hypothesis, not result)

Near-zero cells certify; monthly < quarterly almost everywhere on net IR;
EM/World negative vs ACWI; DM cap_tilt positive but composition-driven.
Exercise documents the whole-family monthly/quarterly behaviour vs ACWI and
closes as a pre-registered negative unless §2 is falsified.
