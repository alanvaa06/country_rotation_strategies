# Whole-family country rotation vs ACWI — 36-cell grid results

- **Date:** 2026-06-23
- **Pre-registration:** docs/research/spec_monthly_world_vsACWI_2026-06-23.md
  (grid / gates / predictions / ledger fixed BEFORE running)
- **Artifacts:** `outputs/research/verdict_*_vsWorld_tca.json` (36),
  `outputs/research/grid_vsACWI_2026-06-23.csv`,
  collator `outputs/research/_collate_vsACWI.py`
- **Data:** 2010-01-01 → 2025-11-14 (vendor max), costed (configs/costs.json),
  active basis, benchmark = ACWI (vendor `World` col).

## Headline

**0 / 36 cells certify. 0 / 36 post positive net-of-50bps-fee IR. Every
segment's mean net IR vs ACWI is negative.** No falsification of the
pre-registered predictions.

| Segment | Mean net IR vs ACWI | Cells net-positive |
|---|---|---|
| DM | −0.29 | 0 / 12 |
| World | −0.44 | 0 / 12 |
| EM | −0.48 | 0 / 12 |

## Grid (sorted by deployable bar = net-of-50bps-mgmt IR)

Only the gross-positive and least-negative cells shown; full 36 in the CSV.

| Seg | Signal | Constr | Cad | Gross IR | t | DSR | MC p | Net(50bps) IR | Turn ×/yr | Breakeven bps | Cert |
|---|---|---|---|---|---|---|---|---|---|---|---|
| World | vm | eqw | @63 | −0.04 | −0.16 | 0.24 | 0.050 | **−0.186** | 4.0 | −11 | n |
| World | amp_ey | eqw | @63 | −0.05 | −0.22 | 0.32 | 0.050 | −0.187 | 3.3 | −19 | n |
| **DM** | **vm** | **cap_tilt** | **@63** | **+0.302** | **+1.22** | **0.84** | **0.040** | −0.202 | 1.0 | **+67** | n |
| DM | full | cap_tilt | @21 | +0.293 | +1.19 | 0.82 | 0.119 | −0.280 | 2.6 | +24 | n |
| DM | vm | cap_tilt | @21 | +0.232 | +0.94 | 0.72 | 0.317 | −0.336 | 2.8 | +19 | n |
| DM | full | cap_tilt | @63 | +0.208 | +0.84 | 0.71 | 0.208 | −0.306 | 1.0 | +45 | n |
| … | | | | | | | | | | | |
| EM | vm | eqw | @21 | −0.431 | −1.75 | 0.01 | 0.792 | −0.670 | 9.6 | −57 | n |
| World | vm | cap_tilt | @21 | −0.297 | −1.20 | 0.05 | 0.782 | −0.709 | 2.1 | −46 | n |

Every **gross-positive** cell in the whole grid is a **DM cap_tilt** book.
All eqw and all non-DM cap_tilt books are gross-negative vs ACWI.

## Prediction A (monthly @21 degrades vs quarterly @63)

**Holds in 14 / 18 matched pairs on net IR.** The 4 exceptions (DM amp_ey
cap_tilt Δ+0.087, DM full cap_tilt Δ+0.026, World amp_ey cap_tilt Δ+0.032,
EM full eqw Δ+0.036) are all cases where **both cadences are net-negative and
neither certifies** — not a falsification (falsification requires a monthly
cell to beat its control AND certify). The biggest degradations are the
eqw books (World vm Δ−0.34, World full Δ−0.37, EM vm Δ−0.36): equal-weight
top-N selection at monthly cadence runs **10–12× annual one-way turnover**,
which costs alone make TC-fatal (breakevens −17 to −57 bps one-way).
cap_tilt damps turnover to ~2–3× but stays net-negative.

## Prediction B (segment books negative vs ACWI)

**Confirmed.** DM/EM/World all 0/12 net-positive. The only book with a real
gross edge — **DM vm cap_tilt @63 (+0.30 IR, MC p 0.040, DSR 0.84,
stability PASS, NW-t 2.77, breakeven 67 bps)** — is the previously-diagnosed
**structural DM−ACWI composition spread, not selection alpha** ([D13]); it is
**fee-fatal** (net of 50 bps mgmt = −0.20).

## Verification

- Independent grep of raw verdict JSONs: 0 `overall=true`.
- Cost-layer monotonicity (gross ≥ net_spread ≥ +expense ≥ +mgmt50): 0
  violations / 36.
- Turnover monotonicity (@21 ≥ @63 per matched cell): 0 violations / 36.
- **Reproduction:** DM vm cap_tilt @63 matches the prior [D13] certified
  verdict to < 1e-9 on sharpe_ann / mc_p / dsr / psr / nw_t — byte-determinism
  holds across the costed re-run.

## Multiple-testing ledger

This exercise = 36 pre-registered cells, **30 net-new trials** (6 vm@63
vsWorld pre-existed). Ledger ~204 → **~234**. Family-wise: 0 cells clear the
Bonferroni-×36 MC bar (α′ ≈ 0.0014); BH-FDR over the 36 net IRs surfaces
nothing (all net IR ≤ 0).

## Conclusion / recommendation

For whole-family country rotation **measured against ACWI**:
- **No certifiable selection alpha** at any segment / signal / construction /
  cadence over 2010–2025.
- **Monthly is the wrong cadence** — slow country signals + 10–12× turnover;
  quarterly dominates net of cost almost everywhere.
- The single positive gross book (DM vm cap_tilt @63) is a **composition bet**,
  fee-fatal, and already documented; nothing here changes that verdict.
- **Action:** close the monthly/ACWI sweep as a pre-registered negative; do
  not deploy any cell; quarterly cadence and the EM-vs-own-index selection
  candidate ([final dossier]) remain the only live leads. Benchmark choice is
  load-bearing — measuring DM/EM books against the global ACWI guarantees a
  composition-driven negative.
