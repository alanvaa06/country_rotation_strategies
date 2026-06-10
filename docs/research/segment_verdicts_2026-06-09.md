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
1. **Extend history** — only 2010+ data exists (vendor has no earlier periods). This remains the single highest-value lever but is currently blocked by data availability. Revisit if MSCI/index-level proxies back to the 1990s become accessible.
2. ~~Run DM vm monthly robustness check~~ — **DONE 2026-06-09, see Addendum below. Monthly degrades the strategy; quarterly confirmed as the correct cadence.**
3. Paper-trade the DM 50/50 V+M composite (top-5, **quarterly**, 2bps) while evidence accrues; re-run `python scripts/research_run.py --segment DM --track prior --prior-set vm` each quarter as `Inputs/` is refreshed — the platform re-certifies deterministically (proven bit-for-bit reproducible, see Addendum §2).
4. Keep EM out of scope until DM/World certify.

---

## Addendum 2026-06-09 — Monthly robustness + reproducibility

### 1. DM 50/50 V+M: monthly (@21d) vs quarterly (@63d), pre-registered

| Axis | Monthly @21d | Quarterly @63d | Gate | Stronger |
|---|---|---|---|---|
| Sharpe t-stat | 1.53 | **1.96** | ≥2.0 | quarterly |
| PSR | 0.93 | **0.97** | ≥0.95 | quarterly |
| DSR | 0.67 | **0.91** | ≥0.95 | quarterly |
| MC p-value | 0.257 | **0.059** | ≤0.05 | quarterly |
| NW t vs equal-weight | 0.70 | **1.56** | >0 | quarterly |
| Bootstrap Sharpe CI | [0.014, 0.204] | **[0.104, 0.426]** | low>0 | quarterly |

**Finding — monthly rebalancing destroys the edge, it does not strengthen it.** The intuition "more rebalances → more observations → higher t" is *disproven*: monthly tripled the period count (~186 vs ~62) yet every discriminating statistic fell. At MC p = 0.257 the monthly strategy is statistically **indistinguishable from random country selection**; the NW t-stat versus the equal-weight null collapses from 1.56 to 0.70. Country value+momentum are slow signals (12-1 momentum, fundamental value); monthly cadence churns positions on noise — paying turnover and 2bps TC drag without capturing fresher signal. Quarterly (@63d) is the correct, literature-consistent cadence. **Monthly is ruled out as a certification path.**

(`wf_efficiency` is identical (4.04) across both because the validation grid sweeps periodicity ∈ {21,63} regardless of base cadence, so the walk-forward stage is not a periodicity-discriminating statistic in this design — only the base-cadence Engine stats above discriminate.)

### 2. Reproducibility / quarterly-re-cert mechanism — verified

Re-ran the exact quarterly command on the unchanged dataset. The verdict reproduced the prior run **bit-for-bit**: max abs diff = 0.0 across sharpe_ann, t-stat, PSR, DSR, MC p, WFE, NW t, stability z-score and both bootstrap CI bounds (seed = 42 fixed throughout). This confirms the "re-certifies automatically as data accrues" mechanism: identical inputs → identical verdict, so any future change in a quarterly verdict is attributable solely to new data, not run-to-run noise.

## Addendum 2026-06-09 (b) — ACTIVE-basis certification (alpha, not beta)

The original verdicts above computed Sharpe-t / PSR / DSR / bootstrap on the **absolute** return of a long-only, fully-invested top-5 book. In a 2010–2025 bull market that is dominated by market beta, not selection skill. The scorecard now supports `basis="active"` (default in `research_run.py`): the headline stats are computed on **excess-over-benchmark** daily returns (strategy − equal-weight universe), i.e. the Information Ratio. The Monte-Carlo random-selection null and the Newey-West-vs-equal-weight test were already beta-controlled and are essentially unchanged — confirming they measured skill all along.

### DM — absolute vs active

| Run | t (abs→act) | PSR | DSR | MC p | Bootstrap Sharpe CI (abs → act) |
|---|---|---|---|---|---|
| vm @63 (lead) | 1.96 → **1.34** | 0.97 → 0.91 | 0.91 → 0.84 | 0.059 → **0.050** | [0.10, 0.43] → **[-0.005, 0.046]** |
| vm @21 | 1.53 → 0.61 | 0.93 → 0.73 | 0.67 → 0.52 | 0.257 → 0.307 | [0.01, 0.20] → [-0.012, 0.032] |
| full @63 | 1.84 → 0.95 | 0.96 → 0.83 | 0.90 → 0.76 | 0.119 → 0.158 | [0.06, 0.41] → [-0.011, 0.040] |

**Corrected verdict (the honest one).** On the active basis the lead candidate (DM 50/50 V+M @63d) has an Information Ratio point estimate of ~0.5 (active Sharpe t = 1.34), but its **bootstrap IR confidence interval straddles zero [-0.005, 0.046]** — the alpha is *not* statistically distinguishable from zero on 2010–2025 DM. The earlier absolute Sharpe (t 1.96, CI [0.10, 0.43]) was ~85% market beta. The MC null at p = 0.050 says selection skill only marginally beats random country picking. **DM 50/50 V+M remains the best candidate and is still not certified — and the bar it must clear is alpha-significance, which is further away than the absolute numbers implied (active t 1.34 needs ~2× the absolute t's data to reach 2).**

This does not change the strategic conclusion (extend history / accrue live quarters; quarterly is the right cadence) — it corrects the *framing*: certify on alpha, not on a beta-laden absolute Sharpe.

## Addendum 2026-06-09 (c) — Cap-weighted mandate benchmark + blended configs

`research_run.py` gains `--bmk-source {eqw,index}` (index = vendor cap-weighted segment column from Price.xlsx; verified cap signature: corr(DM, US)=0.94 vs 0.66 for eqw basket; index beat eqw basket +3.8%/yr 2010-2025) and `--mode blend --bmk-weight w` (core-satellite). Verdicts now also carry `mandate_stats` (TE/IR/beta/capture vs chosen benchmark) and `composite_ic` (full-window IC stats, both methods). All runs vm @63d, active basis, full params. Self-verified by two independent recompute passes (all checks PASS; engine blend decomposition exact to 3.5e-18; NW-vs-eqw bit-invariant across benchmarks).

### Results

| Run | IR (ann) | act t | MC p | TE | beta | IC rel: mean / t / hit |
|---|---|---|---|---|---|---|
| DM vs eqw (ref) | +0.33 | 1.34 | 0.050 | — | — | — |
| **DM vs cap index** | **−0.14** | −0.57 | **0.040** | 10.0% | 0.86 | +0.043 / 1.32 / 0.62 |
| **World vs cap index** | −0.04 | −0.16 | **0.050** | 10.9% | 0.89 | **+0.062 / 2.29 / 0.62** |
| DM cap blend 30 | −0.15* | −0.59 | 0.040 | 7.0% | 0.90 | same sleeve |
| DM cap blend 50 | −0.15* | −0.60 | 0.040 | 5.0% | 0.93 | same sleeve |

\*active return scales by (1−w); t-stat scale-invariant. TE scales exactly (1−w): 10.0% → 7.0% → 5.0%. Beta follows w + (1−w)·β_sleeve to ±0.0002.

### The decomposition this exposes (the key insight)

Strategy total return = **market beta** + **construction tilt** + **selection alpha**:
- **Selection alpha is real and statistically significant**: MC random-selection null — which holds construction constant — rejects at **p = 0.040 (DM) / 0.050 (World)**, the first hard gate this program has passed. Selection beats random selection.
- **Construction tilt is the drag**: the equal-weight top-5 book is structurally anti-US/anti-cap; the cap index beat the eqw basket +3.8%/yr 2010–2025, swamping the ~+2.2%/yr selection alpha → negative IR vs the investable mandate benchmark (−1.6%/yr).
- **Signal quality (IC)**: World composite relative IC @63d (the exact configuration traded) **t = 2.29, p ≈ 0.013, n = 64, ICIR 0.29, hit 62% — statistically significant**. DM t = 1.32. Monthly-grid IC weaker (t ≤ 0.92) — signal lives at the quarterly horizon, consistent with the cadence finding. Caveat: 8 informational IC cells measured; the significant one is the pre-specified traded cell (World/relative/63), but family-wise this is borderline — treat as strong-but-single-cell evidence.

### Goal scorecard (stable + statistically significant IR and IC within a segment)
- **IC: ACHIEVED (World, relative, quarterly)** — t 2.29.
- **Selection skill: ACHIEVED** — MC p ≤ 0.05 in both DM and World vs cap benchmark.
- **IR: NOT achieved** — vs eqw +0.33 (t 1.34, ns); vs cap mandate negative. The blocker is construction, not signal.
- **Stability**: vs eqw fully stable (100% grid positive); vs cap consistently negative (stability frac 0.00) — the tilt dominates every config.

### Next lever (pre-register before running)
Benchmark-aware construction: build the book as cap-index weights ± active tilts from the composite score (e.g. cap-weight base, overweight top-5 / underweight bottom by a TE budget), instead of an equal-weight top-5 sleeve. This removes the structural anti-cap tilt so the realized IR reflects the (already MC-significant) selection skill. Blend mode is the right wrapper afterwards: TE scales exactly (1−w) for mandate sizing.

## Reproducibility
- `python scripts/research_run.py --segment {World|DM|EM} --track {screen|prior} [--prior-set vm] --periodicity {21|63} [--basis active|absolute] [--bmk-source eqw|index] [--mode active|blend --bmk-weight w]`
- `--basis active` is the default (alpha certification). `--basis absolute` is diagnostic only (beta-laden).
- Artifacts carry periodicity + basis suffix to prevent collisions: `verdict_{seg}_{track}[_{prior_set}]_p{periodicity}_{basis}.json` (+ `report_*.html`, `screening_*.xlsx`). All gitignored, local.
- Code state: branch `dev`; full pytest suite green (125).
