# Final Evidence Dossier — Positive Excess-Return Strategy Search

**Date:** 2026-06-09 · **Data:** 34 countries, daily, 2010–2025 (no earlier vendor access) · **All claims pre-registered or labeled; every segment benchmarked against BOTH its vendor cap-weighted index and its equal-weight universe null.**

## The answer

**A positive excess-return strategy exists in EM: the Benchmark-Aware Cap-Tilt book on the 50/50 Value+Momentum composite.**

| Evidence dimension | EM Cap-Tilt (vm signal, @63d, vs EM cap index) | Significance |
|---|---|---|
| Information Ratio (annualized) | **+0.29** (active return ≈ +1.2%/yr at TE 4.3%) | point estimate |
| Monte-Carlo selection-skill null (200 random books, same construction/costs) | **p = 0.030** | **significant at 5%** |
| One-sided stationary-bootstrap p(IR ≤ 0), 2,000 resamples | **p = 0.096** (90.5% of resamples positive; CI90 [−0.08, +0.63] ann.) | significant at 10%, **not** at 5% |
| Walk-forward (anchored, per-fold reselection) | WFE 1.45, **80% of OOS folds positive** | robust |
| Parameter stability | 100% of grid configs positive z=0.55 | robust |
| Mandate profile | TE 4.3%, beta 1.01, up/down capture 1.01/0.99 | implementable |

**Why EM:** the EM cap index carries China-concentration drag that the cap-tilted book systematically avoids while harvesting the (MC-significant) Value+Momentum country selection. The mirror effect kills DM/World vs their cap indexes (US concentration premium).

**Honest statement of the claim:** the EM strategy's excess return is positive in point estimate, its *selection skill* is statistically significant at the 5% level (Monte-Carlo, construction held constant), and its *realized IR* is significant at the 10% — not 5% — level on 15.5 years of data. A true IR of ~0.3 needs ~45 years for 5% bootstrap significance; this is a data-span limit, quantified, not a methodology gap.

## Per-segment scoreboard (vs both benchmarks, @63d, active basis)

| Segment | vs cap index (mandate) | vs equal-weight null | Composite IC (relative) |
|---|---|---|---|
| **EM** | **Cap-Tilt IR +0.29, MC p 0.030** ✓ | negative (eqw EM basket strong) | t 0.88 (noisy — skill shows in MC, not IC) |
| DM | Cap-Tilt IR −0.04 (flat; best of 3 constructions) | vm +0.33, t 1.34, MC p 0.050 | t 1.32 |
| World | Cap-Tilt IR −0.10, MC p 0.030 (skill yes, IR no) | ~flat | **t 2.29 — significant** |

Each segment's full three-strategy comparison (Cap-Tilt / Active-EqW / Core-Satellite-50) is in `outputs/research/strategy_dashboard.html` (toggle, default Cap-Tilt) and segment_verdicts_2026-06-09.md Addendum (d).

## Signal redesign — attempted, evaluated, rejected (Addendum (e) summary)

Per the goal's "rethink the signal development" mandate, a **pre-registered 6-spec tournament** (literature anchors: AMP 2013 canonical mom+value ranks, Balvers-Wu momentum, level vs change selection) was run with screen/lockbox discipline and BH-FDR across specs:
- Screen-window winner S5 (0.5·rank(Mom 12-1) + 0.5·rank(E/P), change-traded): IC t 2.72 (World, q 0.019) / 2.23 (DM, q 0.093).
- **Full scorecard REJECTED S5**: realized books worse everywhere (World cap-tilt IR −0.26, MC p 0.55; DM −0.24, MC p 0.73) — higher concentration, lost MC significance. The vm 6-factor blend's diversification is load-bearing.
- Canonical *level*-selection specs (AMP exact form) were flat-to-negative in this universe — the tradable edge here is in **signal change** (fundamental/score revisions momentum), not signal level.
- Methodological note: this is the system working as designed — in-sample IC advantages that don't survive construction are caught before deployment, and the 6 extra trials are logged in the multiple-testing ledger.

## Addendum — ACWI sole-benchmark evaluation (2026-06-09/10)

All six books (World/DM/EM × Cap-Tilt/EqW-active, vm @63d) re-measured against the vendor **World index = verified ACWI-equivalent** (regression vs DM+EM: 0.887/0.111, residual TE 0.14%/yr). Only positive-IR book: **DM Cap-Tilt vs ACWI — IR +0.30, active t 1.22, MC p 0.044, stability PASS (1.00), 100% OOS folds positive, TE 2.1%, beta 0.98, bootstrap one-sided p(IR≤0) = 0.054**.

**Forensics attribution kill (outputs/research/overfit_forensics_DM_vsWorld.md):** not parameter-overfit (1/7 signatures, all tuning probes clean, DSR failure = power artifact needing ~45y) — but the exact return decomposition shows the entire active mean is the **passive DM−ACWI composition spread (+0.75%/yr, IR +0.46, NW t +2.39)**, while within-DM selection contributed **−0.10%/yr** on this sample. Family-wise (1-of-6 books) Bonferroni MC p = 0.26.

**ACWI verdict, honestly stated:** the goal's three conditions are met only superficially by DM Cap-Tilt (positive alpha +0.30, positive IC +0.043, MC significance 0.044) — attribution shows the alpha is a **structural DM-overweight bet vs ACWI**, itself NW-significant (t 2.39) but beta-composition, not selection skill. No book currently delivers significant *selection* alpha over ACWI. The platform caught this before deployment — exactly its job.

## Standing decisions
1. **Deploy-candidate (own-segment mandate): EM Cap-Tilt vm @63d vs EM cap index** (paper-trade; quarterly re-cert via runbook). Caveat: does NOT survive an ACWI measuring stick (−0.28) — mandate benchmark choice is load-bearing.
2. **ACWI-relative:** DM Cap-Tilt is a disguised DM-composition bet (significant as a passive spread, t 2.39; not selection alpha). Do not market as skill. If the *composition* bet is wanted, hold DM index vs ACWI directly — cheaper.
3. World: signal significant (IC t 2.29), no mandate-positive book — revisit with free-float base construction or new data.
4. Multiple-testing ledger now ≈ 200 trials (legacy 150 + segment runs + tournament + S5 + 6 ACWI books); every future variant logs here.
