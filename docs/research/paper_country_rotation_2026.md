# Factor-Based Country Rotation Under Honest Validation: A Pre-Registered Evidence Cycle in Developed and Emerging Equity Markets

**Alan Vazquez, CFA**
*June 2026*

---

## Abstract

We subject factor-based country equity rotation to a pre-registered, multiple-testing-aware validation protocol on 34 countries over 2010–2025 and report what survives. Almost nothing does — and the exceptions are instructive. Of 45 candidate country-level factors, **zero** survive a Benjamini–Hochberg false-discovery screen on out-of-sample information coefficients. A literature-prior composite (50/50 value and momentum, Asness–Moskowitz–Pedersen 2013) traded as a benchmark-aware cap-tilt book produces two candidates: an emerging-markets book with an information ratio of +0.29 against its capitalization-weighted index, whose selection skill rejects a random-allocation Monte-Carlo null (p = 0.030) but whose bootstrap IR confidence interval straddles zero; and a developed-markets book with IR +0.30 against an ACWI-equivalent benchmark whose entire active mean is shown — by an exact return decomposition — to be a passive composition spread (+0.75%/yr, Newey–West t = 2.39) rather than within-segment selection (−0.10%/yr). Deflated-Sharpe forensics attribute the certification failures to statistical power (an IR of 0.3 requires roughly 45–87 years to clear a DSR ≥ 0.95 gate; we have 16), not parameter overfitting: walk-forward efficiency exceeds 1, parameter neighborhoods are flat, and the books survive realistic trading costs with breakevens of 67–102 bps against modeled one-way costs of 6–26 bps. Fee layers, not trading costs, are the binding economic constraint: both books turn negative net of a 50 bps management fee. We argue that honest reporting of this evidentiary state — *suggestive selection skill, insufficient power, structurally confounded benchmarks* — is the correct scientific posture for country allocation research, where the cross-section is shallow (N ≤ 34) and the literature's headline premia were estimated on spans three times longer than what survivorship-clean vendor data now affords.

**Keywords:** country rotation, factor investing, multiple testing, deflated Sharpe ratio, walk-forward validation, transaction-cost analysis, emerging markets.

---

## 1. Introduction

Country selection is among the oldest ideas in quantitative asset allocation: overweight cheap, rising markets; underweight expensive, falling ones. The academic record appears supportive. Asness, Moskowitz and Pedersen (2013) document a 12-month-minus-1 country momentum premium of 8.7%/yr (t = 4.14) and a value (book-to-market) premium of 6.0%/yr (t = 3.45), with a 50/50 combination Sharpe of 1.16. Balvers and Wu (2006) report momentum-plus-reversion strategies earning 1.1–1.7% per month. Calice and Lin (2021) find profitability, value, momentum and default-risk factors priced in a 45-country panel.

Yet the same record carries three warnings. First, these estimates derive from long histories (1978–2011 and earlier) that pre-date the post-2010 institutionalization of country ETFs. Second, Harvey, Liu and Zhu (2016) demonstrate that the cross-section of published anomalies is contaminated by multiple testing, proposing t > 3.0 as the relevant discovery threshold. Third, out-of-sample audits of country-timing rules are sobering: Faber's (2012) CAPE thresholds improved returns in-sample but failed to beat buy-and-hold in the 1900–2015 out-of-sample test of Asness, Ilmanen and Maloney (2017).

This paper asks a narrow question with discipline: **on data an institutional allocator can actually buy today (34 countries, daily, 2010–2025), does factor-based country rotation produce certifiable selection alpha after honest accounting for multiple testing, look-ahead bias, benchmark choice, and implementation costs?**

Our contribution is methodological as much as empirical:

1. **A leak-free pipeline.** Every transform is expanding-window with a one-day shift; gap-filling is forward-only; leakage is *tested*, not asserted, via input-perturbation checks (perturb data after a cutoff, require bit-identical outputs at and before the cutoff).
2. **Pre-registered evidence gates.** Certification requires jointly: Lo (2002) Sharpe t ≥ 2.0, PSR ≥ 0.95 (Bailey–López de Prado 2012), DSR ≥ 0.95 (Bailey–López de Prado 2014), Monte-Carlo random-selection p ≤ 0.05, walk-forward efficiency ≥ 0.5 with ≥ 50% of out-of-sample folds positive, parameter-sweep stability, bootstrap IR confidence interval excluding zero, and a positive Newey–West t against an equal-weight null. Gates were fixed before results were seen.
3. **A multiple-testing ledger.** Roughly 200 configuration trials are logged across the program's history; deflated-Sharpe computations use this ledger rather than the flattering "one trial" assumption.
4. **An exact alpha decomposition** separating within-segment selection from passive segment-composition spread — which reverses the headline conclusion for our developed-markets candidate.
5. **Honest negatives.** We report, with the same prominence as positives: a 0/45 factor screen, a signal-redesign tournament whose screen-window winner failed out-of-sample, a rejected monthly cadence, and a benchmark substitution that flips our best candidate's sign.

The empirical answer is: *not yet certifiable, but not dismissible*. The emerging-markets cap-tilt book exhibits selection skill that is individually Monte-Carlo significant and survives every direct overfitting probe; what it lacks is span — the realized IR of +0.29 would need most of a century of data to clear a deflated-Sharpe bar of 0.95. We propose live quarterly re-certification, rather than further in-sample searching, as the only honest path to more evidence: every additional design variant burns trials against the DSR budget, while live quarters accrue genuinely out-of-sample observations.

The remainder of the paper proceeds as follows. Section 2 reviews the literature underlying our priors. Section 3 describes data. Section 4 details methodology. Section 5 reports results, including negative ones. Section 6 presents overfitting forensics. Section 7 covers transaction-cost analysis. Section 8 discusses implications, Section 9 limitations, and Section 10 concludes.

---

## 2. Literature and Priors

Our factor priors were fixed *before* composite construction, from a 23-source adversarially-verified literature review. The claims that survived verification:

**Country momentum.** Asness, Moskowitz and Pedersen (2013, *Journal of Finance*) remains the canonical estimate: 12-1 momentum on country indices earns 8.7%/yr long-short (t = 4.14, Sharpe 0.73). This is the strongest documented country-level factor.

**Country value.** The same study places book-to-market country value at 6.0%/yr (t = 3.45, Sharpe 0.61); value and momentum correlate at −0.34 to −0.37 across asset classes, making the 50/50 blend's Sharpe (1.16, t = 6.62) exceed either leg. Asness, Ilmanen and Maloney (2017) recommend exactly this blend for market timing, "sinning a little."

**Breadth of weaker factors.** Calice and Lin (2021, *Journal of Empirical Finance*) examine ~90 candidate variables across 45 countries (2002–2018) and find priced factors clustering in profitability (EBIT/EV), earnings yield, momentum, and default risk. Crucially, per-factor predictive power is small — cross-sectional R² of roughly 0.30–0.42% per month, implying an information coefficient near 0.05. Country selection is a *breadth* game, not a *strength* game.

**Mean reversion.** Balvers and Wu (2006) combine momentum with long-run mean reversion; Malin and Bornholt (2013) find the reversion leg absent post-1989. We carry momentum, not reversion.

**Valuation timing skepticism.** Keimling (2016) documents CAPE's country-level predictive heterogeneity (R² from ~90% for Japan to ~1% for Canada) at 5–15 year horizons only, with EM standard errors materially larger than DM. Faber's (2012) absolute CAPE thresholds failed out-of-sample (Asness–Ilmanen–Maloney 2017). We therefore use valuation *cross-sectionally and relatively*, never as absolute thresholds.

**Multiple-testing discipline.** Harvey, Liu and Zhu (2016) motivate our t > 3 "strong factor" labeling; Bailey and López de Prado (2012, 2014) supply the PSR/DSR machinery; López de Prado (2018) the general warning that backtest overfitting is the field's default failure mode.

Claims we explicitly declined to rely on after verification: Angelidis and Tessaromatis's (2017) per-factor outperformance magnitudes (2.48–8.42%/yr; could not be reproduced at stated magnitudes), Calice–Lin's composite long-short return (0.507%/month; not replicated here), and Keimling's universality claims for CAPE.

---

## 3. Data

**Source and coverage.** Proprietary vendor workbooks (Bloomberg/FactSet-class), 34 input metrics across 34 countries, daily, 2010-03-05 to 2025-11-14 (≈4,100 trading days, 15.7 calendar years). The repository is code-only; data are gitignored and not redistributable.

**Segments.** The classified universe splits into **Developed Markets** (19 countries: United States, Japan, United Kingdom, Canada, France, Switzerland, Germany, Australia, Netherlands, Ireland, Denmark, Sweden, Spain, Hong Kong, Italy, Singapore, Finland, Israel, Norway) and **Emerging Markets** (15 countries: China, India, Taiwan, South Korea, Brazil, Mexico, South Africa, Belgium, Indonesia, Thailand, Malaysia, Poland, Chile, Peru, Colombia), plus the union (**World**, 34). Vendor capitalization-weighted index levels for DM, EM and World accompany the country series.

**Benchmark verification.** Before using the vendor "World" index as an ACWI-equivalent measuring stick, we regressed its daily returns on the DM and EM vendor indices: coefficients 0.887 (DM) and 0.111 (EM), residual tracking error 0.14%/yr — economically indistinguishable from a cap-weighted DM+EM composite. We treat it as ACWI-equivalent throughout.

**Input metrics.** Prices; market capitalization; trailing and forward valuation (P/E, P/B, P/S, P/CF, EV/EBITDA, EV/EBIT, dividend yield and forward variants); profitability (ROE, forward ROE, return on capital, margins reconstructed from EBIT/EBITDA/Revenue); balance-sheet aggregates (assets, equity, debt, liabilities; leverage ratios); estimates (forward consensus ratios); rates (10-year yields); flows and short interest where available; M2 and GDP (used only in ratio/change form).

**Known limitations.**
1. *Span.* No vendor access to pre-2010 history (confirmed 2026-06-09). All power calculations in Section 6 trace to this constraint.
2. *Point-in-time discipline.* Fundamentals are as-published vendor panels, not vintage snapshots; configurable publication lags shift series to respect availability, but true vintage data would be stricter.
3. *Free float.* The Market_Cap input is full market cap, not free-float-adjusted; the residual gap between our cap-tilt base and the vendor index appears as a small structural tracking-error floor.
4. *Survivorship.* All 34 countries persist over the span; country-level survivorship is mild relative to single-name data, but EM index composition pre-2010 would differ.

---

## 4. Methodology

### 4.1 Factor catalog

Raw vendor levels (price, GDP, market cap, EPS, …) are **not** factors: levels are not comparable across countries. The catalog admits only ratios, yields, spreads and changes, organized into four categories with literature-fixed directions:

| Category | Representative factors | Direction |
|---|---|---|
| Valuation | Earnings yield (TTM, FWD), P/E, P/B, CAPE-style ratios | cheap = good |
| Momentum | 12-1 and 6-1 price momentum | high = good |
| Profitability | ROE, forward ROE, EBIT/EV, return on capital | high = good |
| Quality / Risk | Rolling 252d volatility (low-risk anomaly), leverage ratios | low risk = good |

The pre-registered **prior-vm** set used by the deployed composites is six factors: Momentum_12_1, Momentum_6_1, EarningsYieldTTM, EarningsYieldFWD, PB, PE, weighted 50% Momentum / 50% Valuation at the category level (AMP 2013's primary construction).

### 4.2 Transforms

Each factor is mapped to four standardized metrics, all leak-free by construction:

1. **Expanding z-score** (shifted one day), mapped through the normal CDF to a percentile;
2. **Absolute expanding percentile** — fraction of own history ≤ current value (`min` method);
3. **Cross-sectional rank** at each date, scaled to [0, 1];
4. **63-day delta percentile** — percentile rank of the 63-day change.

Metrics average (equal weights, ¼ each) into a factor score; factor scores average within category; categories combine with the registered weights into a composite, which is cross-sectionally normalized to [0, 1] at each date. Per-category contributions are retained and rebased so that contributions sum exactly to the normalized score, enabling the signal decompositions used in reporting.

**Leakage controls.** Gap-filling is forward-only (`ffill` with limits); backward fill is classified as leakage and rejected by an automated perturbation test: inputs after a cutoff date are multiplied and shifted, and any change in outputs at or before the cutoff fails the build. The same harness guards every derived metric in the processing layer.

### 4.3 Portfolio construction

**Selection.** At each rebalance (every 63 trading days — quarterly), countries are ranked by the *change* in composite score over the trailing period ("relative" selection); the top 5 are selected. Signal-*level* selection (canonical in AMP 2013) was implemented and tested; it is dominated by change-selection on this panel (Section 5.3).

**Weighting.** Three constructions are maintained:
- *Equal-weight active* (eqw): top-5 at 20% each — maximal selection expression, but carries a structural anti-cap (anti-US, anti-China) tilt: cap-weighted indices beat equal-weight universes by +3.8%/yr over 2010–2025 in DM, swamping ~+2.2%/yr of selection alpha.
- *Cap-Tilt* (deployed): start from cap weights at the selection date; add `active_share/N` to each top-N country and subtract `min(active_share/N, base)` from each bottom-N country (same signal, mirrored), clip long-only, renormalize. `active_share = 0.30`. This caps tracking error at 2–4% and removes the anti-cap confound.
- *Blend*: convex combination of book and benchmark; TE scales exactly as (1−w). Used for mandate calibration, not certification.

**Costs.** The engine prices each country's traded notional at its own one-way cost (configs/costs.json: DM 5–10 bps spread, EM 12–25 bps, +1 bp commission), charged at rebalances. Expense ratios (DM 50 / EM 65 bps) and management-fee scenarios (0/50 bps) accrue daily in the TCA layer system (Section 7).

### 4.4 Benchmarks

Every book is evaluated **actively** (excess over benchmark) on three measuring sticks: the equal-weight universe (the honest "is selection better than naive breadth" null), the segment's vendor cap-weighted index (the investable mandate), and the ACWI-equivalent World index (the global allocator's stick). The basis matters: an earlier absolute-return headline for DM (Sharpe-t 1.96) proved ~85% market beta on decomposition; all certification statistics in this paper are active-basis.

### 4.5 Validation protocol

All formulas are documented with hand-computed unit tests at 1e-12 tolerance (a discipline that caught one kurtosis-term and one mean-term bug against canonical sources).

- **Lo (2002) Sharpe t:** `SE(SR) = sqrt((1 + 0.5·SR²)/n)` on the daily curve; t = SR/SE.
- **PSR** (Bailey–López de Prado 2012): `PSR = Φ( (SR−SR*)·√(n−1) / √(1 − γ₃·SR + ((γ₄−3)/4)·SR²) )` with non-excess kurtosis.
- **DSR** (Bailey–López de Prado 2014): PSR evaluated at the expected-maximum null Sharpe `SR₀ = √V({SR_trials})·[(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]`, with the trial ledger supplying N.
- **Newey–West (1987)** HAC t on the daily active series vs the equal-weight null, Bartlett lags `L = ⌊4(n/100)^{2/9}⌋`.
- **Stationary bootstrap** (Politis–Romano 1994) for IR confidence intervals and one-sided p(IR ≤ 0), n = 2,000 in forensics runs.
- **Anchored walk-forward:** five expanding-window folds; in each, the configuration is selected on data strictly before the fold and evaluated inside it. Walk-forward efficiency = mean(OOS Sharpe)/mean(IS Sharpe). A lesson incorporated after an early error: significance is *never* tested on fold means (overlapping anchored folds shrink the variance of means mechanically); per-period IC series with df = n−1 are the honest grid.
- **Monte-Carlo selection null:** the identical engine, costs, and construction with random top-N selection; p = (#null ≥ actual + 1)/(n + 1).
- **Screening track:** per-factor, per-period Spearman ICs; one-sided t (Grinold–Kahn df); Benjamini–Hochberg FDR across the full factor family at q = 0.10; HLZ weak labels for |t| < 3; a terminal 20% lockbox never read during screening.

### 4.6 Pre-registration and trial accounting

Composite definitions, gates, segments, and cadence were registered before evaluation (8 runs: World/DM/EM × screen/prior(/vm)). Subsequent design extensions (cap-tilt construction, benchmark substitutions, the 6-spec tournament, cost overlays) were each registered before execution and appended to the trial ledger (~200 trials inclusive of legacy parameter exploration). DSR is reported against this ledger; we additionally quote Bonferroni-style family-wise adjustments wherever a candidate was selected *after* observing siblings (EM = best of 3 segments; DM-vs-ACWI = best of 6 ACWI books).

---

## 5. Results

### 5.1 Single factors: nothing survives

Across 45 catalog factors at the quarterly horizon, the best raw p-value (AssetsEquity, World segment, p = 0.003) maps to a BH-FDR q of 0.137 — above the 0.10 gate. **Zero factors survive in any segment.** Given Calice–Lin-scale ICs (≈0.05) and ~64 quarterly observations, per-factor power is simply insufficient; this outcome *validates* the screen's honesty rather than condemning the factor set. It also mandates composites: breadth must do what strength cannot.

### 5.2 Pre-registered composites

The 50/50 value+momentum composite at quarterly cadence, active basis, across segments (gross of holding-layer costs; engine includes 2 bps flat trading cost in the original runs):

| Book (vs benchmark) | IR | Sharpe-t | MC p | DSR | Bootstrap IR CI |
|---|---|---|---|---|---|
| World cap-tilt vs World cap | −0.10 | — | 0.030* | — | [−0.029, +0.015] |
| DM cap-tilt vs DM cap | −0.04 | — | 0.050 | — | [−0.022, +0.017] |
| **EM cap-tilt vs EM cap** | **+0.29** | 1.18 | **0.030** | 0.76 | [−0.008, +0.043] |
| DM cap-tilt vs ACWI | +0.30 | 1.22 | 0.040 | 0.84 | [−0.003, +0.043] |
| EM cap-tilt vs ACWI | −0.28 | — | — | — | — |

\*MC tests selection skill against random books under identical construction; a negative-IR book can still pass it when construction (not selection) causes the drag.

Two facts organize everything that follows. First, **selection skill is real but small**: Monte-Carlo nulls reject at 3–5% in three books; composite relative IC is positive everywhere (World t = 2.29 at its strongest). Second, **realized IR depends on construction and benchmark more than on selection**: equal-weight books destroy the alpha via anti-cap tilt; the EM book's +0.29 against its own index becomes −0.28 against ACWI.

### 5.3 The signal-redesign tournament: a cautionary positive control

A pre-registered six-spec tournament (blend/AMP-E-P/AMP-BP/momentum × level/change selection; screening on the first 80% of dates; BH-FDR across specs; one-shot lockbox) produced a screen winner — S5: 0.5·rank(Momentum_12_1) + 0.5·rank(EarningsYieldTTM), change-traded — with screen-window IC t-statistics of 2.72 (World) and 2.23 (DM), q = 0.019/0.093. Carried to full scorecards, **S5 failed everywhere**: realized IR −0.26 (World cap-tilt) and −0.24 (DM cap-tilt), MC p ≥ 0.55. Composite IC improved while the tradable book worsened.

Three lessons: (i) screen-window IC is not book alpha; (ii) the six-factor blend's internal diversification is load-bearing — concentrated two-factor specs are fragile; (iii) the tradable edge on this panel lives in signal *change*, not level — canonical level-selection was flat-to-negative in all segments. The tournament consumed its lockbox; the family is closed.

### 5.4 Cadence

A pre-registered monthly (@21d) variant of the lead DM composite degraded every statistic relative to quarterly (Sharpe-t 1.53 vs 1.96; DSR 0.67 vs 0.91; MC p 0.257 vs 0.059; NW t 0.70 vs 1.56). Slow signals (12-1 momentum, valuation) churned monthly trade noise. Quarterly is the registered cadence; monthly is closed.

### 5.5 The EM candidate in full

EM Cap-Tilt, vm composite, quarterly, vs EM cap index, with the per-country cost vector applied (net-of-spread statistics in parentheses):

- **IR +0.29 gross (+0.24 net of spread+commission)**; active return ≈ +1.25%/yr on TE 4.3%; beta 1.01; up/down capture 1.01/0.99.
- **MC selection null: p = 0.030 gross / 0.119 net-vs-uncosted-null (conservative by construction)**; Sharpe-t 1.18; PSR 0.88; DSR 0.76; bootstrap IR CI [−0.008, +0.043], one-sided p(IR ≤ 0) = 0.096.
- **Walk-forward: efficiency 2.85, 4/5 OOS folds positive.** Parameter sweep: 100% of neighborhood configs positive; the default sits at z = 0.55 from the sweep mean — no peak-spike signature.
- **Turnover 1.27× ann. one-way; breakeven flat cost 102 bps** against modeled 12–26 bps EM one-way costs.
- **Time profile (the honest risk):** first-half IR +0.56, second-half +0.08; composite IC +0.072 (t 1.30) first half, −0.004 second half. Alpha is front-loaded.

### 5.6 The DM-vs-ACWI candidate: an exact decomposition reverses the headline

DM Cap-Tilt vs ACWI-equivalent: IR +0.30, MC p 0.040, TE 2.1%, beta 0.98, WFE 2.40, 5/5 OOS folds positive — superficially the strongest book in the program. The daily active return admits an exact identity (max abs error 4.3e-19):

> active (book − ACWI) = **segment spread** (DM index − ACWI) + **selection** (book − DM index)

| Component | Mean (ann.) | IR | NW t |
|---|---|---|---|
| Total active | +0.65% | +0.30 | — |
| Passive DM−ACWI spread | **+0.75%** | +0.46 | **+2.39** |
| Within-DM selection | **−0.10%** | −0.04 | −0.22 |

The *entire* active mean — 115% of it — is the passive composition bet that DM outperforms ACWI; the selection overlay subtracts value (consistent with the same book's −0.04 IR against its own DM index). Correlation between legs is −0.55. Family-wise, the book is 1-of-6 ACWI evaluations: Bonferroni-adjusted MC p ≈ 0.26.

**Verdict:** not selection alpha. If the DM-overweight bet is wanted, a passive DM index implements it at lower cost. The book remains in the production registry *labeled as a composition bet with a selection-neutral overlay*, never marketed as country-picking skill.

---

## 6. Overfitting Forensics

For both candidates we ran a dedicated forensic battery (`scripts/overfit_forensics.py`): overfitting-signature checklist, DSR power decomposition, parameter-neighborhood mapping, family-wise adjustments, alpha decomposition, stationary-bootstrap p-values (n = 2,000), and cost-sensitivity ladders.

**Signatures absent in both books:** in-sample ≫ out-of-sample (WFE is 2.4–2.9, the *opposite* tail); lone-peak parameter tuning (default configs sit at z ≤ 0.76 inside 100%-positive neighborhoods); random-signal null acceptance (MC rejects at 3–4.4%); cost fragility (edges survive 20 bps flat).

**Signature present in both books:** walk-forward parameter choices rotate across folds (modal share 20–40%) — benign given flat, all-positive neighborhoods, but consistent with a weak signal; and **front-loaded alpha** with second-half ICs near zero. The latter is the genuine standing risk and is *unfalsifiable in-sample*: only live accrual distinguishes "weak but persistent" from "decayed."

**Power decomposition.** Inverting the DSR gate: to clear DSR ≥ 0.95 with the observed trial variance, the EM book needs IR ≈ 0.52 over 16.3 years — or its actual IR ≈ 0.29 sustained for ≈ 87 years. The DM-vs-ACWI book needs IR 0.47, or its 0.30 for ≈ 45 years. The DSR failures are therefore *arithmetic consequences of span at this effect size*, not evidence of fitting. Under the full ~200-trial worst-case ledger, the EM deflation hurdle rises above the realized IR — there is **no path to in-sample certification at this effect size**; only fresh out-of-sample quarters change the calculus.

**Family-wise honesty.** EM was selected as the best of three segments (Bonferroni ×3: MC p ≈ 0.09–0.11); DM-vs-ACWI as the best of six ACWI books (×6: p ≈ 0.26). We quote both unadjusted and adjusted figures wherever the candidate is discussed.

---

## 7. Transaction Costs and Fees

The TCA layer system charges each country's traded notional at its own one-way rate, then stacks holding-layer drags (cumulative annualized IRs on the daily active curve):

| Layer | EM book | DM-vs-ACWI book |
|---|---|---|
| Gross | +0.292 | +0.302 |
| − spread + commission (22 / 8 bps ann.) | +0.242 | +0.265 |
| − ETF expense (65 / 50 bps) | +0.090 | +0.032 |
| − management fee 50 bps | **−0.027** | **−0.202** |
| Breakeven flat one-way cost | 102 bps | 67 bps |
| Ann. one-way turnover | 1.27× | 1.00× |

**Trading costs are survivable** — breakevens sit 4–8× above modeled spreads. **Fee layers are binding**: both books are negative net of a 50 bps management fee. Any live mandate requires either a fee structure under ~30 bps or cheaper instruments (futures where available) before the arithmetic works. This conclusion is invariant to every statistical question in Sections 5–6: even taking the alpha point-estimates at face value, the current fund-wrapper economics consume them.

---

## 8. Discussion

**Benchmark choice is load-bearing.** The EM book is the program's best selection-skill candidate against its own index and a *negative* book against ACWI; the DM book is the reverse composition. Neither fact is a contradiction — they jointly say that segment composition effects (DM-vs-EM relative performance, intra-EM China concentration) are first-order at this asset level, comparable in magnitude to any selection alpha. Country-rotation research that reports a single benchmark is underdetermined.

**Construction can destroy real skill.** MC-significant selection (p = 0.03–0.05) coexisted with negative IRs for two years of this program's history until the cap-tilt construction removed the anti-cap confound. The order of operations — prove selection skill exists, then engineer construction to express it — matters; the reverse order (optimize construction metrics) is how overfitting happens.

**Screen IC does not transfer.** The S5 tournament is a clean demonstration inside one program: a spec can win the screening window at t = 2.7 and lose money traded. Validation must happen at the *book* level, on the *traded* signal, against the *mandate* benchmark.

**The honest evidentiary state.** Three claims survive every test we could throw at them: (i) the composite contains positive, MC-significant country-selection information; (ii) no statistic distinguishes its realized alpha from zero at conventional levels once power and family-wise corrections are applied; (iii) fee layers currently consume the point estimate. A research culture that forces a binary deploy/reject verdict would mislabel this state in either direction. The correct outputs are the ones this program produces: a *power-limited* evidence grade, a paper-trading registry with quarterly re-certification, and a pre-registered kill-switch (sustained negative rolling 252-day IR).

---

## 9. Limitations

1. **Span (16 years) is the binding constraint** on every power calculation; pre-2010 vendor history is unavailable to us. Index-proxy reconstruction (MSCI country total-return series) is the highest-value extension.
2. **Point-in-time vintages** for fundamentals are approximated by publication lags, not true snapshots.
3. **Free-float adjustment** of the cap-tilt base is pending; it accounts for a small structural TE versus vendor indices.
4. **The trial ledger is partly reconstructed** for the legacy (pre-platform) period; we bound it at ~200, but the true historical count is uncertain in both directions.
5. **Cost models are tiered assumptions** (country-ETF spreads), not realized execution data.
6. **Two candidate books share the same composite**; their evidence is correlated, not independent.

---

## 10. Conclusion

Applied with pre-registered gates, leak-tested transforms, deflated-Sharpe accounting, and exact attribution, factor-based country rotation on 2010–2025 data yields: zero certifiable single factors; one power-limited selection-skill candidate (EM cap-tilt, IR +0.29, MC p 0.030, bootstrap CI straddling zero); one benchmark-relative book whose alpha is exactly a passive composition spread; and a fee arithmetic that currently consumes both point estimates. The literature's country premia are not refuted — at AMP-scale effect sizes our tests would have certified them — but on the span an allocator can buy today, the honest verdict is *suggestive, unproven, and economically unbankable at standard fees*. We commit the entire pipeline, gates, ledger and forensics to a reproducible codebase and adopt live quarterly re-certification as the only further evidence source that does not debit the multiple-testing budget.

---

## References

- Asness, C., Ilmanen, A., Maloney, T. (2017). "Market Timing: Sin a Little." *Journal of Investment Management*, 15(3).
- Asness, C., Moskowitz, T., Pedersen, L. H. (2013). "Value and Momentum Everywhere." *Journal of Finance*, 68(3), 929–985.
- Bailey, D. H., López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*, 15(2).
- Bailey, D. H., López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio Management*, 40(5), 94–107.
- Balvers, R. J., Wu, Y. (2006). "Momentum and Mean Reversion Across National Equity Markets." *Journal of Empirical Finance*, 13(1), 24–48.
- Benjamini, Y., Hochberg, Y. (1995). "Controlling the False Discovery Rate." *Journal of the Royal Statistical Society B*, 57(1), 289–300.
- Calice, G., Lin, M.-T. (2021). "Exploring Risk Premium Factors for Country Equity Returns." *Journal of Empirical Finance*, 63, 294–322.
- Faber, M. (2012). "Global Value: Building Trading Models with the 10 Year CAPE." Cambria white paper / SSRN 2129474.
- Grinold, R. C., Kahn, R. N. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill.
- Harvey, C. R., Liu, Y., Zhu, H. (2016). "…and the Cross-Section of Expected Returns." *Review of Financial Studies*, 29(1), 5–68.
- Keimling, N. (2016). "Predicting Stock Market Returns Using the Shiller CAPE." StarCapital / SSRN 2736423.
- Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal*, 58(4), 36–52.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Malin, M., Bornholt, G. (2013). "Long-Term Return Reversal: Evidence from International Market Indices." *Journal of International Financial Markets, Institutions and Money*, 25, 1–17.
- Newey, W. K., West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708.
- Politis, D. N., Romano, J. P. (1994). "The Stationary Bootstrap." *Journal of the American Statistical Association*, 89(428), 1303–1313.

---

## Appendix A — Certification gates

| Gate | Threshold | Family |
|---|---|---|
| Deflated Sharpe Ratio | ≥ 0.95 | no-overfitting |
| Walk-forward efficiency | ≥ 0.50 | no-overfitting |
| OOS folds positive | ≥ 50% | no-overfitting |
| Monte-Carlo null p | ≤ 0.05 | no-overfitting |
| Sweep configs positive | ≥ 70% | stability |
| Default config \|z\| in sweep | ≤ 1.50 | stability |
| Lo Sharpe t | ≥ 2.00 | significance |
| PSR | ≥ 0.95 | significance |
| Bootstrap IR CI low | > 0 | significance |
| Newey–West t vs eqw null | > 0 | significance |

## Appendix B — Reproducibility

All results regenerate from the public repository (github.com/alanvaa06/country_rotation_strategies) given schema-conformant vendor data: `python scripts/pipeline.py quarterly` runs re-certification, production artifacts and dashboards end-to-end; `scripts/research_run.py` reproduces any single verdict; `scripts/overfit_forensics.py` reproduces Section 6; `scripts/spec_tournament.py` reproduces Section 5.3. Artifacts are byte-deterministic given identical inputs (seeded validation; exactly-rounded scalar reductions). 227 automated tests cover transforms, engine parity with the legacy implementation, leakage guards, statistics against hand-computed references at 1e-12, and end-to-end script smokes on synthetic fixtures.
