# Pitch Script — EM Country Rotation, Cap-Tilt Construction
## "Selection skill you can measure, sized to what the evidence supports"

**Audience:** investment committee / sophisticated allocator.
**Duration:** ~45 minutes + Q&A.
**Posture:** this is an *evidence pitch*, not a salesman's pitch. Every number below is reproducible from the repository (`python scripts/pipeline.py quarterly`); every weakness is stated before the committee finds it. The deck's credibility IS the product.

---

## §0. The one-slide summary (say this first, verbatim)

> "We built a country-selection signal for emerging markets from literature-fixed factors — no data mining — and validated it the way a journal referee would: pre-registered gates, deflated Sharpe ratios, walk-forward, Monte-Carlo nulls, and per-country transaction costs.
>
> The result: a +0.29 information ratio versus the EM cap index at 4.3% tracking error, whose **selection skill rejects randomness at p = 0.030** — and whose realized alpha is **not yet statistically distinguishable from zero** because sixteen years of data cannot certify an IR of 0.3. Both halves of that sentence are true, we can prove each one, and the sizing and governance I'll propose follow from taking both seriously.
>
> We are asking for a paper-trade / pilot sleeve with quarterly re-certification and a pre-registered kill switch — not a flagship allocation."

---

## §1. Why EM country rotation — the economic prior (5 min)

### 1.1 The opportunity set is wide
- EM country index dispersion is structurally high: annual cross-sectional spread between best and worst EM country indices routinely exceeds 30–40 percentage points. Selection has *room* that DM (one ~65% US block) does not offer.
- The EM cap index is concentrated: **China ≈ 39%, India ≈ 20%** of our 15-country investable set at the latest rebalance. Concentration means the *benchmark itself* is a bet — a disciplined tilt away from/toward it is a real decision an allocator already makes implicitly.

### 1.2 The literature prior (fixed BEFORE we touched the data)
- **Asness–Moskowitz–Pedersen (2013):** country 12-1 momentum 8.7%/yr (t = 4.14); country value 6.0%/yr (t = 3.45); 50/50 blend Sharpe 1.16 — value and momentum correlate −0.34 to −0.37, so the blend beats both legs.
- **Calice–Lin (2021):** across 45 countries, priced factors cluster in earnings yield, momentum, profitability; per-factor IC ≈ 0.05. Read that honestly: country selection is a **breadth game with weak per-period signals** — which dictates everything about how we validate.
- **What the literature does NOT support:** absolute valuation timing (Faber's CAPE thresholds failed out-of-sample, Asness–Ilmanen–Maloney 2017), long-run reversion post-1989 (Malin–Bornholt 2013). We use neither.

### 1.3 Why this is hard, stated up front
- N = 15 investable EM countries. A 64-quarter history. Per-period IC ≈ 0.03–0.05. The information is real but thin — anyone who shows you a country strategy with a *t-stat of 4* on 16 years of data is showing you an artifact. Our pitch is calibrated to what this data can prove.

---

## §2. Data exploration — what we have, what we checked (7 min)

### 2.1 Inventory
- **34 vendor workbooks** (Bloomberg/FactSet-class), 34 countries, **daily, 2010-03-05 → 2025-11-14** (~4,100 trading days).
- **EM universe (15):** China, India, Taiwan, South Korea, Brazil, Mexico, South Africa, Belgium, Indonesia, Thailand, Malaysia, Poland, Chile, Peru, Colombia.
- Raw inputs: prices, market cap, trailing + forward valuation ratios (PE, PB, PS, PCF, EV/EBITDA, EV/EBIT, dividend yields), profitability (ROE, fwd ROE, return on capital), balance-sheet stocks (assets, equity, debt, liabilities), consensus estimates, 10-year yields, short interest, M2, GDP.
- Vendor cap-weighted index columns for EM/DM/World ride in the same Price workbook — used as benchmarks, never as candidate factors.

### 2.2 The exploration findings that shaped the design
1. **Raw levels are not factors.** Price, GDP, market cap, EPS levels are not cross-country comparable (currency, scale, accounting). The legacy catalog had treated them as factors; the rebuilt catalog admits **only ratios, yields, spreads and changes**. This single decision removed ~10 spurious "factors."
2. **Data gaps are real and dangerous.** Fundamentals arrive at country-specific lags with vendor gaps. We found a backward-fill (`bfill`) in the legacy pipeline — that is *look-ahead by construction* (it copies tomorrow's print into today). Fixed to forward-fill-only with limits, and — this is the important part — **we built an automated perturbation test**: multiply and shift all data *after* date T; if any output at or before T changes by one bit, the build fails. Leakage is now a *tested invariant*, not a code-review hope.
3. **Publication lags are configurable per metric** (e.g., fundamentals shifted to respect real availability). Consensus data are as-published panels, not true vintages — a documented limitation, not a hidden one.
4. **The benchmark identity was verified, not assumed.** The vendor "World" column regresses on DM+EM at 0.887/0.111 with 0.14%/yr residual TE — ACWI-equivalent, confirmed. The EM column behaves as the cap-weighted EM composite (China-concentration signature: correlation patterns and the +3.8%/yr cap-vs-equal-weight spread in DM, mirrored by China's drag in EM over this span).
5. **Coverage matrix:** all 15 EM countries have usable price + valuation + momentum data across the full span; profitability/quality coverage is thinner early. The composite tolerates per-date missing factors by renormalizing weights across what exists.

### 2.3 What we did NOT do
- No factor was added because it "worked." The factor set was frozen from the literature review **before** any backtest: Momentum_12_1, Momentum_6_1, EarningsYieldTTM, EarningsYieldFWD, PB, PE — category weights 50% Momentum / 50% Valuation. (A 4-category variant including Profitability/Quality was pre-registered as the primary; the 50/50 V+M was the pre-registered secondary that the evidence favored.)

---

## §3. Signal development — from raw factor to traded score (8 min)

### 3.1 The transform pipeline (identical for every factor)
Each factor becomes four standardized metrics, all leak-free:
1. **Expanding z-score** (shifted 1 day), mapped to a percentile — "where is today vs this country's own history, using only the past";
2. **Absolute expanding percentile** — fraction of own history at or below today;
3. **Cross-sectional rank** across the 15 countries at each date, scaled 0–1;
4. **63-day delta percentile** — percentile of the 63-day change ("is the factor improving").

Equal-weighted (¼ each) into the factor score → factors equal-weighted within category → **50/50 Momentum/Valuation** at category level → cross-sectional min-max normalization to [0,1] per date. Per-category contributions are preserved and rebased to sum exactly to the final score — the dashboard's stacked-bar ranking chart is that decomposition, per country, updated every run.

### 3.2 The selection rule — and why it's *change*, not level
- At each quarterly rebalance we rank countries by the **change in composite score over the trailing 63 trading days** and take the **top 5**.
- Why change? We tested the canonical alternative honestly. A **pre-registered 6-spec tournament** (level- vs change-selection × blend / AMP-style two-factor specs, BH-FDR across the family, one-shot lockbox) found: change-selection dominated level-selection in **all** segments; and the tournament's own screen winner (S5: 0.5·rank(Mom12-1) + 0.5·rank(E/P)) — screen IC t 2.7 — **lost money when traded** (IR −0.24 to −0.26, MC p ≥ 0.55). Two lessons we now treat as law: *screen-window IC is not book alpha*, and *the 6-factor blend's diversification is load-bearing*. The vm blend stays; the tournament family is closed.

### 3.3 Why quarterly
Pre-registered monthly-vs-quarterly comparison on the lead composite: monthly degrades **every** statistic (MC p 0.257 vs 0.059 — monthly is indistinguishable from random churn; DSR 0.67 vs 0.91; NW t 0.70 vs 1.56). Slow signals trade noise at monthly cadence. Quarterly (63 trading days) is the registered cadence.

### 3.4 Signal health today (latest production run, data end 2025-11-14)
- Composite relative IC (full sample, 64 obs): mean +0.034, ICIR 0.11. Positive but individually weak — exactly the Calice–Lin breadth regime (§1.2).
- Latest top-5 signal (next rebalance 2026-02-11): China 38.8%, India 19.7%, Indonesia 7.2%, Brazil 6.7%, Mexico 5.6% (Cap-Tilt weights, see §4).
- **Honest disclosure: the IC is front-loaded** — first half +0.072 (t 1.30), second half −0.004. This is THE standing risk; §7 covers how we govern it.

---

## §4. Portfolio construction — Cap-Tilt, the step that made skill investable (6 min)

### 4.1 The construction problem we found (and you should ask every country-rotation manager about)
Equal-weighting a top-5 selection is the textbook construction — and it embeds a **structural anti-cap bet**: equal weight is permanently short China/index concentration. Over 2010–2025, cap beat equal weight by ~3.8%/yr in DM (mirrored dynamics in EM via China). Result: our early equal-weight books had **MC-significant selection skill (p ≈ 0.03–0.05) and negative IRs** — real skill, destroyed by construction. If a manager shows you equal-weight country books vs a cap benchmark, the IR is mostly the cap-vs-equal-weight spread, in either direction.

### 4.2 Cap-Tilt: express selection, neutralize the structural bet
At each rebalance:
- Base = **cap weights** of the 15 countries (from Market_Cap, renormalized);
- **+ active_share/5** to each top-5 country by the selection signal;
- **− min(active_share/5, base)** from each bottom-5 country (same signal, mirrored);
- clip long-only, renormalize to Σw = 1. **active_share = 0.30.**

Properties: TE drops from ~10% (equal-weight) to **4.3%**; beta 1.01; up/down capture 1.01/0.99 — a pure selection overlay on the index an EM allocator already owns. The signal is identical; only the expression changed.

### 4.3 Costs are inside the engine, not a footnote
Every country trades at its own one-way cost (EM tier: 12–25 bps spread + 1 bp commission). The engine charges |Δw|/2 per country per rebalance at that country's rate. All validation — walk-forward, Monte-Carlo nulls, parameter sweeps — prices books **like-for-like** with the same vector.

---

## §5. Backtest results — the full, unflattered record (10 min)

**Configuration:** EM Cap-Tilt, vm 50/50 composite, top-5 change-selection, quarterly, 2010-03-05 → 2025-11-14, vs vendor EM cap index, per-country costs.

### 5.1 Headline (gross of holding-layer fees)
| Metric | Value |
|---|---|
| Information ratio (ann.) | **+0.29** (net of spread+commission: **+0.24**) |
| Active return (ann.) | ≈ +1.25% |
| Tracking error | 4.3% |
| Beta / up / down capture | 1.01 / 1.01 / 0.99 |
| Book ann. return / vol / Sharpe / maxDD | 3.7% / 16.6% / 0.22 / −42.3% (EM beta, not the product — the product is the overlay) |
| Turnover (one-way, ann.) | 1.27× |
| Win rate (quarterly) | 55% |

### 5.2 The statistical scorecard — pass and fail, both shown
| Test | Result | Gate | Verdict |
|---|---|---|---|
| **Monte-Carlo selection null** (random books, same construction+costs) | **p = 0.030** | ≤ 0.05 | **PASS — the load-bearing positive** |
| Walk-forward efficiency (OOS/IS Sharpe, 5 anchored folds) | **2.85** | ≥ 0.5 | **PASS** (OOS *beat* IS) |
| OOS folds positive | **4/5 (80%)** | ≥ 50% | **PASS** |
| Parameter-sweep stability | 100% of neighborhood positive, default z = 0.55 | ≥ 70%, \|z\| ≤ 1.5 | **PASS** |
| Lo Sharpe t (daily active curve) | 1.18 | ≥ 2.0 | **FAIL** |
| PSR | 0.88 | ≥ 0.95 | **FAIL** |
| DSR (trial-ledger deflated) | 0.76 | ≥ 0.95 | **FAIL** |
| Bootstrap IR 90% CI | [−0.008, +0.043] | low > 0 | **FAIL** (straddles zero) |
| Bootstrap one-sided p(IR ≤ 0) | 0.096 | ≤ 0.05 | sig. at 10%, not 5% |

Say it plainly: **the selection process is demonstrably better than random; the realized alpha is not yet provably positive.** Those are different statements with different standards of proof, and at IR ≈ 0.3 on 16 years, the second is mathematically out of reach (§6).

### 5.3 Sub-period honesty
| Window | Active (ann.) | IR | Hit rate |
|---|---|---|---|
| Full (2010–2025) | +1.25% | +0.29 | 55% |
| First half (2010–2018) | +2.10% | **+0.56** | 73% |
| Second half (2018–2025) | +0.40% | **+0.08** | 38% |
| Thirds | +3.11% / −0.33% / +0.97% | +0.88 / −0.09 / +0.18 | 73/50/41% |

Alpha is **front-loaded**. The recovery in the final third is encouraging but thin. We do not smooth this over; it is the reason the ask is a pilot, not a flagship.

### 5.4 Cost and fee waterfall (the slide most pitches omit)
| Layer | IR after layer |
|---|---|
| Gross | +0.292 |
| − spread + commission (≈22 bps/yr at 1.27× turnover) | +0.242 |
| − ETF expense ratio (65 bps) | +0.090 |
| − management fee 50 bps | **−0.027** |
| **Breakeven flat one-way trading cost** | **102 bps** (vs 12–26 modeled) |

Trading costs are a non-issue (8× headroom). **Fees are the binding constraint**: the strategy only works in a cheap wrapper — fee load under ~30 bps, or futures/swap implementation where available. We state this before you ask.

### 5.5 Benchmark dependence (the second slide most pitches omit)
Versus its own EM cap index: +0.29. **Versus ACWI: −0.28.** The EM-vs-World composition drag over this span dominates the selection alpha if you measure against a global stick. This product is an **EM-sleeve overlay for an allocator who has already decided to own EM at benchmark weight** — it is not a global-alpha claim. Mandate fit is part of the evidence.

---

## §6. Overfitting forensics — why we believe the failures are power, not fitting (5 min)

We ran a dedicated forensic battery (`scripts/overfit_forensics.py`) instead of asking you to trust us:

1. **Overfitting signatures, checked one by one:** IS ≫ OOS? **No** — WFE 2.85, the opposite. Lone-peak parameters? **No** — the forensic probe puts the default config dead-center (z ≈ 0.00) in a 100%-positive neighborhood (z = 0.55 in the latest costed sweep). Random-signal null accepted? **No** — p = 0.036 (n = 500 forensic re-run). Cost-fragile? **No** — edge survives 20 bps flat.
2. **Power decomposition:** clearing DSR ≥ 0.95 at this trial variance needs IR ≈ 0.52 on 16.3 years — or IR 0.29 sustained for ≈ **87 years**. The DSR failure is arithmetic, not evidence of fitting. Under the worst-case ~200-trial ledger, *no in-sample path to certification exists at this effect size* — which is exactly why our governance is built on live out-of-sample accrual, the only evidence that doesn't debit the multiple-testing budget.
3. **What forensics did flag — the two honest risks:**
   - **Front-loaded alpha** (second-half IC ≈ 0): unfalsifiable in-sample; only live quarters resolve it.
   - **Post-hoc segment choice** (EM picked as best of 3 segments): Bonferroni ×3 family-wise MC p ≈ 0.09–0.11 — marginal, not clean. We quote it unprompted.

---

## §7. Implementation, governance, and the ask (4 min)

### 7.1 Production discipline (already running, not a promise)
- One command (`python scripts/pipeline.py quarterly`) re-certifies, regenerates allocations, and rebuilds dashboards from the registry. Artifacts are **byte-deterministic** given the same inputs; every run carries the git commit + data-end stamp. 227 automated tests, including engine parity locks, leakage perturbation guards, and statistics verified against hand computations at 1e-12.
- Latest allocation (data end 2025-11-14, next rebalance 2026-02-11): China 38.8%, India 19.7%, Indonesia 7.2%, Brazil 6.7%, Mexico 5.6%, remainder in cap-weight base. Quarterly TCA artifact (turnover, cost layers, breakeven) ships with every run.

### 7.2 Pre-registered kill switch
- **Rolling 252-day IR** is the monitor. Sustained negative territory (and specifically failure to re-clear the MC null at quarterly re-certification) = de-risk; this is in the runbook *now*, before any P&L exists to argue with.
- Quarterly re-certification re-runs the full gate suite on refreshed data; evidence grades (certified / power-limited / weak / negative) are computed mechanically, not narratively.

### 7.3 The ask
- **Paper-trade or pilot sleeve** at a size where 4.3% TE is immaterial to the institution;
- **Fee structure ≤ 30 bps all-in** or futures-based implementation — at 50 bps management fee this strategy is arithmetically dead (§5.4) and we will not pretend otherwise;
- **8–12 quarters of live accrual** before any scale decision: at IR 0.3, that adds the out-of-sample observations that no further backtesting can.

### 7.4 Closing (verbatim)
> "Most country-rotation pitches show you a Sharpe ratio. We're showing you a *process*: a leak-tested pipeline, pre-registered gates, a Monte-Carlo-significant selection signal, an honest power calculation that says sixteen years can't certify it, and a fee arithmetic that tells you exactly what wrapper it needs. If the next eight quarters look like the last sixteen years, the evidence compounds. If they don't, the kill switch fires and we'll tell you *that* with the same numbers. That's the product."

---

## Appendix — anticipated Q&A

**Q: Why should we believe IR 0.29 if the CI includes zero?**
A: You shouldn't *believe* it; you should price it. The MC test (p = 0.030) says the selection process beats random construction-for-construction; the CI says 16 years can't pin the realized mean. The pilot structure is exactly the bet that prices this state: small, cheap, evidence-accruing.

**Q: Why is Belgium in your EM list?**
A: Vendor classification quirk in this dataset — the classification sheet routes it to the EM segment; it carries ~1% cap weight and never enters the top-5 tilts materially. Reclassifying it is on the data-hygiene list and does not move any headline number.

**Q: China is 39% of the book. Isn't this a China fund?**
A: The *base* is the cap index — China is 39% of the benchmark; the strategy's decision is the ±6% tilts around it. Active share is capped at 30% by construction; the China position is benchmark inheritance, not signal conviction. The right comparison is active weights, which the dashboard shows per rebalance.

**Q: What if EM concentration shifts the cap base further?**
A: Cap-Tilt inherits the benchmark automatically at each rebalance — that is the point of the construction. The signal only ever spends the 30% active-share budget.

**Q: Why not extend history with MSCI index data?**
A: We tried to source pre-2010 fundamentals and failed (no vendor access). Price-only momentum history could be backfilled; valuation factors cannot, so a pre-2010 backtest of *this* composite would be a different strategy. It remains the highest-value data acquisition on the roadmap.

**Q: How many things did you try before this?**
A: The ledger says ~200 trials including legacy explorations, 8 pre-registered platform runs, an 18-trial tournament, and 6 ACWI-benchmark variants. DSR is computed against that ledger. The number is in the verdict JSON of every run — auditable, not anecdotal.
