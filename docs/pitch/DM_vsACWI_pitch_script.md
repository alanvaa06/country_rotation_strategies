# Pitch Script — DM Cap-Tilt vs ACWI
## "A composition bet with a selection-neutral overlay — sold as exactly that"

**Audience:** investment committee / global allocator.
**Duration:** ~40 minutes + Q&A.
**Posture:** this script is deliberately built around a finding that *kills the naive pitch*. The strategy's +0.30 IR vs ACWI decomposes exactly into a passive DM-overweight spread (all of it) and a within-DM selection overlay (slightly negative). We pitch what the evidence supports — a disciplined, cost-aware DM-overweight implementation with a research overlay in live evaluation — and we show the decomposition ourselves, because any competent due-diligence team would find it, and the firm that volunteers it wins the relationship.

---

## §0. The one-slide summary (verbatim)

> "We're going to show you a book that beats ACWI with an information ratio of +0.30 at 2.1% tracking error, with a Monte-Carlo selection test at p = 0.040 and five-out-of-five positive walk-forward folds. Then we're going to show you the attribution that most managers would not put on a slide: **the entire excess return is the passive developed-markets-over-ACWI spread (+0.75%/yr, Newey–West t = 2.39). The country-selection overlay inside DM contributed −0.10%/yr.**
>
> So what we're actually offering is: (1) a structural DM-overweight position, implemented at index-like cost and risk, for allocators who hold that view; and (2) a live, quarterly-audited country-selection overlay that currently earns its keep as *risk-neutral signal R&D* — it has measurable selection information (MC p ≈ 0.04–0.05) that has not yet converted into net selection alpha inside DM.
>
> If you want the DM bet, we'll implement it honestly and cheaply. If you want selection alpha, the evidence today says: take our EM sleeve instead, or wait for this overlay's live record. We'll show you why."

---

## §1. The strategic question: DM vs ACWI is a real allocation decision (4 min)

- ACWI ≈ 88.7% DM + 11.1% EM (we verified the vendor World index against DM+EM composites: regression weights 0.887/0.111, residual TE 0.14%/yr — it *is* ACWI for our purposes).
- Over 2010–2025, DM outperformed ACWI by ≈ +0.75%/yr with remarkable persistence (NW t = 2.39 on the daily spread). That is the "composition dividend" of not owning the EM sleeve over this span.
- Most global mandates carry this decision implicitly. The honest question is not "should you bet on DM vs EM?" — you already are, one way or the other — but "is your DM tilt deliberate, sized, monitored, and cheaply implemented?"
- **What this product is:** a deliberate version of that bet, plus a fully-instrumented selection overlay that is *free at the margin* (TE budget already paid) and audited live.

---

## §2. Data exploration — same platform, DM specifics (5 min)

### 2.1 Universe and inputs
- **DM universe (19):** United States, Japan, United Kingdom, Canada, France, Switzerland, Germany, Australia, Netherlands, Ireland, Denmark, Sweden, Spain, Hong Kong, Italy, Singapore, Finland, Israel, Norway.
- Daily vendor data 2010-03-05 → 2025-11-14 (~4,100 days): prices, market caps, trailing/forward valuation, profitability, balance-sheet aggregates, estimates, yields.
- **Cap structure is the dominant data fact: the US is ≈ 65% of DM cap.** Any DM country-selection design that ignores this (e.g., equal-weighting 19 countries) is implicitly running a massive anti-US bet. Over 2010–2025, DM cap beat the DM equal-weight universe by **+3.8%/yr** — an equal-weight book must overcome that drag before any skill shows.

### 2.2 Hygiene (identical to the EM pitch; summarized)
- Levels are never factors (ratios/yields/changes only).
- Forward-fill-only gap handling; look-ahead is *tested* by automated input-perturbation checks (perturb post-T data → outputs through T must be bit-identical), not asserted.
- Publication lags configurable per metric; consensus panels are as-published (limitation, disclosed).
- 227 automated tests, engine parity-locked to the validated legacy implementation; artifacts byte-deterministic.

### 2.3 Exploration findings specific to DM
- DM factor cross-section is *narrower* than EM: valuation/momentum spreads across DM countries are tighter, and per-factor ICs are lower (best prior-factor full-sample ICs in the deployed verdict: Momentum_6_1 +0.023, Momentum_12_1 +0.010, with valuation factors near zero to negative as standalone — only the blend carries information).
- Cross-sectional composite IC (relative method): **mean +0.043, t = 1.32, hit rate 62.5%** — positive, stable in sign across halves (+0.078 → +0.008), individually sub-significant. Breadth-not-strength again.

---

## §3. Signal and construction — one platform, two books (5 min)

Identical machinery to the EM sleeve (one codebase, one registry — this is a feature, not laziness):
- **Composite:** 50/50 Momentum/Valuation, six literature-prior factors (Mom 12-1, Mom 6-1, E/P TTM, E/P FWD, P/B, P/E), four leak-free transforms each, cross-sectionally normalized.
- **Selection:** top-5 by 63-day composite-score *change*, quarterly. (Level-selection and concentrated two-factor specs were tournament-tested and rejected — the screen winner S5 lost money traded; full story in the EM script §3.2 and the paper.)
- **Construction: Cap-Tilt** — cap-weight base ± active_share/N tilts from the same signal, long-only, Σw = 1, active_share = 0.30. On DM this means the US stays ~60–70% of the book; the strategy spends its 30% active budget on the *relative* ordering of the other 18 markets plus the US itself.
- **Benchmark: ACWI-equivalent World index.** This is the deliberate difference from the EM sleeve (which is measured against its own segment index). Measuring a DM-only book against ACWI *bundles* the composition bet with selection — which is exactly why the attribution in §5 matters so much.
- **Costs inside the engine:** DM tier 5–10 bps one-way spread + 1 bp commission, per-country vector, charged on traded notional at every rebalance, threaded through all validation (walk-forward, MC nulls, sweeps priced like-for-like).

---

## §4. Backtest results — the headline that demands its own autopsy (8 min)

**Configuration:** DM Cap-Tilt, vm composite, top-5 change-selection, quarterly, 2010–2025, vs ACWI-equivalent, per-country costs.

### 4.1 Headline numbers
| Metric | Value |
|---|---|
| Information ratio vs ACWI (ann.) | **+0.30** (net of spread+commission: **+0.27**) |
| Active return (ann.) | ≈ +0.65% |
| Tracking error | **2.14%** |
| Beta vs ACWI | 0.98 |
| Up / down capture | 0.99 / 0.97 |
| Book return / vol / Sharpe / maxDD | 8.8% / 14.1% / 0.63 / −32.5% |
| Turnover (one-way, ann.) | 1.00× |
| Quarterly hit rate vs ACWI | 56% |

### 4.2 Statistical scorecard
| Test | Result | Gate | Verdict |
|---|---|---|---|
| Monte-Carlo selection null | **p = 0.040** | ≤ 0.05 | PASS |
| Newey–West t vs equal-weight null | **2.77** | > 0 | PASS |
| Walk-forward efficiency | **2.40** | ≥ 0.5 | PASS (OOS > IS) |
| OOS folds positive | **5/5 (100%)** | ≥ 50% | PASS |
| Parameter stability | 100% positive, default z = 0.72 | ≥ 70%, \|z\| ≤ 1.5 | PASS |
| Lo Sharpe t | 1.22 | ≥ 2.0 | FAIL |
| PSR / DSR | 0.89 / 0.84 | ≥ 0.95 | FAIL (power) |
| Bootstrap IR 90% CI | [−0.003, +0.043] | low > 0 | FAIL (marginal) |
| Bootstrap one-sided p(IR ≤ 0) | 0.053 | ≤ 0.05 | FAIL (by a hair) |
| Family-wise (best of 6 ACWI books, Bonferroni) | MC p ≈ 0.26 | — | **disclosed, not significant** |

On its face: the strongest scorecard in the program. Which is precisely why we ran the attribution before anyone could fall in love with it.

### 4.3 The attribution that reverses the conclusion (THE slide)
The daily active return obeys an exact identity (machine-precision, max error 4e-19):

> **active (book − ACWI) = segment spread (DM index − ACWI) + selection (book − DM index)**

| Component | Mean (ann.) | IR | NW t |
|---|---|---|---|
| Total active | **+0.65%** | +0.30 | — |
| Passive DM − ACWI spread | **+0.75%** | +0.46 | **+2.39** |
| Within-DM selection (book − DM index) | **−0.10%** | −0.04 | −0.22 |

- The composition spread explains **~115%** of the active mean; selection subtracts ~15%.
- Cross-check: the *same book* measured against its own DM cap index shows IR −0.04 — consistent, no sleight of hand.
- The legs correlate at **−0.55**: the selection overlay has functioned as a (mildly costly) diversifier of the spread, which is why total TE (2.1%) is lower than either leg's standalone behavior suggests.

**Plain-language verdict:** the alpha is *position*, not *picking*. A passive DM index fund delivers the same bet cheaper. We will not market this as selection skill — and you should treat any manager whose DM-vs-global excess return survives this decomposition with selection intact as genuinely rare.

### 4.4 Sub-periods (same honesty as the EM sleeve)
| Window | Active (ann.) | IR vs ACWI | Hit rate |
|---|---|---|---|
| Full | +0.65% | +0.30 | 56% |
| First half (2010–2018) | +1.05% | +0.48 | 64% |
| Second half (2018–2025) | +0.24% | +0.11 | 50% |
| Thirds | +1.49% / +0.09% / +0.35% | +0.66 / +0.05 / +0.17 | 73/41/50% |

Front-loaded, like the EM sleeve, with sign stability (no negative half).

### 4.5 Cost and fee waterfall
| Layer | IR after layer |
|---|---|
| Gross | +0.302 |
| − spread + commission (≈8 bps/yr at 1.0× turnover) | +0.265 |
| − ETF expense (50 bps) | +0.032 |
| − management fee 50 bps | **−0.202** |
| **Breakeven flat one-way cost** | **67 bps** (vs 6–11 modeled) |

Trading is a rounding error (6× breakeven headroom). **Fees are fatal at standard levels** — at 50 bps management fee, the product destroys value with certainty *even granting the alpha point-estimate*. The only honest wrappers: a DM index fund/futures position for the composition bet (single-digit bps), plus the overlay run at internal-research cost until it earns a fee.

---

## §5. Overfitting forensics (4 min)

Same battery as the EM sleeve (`scripts/overfit_forensics.py --segment DM --bmk-index World`):
- **Not parameter-overfit:** 1 of 7 signatures present (walk-forward parameter rotation, modal share 40% — benign in a 100%-positive flat neighborhood, default z = +0.76, not a peak).
- **Power decomposition:** DSR ≥ 0.95 at this trial variance needs IR ≈ 0.47 on 16.3 years; the realized 0.30 would need ≈ **45 years**. The DSR/PSR failures are span arithmetic.
- **Stationary bootstrap (n = 2,000):** p(IR ≤ 0) = 0.053.
- **The two disclosed weaknesses:** (i) family-wise position — this book is the best of six ACWI-benchmark evaluations, Bonferroni MC p ≈ 0.26, *not* significant; (ii) the attribution in §4.3 — the alpha is structurally sourced. Either alone would block a "selection alpha" claim; together they define what this product *is* (a composition bet) and *is not* (a stock-picker's record).

---

## §6. What we propose (and what we refuse to sell) (4 min)

### 6.1 The offering, restructured around the evidence
1. **Core: deliberate DM-overweight vs ACWI**, implemented via index instruments at single-digit bps. Sized by the client's conviction on the DM-vs-EM composition question — *our* evidence says the spread has been persistent (NW t 2.39) but it is a macro position, and we present it as one. No alpha fee on beta.
2. **Overlay: the Cap-Tilt selection book, run live in paper/pilot form** alongside the EM sleeve, re-certified quarterly by the same gate suite, with its alpha decomposition recomputed every run. The overlay's selection information is real (MC p 0.040–0.050 across benchmarks, NW t vs equal-weight 2.77, composite IC hit rate 62%) but has not netted positive *within DM* (−0.10%/yr) — our own analysis says don't pay for it yet.
3. **Kill switch / promotion ladder, pre-registered:** the overlay earns a fee if and when its within-DM selection leg (book − DM index) shows a positive rolling IR over a sustained live window AND re-clears the MC gate; the composition core de-risks on the client's macro schedule, not ours.

### 6.2 Why pitch it this way
- Any diligence team that runs one regression finds §4.3. Volunteering it converts a fatal finding into the firm's credibility asset.
- The platform — leak-tested pipeline, pre-registered gates, deflated-Sharpe ledger, per-country TCA, byte-deterministic quarterly production, public code — *is* the durable product. Books rotate; the evidence factory compounds.

### 6.3 Closing (verbatim)
> "We found a book that looks like alpha and proved to ourselves that it's position. We're showing you the proof instead of the pitch, because the next book we bring you — and the EM sleeve already in pilot — will be judged by the same machinery, and you'll know it. If you want the DM bet, we'll build it for you at index cost, monitored like a strategy instead of drifting like an accident. And you'll get the selection overlay's live record for free, until it earns the right to charge you."

---

## Appendix — anticipated Q&A

**Q: If selection is −0.10%/yr inside DM, why run the overlay at all?**
A: Three reasons. The point estimate is statistically zero (NW t −0.22), not negative-with-confidence; the same signal architecture shows MC-significant skill (p = 0.030) and +0.29 IR in EM, so the information content is established at platform level; and the overlay's −0.55 correlation to the spread *reduced* total TE. We run it at zero marginal fee and let live data decide. What we won't do is charge alpha fees for it today.

**Q: Why is the MC test significant (p = 0.040) if selection loses money?**
A: The MC null randomizes *selection* under identical construction and costs. The book beats random selections — the signal orders DM countries better than chance — but the top-5 tilt sizing inside DM hasn't monetized that ordering net of the US-block structure. Skill in ranking ≠ profit in this construction; that gap is precisely what the live overlay record is for.

**Q: Couldn't you fix the construction to monetize the ranking?**
A: Perhaps — and that is a *new trial* against a multiple-testing ledger of ~200, where deflation already prices further in-sample search at near-zero evidentiary value. The honest path is the live record, not another backtest. (The tournament that tried exactly this for signals — six pre-registered specs — produced a screen winner that lost money traded. We learn from our own positive controls.)

**Q: Is the DM-vs-ACWI spread just the last cycle's US tech run?**
A: Largely, yes — that is what "composition bet" means and why it's priced as macro, not alpha. The NW t of 2.39 says the 2010–2025 spread wasn't daily noise; it says nothing structural about the next decade. Sizing belongs to the allocator's macro process; our machinery contributes measurement, implementation, and discipline.

**Q: What does the EM flip tell us (EM book: +0.29 vs own index, −0.28 vs ACWI)?**
A: That benchmark choice is load-bearing at the country-allocation level — segment composition effects rival selection alpha in magnitude. It's the same lesson as §4.3 from the opposite direction, and it's why every book in our registry declares its benchmark identity in the artifact manifest.

**Q: Production readiness?**
A: One command (`python scripts/pipeline.py quarterly`) re-certifies both registry strategies net-of-cost, produces allocations (latest: US 65.7%, HK 9.2%, NL 3.7%, CA 3.5%, UK 3.4% top-5; next rebalance 2026-02-11), TCA artifacts, and dashboards. Byte-deterministic, git-stamped, 227 tests. The runbook pre-registers the gate-reading and kill-switch protocol.
