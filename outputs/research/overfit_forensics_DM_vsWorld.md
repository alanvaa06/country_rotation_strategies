# Overfitting Forensics — DM Cap-Tilt (vm @63d vs vendor 'World' index (ACWI-equivalent))

**Question:** is the DM Cap-Tilt book's IR of +0.30 vs the vendor 'World' index (ACWI-equivalent) (t 1.22, DSR 0.838 < 0.95) overfitted, or robust?  **Data:** 4096 active daily returns (16.3y, 2010-03-05 to 2025-11-14). Reconstruction reproduces the certified verdict `verdict_DM_prior_vm_p63_active_captilt_vsWorld.json` bit-for-bit (sharpe_ann / t / PSR / DSR all match to <1e-8). Script: `python scripts/overfit_forensics.py --segment DM --bmk-index World`; machine-readable results in `outputs/research/overfit_forensics_DM_vsWorld.json` (gitignored, regenerable).

## 1. DSR power decomposition

The certified DSR uses the validation sweep's 4 trial Sharpes (sigma_daily = 0.00335) giving a deflated benchmark SR0 = 0.00353/day (+0.056 annualized). Inverting PSR(SR0) = 0.95 at the observed sample size, skew (-0.18) and kurtosis (7.7):

| Quantity | Actual | Required for DSR >= 0.95 |
|---|---|---|
| Annualized IR (active Sharpe) | +0.302 | +0.465 |
| Lo (2002) t-stat | 1.22 | 1.88 |
| DSR | 0.838 | 0.950 |

At the CURRENT IR of +0.302, the sample needed is **45 years** for DSR >= 0.95, 30 years for plain PSR >= 0.95, and 44 years for Sharpe t >= 2 (t ~ IR x sqrt(years)). The strategy has 16.3 years. A true IR ~0.3 cannot mathematically clear a t~2-equivalent gate on this span regardless of whether the edge is real — the DSR gate is power-limited here, by construction.

## 2. Monte-Carlo random-signal null (n = 500)

| Metric | Value |
|---|---|
| Actual IR (ann) | +0.302 |
| Null mean / std | +0.120 / 0.109 |
| Null q95 | +0.292 |
| Actual percentile in null | 95.8 |
| p-value (add-one) | **0.0439** |
| Family-wise (x6, Bonferroni) | 0.263 |

The null holds the Cap_Tilt construction, costs and benchmark constant and randomizes only the signal — this isolates selection skill from construction. Family for the Bonferroni line: 6 ACWI-relative books (World/DM/EM x cap_tilt/eqw) evaluated in this round.

## 3. Stationary-bootstrap significance (n = 2000, seed 42)

Politis-Romano stationary bootstrap of the daily active series (expected block length sqrt(n) = 64.0 days, wrap-around), 2000 resamples; one-sided test of H0: IR <= 0 with add-one smoothing.

| Metric | Value |
|---|---|
| Point IR (ann) | +0.302 |
| Bootstrap IR 90% CI | [-0.005, +0.589] |
| Bootstrap IR 95% CI | [-0.066, +0.662] |
| Resamples with IR <= 0 | 5.30% |
| **p (one-sided, IR <= 0)** | **0.0535** |

## 4. Alpha decomposition — segment bet vs selection

Exact additive split of the daily active return on 4096 common days (identity max abs error 4.3e-19):  `active (book - vendor 'World' index (ACWI-equivalent)) = spread (DM cap index - benchmark) + selection (book - DM cap index)`.

| Component | Ann mean | IR (ann) | NW t | Ann vol |
|---|---|---|---|---|
| Active (book - benchmark) | +0.65% | +0.302 | — | — |
| Segment spread (DM - World, structural) | +0.75% | +0.464 | +2.39 | 1.61% |
| Within-DM selection (book - DM) | -0.10% | -0.039 | -0.22 | 2.55% |

corr(selection, spread) = -0.550. Mean attribution: spread +0.75%/yr + selection -0.10%/yr = active +0.65%/yr. Variance shares of the active return: selection 142%, spread 56%, 2cov -98%; active~spread regression beta = +0.13, R^2 = 0.01. Read: in RETURN terms the active mean is carried ENTIRELY by the structural spread (115% of the mean; selection nets -0.10%/yr). The low R^2 only says the day-to-day tracking RISK is selection-driven — risk is not return: what the IR monetizes is the segment composition bet.

## 5. Walk-forward forensics (5 anchored folds)

| Fold | OOS window | Chosen params | IS Sharpe (per-period) | OOS Sharpe (per-period) |
|---|---|---|---|---|
| 1 | 2010-06-28 to 2013-07-24 | relative_selection_score=5, periodicity=21 | -0.1233 | +0.2040 |
| 2 | 2013-07-25 to 2016-08-22 | relative_selection_score=5, periodicity=63 | +0.2693 | +0.2862 |
| 3 | 2016-08-23 to 2019-09-19 | relative_selection_score=5, periodicity=63 | +0.3099 | +0.2980 |
| 4 | 2019-09-20 to 2022-10-18 | relative_selection_score=7, periodicity=63 | +0.2495 | +0.1410 |
| 5 | 2022-10-19 to 2025-11-14 | relative_selection_score=3, periodicity=63 | +0.2262 | +1.2854 |

WF efficiency = **2.38** (OOS mean / IS mean; >1 means OOS BEAT in-sample selection — the opposite of the overfit signature IS >> OOS). 100% of folds OOS-positive. Param choices: 4 distinct combination(s) across 5 folds (modal share 40%).

## 6. Subperiod robustness

| Period | Window | Ann active | IR | Quarterly hit rate |
|---|---|---|---|---|
| Full | 2010-03-05 to 2025-11-14 | +0.65% | +0.30 | 56% (64q) |
| Half 1 | 2010-03-05 to 2018-01-09 | +1.05% | +0.48 | 64% (33q) |
| Half 2 | 2018-01-10 to 2025-11-14 | +0.24% | +0.11 | 50% (32q) |
| Third 1 | 2010-03-05 to 2015-05-29 | +1.49% | +0.66 | 73% (22q) |
| Third 2 | 2015-06-01 to 2020-08-21 | +0.09% | +0.05 | 41% (22q) |
| Third 3 | 2020-08-24 to 2025-11-14 | +0.35% | +0.17 | 50% (22q) |

## 7. Rolling 252-day IR

59.4% of 3845 rolling windows positive; median +0.20; worst -2.10 (2024-05-01); best +2.68 (2014-05-27); latest +0.39.

## 8. Cost stress

Annualized one-sided turnover = 92.0%. Headline IR is gross of transaction costs (the engine's daily curve carries no TC; costs only hit period net returns). Stress deducts turnover*tc at each period end.

| tc (bps) | IR (ann) | Ann active return | Ann cost drag |
|---|---|---|---|
| 0 | +0.302 | +0.65% | 0.000% |
| 2 | +0.293 | +0.63% | 0.018% |
| 5 | +0.279 | +0.60% | 0.046% |
| 10 | +0.257 | +0.55% | 0.092% |
| 20 | +0.211 | +0.45% | 0.184% |

## 9. Parameter neighborhood (1-D sweeps around the default)

| Config (delta vs default) | IR (ann) | Total active return | Max DD (active eq.) |
|---|---|---|---|
| default (N=5, p=63, as=0.30) | +0.302 | +10.66% | -4.76% |
| relative_selection_score=3 | +0.205 | +7.19% | -4.76% |
| relative_selection_score=4 | +0.231 | +8.17% | -5.49% |
| relative_selection_score=6 | +0.299 | +10.35% | -4.41% |
| relative_selection_score=7 | +0.314 | +11.12% | -3.99% |
| periodicity=42 | +0.230 | +8.53% | -6.47% |
| periodicity=84 | +0.161 | +5.84% | -5.96% |
| active_share=0.2 | +0.322 | +10.61% | -4.40% |
| active_share=0.4 | +0.272 | +10.25% | -5.31% |

9 configurations: **100% positive**, IR mean +0.259 (std 0.056), range [+0.161, +0.322]. Default config z-score = +0.76 — the default is NOT a lone spike in its neighborhood.

## 10. Composite relative IC @63d — time stability

| Window | Dates | n | Mean IC | t-stat | Hit rate |
|---|---|---|---|---|---|
| full | 2010-06-02 to 2025-08-19 | 64 | +0.0432 | +1.32 | 62% |
| first_half | 2010-06-02 to 2017-11-27 | 32 | +0.0784 | +1.65 | 66% |
| second_half | 2018-02-22 to 2025-08-19 | 32 | +0.0080 | +0.18 | 59% |

Note: the IC here is signal-vs-segment-countries (benchmark plays no role) — it measures the raw ranking skill the book monetizes.

## 11. Trial-count honesty — DSR at the cross-run ledger N = 186

Ledger: ~150 legacy scenarios + 8 segment runs + 18 tournament specs + 4 S5 confirmation books + 6 ACWI-relative books = 186 trials (~200). Using the sweep trial sigma (0.00335/day) with N = 186 raises the deflated benchmark to SR0 = +0.146 annualized — vs the actual IR of +0.302 — and gives **DSR = 0.734** (vs 0.838 at the certified N = 4; at the round N = 200: SR0 +0.147, DSR 0.733). The required IR rises to +0.556; years needed at the current IR: 112 years. Caveat: the sigma behind SR0 is estimated from only 4 sweep trials and those 186 ledger trials are highly correlated (variants of the same V+M signal on overlapping data), so this is a worst-case deflation, not a point estimate. The qualitative conclusion — the DSR gate is unreachable on this span at IR ~0.3 — holds at any N >= 4.

## 12. Verdict

### Overfitting signature scoreboard

| Signature | Present? |
|---|---|
| wf_is_much_greater_than_oos (WFE<0.5) | absent |
| wf_unstable_param_choices (modal share<0.6) | **PRESENT** |
| alpha_concentrated_in_one_subperiod (<=1 of 3 thirds IR>0) | absent |
| fails_random_signal_null (MC p>0.05) | absent |
| default_config_lone_spike (abs z>1.5 or frac+<0.7) | absent |
| edge_dies_at_realistic_costs (IR<=0 at 10bps) | absent |
| ic_sign_flips_between_halves | absent |

**1 of 7 overfitting signatures present** (discussed below; none of the flagged items are parameter-tuning signatures).

### Significance summary

| Test | Statistic | Reads |
|---|---|---|
| MC random-signal null (n=500) | p = 0.0439 (percentile 95.8) | significant at 5% |
| MC family-wise (Bonferroni x6) | p = 0.263 | NOT significant at 5% |
| Stationary bootstrap, one-sided IR<=0 (n=2000) | p = 0.0535 | NOT significant at 5% |
| Lo t-stat / PSR / DSR (certified) | t = 1.22, PSR = 0.887, DSR = 0.838 | power-limited at this span (Section 1) |
| DSR at honest ledger N = 186 | DSR = 0.734 | worst-case deflation (Section 11) |

### Conclusion

**Not overfitted in the parameter / selection-tuning sense, and the DSR failure is a statistical-power artifact — BUT the decomposition shows the realized IR is a structural DM-vs-World composition bet, not within-DM selection alpha, and headline significance is marginal (bootstrap p 0.053, family-wise MC p 0.26).** Evidence by leg:

1. **Power.** At the realized IR of +0.30, DSR >= 0.95 needs ~45 years of data (plain PSR: 30y; t >= 2: 44y); the sample has 16.3. Equivalently the gate demands IR +0.47 (t 1.88) on this span — 1.5x the point estimate. A true IR-0.3 strategy fails this gate by construction; the failure carries no information about overfitting.
2. **Parameter-tuning probes.** Walk-forward WFE 2.38 with 100% of folds OOS-positive (OOS beat IS selection — the opposite of the IS>>OOS signature); the parameter neighborhood is 100% positive with the default at z = +0.76 (not the peak — a tuned config would sit at the peak); the edge survives 10x assumed costs (IR +0.21 at 20 bps, turnover 92%/yr); the random-signal null — construction held constant — gives p = 0.044 on 500 sims (book at the 96th percentile); and the stationary bootstrap puts p(IR<=0) at 0.053 with a 90% IR CI of [-0.00, +0.59]. Flagged-but-benign: WF param choices rotate across folds (modal share 40%) because the neighborhood is flat — IS Sharpes are statistically indistinguishable, so the argmax is noise (corroborated by 100% of configs positive) — and the deployed params were fixed ex-ante by literature priors, not selected by this WF.
3. **Segment bet vs selection.** The active return splits exactly into the passive DM-vs-World spread (ann +0.75%, IR +0.46, NW t +2.39) and within-DM selection (ann -0.10%, IR -0.04, NW t -0.22), corr -0.55; variance shares selection 142% / spread 56% / 2cov -98%. The realized +0.30 IR vs World is carried entirely by the structural DM overweight; the V+M overlay nets -0.10%/yr against the vendor DM cap index on this sample. Reconciliation with the MC result: random-signal tilts under the SAME construction average only +0.12 IR vs the +0.46 passive spread (the Cap_Tilt machinery itself dilutes the spread), so the signal beats random tilts (p = 0.044) — but 'better than random tilts' is a weaker claim than 'adds value over the passive DM index', and the data do not support the latter.
4. **Time profile.** Half-1 IR +0.48 (q-hit 64%) vs half-2 IR +0.11 (q-hit 50%); thirds +0.66 / +0.05 / +0.17; composite IC +0.078 -> +0.008 across halves; 59% of rolling 252d windows positive (latest +0.39). The alpha is front-loaded.

### Residual risks (stated honestly)

1. **Post-hoc candidate selection (family-wise honesty).** This book was chosen as the lead candidate AFTER observing the full evaluation round (6 ACWI-relative books (World/DM/EM x cap_tilt/eqw) evaluated in this round — 1-of-6); Bonferroni across the family puts the MC evidence at ~0.26 family-wise — suggestive, not significant at 5%. The honest claim is 'signal-beats-random-tilts significant within this book, but only marginal family-wise'.
2. **Big-N deflation.** Against the full cross-run ledger (N~186) the DSR drops to 0.734 and the deflated benchmark (+0.15) approaches the realized IR — under worst-case trial accounting no amount of history at this IR certifies. The program has consumed many trials; only fresh OOS data (live quarters) repays that debt.
3. **Alpha time profile.** The alpha is front-loaded (Sections 6/10); the per-period IC t-stats are individually insignificant throughout (skill shows in the book-level MC test, not the period IC), so the edge rests on construction + breadth — monitor the rolling IR and quarterly re-certs; a sustained negative rolling-IR trend is grounds to pull the candidate regardless of the full-sample stats.
4. **Structural-bet dependence.** The active mean is entirely the passive DM-vs-World spread (+0.75%/yr of the +0.65%/yr active mean; 56% of active variance); if the DM-vs-World regime reverses (e.g. an EM-led cycle), that leg of the IR reverses with it, independent of selection skill.

*Generated by `scripts/overfit_forensics.py` in 229.4s; deterministic (seeded); inputs are the gitignored vendor data.*
