# Overfitting Forensics — EM Cap-Tilt (vm @63d vs EM cap index)

**Question:** is the strategy overfitted, or is the DSR failure (0.762 < 0.95) a statistical-power artifact?  **Data:** 4096 active daily returns (16.3y, 2010-03-05 to 2025-11-14). Reconstruction reproduces the certified verdict bit-for-bit (sharpe_ann / t / PSR / DSR all match to <1e-8). Script: `scripts/overfit_forensics.py`; machine-readable results in `outputs/research/overfit_forensics_EM.json` (gitignored, regenerable).

## 1. DSR power decomposition

The certified DSR uses the validation sweep's 4 trial Sharpes (sigma_daily = 0.00696) giving a deflated benchmark SR0 = 0.00733/day (+0.116 annualized). Inverting PSR(SR0) = 0.95 at the observed sample size, skew (+0.79) and kurtosis (13.3):

| Quantity | Actual | Required for DSR >= 0.95 |
|---|---|---|
| Annualized IR (active Sharpe) | +0.292 | +0.520 |
| Lo (2002) t-stat | 1.18 | 2.09 |
| DSR | 0.762 | 0.950 |

At the CURRENT IR of +0.292, the sample needed is **87 years** for DSR >= 0.95, 31 years for plain PSR >= 0.95, and 47 years for Sharpe t >= 2 (t ~ IR x sqrt(years)). The strategy has 16.3 years. A true IR ~0.3 cannot mathematically clear a t~2-equivalent gate on this span regardless of whether the edge is real — the DSR gate is power-limited here, by construction.

## 2. Monte-Carlo random-signal null (n = 500)

| Metric | Value |
|---|---|
| Actual IR (ann) | +0.292 |
| Null mean / std | +0.130 / 0.098 |
| Null q95 | +0.280 |
| Actual percentile in null | 96.6 |
| p-value (add-one) | **0.0359** |
| Family-wise (x3 segments, Bonferroni) | 0.108 |

The null holds the Cap_Tilt construction, costs and benchmark constant and randomizes only the signal — this isolates selection skill from construction. 500 sims confirm the certified n=100 result.

## 3. Walk-forward forensics (5 anchored folds)

| Fold | OOS window | Chosen params | IS Sharpe (per-period) | OOS Sharpe (per-period) |
|---|---|---|---|---|
| 1 | 2010-06-28 to 2013-07-24 | relative_selection_score=3, periodicity=21 | +0.1491 | +0.0597 |
| 2 | 2013-07-25 to 2016-08-22 | relative_selection_score=3, periodicity=63 | +0.0651 | +0.0591 |
| 3 | 2016-08-23 to 2019-09-19 | relative_selection_score=7, periodicity=63 | +0.0951 | +0.2649 |
| 4 | 2019-09-20 to 2022-10-18 | relative_selection_score=5, periodicity=63 | +0.0855 | -0.0361 |
| 5 | 2022-10-19 to 2025-11-14 | relative_selection_score=7, periodicity=21 | +0.0202 | +0.2537 |

WF efficiency = **1.45** (OOS mean / IS mean; >1 means OOS BEAT in-sample selection — the opposite of the overfit signature IS >> OOS). 80% of folds OOS-positive. Param choices: 5 distinct combination(s) across 5 folds (modal share 20%).

## 4. Subperiod robustness

| Period | Window | Ann active | IR | Quarterly hit rate |
|---|---|---|---|---|
| Full | 2010-03-05 to 2025-11-14 | +1.25% | +0.29 | 55% (64q) |
| Half 1 | 2010-03-05 to 2018-01-09 | +2.10% | +0.56 | 73% (33q) |
| Half 2 | 2018-01-10 to 2025-11-14 | +0.40% | +0.08 | 38% (32q) |
| Third 1 | 2010-03-05 to 2015-05-29 | +3.11% | +0.88 | 73% (22q) |
| Third 2 | 2015-06-01 to 2020-08-21 | -0.33% | -0.09 | 50% (22q) |
| Third 3 | 2020-08-24 to 2025-11-14 | +0.97% | +0.18 | 41% (22q) |

## 5. Rolling 252-day IR

61.7% of 3845 rolling windows positive; median +0.36; worst -1.98 (2019-07-11); best +2.40 (2015-01-02); latest -0.35.

## 6. Cost stress

Annualized one-sided turnover = 118.0%. Headline IR is gross of transaction costs (the engine's daily curve carries no TC; costs only hit period net returns). Stress deducts turnover*tc at each period end.

| tc (bps) | IR (ann) | Ann active return | Ann cost drag |
|---|---|---|---|
| 0 | +0.292 | +1.25% | 0.000% |
| 2 | +0.286 | +1.23% | 0.024% |
| 5 | +0.277 | +1.19% | 0.059% |
| 10 | +0.263 | +1.13% | 0.118% |
| 20 | +0.235 | +1.01% | 0.236% |

## 7. Parameter neighborhood (1-D sweeps around the default)

| Config (delta vs default) | IR (ann) | Total active return | Max DD (active eq.) |
|---|---|---|---|
| default (N=5, p=63, as=0.30) | +0.292 | +20.71% | -8.50% |
| relative_selection_score=3 | +0.305 | +21.82% | -10.07% |
| relative_selection_score=4 | +0.298 | +21.39% | -8.57% |
| relative_selection_score=6 | +0.250 | +16.94% | -8.82% |
| relative_selection_score=7 | +0.258 | +17.23% | -11.02% |
| periodicity=42 | +0.258 | +17.58% | -9.98% |
| periodicity=84 | +0.392 | +29.71% | -8.08% |
| active_share=0.2 | +0.269 | +17.72% | -8.94% |
| active_share=0.4 | +0.304 | +23.14% | -9.23% |

9 configurations: **100% positive**, IR mean +0.292 (std 0.043), range [+0.250, +0.392]. Default config z-score = +0.00 — the default is NOT a lone spike in its neighborhood.

## 8. Composite relative IC @63d — time stability

| Window | Dates | n | Mean IC | t-stat | Hit rate |
|---|---|---|---|---|---|
| full | 2010-06-02 to 2025-08-19 | 64 | +0.0342 | +0.88 | 47% |
| first_half | 2010-06-02 to 2017-11-27 | 32 | +0.0720 | +1.30 | 53% |
| second_half | 2018-02-22 to 2025-08-19 | 32 | -0.0037 | -0.07 | 41% |

## 9. Trial-count honesty — DSR at the cross-run ledger N = 180

Ledger: ~150 legacy scenarios + 8 segment runs + 18 tournament specs + 4 S5 confirmation books. Using the sweep trial sigma (0.00696/day) with N = 180 raises the deflated benchmark to SR0 = +0.302 annualized — ABOVE the actual IR of +0.292 — and gives **DSR = 0.484** (vs 0.762 at the certified N = 4). The required IR rises to +0.704; years needed at the current IR: INFINITE — the current IR sits below the deflated benchmark, so no sample length clears DSR at this IR under honest big-N accounting; only a higher realized IR (or a smaller honest trial family) can. Caveat: the sigma behind SR0 is estimated from only 4 sweep trials and those 180 ledger trials are highly correlated (variants of the same V+M signal on overlapping data), so this is a worst-case deflation, not a point estimate. The qualitative conclusion — the DSR gate is unreachable on this span at IR ~0.3 — holds at any N >= 4.

## 10. Verdict

### Overfitting signature scoreboard

| Signature | Present? |
|---|---|
| wf_is_much_greater_than_oos (WFE<0.5) | absent |
| wf_unstable_param_choices (modal share<0.6) | **PRESENT** |
| alpha_concentrated_in_one_subperiod (<=1 of 3 thirds IR>0) | absent |
| fails_random_signal_null (MC p>0.05) | absent |
| default_config_lone_spike (abs z>1.5 or frac+<0.7) | absent |
| edge_dies_at_realistic_costs (IR<=0 at 10bps) | absent |
| ic_sign_flips_between_halves | **PRESENT** |

**2 of 7 overfitting signatures present** (both discussed below; neither is a parameter-tuning signature).

### Conclusion

**The strategy is NOT overfitted in the parameter / selection-tuning sense, and the DSR failure is a statistical-power artifact — but the alpha is front-loaded, which is a real (separate) risk.** Three lines of evidence support each leg:

1. **Power, decisively.** At the realized IR of +0.29, DSR >= 0.95 needs ~87 years of data (plain PSR: 31y; t >= 2: 47y); the sample has 16.3. Equivalently the gate demands IR +0.52 (t 2.09) on this span — 1.8x the point estimate. A true IR-0.3 strategy fails this gate by construction; the failure carries no information about overfitting.
2. **Every parameter-tuning probe is clean.** Walk-forward OOS folds BEAT in-sample selection (WFE 1.45, 80% folds positive — the opposite of the IS>>OOS signature); the parameter neighborhood is 100% positive with the default at z = +0.00 (dead-center, and NOT the peak — p=84 scores higher at +0.39; a tuned config would sit at the peak); the edge survives 10x assumed costs (IR +0.23 at 20 bps); and the random-signal null — construction held constant — rejects at p = 0.036 on 500 sims (actual book at the 97th percentile of the null). The two flagged signatures are benign variants: WF param choices rotate across folds precisely BECAUSE the neighborhood is flat (IS Sharpes are statistically indistinguishable, so the argmax is noise — corroborated by 100% of configs positive), and the deployed params were fixed ex-ante by literature priors, not selected by this WF.
3. **The honest weakness is time-decay, not tuning.** The alpha is front-loaded: half-1 IR +0.56 (q-hit 73%) vs half-2 IR +0.08 (q-hit 38%); thirds +0.88 / -0.09 / +0.18; composite IC fades from +0.072 to -0.004 across halves. Two of three thirds and 62% of rolling windows are still positive, so the alpha is not a single-episode artifact, but the recent-sample evidence alone would not justify deployment.

### Residual risks (stated honestly)

1. **Post-hoc segment selection.** EM was chosen as the deploy candidate AFTER observing all three segments; a Bonferroni-style correction across the 3-segment family puts the MC evidence at ~0.11 family-wise — suggestive, not significant at 5%. The honest claim is 'selection skill significant within EM, marginal family-wise'.
2. **Big-N deflation.** Against the full cross-run ledger (N~180) the DSR drops to 0.484 and the deflated benchmark (+0.30) exceeds the realized IR — under worst-case trial accounting no amount of history at this IR certifies. The program has consumed many trials; only fresh OOS data (live quarters) repays that debt.
3. **Alpha decay.** Section 4/8: most of the realized alpha is pre-2018 and the second-half IC is ~0. The per-period IC t-stats are individually insignificant in EM throughout (skill shows in the MC book-level test, not the period IC), so the edge rests on construction + breadth — monitor the rolling IR and quarterly re-certs; a continued flat second-half trend is grounds to pull the candidate regardless of the full-sample stats.

*Generated by `scripts/overfit_forensics.py` in 107.8s; deterministic (seeded); inputs are the gitignored vendor data.*
