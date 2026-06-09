# Canonical Validation Formulas (ground truth for validation/statistics.py)

Conventions: daily simple returns `r = equity.pct_change().dropna()`; `n = len(r)`;
daily Sharpe `SR = mean(r)/std(r, ddof=1)`; annualized `SR_ann = SR*sqrt(252)`;
skew `γ3` unbiased; **non-excess** kurtosis `γ4` (Normal → 3), i.e. `scipy.stats.kurtosis(r, fisher=False, bias=False)`.

## Lo (2002) IID Sharpe standard error
`SE(SR) = sqrt((1 + 0.5*SR^2) / n)` ; `t = SR / SE` (annualization-invariant).

## Probabilistic Sharpe Ratio — Bailey & López de Prado (2012)
`PSR(SR*) = Φ( (SR − SR*) · sqrt(n − 1) / sqrt(1 − γ3·SR + ((γ4 − 1)/4)·SR²) )`
with SR, SR* in the same (daily) units. Negative radicand → clamp to small positive (pathological).

## Deflated Sharpe Ratio — Bailey & López de Prado (2014)
`SR0 = sqrt(Var({SR_trials})) · ( (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) )`
where γ = Euler–Mascheroni 0.5772156649…, N = number of trials, Var across observed trial Sharpes (daily units), **no mean term** (null assumes E[SR]=0).
`DSR = PSR(SR0)`. Edge cases: N ≤ 1 or Var ≤ 0 → SR0 = 0.
Property: more trials with same observed SR ⇒ higher SR0 ⇒ lower DSR.

## Newey–West (1987) HAC t-stat of mean
Input: a RETURN series x (e.g. strategy-minus-null daily returns), NOT an equity curve.
Default lag `L = floor(4·(n/100)^(2/9))`, min 1. Bartlett weights `w_l = 1 − l/(L+1)`.
`γ_l = (1/n)·Σ e_t e_{t−l}` with `e = x − mean(x)`;
`Var_HAC = γ_0 + 2·Σ_{l=1..L} w_l·γ_l` ; `SE = sqrt(Var_HAC/n)` ; `t = mean(x)/SE`.

## Stationary bootstrap — Politis & Romano (1994)
Geometric blocks, mean block length `b` (p = 1/b), wrap-around indices, seeded RNG.
CI from percentiles (α/2, 1−α/2) of resampled Sharpe distribution. Deterministic under fixed seed.

## Scorecard thresholds (design policy, spec §7)
DSR ≥ 0.95; PSR ≥ 0.95; MC p ≤ 0.05; WFE ≥ 0.5 with ≥50% folds OOS-positive;
Sharpe t ≥ 2; bootstrap CI low > 0; NW t vs equal-weight-B&H > 0;
stability: ≥70% grid Sharpes positive and |default z| ≤ 1.5;
HLZ: raw t < 3 ⇒ "weak" label; BH-FDR q = 0.10 across factor family.
