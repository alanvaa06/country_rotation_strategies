# Runbook — Quarterly Re-Certification (DM 50/50 Value+Momentum)

Lead candidate: **DM 50/50 Value+Momentum, top-5 relative, quarterly (63d) rebalance, 2bps TC.**
Status as of 2026-06-09: nearest miss, not certified (t 1.96, MC p 0.059, DSR 0.91). Data limited to 2010+.

## When to run
Each calendar quarter, after `Inputs/` is refreshed with the new quarter of vendor data. (On unchanged data the verdict is bit-for-bit reproducible — re-running adds no information.)

## Command
```
python scripts/research_run.py --segment DM --track prior --prior-set vm --periodicity 63
```
`--basis active` is the default — headline stats are computed on **excess-over-benchmark** (alpha / Information Ratio), not the beta-laden absolute book return. Writes (gitignored): `outputs/research/verdict_DM_prior_vm_p63_active.json`, `report_DM_prior_vm_p63_active.html`.

## Read the verdict — certification gates (all must pass), ACTIVE basis
| Field in `stats` | Gate | Current (active) |
|---|---|---|
| `sharpe_t_stat` (active = IR t) | ≥ 2.0 | 1.34 |
| `psr` | ≥ 0.95 | 0.91 |
| `dsr` | ≥ 0.95 | 0.84 |
| `mc_p_value` | ≤ 0.05 | 0.050 |
| `bootstrap_ci[0]` (IR CI low) | > 0 | **−0.005** |
| `nw_t_vs_eqw` | > 0 | 1.56 |
| `wf_efficiency` | ≥ 0.5 | 4.04 ✓ |
| `frac_oos_positive` | ≥ 0.5 | 1.00 ✓ |
| `stability_frac_positive` ≥ 0.7 & \|`stability_default_zscore`\| ≤ 1.5 | — | ✓ |

`verdict.overall == true` ⇒ certified. Current active verdict: **not certified** — the alpha (IR ≈ 0.5 point estimate) is not yet statistically distinguishable from zero (bootstrap IR CI straddles 0).

## Expected trajectory
With a true active Sharpe (IR) ≈ 0.5, expected active t ≈ 0.5·√(years). At ~6.5 effective years of usable history the active t is ~1.34; certification (active t ≥ 2 ⇒ ~16 years of edge, DSR ≥ 0.95 a bit more) is reachable late this decade **only if the alpha persists**. Each live quarter nudges the active t, PSR, DSR upward if the strategy keeps delivering excess return — but the bar is alpha-significance, not absolute return, and it is genuinely far. Watch the **bootstrap IR CI low crossing above 0** as the first leading indicator.

## Do NOT
- Lower thresholds to force a pass.
- Switch to monthly (@21d) — proven to degrade the edge (MC p 0.257; see segment_verdicts_2026-06-09.md Addendum §1).
- Add factors post-hoc to the prior set — that re-opens multiple-testing; the prior set is fixed ex-ante from the literature.

## Optional cross-checks (same quarter)
```
python scripts/research_run.py --segment DM    --track prior --prior-set full --periodicity 63   # 4-category sanity
python scripts/research_run.py --segment World --track prior --prior-set vm   --periodicity 63   # broader universe
```
