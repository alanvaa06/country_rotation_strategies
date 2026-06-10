# Runbook — Quarterly Re-Certification (DM 50/50 Value+Momentum)

Lead candidate: **DM 50/50 Value+Momentum, top-5 relative, quarterly (63d) rebalance, 2bps TC.**
Status as of 2026-06-09: nearest miss, not certified (t 1.96, MC p 0.059, DSR 0.91). Data limited to 2010+.

## When to run
Each calendar quarter, after `Inputs/` is refreshed with the new quarter of vendor data. (On unchanged data the verdict is bit-for-bit reproducible — re-running adds no information.)

## Command
```
python scripts/research_run.py --segment DM --track prior --prior-set vm --periodicity 63
```
Writes (gitignored): `outputs/research/verdict_DM_prior_vm_p63.json`, `report_DM_prior_vm_p63.html`.

## Read the verdict — certification gates (all must pass)
| Field in `stats` | Gate | Current |
|---|---|---|
| `sharpe_t_stat` | ≥ 2.0 | 1.96 |
| `psr` | ≥ 0.95 | 0.97 |
| `dsr` | ≥ 0.95 | 0.91 |
| `mc_p_value` | ≤ 0.05 | 0.059 |
| `bootstrap_ci[0]` | > 0 | 0.104 |
| `nw_t_vs_eqw` | > 0 | 1.56 |
| `wf_efficiency` | ≥ 0.5 | 4.04 ✓ |
| `frac_oos_positive` | ≥ 0.5 | 1.00 ✓ |
| `stability_frac_positive` ≥ 0.7 & \|`stability_default_zscore`\| ≤ 1.5 | — | 1.00 / 0.54 ✓ |

`verdict.overall == true` ⇒ certified. Until then it stays a watch-list candidate.

## Expected trajectory
With a true active Sharpe ≈ 0.5, expected t ≈ 0.5·√(years). Certification (t≥2 ⇒ ~16y, DSR≥0.95 a bit more) becomes reachable around **2026–2028** purely from accumulating live quarters, assuming the edge persists. Each quarter nudges t, PSR, DSR upward if the strategy keeps delivering.

## Do NOT
- Lower thresholds to force a pass.
- Switch to monthly (@21d) — proven to degrade the edge (MC p 0.257; see segment_verdicts_2026-06-09.md Addendum §1).
- Add factors post-hoc to the prior set — that re-opens multiple-testing; the prior set is fixed ex-ante from the literature.

## Optional cross-checks (same quarter)
```
python scripts/research_run.py --segment DM    --track prior --prior-set full --periodicity 63   # 4-category sanity
python scripts/research_run.py --segment World --track prior --prior-set vm   --periodicity 63   # broader universe
```
