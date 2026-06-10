# Runbook — Quarterly Production Cycle & Re-Certification

Deployed registry (`configs/production.json`), status as of 2026-06-10:

| id | Book | Benchmark | Status |
|---|---|---|---|
| `EM_captilt_vsEM` | EM Cap-Tilt vm @63d, active_share 0.30 | EM vendor cap index | **Power-limited candidate** — IR +0.29, MC p 0.030, bootstrap CI straddles 0; paper-trade |
| `DM_captilt_vsACWI` | DM Cap-Tilt vm @63d, active_share 0.30 | World (ACWI-equiv) | **Composition bet** — alpha = passive DM−ACWI spread (NW t 2.39); selection leg −0.10%/yr; never market as selection skill |

## When to run
Each calendar quarter, after `Inputs/` is refreshed with the new quarter of vendor data. On unchanged data every artifact is bit-for-bit reproducible — re-running adds no information.

## The one command
```
python scripts/pipeline.py quarterly
```
Sequence (each stage independently re-runnable; the plan is registry-derived):
1. `recert:{id}` — `research_run.py` per deployed strategy, net-of-cost (`--costs configs/costs.json`), writing `outputs/research/verdict_*_tca.json` + HTML reports;
2. `production` — `production_run.py`: allocations / signals / metrics / TCA under `outputs/production/run_{data_end}/`;
3. `dashboards` — production + research HTML dashboards.

`--dry-run` validates the registry and inputs and prints the exact commands.
`--quick` for smoke runs only — never for a certification read.

## Read the verdicts — certification gates (ALL must pass), ACTIVE basis
| Field in `stats` | Gate | EM (2026-06-10) | DM-vsACWI (2026-06-10) |
|---|---|---|---|
| `sharpe_t_stat` (IR t) | ≥ 2.0 | 1.18 | 1.22 |
| `psr` | ≥ 0.95 | 0.88 | 0.89 |
| `dsr` | ≥ 0.95 | 0.76 | 0.84 |
| `mc_p_value` | ≤ 0.05 | **0.030 ✓** | **0.040 ✓** |
| `bootstrap_ci[0]` (IR CI low) | > 0 | −0.008 | −0.003 |
| `nw_t_vs_eqw` | > 0 | 0.54 ✓ | 2.77 ✓ |
| `wf_efficiency` | ≥ 0.5 | 2.85 ✓ | 2.40 ✓ |
| `frac_oos_positive` | ≥ 0.5 | 0.80 ✓ | 1.00 ✓ |
| `stability_frac_positive` ≥ 0.7 & \|`stability_default_zscore`\| ≤ 1.5 | — | ✓ | ✓ |

`verdict.overall == true` ⇒ certified. Neither book certifies today; the failures are
**power gates** (t/PSR/DSR/CI) — see `outputs/research/overfit_forensics_*.md`: at IR ≈ 0.3,
DSR ≥ 0.95 needs ~45–87 years. The dashboard's evidence-grade banner computes this
mechanically (certified / power-limited / weak / negative).

## Kill-switch monitors (check EVERY quarter)
1. **Rolling 252-day IR** (`metrics.json: last_252d.ir_ann`) — sustained negative ⇒ de-risk.
2. **MC gate** — `mc_p_value` failing (> 0.05) at re-cert ⇒ selection skill no longer distinguishable from random ⇒ de-risk.
3. **Alpha front-loading** — second-half IC was ~0 at deployment for both books; if fresh quarters do not improve the trailing IC, the "decayed signal" hypothesis gains.
4. **DM book only:** re-run the alpha decomposition (`python scripts/overfit_forensics.py --segment DM --bmk-index World`) — if the selection leg turns *significantly* negative, the overlay is paying for nothing.

## Do NOT
- Lower thresholds to force a pass.
- Switch to monthly (@21d) — pre-registered comparison degraded every stat (MC p 0.257; segment_verdicts_2026-06-09.md Addendum §1).
- Add factors or specs post-hoc — the 6-spec tournament consumed its lockbox; every new variant debits the ~200-trial DSR ledger. Only live quarters add evidence for free.
- Read `--quick` runs as certification evidence (validation suite is truncated).

## Optional cross-checks (same quarter)
```
python scripts/research_run.py --segment DM    --track prior --prior-set full --periodicity 63   # 4-category sanity
python scripts/research_run.py --segment World --track prior --prior-set vm   --periodicity 63   # broader universe
```

## Provenance
Every `manifest.json` records git commit, package version, data end date and file inventory;
`metrics.json` is timestamp-free by design — identical inputs ⇒ identical bytes (verified
2026-06-10: 20/20 artifacts hash-identical across consecutive full runs).
