# Rebalancing Calendar — Deployed Strategies

Both deployed books rebalance on the **same 63-trading-day grid** (quarterly,
the certified cadence). Anchor: last executed rebalance **2025-11-14**
(production run `run_20251114`). Generated 2026-06-10 via
`python scripts/pipeline.py calendar --periods 8`.

## Schedule (next ~2 years)

| # | Rebalance date | Status (as of 2026-06-10) |
|---|---|---|
| 1 | **2026-02-11** | **OVERDUE** — elapsed without data refresh |
| 2 | **2026-05-11** | **OVERDUE** — elapsed without data refresh |
| 3 | 2026-08-06 | next actionable |
| 4 | 2026-11-03 | |
| 5 | 2027-01-29 | |
| 6 | 2027-04-28 | |
| 7 | 2027-07-26 | |
| 8 | 2027-10-21 | |

Applies to both `EM_captilt_vsEM` and `DM_captilt_vsACWI` (identical grid).

## How to run a rebalance

A few days before each date, refresh `Inputs/` + `Classification.xlsx` with
data through (at least) the rebalance date, then:

```bash
# Both strategies — the normal quarterly cycle:
python scripts/pipeline.py quarterly

# One strategy only:
python scripts/pipeline.py quarterly --strategy EM_captilt_vsEM
python scripts/pipeline.py quarterly --strategy DM_captilt_vsACWI

# Allocations only (skip re-certification):
python scripts/pipeline.py production --strategy EM_captilt_vsEM
```

New weights land in
`outputs/production/run_{data_end}/{strategy_id}/allocations_latest.json`,
which also re-emits the **authoritative** next rebalance date.

## Caveats

- Dates beyond the next one are **holiday-approximate**: the engine steps 63
  *business* days without an exchange holiday calendar, and the realized grid
  follows the data's actual trading days. Expect ±1–3 day drift per step;
  always trust `allocations_latest.json: next_rebalance_date` after each run.
- Regenerate this table after every production run:
  `python scripts/pipeline.py calendar` (it anchors on the latest run
  directory automatically and flags overdue dates).
- The two missed rebalances (2026-02-11, 2026-05-11) are recoverable: one
  data refresh + one `pipeline.py quarterly` produces the current allocation
  and converts the elapsed period into out-of-sample evidence at the next
  re-certification.
