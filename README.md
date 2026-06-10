# Country Rotation Strategy

A factor-based country equity rotation research platform for systematic investment decisions.

**Author:** Alan Vazquez, CFA  
**Last Updated:** June 2026  
**Public repo:** https://github.com/alanvaa06/country_rotation_strategies

---

## Overview

This platform implements a rigorous, leak-free research pipeline for building and validating factor-based country rotation strategies. It segments the investable universe into **World / DM / EM** and scores countries on multi-factor models across Valuation, Quality, Profitability, and Momentum. Every result is statistically validated before any inference is drawn.

Key design commitments:
- **No look-ahead leakage** — ffill-only gap fill with configurable per-metric publication lags; perturbation-tested via `data/integrity.py`.
- **OOS-honest factor screening** — per-period IC t-stats (Grinold-Kahn), Benjamini-Hochberg FDR, HLZ weak labels, and a terminal lockbox that is never touched during selection.
- **Literature-grounded factor catalog** — corrected vs. legacy (no raw levels as factors; 12-1 / 6-1 momentum; ROE in Profitability, not Valuation). See [docs/research/country_rotation_literature.md](docs/research/country_rotation_literature.md).
- **Statistically validated** — DSR / PSR / Lo (2002) Sharpe t / anchored walk-forward / Monte-Carlo random-selection null; hard scorecard thresholds. See [docs/references/validation_formulas.md](docs/references/validation_formulas.md).

---

## Architecture

### Package: `country_rotation/`

```
country_rotation/
├── config.py            # Frozen dataclass platform config (DataConfig, BacktestConfig)
├── data/
│   ├── ingestion.py     # Load Excel workbooks from Inputs/ into aligned DataFrames
│   ├── processing.py    # Derived metrics: yields, spreads, margins, 12-1/6-1 momentum
│   └── integrity.py     # Leakage guards: lookahead_check (perturbation test) + coverage_matrix
├── factors/
│   ├── catalog.py       # Factor registry: name, category, direction, exploratory flag
│   ├── transforms.py    # 4 standardized metric transforms (zscore, abs_pct, rel_rank, delta_pct)
│   └── redundancy.py    # Hierarchical-clustering factor deduplication (ClusterResult)
├── signals/
│   └── composite.py     # Weighted metric avg → category aggregation → composite → cross-sectional norm
├── selection/
│   └── walkforward.py   # OOS-honest factor screening: BH-FDR, HLZ weak labels, lockbox
├── backtest/
│   ├── engine.py        # Full backtest engine (Engine + EngineResult); absolute/relative selection; Equal/Risk_Parity weights
│   ├── metrics.py       # Annualized return/vol/Sharpe, max drawdown, beta, IR, turnover
│   ├── ic.py            # Information coefficient calculation and IC statistics
│   └── benchmarks.py   # Equal-weight buy-and-hold null benchmark constructor
├── validation/
│   ├── statistics.py    # Lo (2002) Sharpe SE, PSR, DSR, Newey-West HAC, stationary bootstrap CI
│   ├── protocols.py     # Parameter sweep, anchored walk-forward, Monte-Carlo random-signal null
│   └── scorecard.py     # compute_validation(): single-call full evidence suite → ValidationReport
└── reporting/
    └── report.py        # HTML report renderer (8 figures + 5 tables, base64 PNG embeds)
```

### Legacy root scripts (deprecated — parity-locked)

The original monolithic scripts (`ProcessData.py`, `FactorTransformer.py`, `FactorTesting.py`, `backtest.py`, `backtesting_tests.py`, `strategy.py`, `test_normalized_scores.py`, `Threshold_Testing.py`) remain at the repo root. They are **not maintained**; they exist solely as a behavioral reference for the regression suite in `tests/test_parity.py`. Do not extend them.

---

## Methodology Highlights

### Factor Transforms

Each raw factor is converted into four standardized, cross-sectionally comparable metrics:

| Transform | Description |
|-----------|-------------|
| `zscore` | Expanding z-score converted to percentile (historical context) |
| `abs_pct` | Historical percentile rank per country (`min` method) |
| `rel_rank` | Cross-sectional rank at each date (`average` method) |
| `delta_pct` | Percentile of 63-day percent changes (momentum of the factor) |

These are equal-weighted (1/4 each) by default; the weight mix is a tunable parameter.

### Factor Catalog (Corrected)

- **No raw levels as factors** — Price, GDP, M2, Market_Cap, EV, Revenue, Debt, Equity, EPS, Ten_Year, SI are not cross-country comparable; they are not factors in the catalog.
- **Momentum_12-1 / Momentum_6-1** added — strongest documented country-level factor (AMP 2013).
- **ROE / Fwd_ROE / Return_Capital** in Profitability, not Valuation (legacy error corrected).
- **RollingVol** in Quality, direction −1 (low-risk anomaly).

### Gap Fill and Publication Lag

Data gaps are forward-filled only (`ffill`, configurable limit). Backward fill is explicitly prohibited — it is classified as look-ahead leakage and will fail `lookahead_check`. Per-metric publication lags (configurable in `DataConfig.publication_lag_days`) shift each series forward to respect real-world data availability.

### OOS-Honest Factor Screening

`selection/walkforward.py` implements a two-stage screen:

1. **Anchored fold IC series** — per-period Spearman IC is computed on rolling OOS folds; t-stat uses Grinold-Kahn (df = n_ic_obs − 1) to avoid the inflated df of correlated expanding means.
2. **BH-FDR** — Benjamini-Hochberg correction across the full factor family (45 factors).  q < 0.10 survives.
3. **HLZ weak labels** — surviving factors are labeled weak/strong based on t-statistic magnitude, consistent with Harvey, Liu & Zhu (2016) multiple-testing standards.
4. **Lockbox** — the final `lockbox_frac` (default 20%) of the time series is never touched during screening. Any post-screening backtest that uses it is thus genuinely out-of-sample.

### Validation Scorecard Thresholds

`validation/scorecard.py` runs a single-call evidence suite. All thresholds must pass:

| Check | Metric | Threshold |
|-------|--------|-----------|
| No overfitting | Deflated Sharpe Ratio (DSR) | ≥ 0.95 |
| No overfitting | Walk-forward efficiency (OOS/IS Sharpe) | ≥ 0.50 |
| No overfitting | Fraction of OOS-positive WF folds | ≥ 0.50 |
| No overfitting | Monte-Carlo null p-value | ≤ 0.05 |
| Parameter stable | Fraction of sweep configs positive | ≥ 0.70 |
| Parameter stable | Default config |z-score| vs sweep | ≤ 1.50 |
| Statistically significant | Lo (2002) Sharpe t-stat | ≥ 2.00 |
| Statistically significant | Probabilistic Sharpe Ratio (PSR) | ≥ 0.95 |
| Statistically significant | Bootstrap CI low end | > 0.00 |
| Statistically significant | Newey-West t-stat vs equal-weight null | > 0.00 |

Formulas documented in [docs/references/validation_formulas.md](docs/references/validation_formulas.md).

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run tests

```bash
python -m pytest
```

### Production pipeline (one command)

The deployed-strategy registry `configs/production.json` drives everything.
The quarterly cycle — after refreshing `Inputs/` — is one command:

```bash
# Full cycle: re-certification runs -> production artifacts -> dashboards
python scripts/pipeline.py quarterly

# Stages individually (each independently re-runnable):
python scripts/pipeline.py recert        # research_run re-certs (net-of-cost) per deployed strategy
python scripts/pipeline.py production    # allocations / signals / TCA -> outputs/production/run_{data_end}/
python scripts/pipeline.py dashboards    # production + research HTML dashboards

# Validate registry + data and print the command plan without executing:
python scripts/pipeline.py quarterly --dry-run
```

Allocations land in `outputs/production/run_{YYYYMMDD}/{strategy_id}/allocations_latest.json`
(latest weights + next rebalance date). The run directory is stamped with the
**data end date**, and artifacts are byte-deterministic given the same inputs
(seeded validation; the run timestamp lives only in `manifest.json`).
Decision protocol (gate reading, registry updates, kill-switch criteria):
[docs/research/RUNBOOK_quarterly_recert.md](docs/research/RUNBOOK_quarterly_recert.md).

### Scripts

All entry-point scripts live in `scripts/`. They require `Inputs/` data (see note below).

| Script | Purpose |
|--------|---------|
| `pipeline.py` | **Quarterly orchestrator** — recert → production → dashboards, registry-driven |
| `production_run.py` | Periodic per-strategy artifacts (allocations, signals, metrics, TCA) |
| `build_production_dashboard.py` | Self-contained production dashboard HTML over the latest run |
| `research_run.py` | Per-segment research pipeline: screen/prior → composite → validation → report |
| `build_dashboard.py` | Research strategy dashboard (segment tabs × strategy toggle) |
| `overfit_forensics.py` | Deep overfitting diagnostics for a candidate (signatures, power, alpha decomposition) |
| `spec_tournament.py` | Pre-registered 6-spec signal tournament (screen-window IC + BH-FDR + lockbox) |
| `build_scores.py` / `run_backtest.py` / `build_report.py` | Single-stage building blocks (scores, backtest, HTML report) |

**Build factor scores (one segment):**
```bash
python scripts/build_scores.py --segment World --output outputs/scores/world_scores.csv
```

**Run backtest:**
```bash
python scripts/run_backtest.py \
    --scores outputs/scores/world_scores.csv \
    --prices Inputs/Price.xlsx \
    --segment World
```

**Build HTML report:**
```bash
python scripts/build_report.py \
    --scores outputs/scores/world_scores.csv \
    --prices Inputs/Price.xlsx \
    --segment World \
    --output outputs/reports/world_report.html
```

**Full research run (factor screen + validation + report):**
```bash
# Literature-prior track (no data-driven selection, lockbox not consumed)
python scripts/research_run.py --segment World --track prior

# OOS-honest screening track (BH-FDR screen + lockbox)
python scripts/research_run.py --segment EM --track screen

# Quick smoke run on synthetic data
python scripts/research_run.py --segment DM --track prior --quick
```

`--segment` accepts: `World`, `DM`, `EM`.  
`--track prior` fixes the factor set from documented literature priors (AMP 2013, Calice & Lin 2021).  
`--track screen` runs OOS-honest BH-FDR screening; consumes `lockbox_frac` of the series.

### Data note

`Inputs/` is **gitignored** — it contains proprietary vendor data (Bloomberg / FactSet workbooks) and is not part of the repository. You must supply your own data files matching the schema expected by `data/ingestion.py`.

---

## Testing

```bash
python -m pytest
```

**227 tests** across:
- Unit tests for every module (transforms, catalog, composite, engine, metrics, IC, benchmarks, statistics, protocols, scorecard, report).
- **Parity tests** (`tests/test_parity.py`) — regression suite that locks the new package to legacy script behavior; any behavioral divergence fails CI.
- **Leakage guards** (`tests/test_integrity.py`) — deliberate look-ahead injection; bfill classified as dirty; ffill classified as clean.
- **Smoke tests** (`tests/test_scripts_smoke.py`) — end-to-end `build_scores` + `run_backtest` on synthetic Excel inputs.

---

## Project Docs Map

| Location | Contents |
|----------|----------|
| `docs/superpowers/` | Architecture specs and implementation plans |
| `docs/research/` | Literature summaries and factor-selection rationale |
| `docs/references/` | Validation formula derivations (`validation_formulas.md`), Python best practices |
| `docs/context/` | Session log, todo tracker, lessons learned, memory |

---

## License

MIT License
