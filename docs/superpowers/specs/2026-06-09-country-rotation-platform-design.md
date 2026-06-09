# Country Rotation Research Platform — Design Spec

**Date:** 2026-06-09
**Status:** Approved for implementation (autonomous program; user directive 2026-06-09)
**Supersedes:** ad-hoc script pipeline (ProcessData.py / FactorTransformer.py / FactorTesting.py / backtest.py / backtesting_tests.py / test_normalized_scores.py / Threshold_Testing.py)

---

## 1. Goal

Turn the existing country rotation script pipeline into a validated, reproducible research platform that can certify (or refute) a profitable rotation strategy for the **World / DM / EM** segments, with:

1. Literature-grounded factor priors (deep-research output → `docs/research/`).
2. Leak-free ingestion/processing preserving every current derived metric.
3. An out-of-sample-honest feature selection process.
4. Composite signal engineering decomposed into building blocks (Valuation / Quality / Profitability / Momentum).
5. A backtest engine supporting absolute and benchmark-relative strategies with honest nulls.
6. A comprehensive visualization report (performance abs/rel, risk abs/rel, IC analysis, score building-block decomposition).
7. A statistically certified strategy verdict per segment (DSR, PSR, walk-forward, Monte-Carlo null).

## 2. Constraints & context

- Data: vendor Excel files in `Inputs/` (gitignored — repo stays code-only, public at github.com/alanvaa06/country_rotation_strategies).
- Daily series, ~2010 onward, ~40 metrics, ~45 countries + regional aggregates; Classification.xlsx maps Segment (DM/EM), Region.
- Current features preserved: 4 metric transforms (expanding z-score shift(1), absolute expanding percentile, cross-sectional rank, 63d delta percentile), category aggregation, composite weighting, cross-sectional min-max normalization, contribution decomposition, redundancy clustering, absolute/relative selection, Equal/Risk-Parity weighting, benchmark blending, turnover-based TC, Spearman IC analysis.
- Stack: Python 3, pandas, numpy, scipy, matplotlib, openpyxl, pytest (add), no new heavy deps.

## 3. Decisions that resolve audit findings

| # | Audit finding | Decision |
|---|---|---|
| D1 | `bfill()` on fundamentals leaks future values backward (ProcessData.py:350) | Replace with `ffill(limit=63)`; add regression test proving no future value enters date t. |
| D2 | No publication-lag handling | Add `publication_lag_days` per metric class in config (default 0 for consensus/market data, documented assumption; fundamentals configurable). Applied as `.shift(lag)` at ingestion. |
| D3 | Factor catalog misclassified (ROE in Valuation; raw levels Price/GDP/M2/MarketCap/EV/Revenue/Ten_Year/RollingVol in Momentum) | New explicit `factors/catalog.py`: every factor = (name, category, direction, transform_eligibility). ROE/Fwd_ROE → Profitability. Raw levels removed from Momentum; levels only enter via delta/derived transforms. RollingVol → LowRisk direction (lower better) under Quality. |
| D4 | No price momentum factor | Add `Momentum_12_1` (252d return skipping last 21d) and `Momentum_6_1`; literature-strongest country factor (Balvers-Wu, Asness et al.). |
| D5 | IC duplicated in 2 files | Single `backtest/ic.py` implementation; both entry points consume it. |
| D6 | In-sample IC filter + in-sample grid search = data snooping | Feature selection and parameter choice move inside anchored walk-forward (§6); full-sample numbers reported only as "in-sample reference". |
| D7 | Only benchmark blending, no relative framework | Engine gains `mode='blend' | 'active'`; active mode reports active weights vs benchmark, TE, IR; nulls: equal-weight B&H of same universe + benchmark itself. |
| D8 | Division-by-zero in 1/PE etc. | Yield/ratio helpers mask nonpositive denominators to NaN. |
| D9 | No tests | pytest suite per module; synthetic fixtures; engine semantics regression-locked. |
| D10 | Hardcoded paths/params | `config.py` dataclasses + one YAML (`configs/default.yaml`); paths resolved relative to repo root. |

## 4. Architecture

```
country_rotation/                  # package (new)
├── __init__.py
├── config.py        # frozen dataclasses: DataConfig, FactorConfig, SignalConfig,
│                    # BacktestConfig, ValidationConfig, ReportConfig; YAML loader
├── data/
│   ├── ingestion.py    # read Inputs/*.xlsx, classification, weekday filter, slicing,
│   │                   # publication lag shift, ffill(limit) — IO layer
│   ├── processing.py   # ALL current derived metrics as pure functions (yields, spreads,
│   │                   # consensus growth, margins, rolling stats, aggregates) + momentum
│   └── integrity.py    # leakage guards: no-bfill assert, monotone index, gap report,
│                       # coverage matrix per factor/country
├── factors/
│   ├── catalog.py      # factor registry: category, direction, eligibility (D3, D4)
│   ├── transforms.py   # 4 metric transforms (pure; current math preserved incl. shift(1))
│   └── redundancy.py   # hierarchical clustering selection (port of Threshold_Testing)
├── signals/
│   └── composite.py    # metric-weighting → factor score → category aggregate →
│                       # composite → cross-sectional normalization → contributions
│                       # (building-block decomposition preserved)
├── selection/
│   └── walkforward.py  # feature selection protocol (§6): fold-wise IC screening,
│                       # redundancy, multiple-testing adjustment
├── backtest/
│   ├── engine.py       # refactored Backtest: absolute/relative selection,
│   │                   # Equal/Risk-Parity, blend + active modes, TC, periodicity
│   ├── benchmarks.py   # equal-weight B&H of universe, benchmark series handling
│   ├── metrics.py      # one implementation of all performance/risk stats
│   └── ic.py           # one Spearman IC implementation (absolute + relative signals)
├── validation/
│   ├── statistics.py   # Lo(2002) SE, PSR, DSR, Newey-West, stationary bootstrap
│   └── protocols.py    # anchored walk-forward, parameter sweep + stability,
│                       # Monte-Carlo random-selection null
└── reporting/
    └── report.py       # HTML report w/ embedded PNGs (§7)

tests/                  # pytest: per-module unit tests + engine regression fixtures
scripts/                # thin CLIs: process_data.py, build_scores.py, run_backtest.py,
                        # run_validation.py, build_report.py
configs/default.yaml
```

Legacy root scripts remain until parity is proven by regression tests, then deleted in a final cleanup commit.

**Data flow:** `Inputs/ → data.ingestion → data.processing → factors.transforms → signals.composite → backtest.engine → validation.protocols → reporting.report`.

**Interface style:** numpy/pandas in → frozen dataclasses / DataFrames out; IO confined to `data.ingestion`, `reporting`, and `scripts/`.

## 5. Engine semantics (preserved + extended)

- Timing: score observed at close(t_sel); weights apply to returns t_sel→t_ret (next rebalance); daily curve shifts weights by 1 day. Optional `execution_lag_days=1` to trade at next close (default 0 = current behavior, documented).
- Selection: absolute (score > θ) and relative (top-N by score change over `periodicity`); preserved exactly.
- Weighting: Equal, Risk-Parity (inverse variance, lookback param); preserved.
- Benchmark: `blend` mode = current (w_bmk·bmk + (1−w_bmk)·active). New `active` mode: portfolio = 100% selected countries; report vs benchmark (active return, TE, IR, active weights, capture).
- TC: one-way turnover × bps; first-period turnover = deployed active fraction; preserved.
- Nulls computed for every run: benchmark B&H; equal-weight B&H of the same filtered universe (the honest rotation null).

## 6. Feature selection protocol (Phase 3) — OOS-honest

1. **Stage 0 — coverage filter:** factor must have ≥ 8 countries × ≥ 3 years history in segment.
2. **Stage 1 — economic prior:** factor must map to a literature-documented country-level premium (from `docs/research/country_rotation_literature.md`); undocumented factors flagged "exploratory" and held to stricter thresholds.
3. **Stage 2 — redundancy:** hierarchical clustering (1−|ρ|, average linkage) at threshold chosen by Threshold-Testing curve; one representative per cluster (coverage rule).
4. **Stage 3 — predictive screen (walk-forward):** anchored expanding folds (≥5). Per fold, factor-level IC (Spearman, forward `periodicity` return) computed **on training window only**. Keep factor iff: mean train IC > 0, sign-consistency ≥ 70% of folds, and Benjamini–Hochberg FDR-adjusted IC t-stat significant at q=0.10 across the factor family (Harvey-Liu-Zhu discipline: raw t < 3 ⇒ "weak" label).
5. **Stage 4 — self-verification:** last ~20% of history is a lockbox never touched by Stages 2–4; selected factor set's composite IC measured once on lockbox; report side-by-side (train vs lockbox IC). Composite-level DSR accounts for number of factor/scenario trials.

## 7. Report (Phase 6)

Single HTML (base64 PNGs) + companion xlsx per run:
1. **Performance:** cumulative abs + vs benchmark + vs equal-weight null; drawdowns; per-year table; rolling 12m return.
2. **Risk:** vol, max DD, beta, capture (abs); TE, IR, active drawdown (rel).
3. **IC analysis:** IC time series, distribution, mean/ICIR/hit-rate per periodicity, absolute vs relative method.
4. **Score decomposition:** stacked building-block contributions per selected country over time (current visualization preserved), latest-date cross-section bar chart.
5. **Validation scorecard:** DSR, PSR, Sharpe t, bootstrap CI, MC p-value, WF efficiency, stability summary, with pass/fail thresholds (DSR≥0.95, PSR≥0.95, MC p≤0.05, WFE≥0.5, bootstrap CI low > 0).

## 8. Strategy selection (Phase 7)

Per segment (World, DM, EM): run selection protocol → composite signal → parameter walk-forward (scenario weights × selection × periodicity chosen per fold IS, evaluated OOS) → stitched OOS curve → validation scorecard → verdict in `docs/context/results.md`. A strategy is "found" only if OOS evidence passes the scorecard; otherwise report the honest negative.

## 9. Testing strategy

- Unit: every transform vs hand-computed mini-fixtures; turnover/TC/blend math; selection edge cases (empty, ties); IC pairing (signal t vs return t→t+1 verified by construction).
- Leakage: `integrity.py` property test — perturbing data after date t cannot change any score/weight at ≤ t (the definitive no-look-ahead test).
- Regression: new engine reproduces current `Backtest` equity (tolerance 1e-9) on a frozen synthetic fixture before legacy deletion.
- Statistics: validation stats tested per the 2026-06-07 methodology plan (ported here).

## 10. Phasing (implementation plans)

1. **Plan A — platform refactor:** package skeleton, config, data layer (D1/D2/D8), catalog (D3/D4), transforms, composite, engine port + active mode (D7), ic/metrics consolidation (D5), tests, parity regression.
2. **Plan B — selection + validation:** walkforward selection (D6), validation statistics + protocols, MC null.
3. **Plan C — reporting:** report.py + scripts.
4. **Plan D — research runs:** Phase 7 execution per segment + verdicts + docs.

## 11. Risks

- Parity risk porting engine → mitigated by regression fixture before any behavior change; behavior changes (bfill→ffill, catalog fixes) land **after** parity lock, each with its own diff-test.
- Expanding-percentile non-stationarity (early-history scores unstable) → min_periods=63 retained; selection Stage 3 folds start ≥ 2 years in.
- Consensus data has no true point-in-time vintage → documented assumption; publication lag configurable (D2); results labeled accordingly.
- Multiple-testing inflation from 150 scenarios → DSR trial count includes all scenario × grid trials ever evaluated per segment.
