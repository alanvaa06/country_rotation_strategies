# Plan B — Selection + Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the statistical validation layer (Lo/PSR/DSR/NW/bootstrap), the research protocols (parameter sweep + stability, anchored walk-forward, Monte-Carlo random-signal null), and the OOS-honest factor selection process (spec §6, decision D6) on top of the Plan-A package.

**Architecture:** `country_rotation/validation/statistics.py` (pure stats on equity/return series — code ported nearly verbatim from docs/superpowers/plans/2026-06-07-strategy-validation.md Tasks 1–5, which were written for equity Series and are codebase-agnostic), `validation/protocols.py` (sweep/WF/MC engines driving `backtest.engine.Engine`), `selection/walkforward.py` (factor screening with BH-FDR + lockbox). TDD throughout; deterministic seeds; no IO.

**Key adaptation vs the 2026-06-07 plan:** that plan targeted a nonexistent `momentum_strategy` package. Statistics functions transfer as-is (input = equity `pd.Series`). Protocol engines are rewritten for `Engine(normalized_score, prices, BacktestConfig)`; the MC null needs NO engine surgery — feed seeded random score DataFrames through the same Engine config (selection machinery identical, signal randomized).

---

### Task B1: validation/statistics.py — five statistics

**Files:** Create `country_rotation/validation/statistics.py`; Test `tests/test_validation_stats.py`.

Port from `docs/superpowers/plans/2026-06-07-strategy-validation.md` Tasks 1–5 VERBATIM (full code lives there — read it): `sharpe_significance` (Lo 2002 SE, `SharpeSignificance`), `probabilistic_sharpe_ratio` (`PSRResult`), `deflated_sharpe_ratio` (`DSRResult` + `_expected_max_sharpe`), `newey_west_tstat` (`NWResult`), `bootstrap_sharpe_ci` (`BootstrapCI` + `_stationary_bootstrap_indices`). Same conventions: daily simple returns, ddof=1, non-excess kurtosis, seeded `default_rng`, NaN-safe degenerate inputs.

Tests: the five failing tests from that plan's Tasks 1–5 Step 1 blocks, verbatim (`test_sharpe_tstat_scales_with_sqrt_n`, `test_psr_monotonic_and_bounded`, `test_dsr_penalizes_many_trials`, `test_newey_west_tstat_positive_drift`, `test_stationary_bootstrap_ci_brackets_and_is_deterministic`), with the helper `_equity_from_daily`.

Commit: `feat(validation): Lo SE, PSR, DSR, Newey-West, stationary-bootstrap CI`

---

### Task B2: validation/protocols.py — sweep, stability, walk-forward, MC null

**Files:** Create `country_rotation/validation/protocols.py`; Test `tests/test_protocols.py`.

```python
def equity_curve(period_returns: pd.Series) -> pd.Series
    # cumprod(1+r), prepend 1.0 at a synthetic start index

@dataclass(frozen=True)
class SweepResult: table: pd.DataFrame; base_key: tuple
def parameter_sweep(scores, prices, base: BacktestConfig, grid: dict[str, tuple]) -> SweepResult
    # grid: field name -> values; 1-D neighborhoods around base for each axis
    # (replace(base, field=v)), dedup by config key, one Engine run each;
    # row = config fields + sharpe_ann + sharpe_daily (from sharpe_significance
    # on equity_curve of portfolio_return_net) + total_return + max_drawdown.
    # Degenerate run (exception) -> skip row.

@dataclass(frozen=True)
class StabilitySummary: n_trials:int; sharpe_mean:float; sharpe_std:float;
    sharpe_min:float; sharpe_max:float; frac_positive:float;
    default_sharpe:float; default_zscore:float
def stability_summary(sweep: SweepResult, base: BacktestConfig) -> StabilitySummary

@dataclass(frozen=True)
class WalkForwardResult: fold_table:pd.DataFrame; oos_returns:pd.Series;
    wf_efficiency:float; oos_sharpe_ann:float; frac_folds_oos_positive:float
def walk_forward(scores, prices, base, grid, n_folds=5) -> WalkForwardResult
    # anchored: usable rebalance-grid dates split into n_folds contiguous segments;
    # fold k: IS = data before segment k (truncate scores+prices), pick config by
    # IS Sharpe over candidate grid (cartesian product, capped — dedup), then run
    # chosen config on data truncated at segment k end and take period returns
    # whose d_ret falls inside segment k as OOS. Stitch OOS period returns
    # chronologically; WFE = mean(OOS sharpe_daily)/mean(IS best sharpe_daily).
    # STRICT: no fold may select on data >= its OOS segment start.

@dataclass(frozen=True)
class MonteCarloNull: actual_sharpe_ann:float; null_sharpes:np.ndarray;
    null_mean:float; null_q95:float; p_value:float; n_sims:int
def monte_carlo_null(scores, prices, cfg, n_sims=200, seed=0) -> MonteCarloNull
    # actual: Engine(scores, prices, cfg). Null sim s: random scores DataFrame
    # (default_rng(seed+s).random(shape), same index/columns as scores) through
    # the SAME Engine cfg. Sharpe_ann via sharpe_significance(equity_curve(net)).
    # p = (null >= actual).mean()  (add-one smoothing optional: document).
```

Tests (use conftest `synthetic_prices`/`synthetic_scores`; periodicity 21, relative/Equal):
- sweep returns table with expected columns, ≥ len(grid values) rows, finite sharpes; stability_summary fields finite, frac_positive in [0,1].
- walk_forward: fold_table has ≤ n_folds rows, each fold's chosen params recorded; oos_returns non-empty; NO-LOOKAHEAD assertion: rerun walk_forward with prices perturbed only after the last IS boundary of fold 1 → fold 1's chosen params identical.
- monte_carlo_null: deterministic under same seed; p_value in [0,1]; null_sharpes size n_sims (use n_sims=25 in test for speed).
- An oracle test: scores = future-return ranks (as in test_ic oracle) → actual sharpe should sit in the top tail (p_value ≤ 0.1) with n_sims=25.

Commit: `feat(validation): sweep+stability, anchored walk-forward, Monte-Carlo random-signal null`

---

### Task B3: selection/walkforward.py — OOS-honest factor selection (D6)

**Files:** Create `country_rotation/selection/walkforward.py`; Test `tests/test_selection.py`.

```python
@dataclass(frozen=True)
class FactorScreenResult:
    fold_ic: pd.DataFrame        # rows=factor, cols=fold mean ICs (train-only)
    kept: tuple                  # factors passing all gates
    weak: tuple                  # kept but raw |t| < 3 (HLZ label)
    dropped: tuple
    bh_qvalues: dict             # factor -> BH-adjusted q
    lockbox_ic: dict | None      # factor -> IC on lockbox (filled only by verify_on_lockbox)

def screen_factors(factor_scores: dict[str, pd.DataFrame], prices: pd.DataFrame,
                   periodicity: int = 21, n_folds: int = 5, lockbox_frac: float = 0.2,
                   min_sign_consistency: float = 0.7, fdr_q: float = 0.10,
                   exploratory: set | None = None) -> FactorScreenResult
    # factor_scores: factor name -> per-country score DataFrame (e.g. equal-weight
    #   metric average of its 4 transforms). Steps:
    # 1. Reserve lockbox = last lockbox_frac of dates. NEVER touched here.
    # 2. Split remaining dates into n_folds expanding (anchored) train windows:
    #    fold k trains on dates[: end_k]. Per fold & factor: ic.information_coefficient
    #    (method='absolute') on the train slice -> mean IC.
    # 3. Gates: mean of fold ICs > 0; sign consistency across folds >= min_sign_consistency;
    #    BH-FDR across factors on the t-stat of fold-mean ICs (q <= fdr_q).
    #    Exploratory factors (set from catalog): require q <= fdr_q/2.
    # 4. weak label: kept but |t| < 3.0 (Harvey-Liu-Zhu).

def verify_on_lockbox(result: FactorScreenResult, factor_scores, prices,
                      periodicity=21, lockbox_frac=0.2) -> FactorScreenResult
    # returns copy with lockbox_ic filled for KEPT factors only (one shot, report side-by-side).

def benjamini_hochberg(pvalues: dict[str, float], q: float) -> dict[str, float]
    # returns adjusted q-values; standard BH step-up.
```

Tests:
- BH: hand-built p-values {a:0.001,b:0.02,c:0.04,d:0.9} → known adjusted q ordering; a survives at q=0.05, d never.
- screen_factors with planted signal: factor_scores["good"] = future-return ranks (oracle), factor_scores["noise1..3"] = seeded random → "good" in kept, noise mostly dropped (assert "good" in kept; assert len(kept) <= 2).
- Lockbox isolation: screening result identical when lockbox slice values are scrambled (perturb last 20% of input data → same kept tuple). THE definitive no-peek test.
- verify_on_lockbox fills lockbox_ic only for kept factors.

Commit: `feat(selection): walk-forward IC screening with BH-FDR, HLZ labels, lockbox verification`

---

### Task B4: validation scorecard aggregator

**Files:** Create `country_rotation/validation/scorecard.py`; Test `tests/test_scorecard.py`.

```python
@dataclass(frozen=True)
class Thresholds:
    dsr: float = 0.95; psr: float = 0.95; mc_p: float = 0.05
    wfe: float = 0.5; frac_oos_positive: float = 0.5
    sharpe_t: float = 2.0; stability_frac_positive: float = 0.7
    stability_max_zscore: float = 1.5

@dataclass(frozen=True)
class Verdict:
    checks: dict            # name -> bool
    overall: bool
    notes: tuple

@dataclass(frozen=True)
class ValidationReport:
    sharpe: SharpeSignificance; psr: PSRResult; dsr: DSRResult
    nw_vs_eqw: NWResult; bootstrap: BootstrapCI
    sweep: SweepResult; stability: StabilitySummary
    walkforward: WalkForwardResult; mc: MonteCarloNull
    verdict: Verdict

def compute_validation(scores, prices, cfg, grid, thresholds=Thresholds(),
                       n_boot=500, n_mc=100, seed=0) -> ValidationReport
    # Runs Engine once for actual equity; eqw null via benchmarks.equal_weight_buy_hold
    # on the same country columns; nw_vs_eqw on (portfolio daily returns - eqw daily returns)
    # aligned; DSR trial sharpes = sweep table sharpe_daily, n_trials = len(table).
    # checks: no_overfitting = dsr>=th AND wfe-pass AND mc-pass;
    #         param_stable = frac_positive>=th AND |default_zscore|<=th;
    #         statistically_significant = t>=th AND psr>=th AND bootstrap.ci_low>0 AND nw.t_stat>0
    # overall = all three.
```

Tests: run on synthetic fixtures with small grid/n_mc (fast); assert checks dict has the 3 keys, all booleans; overall == all(checks.values()); a doctored strong-oracle scores input flips statistically_significant to True.

Commit: `feat(validation): scorecard aggregator with evidence thresholds`

---

### Gate
Full suite green; push origin dev.

## Self-Review
- Spec coverage: D6 (B3), validation harness (B1/B2/B4) per spec §5/§6/§8. ✓
- No placeholders: stats code referenced to its verbatim source (2026-06-07 plan, in-repo); new code has signatures + behavioral contracts + concrete tests. ✓
- Type consistency: BacktestConfig reused; ic.information_coefficient consumed by B3; sharpe_significance consumed by B2/B4. ✓
