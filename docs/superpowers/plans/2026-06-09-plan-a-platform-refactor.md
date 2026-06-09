# Plan A — Platform Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the script pipeline into the `country_rotation/` package per `docs/superpowers/specs/2026-06-09-country-rotation-platform-design.md`, preserving current behavior (parity-locked), then landing the audited fixes (bfill leak, factor catalog, active mode, IC consolidation) each behind its own test.

**Architecture:** Pure-function layers (`data` → `factors` → `signals` → `backtest`) with IO confined to ingestion/scripts. Legacy root scripts stay untouched until the parity regression passes; behavior changes land after parity, each with a dedicated diff test.

**Tech Stack:** Python 3, pandas, numpy, scipy, matplotlib, openpyxl, pytest. Config = stdlib `json` + frozen dataclasses (no new deps; spec's "YAML" amended to JSON for zero-dependency reproducibility).

**Source-of-truth note for porters:** Tasks marked PORT copy logic from legacy files at the cited lines. Read the legacy code before porting; preserve math exactly. Do not "improve" ported logic in PORT tasks — improvements land in later FIX tasks.

**Run all commands from repo root `C:\Proyectos\country_rotation`. Python: use `python` (system/conda env in use). If `pytest` missing: `pip install pytest`.**

---

### Task 1: Package skeleton + pytest wiring

**Files:**
- Create: `country_rotation/__init__.py`, `country_rotation/data/__init__.py`, `country_rotation/factors/__init__.py`, `country_rotation/signals/__init__.py`, `country_rotation/selection/__init__.py`, `country_rotation/backtest/__init__.py`, `country_rotation/validation/__init__.py`, `country_rotation/reporting/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Modify: `pyproject.toml` (add pytest config + package discovery if absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_package.py
def test_package_imports():
    import country_rotation
    from country_rotation import data, factors, signals, backtest, validation, reporting, selection
    assert country_rotation.__version__ == "0.2.0"
```

- [ ] **Step 2: Run** `python -m pytest tests/test_package.py -v` → FAIL (no package).

- [ ] **Step 3: Create the skeleton.** Every `__init__.py` empty except top-level:

```python
# country_rotation/__init__.py
"""Country rotation research platform."""
__version__ = "0.2.0"
```

Create `tests/conftest.py` with the shared synthetic fixture used by ALL later tasks:

```python
# tests/conftest.py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices():
    """10 countries + 'World' benchmark, 800 business days, deterministic drifts."""
    rng = np.random.default_rng(0)
    n = 800
    idx = pd.bdate_range("2015-01-02", periods=n)
    names = [f"C{i}" for i in range(10)]
    drifts = np.linspace(-0.0003, 0.0009, 10)
    cols = {}
    for k, name in enumerate(names):
        r = rng.normal(drifts[k], 0.012, n)
        cols[name] = 100.0 * np.cumprod(1.0 + r)
    df = pd.DataFrame(cols, index=idx)
    df["World"] = df[names].mean(axis=1)
    return df


@pytest.fixture
def synthetic_scores(synthetic_prices):
    """Normalized scores in [0,1] for the 10 countries (not the benchmark)."""
    rng = np.random.default_rng(1)
    countries = [c for c in synthetic_prices.columns if c != "World"]
    raw = rng.random((len(synthetic_prices), len(countries)))
    df = pd.DataFrame(raw, index=synthetic_prices.index, columns=countries)
    # smooth so 'relative' selection has persistent changes
    return df.rolling(21, min_periods=1).mean()


@pytest.fixture
def synthetic_factor():
    """Single-factor DataFrame: 6 countries x 300 days with trends + NaN block."""
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2018-01-01", periods=300)
    df = pd.DataFrame(
        rng.normal(0, 1, (300, 6)).cumsum(axis=0),
        index=idx, columns=list("ABCDEF"),
    )
    df.iloc[:40, 5] = np.nan  # late-starting country
    return df
```

If `pyproject.toml` lacks them, add:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.setuptools.packages.find]
include = ["country_rotation*"]
```

(Read the existing file first; merge, don't clobber existing sections.)

- [ ] **Step 4: Run** `python -m pytest tests/test_package.py -v` → PASS.
- [ ] **Step 5: Commit** `git add country_rotation tests pyproject.toml && git commit -m "feat: package skeleton + shared test fixtures"`

---

### Task 2: config.py — frozen dataclasses + JSON loader

**Files:**
- Create: `country_rotation/config.py`, `configs/default.json`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from country_rotation.config import PlatformConfig, load_config


def test_default_config_loads():
    cfg = load_config("configs/default.json")
    assert isinstance(cfg, PlatformConfig)
    assert cfg.data.inputs_folder == "Inputs"
    assert cfg.data.ffill_limit == 63
    assert cfg.data.publication_lag_days == {}
    assert cfg.backtest.periodicity == 63
    assert cfg.backtest.transaction_cost_bps == 2.0
    assert cfg.backtest.mode in ("blend", "active")


def test_config_is_frozen():
    cfg = load_config("configs/default.json")
    import pytest
    with pytest.raises(Exception):
        cfg.backtest.periodicity = 5
```

- [ ] **Step 2: Run** `python -m pytest tests/test_config.py -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# country_rotation/config.py
"""Platform configuration: frozen dataclasses loaded from JSON."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    inputs_folder: str = "Inputs"
    classification_file: str = "Classification.xlsx"
    target_date: str = "2010-01-01"
    columns_to_drop: tuple = ("Saudi Arabia",)
    ffill_limit: int = 63
    publication_lag_days: dict = field(default_factory=dict)  # metric -> days


@dataclass(frozen=True)
class BacktestConfig:
    selection_criteria: str = "relative"       # 'absolute' | 'relative'
    absolute_selection_score: float = 0.75
    relative_selection_score: int = 5
    weighting_method: str = "Equal"            # 'Equal' | 'Risk_Parity'
    risk_parity_lookback: int = 120
    bmk: str = "World"
    bmk_weight: float = 0.5
    mode: str = "blend"                        # 'blend' | 'active'
    periodicity: int = 63
    transaction_cost_bps: float = 2.0
    execution_lag_days: int = 0


@dataclass(frozen=True)
class PlatformConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def load_config(path: str | Path) -> PlatformConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = raw.get("data", {})
    if "columns_to_drop" in data:
        data["columns_to_drop"] = tuple(data["columns_to_drop"])
    return PlatformConfig(
        data=DataConfig(**data),
        backtest=BacktestConfig(**raw.get("backtest", {})),
    )
```

```json
// configs/default.json  (JSON has no comments — file content below, no comment line)
{
  "data": {
    "inputs_folder": "Inputs",
    "classification_file": "Classification.xlsx",
    "target_date": "2010-01-01",
    "columns_to_drop": ["Saudi Arabia"],
    "ffill_limit": 63,
    "publication_lag_days": {}
  },
  "backtest": {
    "selection_criteria": "relative",
    "absolute_selection_score": 0.75,
    "relative_selection_score": 5,
    "weighting_method": "Equal",
    "risk_parity_lookback": 120,
    "bmk": "World",
    "bmk_weight": 0.5,
    "mode": "blend",
    "periodicity": 63,
    "transaction_cost_bps": 2.0,
    "execution_lag_days": 0
  }
}
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: frozen dataclass config + configs/default.json"`

---

### Task 3 (PORT): data/ingestion.py

**Files:**
- Create: `country_rotation/data/ingestion.py`
- Test: `tests/test_ingestion.py`
- Legacy source: `ProcessData.py` — `read_excel_files_to_dict` (~:80-126), classification loading (~:128-164), `remove_weekends_optimized` (~:166-199), date slicing (~:200+)

Port the legacy functions as module-level pure-ish functions (IO allowed here). Signatures:

```python
read_inputs(folder: str, skiprows: int = 2) -> dict[str, pd.DataFrame]
load_classification(path: str) -> pd.DataFrame
remove_weekends(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]
slice_by_date(dfs: dict[str, pd.DataFrame], target_date: str) -> dict[str, pd.DataFrame]
drop_countries(dfs: dict[str, pd.DataFrame], columns_to_drop: tuple) -> dict[str, pd.DataFrame]
apply_publication_lag(dfs: dict[str, pd.DataFrame], lag_days: dict[str, int]) -> dict[str, pd.DataFrame]
fill_gaps(dfs: dict[str, pd.DataFrame], metrics: tuple, limit: int) -> dict[str, pd.DataFrame]
```

`apply_publication_lag`: for each metric in `lag_days`, `df.shift(lag_days[metric])`. `fill_gaps`: `df.ffill(limit=limit)` — this is the D1 fix; legacy used `bfill` (ProcessData.py:350) — do NOT port bfill.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingestion.py
import numpy as np
import pandas as pd
from country_rotation.data import ingestion


def _mini():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")  # includes weekend
    df = pd.DataFrame({"A": np.arange(10.0), "B": np.arange(10.0) * 2}, index=idx)
    return {"PE": df.copy(), "EBITDA": df.copy()}


def test_remove_weekends():
    out = ingestion.remove_weekends(_mini())
    assert all(out["PE"].index.dayofweek < 5)


def test_slice_and_drop():
    dfs = _mini()
    out = ingestion.slice_by_date(dfs, "2020-01-05")
    assert out["PE"].index.min() >= pd.Timestamp("2020-01-05")
    out2 = ingestion.drop_countries(dfs, ("B",))
    assert "B" not in out2["PE"].columns


def test_publication_lag_shifts_data():
    dfs = _mini()
    out = ingestion.apply_publication_lag(dfs, {"EBITDA": 2})
    assert out["EBITDA"]["A"].iloc[2] == dfs["PE"]["A"].iloc[0]  # shifted by 2
    assert out["PE"].equals(dfs["PE"])  # unlagged metric untouched


def test_fill_gaps_never_uses_future():
    dfs = _mini()
    dfs["EBITDA"].iloc[3:5, 0] = np.nan
    out = ingestion.fill_gaps(dfs, metrics=("EBITDA",), limit=63)
    # gap filled with PAST value (2.0), never the future value (5.0)
    assert out["EBITDA"]["A"].iloc[3] == 2.0
    assert out["EBITDA"]["A"].iloc[4] == 2.0
    # leading NaN stays NaN (nothing in the past to fill from)
    dfs2 = _mini()
    dfs2["EBITDA"].iloc[0:2, 0] = np.nan
    out2 = ingestion.fill_gaps(dfs2, metrics=("EBITDA",), limit=63)
    assert np.isnan(out2["EBITDA"]["A"].iloc[0])
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Port legacy logic for `read_inputs`/`load_classification`/`remove_weekends`/`slice_by_date` from the cited lines (keep `skiprows=2, index_col=0, parse_dates=True` and the `dayofweek < 5` mask). New functions:

```python
def apply_publication_lag(dfs, lag_days):
    out = dict(dfs)
    for metric, lag in lag_days.items():
        if metric in out and lag > 0:
            out[metric] = out[metric].shift(lag)
    return out


def fill_gaps(dfs, metrics, limit):
    out = dict(dfs)
    for metric in metrics:
        if metric in out:
            out[metric] = out[metric].ffill(limit=limit)
    return out
```

- [ ] **Step 4: Run** → PASS. Also run `python -m pytest tests -v` (all green).
- [ ] **Step 5: Commit** `git commit -m "feat(data): ingestion layer with publication lag + ffill-only gap filling (fixes bfill leak)"`

---

### Task 4 (PORT): data/processing.py — derived metrics + momentum

**Files:**
- Create: `country_rotation/data/processing.py`
- Test: `tests/test_processing.py`
- Legacy source: `ProcessData.py` `transform_process_data` (:306-559) — consensus growth (:408-418), yields (:425-435), spreads (:442-459), reconstructions (:466-482), margins (:489-503), rolling stats (:510-520), AssetsEquity (:527-528), regional aggregations (:354-401)

Port each step as a pure function taking/returning `dict[str, pd.DataFrame]`. Add a safe-division helper and the NEW momentum factors (D4):

```python
def _safe_div(num, den):
    """Element-wise num/den with nonpositive/zero denominators -> NaN (D8)."""
    den = den.where(den > 0)
    return num / den

def add_price_momentum(dfs):
    """Momentum_12_1 = P(t-21)/P(t-252) - 1 ; Momentum_6_1 = P(t-21)/P(t-126) - 1."""
    px = dfs["Price"]
    out = dict(dfs)
    out["Momentum_12_1"] = px.shift(21) / px.shift(252) - 1.0
    out["Momentum_6_1"] = px.shift(21) / px.shift(126) - 1.0
    return out
```

Yields use `_safe_div(1.0, dfs["PE"])` etc. Everything else: port formulas exactly as cited.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_processing.py
import numpy as np
import pandas as pd
from country_rotation.data import processing


def _dfs():
    idx = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(3)
    px = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300)),
                       "B": 100 * np.cumprod(1 + rng.normal(0.0, 0.01, 300))}, index=idx)
    pe = pd.DataFrame({"A": np.full(300, 15.0), "B": np.full(300, -5.0)}, index=idx)
    return {"Price": px, "PE": pe}


def test_earnings_yield_safe_division():
    out = processing.add_yields(_dfs())
    ey = out["EarningsYieldTTM"]
    assert abs(ey["A"].iloc[0] - 1 / 15.0) < 1e-12
    assert np.isnan(ey["B"].iloc[0])  # negative PE -> NaN, not -0.2


def test_momentum_12_1_window_and_skip():
    dfs = _dfs()
    out = processing.add_price_momentum(dfs)
    mom = out["Momentum_12_1"]
    px = dfs["Price"]["A"]
    t = 280
    expected = px.iloc[t - 21] / px.iloc[t - 252] - 1.0
    assert abs(mom["A"].iloc[t] - expected) < 1e-12
    assert np.isnan(mom["A"].iloc[100])  # warmup
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** (port + new code above). Function inventory to expose: `add_consensus_growth`, `add_yields`, `add_spreads`, `add_reconstructions`, `add_margins`, `add_rolling_stats`, `add_leverage`, `add_regional_aggregates`, `add_price_momentum`, plus `run_processing(dfs, classification) -> dict` chaining them in legacy order then momentum.
- [ ] **Step 4: Run** → PASS; full suite green.
- [ ] **Step 5: Commit** `git commit -m "feat(data): pure derived-metric functions + 12-1/6-1 price momentum + safe division"`

---

### Task 5: data/integrity.py — leakage guard

**Files:**
- Create: `country_rotation/data/integrity.py`
- Test: `tests/test_integrity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_integrity.py
import numpy as np
import pandas as pd
from country_rotation.data import integrity
from country_rotation.data import processing
from country_rotation.factors import transforms  # available after Task 6; see note


def test_no_lookahead_in_processing(synthetic_factor):
    """Perturbing data AFTER date t must not change any derived value at <= t."""
    dfs = {"Price": synthetic_factor.abs() + 1.0, "PE": synthetic_factor.abs() + 5.0}
    cut = synthetic_factor.index[200]
    report = integrity.lookahead_check(
        pipeline=lambda d: processing.add_price_momentum(processing.add_yields(d)),
        dfs=dfs, cutoff=cut, perturb_scale=100.0,
    )
    assert report.clean, f"leaking outputs: {report.dirty_outputs}"


def test_coverage_matrix(synthetic_factor):
    cov = integrity.coverage_matrix({"X": synthetic_factor})
    assert cov.loc["X", "n_countries"] == 6
    assert cov.loc["X", "first_date"] == synthetic_factor.index[0]
```

(Note: if Task 6 not yet done, drop the transforms import — only processing is exercised here.)

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**

```python
# country_rotation/data/integrity.py
"""Leakage guards and coverage diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LookaheadReport:
    clean: bool
    dirty_outputs: tuple


def lookahead_check(pipeline, dfs, cutoff, perturb_scale=100.0):
    """Run pipeline on original and on future-perturbed inputs; any difference
    at dates <= cutoff is look-ahead leakage."""
    base = pipeline({k: v.copy() for k, v in dfs.items()})
    perturbed_in = {}
    for k, v in dfs.items():
        p = v.copy()
        p.loc[p.index > cutoff] = p.loc[p.index > cutoff] * perturb_scale + perturb_scale
        perturbed_in[k] = p
    pert = pipeline(perturbed_in)
    dirty = []
    for name, df in base.items():
        if name not in pert:
            continue
        a = df.loc[df.index <= cutoff]
        b = pert[name].loc[pert[name].index <= cutoff]
        if not np.allclose(a.fillna(-9e9).to_numpy(), b.fillna(-9e9).to_numpy(), equal_nan=False):
            dirty.append(name)
    return LookaheadReport(clean=not dirty, dirty_outputs=tuple(dirty))


def coverage_matrix(dfs):
    rows = {}
    for name, df in dfs.items():
        valid = df.notna()
        rows[name] = {
            "n_countries": int((valid.sum() > 0).sum()),
            "first_date": df.index[0] if len(df) else pd.NaT,
            "last_date": df.index[-1] if len(df) else pd.NaT,
            "pct_missing": float(df.isna().mean().mean()),
        }
    return pd.DataFrame(rows).T
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(data): lookahead perturbation guard + coverage matrix"`

---

### Task 6 (PORT): factors/transforms.py — the 4 metric transforms

**Files:**
- Create: `country_rotation/factors/transforms.py`
- Test: `tests/test_transforms.py`
- Legacy source: `FactorTransformer.py` — zscore (:314-331, expanding(min_periods=63), `.shift(1)`, `norm.cdf`), absolute_pct (:333-345, `expanding().rank(pct=True)`), relative_rank (:347-357, `rank(axis=1, pct=True)`), delta_pct (:359-370, `pct_change(63)` then expanding rank), direction inversion (:392-398, `1.0 - x`)

Signatures (pure, one DataFrame in → one out):

```python
zscore_percentile(df, min_periods=63) -> pd.DataFrame
absolute_percentile(df) -> pd.DataFrame
relative_rank(df) -> pd.DataFrame
delta_percentile(df, window=63, min_periods=63) -> pd.DataFrame
apply_direction(df, direction: int) -> pd.DataFrame   # -1 -> 1.0 - df
transform_factor(df, direction) -> dict[str, pd.DataFrame]  # all four, direction applied
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transforms.py
import numpy as np
import pandas as pd
from scipy.stats import norm
from country_rotation.factors import transforms


def test_zscore_uses_only_past_stats(synthetic_factor):
    z = transforms.zscore_percentile(synthetic_factor)
    col = synthetic_factor["A"]
    t = 150
    hist = col.iloc[:t]              # expanding stats up to t-1, shift(1)
    expected = norm.cdf((col.iloc[t] - hist.mean()) / hist.std())
    assert abs(z["A"].iloc[t] - expected) < 1e-9


def test_zscore_no_lookahead(synthetic_factor):
    z1 = transforms.zscore_percentile(synthetic_factor)
    pert = synthetic_factor.copy()
    pert.iloc[200:] += 1000.0
    z2 = transforms.zscore_percentile(pert)
    pd.testing.assert_frame_equal(z1.iloc[:200], z2.iloc[:200])


def test_absolute_percentile_expanding(synthetic_factor):
    p = transforms.absolute_percentile(synthetic_factor)
    col = synthetic_factor["A"]
    t = 100
    expected = (col.iloc[: t + 1] <= col.iloc[t]).mean()  # rank pct incl. self
    assert abs(p["A"].iloc[t] - expected) < 1e-9


def test_relative_rank_cross_sectional(synthetic_factor):
    r = transforms.relative_rank(synthetic_factor)
    row = synthetic_factor.iloc[250]
    assert r.iloc[250][row.idxmax()] == r.iloc[250].max()


def test_direction_inversion(synthetic_factor):
    base = transforms.transform_factor(synthetic_factor, direction=1)
    inv = transforms.transform_factor(synthetic_factor, direction=-1)
    pd.testing.assert_frame_equal(inv["zscore"], 1.0 - base["zscore"])
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Port** from cited lines. Math identical; keys of `transform_factor` output: `"zscore"`, `"absolute_pct"`, `"relative_rank"`, `"delta_pct"`. If a legacy test expectation mismatches (e.g. rank `method=` default), match LEGACY behavior and adjust the test's expected formula accordingly — parity wins; note any such adjustment in the commit message.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(factors): port 4 metric transforms (parity with FactorTransformer)"`

---

### Task 7: factors/catalog.py — corrected factor registry (D3/D4)

**Files:**
- Create: `country_rotation/factors/catalog.py`
- Test: `tests/test_catalog.py`
- Reference: legacy category lists in `FactorTesting.py:20-44` (note: lines 20-31 are dead code overridden by 33-44) and `FactorTransformer.py:180-295`

Registry = tuple of `FactorSpec(name, category, direction, exploratory)`. Corrections vs legacy:
- `ROE`, `Fwd_ROE`, `Return_Capital` → **Profitability** (legacy had ROE in Valuation).
- REMOVE raw levels from Momentum: `Price`, `GDP`, `M2`, `Market_Cap`, `EV`, `Revenue`, `Ten_Year`, `SI`, `SI_Ratio` are NOT factors (levels are not cross-country comparable). `Flows` stays only as `CumFlow`.
- ADD `Momentum_12_1` (+1), `Momentum_6_1` (+1) → Momentum.
- `RollingVol` → Quality with direction −1 (low-risk).
- Valuation directions: multiples (PE, PB, PS, PCF, EV_EBIT, EV_EBITDA + Fwd variants) −1; yields/spreads/DVD +1.
- Quality: `Debt_to_Equity`, `Net_Debt_Ebitda`, `AssetsEquity` −1; `CF`, `FwdCF` +1; drop bare `Debt`, `Equity`, `Liabilities`, `Assets`, `FwdRevenue` (levels).
- Profitability: margins +1, consensus growth +1, `EPS`/`Earnings`/`FwdEarnings`/`EBIT`/`EBITDA`/`FwdEBITDA` dropped (levels) — growth captured via `RollingEarnings`/`FwdRollingEarnings` (Momentum, +1, as legacy).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_catalog.py
from country_rotation.factors.catalog import CATALOG, by_category, get_spec


def test_no_raw_levels_in_catalog():
    names = {f.name for f in CATALOG}
    for lvl in ("Price", "GDP", "M2", "Market_Cap", "EV", "Revenue", "Debt", "Equity"):
        assert lvl not in names


def test_roe_is_profitability():
    assert get_spec("ROE").category == "Profitability"
    assert get_spec("ROE").direction == 1


def test_momentum_has_price_momentum():
    moms = {f.name for f in by_category("Momentum")}
    assert {"Momentum_12_1", "Momentum_6_1"}.issubset(moms)


def test_directions_sane():
    assert get_spec("PE").direction == -1
    assert get_spec("EarningsYieldTTM").direction == 1
    assert get_spec("RollingVol").direction == -1
    assert all(f.category in {"Valuation", "Quality", "Profitability", "Momentum"} for f in CATALOG)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**

```python
# country_rotation/factors/catalog.py
"""Factor registry: every usable factor with category and direction.

Direction: +1 = higher is better, -1 = lower is better.
exploratory=True: no documented country-level literature support; held to
stricter selection thresholds (spec §6 Stage 1).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorSpec:
    name: str
    category: str
    direction: int
    exploratory: bool = False


CATALOG: tuple = (
    # Valuation — multiples (lower better)
    FactorSpec("PE", "Valuation", -1), FactorSpec("Fwd_PE", "Valuation", -1),
    FactorSpec("PB", "Valuation", -1), FactorSpec("PS", "Valuation", -1),
    FactorSpec("Fwd_PS", "Valuation", -1), FactorSpec("PCF", "Valuation", -1),
    FactorSpec("Fwd_PCF", "Valuation", -1), FactorSpec("EV_EBIT", "Valuation", -1),
    FactorSpec("EV_EBITDA", "Valuation", -1), FactorSpec("Fwd_EV_EBITDA", "Valuation", -1),
    # Valuation — yields & spreads (higher better)
    FactorSpec("EarningsYieldTTM", "Valuation", 1), FactorSpec("EarningsYieldFWD", "Valuation", 1),
    FactorSpec("CashFlowYieldTTM", "Valuation", 1), FactorSpec("CashFlowYieldFWD", "Valuation", 1),
    FactorSpec("DVD", "Valuation", 1), FactorSpec("Fwd_DVD", "Valuation", 1),
    FactorSpec("EarningsYieldTTMSpread", "Valuation", 1), FactorSpec("EarningsYieldFWDSpread", "Valuation", 1),
    FactorSpec("CashFlowYieldTTMSpread", "Valuation", 1), FactorSpec("CashFlowYieldFWDSpread", "Valuation", 1),
    FactorSpec("DvdYieldTTMSpread", "Valuation", 1), FactorSpec("DvdYieldFWDSpread", "Valuation", 1),
    # Quality
    FactorSpec("Debt_to_Equity", "Quality", -1), FactorSpec("Net_Debt_Ebitda", "Quality", -1),
    FactorSpec("AssetsEquity", "Quality", -1), FactorSpec("CF", "Quality", 1),
    FactorSpec("FwdCF", "Quality", 1), FactorSpec("RollingVol", "Quality", -1),
    # Profitability
    FactorSpec("ROE", "Profitability", 1), FactorSpec("Fwd_ROE", "Profitability", 1),
    FactorSpec("Return_Capital", "Profitability", 1),
    FactorSpec("EbitMargin", "Profitability", 1), FactorSpec("EbitdaMargin", "Profitability", 1),
    FactorSpec("NetMargin", "Profitability", 1), FactorSpec("FwdEBITDAMargin", "Profitability", 1),
    FactorSpec("FwdNetMargin", "Profitability", 1),
    FactorSpec("ConsensusSalesGrowth", "Profitability", 1, exploratory=True),
    FactorSpec("ConsensusEbitdaGrowth", "Profitability", 1, exploratory=True),
    FactorSpec("ConsensusEarningsGrowth", "Profitability", 1, exploratory=True),
    FactorSpec("ConsensusCashFlowGrowth", "Profitability", 1, exploratory=True),
    # Momentum
    FactorSpec("Momentum_12_1", "Momentum", 1), FactorSpec("Momentum_6_1", "Momentum", 1),
    FactorSpec("RollingEarnings", "Momentum", 1), FactorSpec("FwdRollingEarnings", "Momentum", 1),
    FactorSpec("CumFlow", "Momentum", 1, exploratory=True),
)

_BY_NAME = {f.name: f for f in CATALOG}


def get_spec(name: str) -> FactorSpec:
    return _BY_NAME[name]


def by_category(category: str) -> tuple:
    return tuple(f for f in CATALOG if f.category == category)
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(factors): explicit catalog with corrected categories/directions; drop raw levels; add price momentum"`

---

### Task 8 (PORT): signals/composite.py

**Files:**
- Create: `country_rotation/signals/composite.py`
- Test: `tests/test_composite.py`
- Legacy source: `FactorTransformer.py` — `calculate_weighted_average` (~:400-460), `aggregate_by_category` (:462-515), `calculate_composite_score` (:517-551), `normalize_and_rebase_contributions` (:553-620; min-max per date :570-582, all-equal → 0.5)

Signatures:

```python
weighted_metric_average(factor_results: dict[str, dict[str, pd.DataFrame]],
                        metric_weights: dict[str, float]) -> dict[str, pd.DataFrame]
aggregate_by_category(weighted: dict[str, pd.DataFrame],
                      catalog) -> dict[str, pd.DataFrame]   # category -> countries df
composite_score(category_scores: dict[str, pd.DataFrame],
                category_weights: dict[str, float]) -> tuple[pd.DataFrame, dict]
normalize_cross_section(composite: pd.DataFrame) -> pd.DataFrame
rebase_contributions(contributions: dict, composite: pd.DataFrame,
                     normalized: pd.DataFrame) -> dict
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composite.py
import numpy as np
import pandas as pd
from country_rotation.signals import composite


def _factor_results(idx, cols):
    rng = np.random.default_rng(4)
    mk = lambda: pd.DataFrame(rng.random((len(idx), len(cols))), index=idx, columns=cols)
    return {
        "PE": {m: mk() for m in ("zscore", "absolute_pct", "relative_rank", "delta_pct")},
        "ROE": {m: mk() for m in ("zscore", "absolute_pct", "relative_rank", "delta_pct")},
    }


def test_weighted_average_math():
    idx = pd.bdate_range("2020-01-01", periods=5)
    fr = _factor_results(idx, ["A", "B"])
    w = {"zscore": 0.5, "absolute_pct": 0.5, "relative_rank": 0.0, "delta_pct": 0.0}
    out = composite.weighted_metric_average(fr, w)
    expected = 0.5 * fr["PE"]["zscore"] + 0.5 * fr["PE"]["absolute_pct"]
    pd.testing.assert_frame_equal(out["PE"], expected)


def test_normalize_cross_section_bounds_and_ties():
    idx = pd.bdate_range("2020-01-01", periods=3)
    comp = pd.DataFrame({"A": [1.0, 2.0, 5.0], "B": [3.0, 2.0, 1.0], "C": [2.0, 2.0, 3.0]}, index=idx)
    norm = composite.normalize_cross_section(comp)
    assert norm.iloc[0].min() == 0.0 and norm.iloc[0].max() == 1.0
    assert (norm.iloc[1] == 0.5).all()  # all-equal date -> 0.5 (legacy behavior)


def test_composite_weights_applied():
    idx = pd.bdate_range("2020-01-01", periods=4)
    cat = {
        "Valuation": pd.DataFrame({"A": [1.0] * 4, "B": [0.0] * 4}, index=idx),
        "Momentum": pd.DataFrame({"A": [0.0] * 4, "B": [1.0] * 4}, index=idx),
    }
    comp, contrib = composite.composite_score(cat, {"Valuation": 0.7, "Momentum": 0.3})
    assert abs(comp["A"].iloc[0] - 0.7) < 1e-12
    assert abs(comp["B"].iloc[0] - 0.3) < 1e-12
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Port** from cited lines; `aggregate_by_category` takes the catalog (Task 7) for the factor→category map instead of legacy hardcoded lists.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(signals): composite scoring pipeline with building-block contributions"`

---### Task 9 (PORT): backtest/metrics.py + backtest/ic.py

**Files:**
- Create: `country_rotation/backtest/metrics.py`, `country_rotation/backtest/ic.py`
- Test: `tests/test_metrics.py`, `tests/test_ic.py`
- Legacy source: metrics `backtest.py:919-1111` (geometric annualization :973, vol :980, Sharpe :987, beta :998-1002, TE :1009, capture :1018-1032, IR :1049, win rate :1059, maxDD :1110); IC `test_normalized_scores.py:193-301` (Spearman :240, signal/return pairing :224-230, absolute :293-296, relative :299-301)

`metrics.summary(returns: pd.Series, bmk_returns: pd.Series, periods_per_year: float) -> dict` — one dict with all stats, names: `ann_return, ann_vol, sharpe, max_drawdown, win_rate, beta, up_capture, down_capture, tracking_error, information_ratio`.

`ic.information_coefficient(scores: pd.DataFrame, prices: pd.DataFrame, periodicity: int, method: str) -> pd.DataFrame` — per-period Spearman IC; `method='absolute'` signal=score[t]; `method='relative'` signal=score[t]−score[t−periodicity]; both predict return t→t+periodicity. `ic.ic_stats(ic_series) -> dict` with `mean_ic, median_ic, std_ic, t_stat, icir, hit_rate`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import numpy as np
import pandas as pd
from country_rotation.backtest import metrics


def test_summary_known_values():
    idx = pd.bdate_range("2020-01-01", periods=252)
    r = pd.Series(0.001, index=idx)          # constant +10bp/day
    b = pd.Series(0.0005, index=idx)
    s = metrics.summary(r, b, periods_per_year=252)
    assert s["ann_vol"] == 0.0 or np.isnan(s["sharpe"]) or s["sharpe"] > 100  # degenerate vol
    assert s["max_drawdown"] == 0.0
    assert s["win_rate"] == 1.0
    assert s["tracking_error"] >= 0.0


def test_max_drawdown_sign():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.10, -0.50, 0.10, 0.10], index=idx)
    s = metrics.summary(r, r * 0, periods_per_year=252)
    assert -0.51 < s["max_drawdown"] < -0.49
```

```python
# tests/test_ic.py
import numpy as np
import pandas as pd
from country_rotation.backtest import ic


def test_perfect_signal_gives_ic_one(synthetic_prices):
    countries = [c for c in synthetic_prices.columns if c != "World"]
    px = synthetic_prices[countries]
    periodicity = 21
    fwd = px.shift(-periodicity) / px - 1.0
    scores = fwd.rank(axis=1, pct=True)      # oracle: score = future-return rank
    out = ic.information_coefficient(scores, px, periodicity, method="absolute")
    assert out["IC"].dropna().mean() > 0.95


def test_random_signal_ic_near_zero(synthetic_prices, synthetic_scores):
    countries = [c for c in synthetic_prices.columns if c != "World"]
    out = ic.information_coefficient(synthetic_scores, synthetic_prices[countries], 21, method="absolute")
    stats = ic.ic_stats(out["IC"])
    assert abs(stats["mean_ic"]) < 0.15
    assert set(stats) >= {"mean_ic", "icir", "t_stat", "hit_rate"}
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Port** from cited lines into single implementations. IC date grid: same `[::-periodicity]` reverse-sampling as legacy (backtest.py:287), signal at grid date i paired with return grid i→i+1.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(backtest): consolidated metrics + single IC implementation"`

---

### Task 10 (PORT): backtest/engine.py — core engine, parity-exact

**Files:**
- Create: `country_rotation/backtest/engine.py`
- Test: `tests/test_engine.py`
- Legacy source: `backtest.py` — init/validation (:36-151), date grid `_filter_dates` (:275-293, `[::-periodicity]`), date tuples (:176-177), absolute selection (:319-344), relative selection (:346-380), equal weights (:431-434), risk parity (:445-560, lookback excl. current date, min 10 obs, fallback equal), blending (:562-598), turnover (:600-666, first period = 1−bmk_weight, then Σ|Δw|/2 excl. bmk), TC application (:211-223, net = gross − turnover×bps_dec), period returns (:689-696 close-to-close d_sel→d_ret), portfolio return dot product (:702-724), daily curve (weights ffill + shift(1), :867-871), result frames (:740-800)

New class, dataclass-config driven:

```python
class Engine:
    def __init__(self, normalized_score: pd.DataFrame, prices: pd.DataFrame, cfg: BacktestConfig): ...
    def run(self) -> EngineResult   # frozen dataclass: period_results df, daily_returns,
                                    # daily_bmk_returns, historical_weights df, turnover, tc
```

`mode='blend'` reproduces legacy exactly. `mode='active'` = same engine with `bmk_weight=0.0` internally but ALSO records benchmark series + active stats; no blending row.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine.py
import numpy as np
import pandas as pd
from country_rotation.backtest.engine import Engine
from country_rotation.config import BacktestConfig


def _cfg(**kw):
    base = dict(selection_criteria="relative", relative_selection_score=3,
                weighting_method="Equal", bmk="World", bmk_weight=0.5,
                mode="blend", periodicity=21, transaction_cost_bps=2.0)
    base.update(kw)
    return BacktestConfig(**base)


def test_blend_weights_sum_to_one(synthetic_prices, synthetic_scores):
    res = Engine(synthetic_scores, synthetic_prices, _cfg()).run()
    w = res.historical_weights.drop(columns=["date"], errors="ignore")
    sums = res.historical_weights.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)
    assert (res.historical_weights["World"].iloc[1:] == 0.5).all()


def test_equal_weighting_among_selected(synthetic_prices, synthetic_scores):
    res = Engine(synthetic_scores, synthetic_prices, _cfg(bmk_weight=0.0)).run()
    w = res.historical_weights.drop(columns="World", errors="ignore")
    nonzero = w.iloc[5][w.iloc[5] > 0]
    assert len(nonzero) == 3
    assert np.allclose(nonzero, 1.0 / 3 * 1.0, atol=1e-9)


def test_tc_reduces_returns(synthetic_prices, synthetic_scores):
    r0 = Engine(synthetic_scores, synthetic_prices, _cfg(transaction_cost_bps=0.0)).run()
    r1 = Engine(synthetic_scores, synthetic_prices, _cfg(transaction_cost_bps=50.0)).run()
    assert r1.period_results["portfolio_return_net"].sum() < r0.period_results["portfolio_return_net"].sum()


def test_active_mode_reports_benchmark_relative(synthetic_prices, synthetic_scores):
    res = Engine(synthetic_scores, synthetic_prices, _cfg(mode="active", bmk_weight=0.0)).run()
    assert "active_return" in res.period_results.columns
    assert "World" not in res.historical_weights.columns or (res.historical_weights["World"] == 0).all()


def test_no_lookahead_in_engine(synthetic_prices, synthetic_scores):
    """Perturbing prices after date t must not change weights chosen at <= t."""
    res1 = Engine(synthetic_scores, synthetic_prices, _cfg()).run()
    cut = synthetic_prices.index[400]
    pert = synthetic_prices.copy()
    pert.loc[pert.index > cut] *= 3.0
    res2 = Engine(synthetic_scores, pert, _cfg()).run()
    w1 = res1.historical_weights.loc[res1.historical_weights.index <= cut]
    w2 = res2.historical_weights.loc[res2.historical_weights.index <= cut]
    pd.testing.assert_frame_equal(w1, w2)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Port** the engine. Keep `d_sel`/`d_ret` tuple loop, selection, weighting, blending, turnover, TC and daily-curve logic mathematically identical to cited lines. Active mode: skip `_construct_active_weights` blending; `period_results` gains `active_return = portfolio_return_net - bmk_return`.
- [ ] **Step 4: Run** → PASS; full suite green.
- [ ] **Step 5: Commit** `git commit -m "feat(backtest): engine port with blend parity + new active mode"`

---

### Task 11: Parity regression — new engine vs legacy Backtest

**Files:**
- Test: `tests/test_parity.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_parity.py
"""New Engine must reproduce legacy Backtest period returns exactly (blend mode)."""
import numpy as np
import pandas as pd
import pytest

from backtest import Backtest               # legacy root module
from country_rotation.backtest.engine import Engine
from country_rotation.config import BacktestConfig


@pytest.mark.parametrize("selection,weighting,bmkw,period", [
    ("relative", "Equal", 0.5, 21),
    ("relative", "Equal", 0.0, 63),
    ("absolute", "Equal", 0.3, 21),
    ("relative", "Risk_Parity", 0.5, 21),
])
def test_period_return_parity(synthetic_prices, synthetic_scores, selection, weighting, bmkw, period):
    legacy = Backtest(
        normalized_score=synthetic_scores, prices=synthetic_prices,
        selection_criteria=selection, absolute_selection_score=0.75,
        relative_selection_score=5, weighting_method=weighting,
        bmk="World", bmk_weight=bmkw, periodicity=period, transaction_cost_bps=2.0,
    )
    legacy_res = legacy.run_backtest()

    cfg = BacktestConfig(selection_criteria=selection, absolute_selection_score=0.75,
                         relative_selection_score=5, weighting_method=weighting,
                         bmk="World", bmk_weight=bmkw, mode="blend",
                         periodicity=period, transaction_cost_bps=2.0)
    new_res = Engine(synthetic_scores, synthetic_prices, cfg).run()

    a = np.asarray(legacy_res["portfolio_return_net"], dtype=float)
    b = np.asarray(new_res.period_results["portfolio_return_net"], dtype=float)
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-9, equal_nan=True)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_parity.py -v`. Investigate ANY mismatch by printing first divergent period; fix the NEW engine (legacy is reference). If legacy result-frame column names differ (inspect `legacy_res.columns` / it may be a DataFrame keyed differently — read `backtest.py:740-800`), adapt the test accessor, not the engine.
- [ ] **Step 3: Run full suite** → all green.
- [ ] **Step 4: Commit** `git commit -m "test: parity lock — new engine reproduces legacy Backtest"`

---

### Task 12 (PORT): factors/redundancy.py + backtest/benchmarks.py

**Files:**
- Create: `country_rotation/factors/redundancy.py`, `country_rotation/backtest/benchmarks.py`
- Test: `tests/test_redundancy.py`, `tests/test_benchmarks.py`
- Legacy source: redundancy `FactorTransformer.py:767-908` (per-country pairwise corr averaged :808-829, min 10 obs :822, distance 1−|ρ| , linkage :833, fcluster by distance threshold, representative selection :853-886 coverage/unique/central)

`redundancy.cluster_factors(factor_dfs: dict[str, pd.DataFrame], threshold: float, linkage_method="average") -> ClusterResult(clusters: dict[int, list[str]], representatives: list[str], distance: pd.DataFrame)`.

`benchmarks.equal_weight_buy_hold(prices: pd.DataFrame, start=None) -> pd.Series` — rebase each column to 1.0 at first common valid date, mean across columns, renormalized to start at 1.0.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_redundancy.py
import numpy as np
import pandas as pd
from country_rotation.factors import redundancy


def test_duplicate_factors_cluster_together():
    idx = pd.bdate_range("2019-01-01", periods=200)
    rng = np.random.default_rng(5)
    base = pd.DataFrame(rng.normal(size=(200, 4)).cumsum(axis=0), index=idx, columns=list("WXYZ"))
    dfs = {"F1": base, "F2": base * 2 + 1, "F3": pd.DataFrame(
        rng.normal(size=(200, 4)).cumsum(axis=0), index=idx, columns=list("WXYZ"))}
    res = redundancy.cluster_factors(dfs, threshold=0.3)
    cl = {tuple(sorted(v)) for v in res.clusters.values()}
    assert ("F1", "F2") in cl              # perfectly correlated -> same cluster
    assert len(res.representatives) == 2   # one per cluster
```

```python
# tests/test_benchmarks.py
import numpy as np
from country_rotation.backtest import benchmarks


def test_eqw_buy_hold_starts_at_one(synthetic_prices):
    countries = [c for c in synthetic_prices.columns if c != "World"]
    eq = benchmarks.equal_weight_buy_hold(synthetic_prices[countries])
    assert abs(eq.iloc[0] - 1.0) < 1e-12
    assert len(eq) == len(synthetic_prices)
    assert eq.isna().sum() == 0
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** (port redundancy from cited lines; benchmarks is new, ~10 lines).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: redundancy clustering port + equal-weight buy-hold null"`

---

### Task 13: scripts/ thin CLIs + full-suite gate

**Files:**
- Create: `scripts/build_scores.py`, `scripts/run_backtest.py`
- Test: smoke only (argparse wiring), `tests/test_scripts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scripts.py
import subprocess, sys


def test_run_backtest_help():
    out = subprocess.run([sys.executable, "scripts/run_backtest.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "--config" in out.stdout


def test_build_scores_help():
    out = subprocess.run([sys.executable, "scripts/build_scores.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Each script: argparse with `--config configs/default.json` (+ `--scores`, `--prices` overrides for run_backtest; `--market`, `--metric-scenario`, `--category-scenario` for build_scores), loads config, calls package functions, writes outputs under `outputs/`. Mirror the orchestration order of legacy `FactorTesting.py:run` and `strategy.py` but through the new package modules only.
- [ ] **Step 4: Run** `python -m pytest tests -v` → entire suite green.
- [ ] **Step 5: Commit + push** `git commit -m "feat: thin CLI scripts over package pipeline" && git push origin dev`

---

## Self-Review (done at write time)

- **Spec coverage:** D1 (Task 3), D2 (Task 3), D3/D4 (Tasks 4, 7), D5 (Task 9), D7 (Task 10), D8 (Task 4), D9 (all), D10 (Task 2). D6 (walk-forward selection) is Plan B by design. Reporting is Plan C. ✓
- **Placeholder scan:** PORT tasks cite exact legacy line ranges + signatures + tests; new logic has full code. No TBDs. ✓
- **Type consistency:** `BacktestConfig` fields match Engine usage (Tasks 2/10/11); transform output keys match composite input keys (Tasks 6/8); fixtures shared via conftest (Task 1). ✓
- **Known risk:** legacy `run_backtest()` return shape assumed DataFrame with `portfolio_return_net` — Task 11 Step 2 explicitly tells the implementer to inspect and adapt the accessor, never the engine.
