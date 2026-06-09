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
    df.iloc[:40, 5] = np.nan
    return df
