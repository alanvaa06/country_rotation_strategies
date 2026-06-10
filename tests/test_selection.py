"""Tests for country_rotation.selection.walkforward (Task B3).

Covers: Benjamini-Hochberg adjusted q-values, planted-signal screening,
lockbox isolation (the definitive no-peek test), one-shot lockbox
verification, and statistical power of the per-period IC series t-test.
"""
import numpy as np
import pandas as pd
import pytest

from country_rotation.selection import walkforward as wf

PERIODICITY = 21
LOCKBOX_FRAC = 0.2


def _countries(prices: pd.DataFrame) -> list:
    return [c for c in prices.columns if c != "World"]


def _oracle_scores(px: pd.DataFrame, periodicity: int = PERIODICITY) -> pd.DataFrame:
    """Forward-return ranks: a perfect (look-ahead) factor for planting signal."""
    fwd = px.shift(-periodicity) / px - 1.0
    return fwd.rank(axis=1, pct=True).fillna(0.5)


def _noise_scores(px: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random(px.shape), index=px.index, columns=px.columns)


def _planted_factors(px: pd.DataFrame) -> dict:
    return {
        "good": _oracle_scores(px),
        "noise1": _noise_scores(px, 11),
        "noise2": _noise_scores(px, 12),
        "noise3": _noise_scores(px, 13),
    }


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_known_values():
    pvalues = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.9}
    q = wf.benjamini_hochberg(pvalues, 0.05)
    # raw step-up with m=4: p*m/rank, then monotone from the largest p down
    assert q["a"] == pytest.approx(0.004)
    assert q["b"] == pytest.approx(0.04)
    assert q["c"] == pytest.approx(0.04 * 4 / 3)   # 0.05333...
    assert q["d"] == pytest.approx(0.9)
    assert q["a"] <= 0.05            # a survives at q=0.05
    assert q["d"] >= 0.9             # d never survives
    assert q["a"] <= q["b"] <= q["c"] <= q["d"]   # monotone in p order


def test_benjamini_hochberg_empty():
    assert wf.benjamini_hochberg({}, 0.10) == {}


# ---------------------------------------------------------------------------
# screen_factors
# ---------------------------------------------------------------------------

def test_screen_factors_planted_signal(synthetic_prices):
    px = synthetic_prices[_countries(synthetic_prices)]
    factors = _planted_factors(px)

    res = wf.screen_factors(factors, px, periodicity=PERIODICITY, n_folds=4)

    assert "good" in res.kept
    assert len(res.kept) <= 2
    assert res.lockbox_ic is None
    assert set(res.fold_ic.index) == set(factors)
    assert res.fold_ic.shape == (len(factors), 4)
    assert list(res.fold_ic.columns) == ["fold_1", "fold_2", "fold_3", "fold_4"]
    assert set(res.bh_qvalues) == set(factors)
    # partition: every factor is either kept or dropped, never both
    assert set(res.kept) | set(res.dropped) == set(factors)
    assert set(res.kept) & set(res.dropped) == set()
    assert set(res.weak) <= set(res.kept)


def test_screen_factors_all_noise(synthetic_prices):
    px = synthetic_prices[_countries(synthetic_prices)]
    factors = {f"noise{i}": _noise_scores(px, 20 + i) for i in range(4)}

    res = wf.screen_factors(factors, px, periodicity=PERIODICITY, n_folds=4)

    assert set(res.kept) <= set(factors)
    assert set(res.kept) | set(res.dropped) == set(factors)
    assert set(res.weak) <= set(res.kept)


def test_lockbox_isolation(synthetic_prices):
    """THE definitive no-peek test: scrambling the lockbox slice of every
    input (factors AND prices) must leave the screening result bit-identical.
    """
    px = synthetic_prices[_countries(synthetic_prices)]
    factors = _planted_factors(px)

    base = wf.screen_factors(factors, px, periodicity=PERIODICITY, n_folds=4)

    # Same floor boundary screen_factors uses: lockbox = last int(0.2*n) dates.
    n = len(px)
    cut = n - int(LOCKBOX_FRAC * n)
    rng = np.random.default_rng(99)

    def scramble(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        block = out.iloc[cut:]
        out.iloc[cut:] = -block.to_numpy() + rng.normal(0.0, 1.0, block.shape)
        return out

    factors_s = {name: scramble(df) for name, df in factors.items()}
    px_s = scramble(px)

    pert = wf.screen_factors(factors_s, px_s, periodicity=PERIODICITY, n_folds=4)

    assert pert.kept == base.kept
    assert pert.weak == base.weak
    assert pert.dropped == base.dropped
    assert pert.bh_qvalues == base.bh_qvalues       # exact: lockbox never read
    pd.testing.assert_frame_equal(pert.fold_ic, base.fold_ic)


# ---------------------------------------------------------------------------
# min_mean_ic gate (Task B4 improvement to B3)
# ---------------------------------------------------------------------------

def test_min_mean_ic_oracle_kept(synthetic_prices):
    """Oracle factor has mean fold IC ~1.0; min_mean_ic=0.5 still keeps it."""
    px = synthetic_prices[_countries(synthetic_prices)]
    factors = _planted_factors(px)

    res = wf.screen_factors(
        factors, px, periodicity=PERIODICITY, n_folds=4, min_mean_ic=0.5
    )

    assert "good" in res.kept, (
        "Oracle factor should survive min_mean_ic=0.5 (its IC is ~1.0)"
    )


def test_min_mean_ic_all_dropped(synthetic_prices):
    """min_mean_ic=1.5 is impossible; no factor (not even the oracle) is kept."""
    px = synthetic_prices[_countries(synthetic_prices)]
    factors = _planted_factors(px)

    res = wf.screen_factors(
        factors, px, periodicity=PERIODICITY, n_folds=4, min_mean_ic=1.5
    )

    assert len(res.kept) == 0, (
        "min_mean_ic=1.5 should drop every factor (IC is capped at 1.0)"
    )


# ---------------------------------------------------------------------------
# series_t / n_ic_obs populated for every factor
# ---------------------------------------------------------------------------

def test_series_t_and_n_ic_obs_populated(synthetic_prices):
    """series_t and n_ic_obs must be present for every input factor."""
    px = synthetic_prices[_countries(synthetic_prices)]
    factors = _planted_factors(px)

    res = wf.screen_factors(factors, px, periodicity=PERIODICITY, n_folds=4)

    assert set(res.series_t) == set(factors), "series_t must cover every factor"
    assert set(res.n_ic_obs) == set(factors), "n_ic_obs must cover every factor"

    for name in factors:
        n = res.n_ic_obs[name]
        assert isinstance(n, int), f"n_ic_obs[{name!r}] must be an int"
        assert n >= 0, f"n_ic_obs[{name!r}] must be non-negative"
        # t-stat is finite for factors with enough observations
        if n >= wf.MIN_IC_OBS:
            assert np.isfinite(res.series_t[name]) or res.series_t[name] in (
                float("inf"), float("-inf")
            ), f"series_t[{name!r}] must be a number when n={n}"


# ---------------------------------------------------------------------------
# Power test: weak-but-persistent signal survives; pure noise is dropped
# ---------------------------------------------------------------------------

def test_series_ttest_power():
    """Per-period IC t-test must keep a weak-but-persistent real signal that
    the old 5-fold test (df=4) would be too underpowered to detect.

    Design
    ------
    * 600 business days, 15 countries, periodicity=21 → ~27 IC obs in screening
      window.  Old df=4 test had critical |t| ≈ 2.13 at α=0.05 (one-sided);
      new df≈26 test has critical |t| ≈ 1.71.  The planted signal is in the
      borderline zone: strong enough for the new test but borderline for the old.

    Signal construction
    -------------------
      scores = 0.8 * (forward-return ranks) + 0.2 * noise
    This gives a persistent but imperfect IC series.  The noise factor is pure
    random, so it should be dropped.
    """
    rng = np.random.default_rng(42)

    n_days = 600
    n_countries = 15
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    countries = [f"C{i}" for i in range(n_countries)]

    # Prices with small positive drift
    log_returns = rng.normal(0.0002, 0.012, (n_days, n_countries))
    px = pd.DataFrame(
        100.0 * np.exp(np.cumsum(log_returns, axis=0)),
        index=idx, columns=countries,
    )

    # Weak-but-persistent signal: 80% future-return ranks + 20% noise
    fwd = px.shift(-PERIODICITY) / px - 1.0
    future_ranks = fwd.rank(axis=1, pct=True).fillna(0.5)
    noise_component = pd.DataFrame(
        rng.random((n_days, n_countries)), index=idx, columns=countries
    )
    signal_factor = 0.8 * future_ranks + 0.2 * noise_component

    # Pure noise factor
    noise_factor = pd.DataFrame(
        rng.random((n_days, n_countries)), index=idx, columns=countries
    )

    factors = {"signal": signal_factor, "pure_noise": noise_factor}

    res = wf.screen_factors(
        factors, px,
        periodicity=PERIODICITY,
        n_folds=5,
        lockbox_frac=0.2,
        fdr_q=0.10,
    )

    assert "signal" in res.kept, (
        f"Weak-but-persistent signal must survive the per-period IC t-test "
        f"(series_t={res.series_t.get('signal'):.3f}, "
        f"n_ic_obs={res.n_ic_obs.get('signal')}, "
        f"bh_q={res.bh_qvalues.get('signal'):.4f})"
    )
    assert "pure_noise" not in res.kept, (
        f"Pure noise must be dropped "
        f"(series_t={res.series_t.get('pure_noise'):.3f}, "
        f"bh_q={res.bh_qvalues.get('pure_noise'):.4f})"
    )


# ---------------------------------------------------------------------------
# verify_on_lockbox
# ---------------------------------------------------------------------------

def test_verify_on_lockbox(synthetic_prices):
    px = synthetic_prices[_countries(synthetic_prices)]
    factors = _planted_factors(px)

    res = wf.screen_factors(factors, px, periodicity=PERIODICITY, n_folds=4)
    assert "good" in res.kept

    out = wf.verify_on_lockbox(
        res, factors, px, periodicity=PERIODICITY, lockbox_frac=LOCKBOX_FRAC
    )

    # original result untouched (frozen dataclass, replace returns a copy)
    assert res.lockbox_ic is None
    # lockbox IC filled for kept factors ONLY
    assert out.lockbox_ic is not None
    assert set(out.lockbox_ic) == set(res.kept)
    # everything else carried over unchanged
    assert out.kept == res.kept
    assert out.weak == res.weak
    assert out.dropped == res.dropped
    assert out.bh_qvalues == res.bh_qvalues
    pd.testing.assert_frame_equal(out.fold_ic, res.fold_ic)
    # the oracle factor predicts the lockbox too (it embeds future returns)
    assert out.lockbox_ic["good"] > 0.5
