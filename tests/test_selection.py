"""Tests for country_rotation.selection.walkforward (Task B3).

Covers: Benjamini-Hochberg adjusted q-values, planted-signal screening,
lockbox isolation (the definitive no-peek test) and one-shot lockbox
verification.
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
