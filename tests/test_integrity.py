import numpy as np
import pandas as pd
import pytest
from country_rotation.data import integrity
from country_rotation.data import processing


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


def test_ffill_past_cutoff_is_clean(synthetic_factor):
    """ffill is backward-looking; a NaN block ending past the cutoff should be CLEAN."""
    idx = synthetic_factor.index
    cutoff = idx[200]

    price = synthetic_factor.abs() + 1.0
    # NaN block spans rows 195-210; cutoff is at row 200
    price.iloc[195:211] = np.nan

    # ffill only carries the last known value forward — no future data used
    report = integrity.lookahead_check(
        pipeline=lambda d: {"f": d["Price"].ffill()},
        dfs={"Price": price},
        cutoff=cutoff,
        perturb_scale=100.0,
    )
    assert report.clean, f"ffill falsely flagged as leaky: {report.dirty_outputs}"


def test_bfill_past_cutoff_is_dirty(synthetic_factor):
    """bfill pulls post-cutoff values into rows <= cutoff — must be DIRTY."""
    idx = synthetic_factor.index
    cutoff = idx[200]

    price = synthetic_factor.abs() + 1.0
    # NaN block spans rows 195-205; bfill will fill those from row 206 onward
    price.iloc[195:206] = np.nan

    report = integrity.lookahead_check(
        pipeline=lambda d: {"bad": d["Price"].bfill()},
        dfs={"Price": price},
        cutoff=cutoff,
        perturb_scale=100.0,
    )
    assert not report.clean, "bfill over NaN-past-cutoff not detected as leaky"
    assert "bad" in report.dirty_outputs


def test_lookahead_check_catches_real_leak(synthetic_factor):
    """A pipeline that back-fills future values into the past must be flagged."""
    # Build a Price df that has NaN rows before the cutoff and real values after.
    # bfill() will propagate future data backward — a textbook look-ahead leak.
    idx = synthetic_factor.index
    cutoff = idx[100]

    price = synthetic_factor.abs() + 1.0
    # Introduce NaN in rows 50-100 so bfill has material future data to pull back.
    price.iloc[50:101] = np.nan

    def leaky_pipeline(d):
        return {"bad": d["Price"].bfill()}

    report = integrity.lookahead_check(
        pipeline=leaky_pipeline,
        dfs={"Price": price},
        cutoff=cutoff,
        perturb_scale=100.0,
    )
    assert not report.clean, "expected leak not detected"
    assert "bad" in report.dirty_outputs
