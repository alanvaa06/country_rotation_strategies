"""Tests for country_rotation/reporting/signal_viz.py.

TDD — written before signal_viz.py exists.  All fixtures are synthetic
(no real / gitignored data required).

Tests
-----
1.  test_ranking_frame_sorting_columns — _ranking_frame returns countries
    sorted by total normalized score DESC with category + meta columns.
2.  test_ranking_frame_handles_missing — None scores/contributions never
    crash; missing contributions become NaN.
3.  test_fig_signal_ranking_png       — base64 decodes to a real PNG.
4.  test_fig_signal_ranking_empty     — None-safe on empty/None input.
5.  test_fig_allocation_history_png   — stacked-area PNG from weights.
6.  test_fig_allocation_history_empty — None-safe on empty input.
7.  test_signal_history_payload       — shape, ISO dates, 4dp rounding,
    NaN -> None for JS embedding.
8.  test_signal_history_payload_empty — empty frame -> empty payload.
"""
from __future__ import annotations

import base64

import numpy as np
import pandas as pd

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _signal_latest(n: int = 6) -> dict:
    """signal_latest.json-shaped payload: 6 countries x 2 categories.

    Scores are chosen NON-monotonic in insertion order so the sorting
    assertion is meaningful.
    """
    scores = [0.42, 0.91, 0.05, 1.0, 0.63, 0.27][:n]
    names = [f"Country{i}" for i in range(n)]
    countries = {}
    for name, s in zip(names, scores):
        countries[name] = {
            "score": s,
            "score_change": round(s - 0.5, 4),
            "weight": round(s / 10.0, 4),
            "base_weight": 1.0 / n,
            "contributions": {
                "Momentum": round(s * 0.6, 6),
                "Valuation": round(s * 0.4, 6),
            },
        }
    return {
        "strategy_id": "TEST_strategy",
        "date": "2025-11-14",
        "periodicity": 63,
        "countries": countries,
    }


def _allocations(n_dates: int = 12, n_countries: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2024-01-02", periods=n_dates, freq="63B")
    raw = rng.random((n_dates, n_countries))
    df = pd.DataFrame(
        raw, index=idx, columns=[f"C{i}" for i in range(n_countries)]
    )
    return df.div(df.sum(axis=1), axis=0)


# ---------------------------------------------------------------------------
# _ranking_frame: data prep is the testable core (PNG ordering is not)
# ---------------------------------------------------------------------------

def test_ranking_frame_sorting_columns():
    from country_rotation.reporting.signal_viz import _ranking_frame

    frame = _ranking_frame(_signal_latest())

    # Countries sorted by total normalized score DESC
    expected_order = ["Country3", "Country1", "Country4", "Country0",
                      "Country5", "Country2"]
    assert list(frame.index) == expected_order
    assert frame["score"].is_monotonic_decreasing

    # Category contribution columns + meta columns
    assert list(frame.columns) == [
        "Momentum", "Valuation", "score", "score_change", "weight",
        "base_weight",
    ]
    # Rebased contributions stack to the score
    stacked = frame["Momentum"] + frame["Valuation"]
    assert np.allclose(stacked.values, frame["score"].values, atol=1e-9)
    # Meta values flow through
    assert frame.loc["Country3", "score"] == 1.0
    assert frame.loc["Country3", "weight"] == 0.1


def test_ranking_frame_handles_missing():
    from country_rotation.reporting.signal_viz import _ranking_frame

    payload = _signal_latest(3)
    payload["countries"]["Country0"]["contributions"]["Momentum"] = None
    payload["countries"]["Country1"]["score_change"] = None

    frame = _ranking_frame(payload)
    assert np.isnan(frame.loc["Country0", "Momentum"])
    assert np.isnan(frame.loc["Country1", "score_change"])
    # Empty -> empty frame
    assert _ranking_frame({}).empty
    assert _ranking_frame({"countries": {}}).empty


# ---------------------------------------------------------------------------
# fig_signal_ranking
# ---------------------------------------------------------------------------

def _decodes_to_png(b64: str) -> bool:
    raw = base64.b64decode(b64)
    return raw[: len(_PNG_MAGIC)] == _PNG_MAGIC and len(raw) > 1000


def test_fig_signal_ranking_png():
    from country_rotation.reporting.signal_viz import fig_signal_ranking

    b64 = fig_signal_ranking(_signal_latest(), "2025-11-14")
    assert isinstance(b64, str) and len(b64) > 0
    assert _decodes_to_png(b64)


def test_fig_signal_ranking_empty():
    from country_rotation.reporting.signal_viz import fig_signal_ranking

    assert fig_signal_ranking({}, "2025-11-14") is None
    assert fig_signal_ranking({"countries": {}}, "2025-11-14") is None
    assert fig_signal_ranking(None, "2025-11-14") is None


# ---------------------------------------------------------------------------
# fig_allocation_history
# ---------------------------------------------------------------------------

def test_fig_allocation_history_png():
    from country_rotation.reporting.signal_viz import fig_allocation_history

    b64 = fig_allocation_history(_allocations())
    assert isinstance(b64, str) and len(b64) > 0
    assert _decodes_to_png(b64)


def test_fig_allocation_history_empty():
    from country_rotation.reporting.signal_viz import fig_allocation_history

    assert fig_allocation_history(pd.DataFrame()) is None
    assert fig_allocation_history(None) is None


# ---------------------------------------------------------------------------
# signal_history_payload
# ---------------------------------------------------------------------------

def test_signal_history_payload():
    from country_rotation.reporting.signal_viz import signal_history_payload

    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    df = pd.DataFrame(
        {
            "Alpha": [0.123456789, np.nan, 0.5],
            "Beta": [1.0, 0.0, 0.98765432],
        },
        index=idx,
    )
    payload = signal_history_payload(df)

    assert payload["dates"] == ["2024-01-31", "2024-02-29", "2024-03-29"]
    assert set(payload["countries"]) == {"Alpha", "Beta"}
    # 4dp rounding + NaN -> None (JSON null)
    assert payload["countries"]["Alpha"] == [0.1235, None, 0.5]
    assert payload["countries"]["Beta"] == [1.0, 0.0, 0.9877]
    # JSON-serializable end to end (no NaN literals)
    import json

    text = json.dumps(payload)
    assert "NaN" not in text and "null" in text


def test_signal_history_payload_empty():
    from country_rotation.reporting.signal_viz import signal_history_payload

    payload = signal_history_payload(pd.DataFrame())
    assert payload == {"dates": [], "countries": {}}
    payload = signal_history_payload(None)
    assert payload == {"dates": [], "countries": {}}


# ---------------------------------------------------------------------------
# fig_ic_distribution
# ---------------------------------------------------------------------------

def _ic_series(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic IC DataFrame with a single column 'IC'."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n, freq="63B")
    return pd.DataFrame({"IC": rng.normal(0.05, 0.2, n)}, index=idx)


def test_fig_ic_distribution_png():
    from country_rotation.reporting.signal_viz import fig_ic_distribution

    ic_df = _ic_series(60)
    b64 = fig_ic_distribution(ic_df, "Absolute (score level)")
    assert isinstance(b64, str) and len(b64) > 0
    assert _decodes_to_png(b64), "Expected a valid PNG output"


def test_fig_ic_distribution_series_input():
    """Accepts a pd.Series as well as a single-column DataFrame."""
    from country_rotation.reporting.signal_viz import fig_ic_distribution

    ic_s = _ic_series(60)["IC"]
    b64 = fig_ic_distribution(ic_s, "Relative (63d score change)")
    assert isinstance(b64, str) and _decodes_to_png(b64)


def test_fig_ic_distribution_empty_returns_none():
    from country_rotation.reporting.signal_viz import fig_ic_distribution

    assert fig_ic_distribution(pd.DataFrame(), "Absolute") is None
    assert fig_ic_distribution(None, "Absolute") is None
    # All-NaN series
    assert fig_ic_distribution(
        pd.DataFrame({"IC": [float("nan"), float("nan")]}), "Absolute"
    ) is None
