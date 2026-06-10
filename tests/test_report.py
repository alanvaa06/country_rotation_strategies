"""Tests for country_rotation/reporting/report.py (Plan C).

TDD — written before report.py exists.

Tests
-----
1. test_build_report_smoke        — minimal build_report end-to-end, ≥4 figures, key sections.
2. test_report_with_contributions_and_scorecard — contributions dict + ValidationReport → extra sections.
3. test_script_help               — scripts/build_report.py --help exits 0, lists --scores.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from country_rotation.config import BacktestConfig


# ---------------------------------------------------------------------------
# Synthetic fixture helpers (independent of conftest to keep test self-contained)
# ---------------------------------------------------------------------------

def _make_prices(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2016-01-04", periods=n)
    names = [f"C{i}" for i in range(8)]
    drifts = np.linspace(-0.0001, 0.0008, 8)
    cols = {}
    for k, name in enumerate(names):
        r = rng.normal(drifts[k], 0.012, n)
        cols[name] = 100.0 * np.cumprod(1.0 + r)
    df = pd.DataFrame(cols, index=idx)
    df["World"] = df[names].mean(axis=1)
    return df


def _make_scores(prices: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    countries = [c for c in prices.columns if c != "World"]
    raw = rng.random((len(prices), len(countries)))
    df = pd.DataFrame(raw, index=prices.index, columns=countries)
    return df.rolling(10, min_periods=1).mean()


def _make_cfg() -> BacktestConfig:
    return BacktestConfig(
        selection_criteria="relative",
        relative_selection_score=3,
        weighting_method="Equal",
        bmk="World",
        bmk_weight=0.0,
        mode="active",
        periodicity=21,
        transaction_cost_bps=2.0,
    )


def _make_contributions(scores: pd.DataFrame) -> dict:
    """Fabricate a {category: DataFrame} contributions dict."""
    rng = np.random.default_rng(99)
    cats = ["Valuation", "Momentum"]
    result = {}
    for cat in cats:
        data = rng.uniform(-0.1, 0.1, scores.shape)
        result[cat] = pd.DataFrame(data, index=scores.index, columns=scores.columns)
    return result


# ---------------------------------------------------------------------------
# Test 1: smoke — build_report returns sensible result and writes valid HTML
# ---------------------------------------------------------------------------

def test_build_report_smoke(tmp_path):
    """build_report produces an HTML file with >=4 base64 images and key sections."""
    from country_rotation.reporting.report import build_report

    prices = _make_prices(400)
    scores = _make_scores(prices)
    cfg = _make_cfg()
    output = tmp_path / "report.html"

    result = build_report(
        scores=scores,
        prices=prices,
        cfg=cfg,
        output_path=str(output),
        validation_report=None,
        contributions=None,
        ic_methods=("absolute", "relative"),
    )

    # File written
    assert output.exists(), "HTML file was not created"

    html = output.read_text(encoding="utf-8")

    # Required sections
    assert "Performance" in html, "Missing 'Performance' section"
    assert "Risk" in html, "Missing 'Risk' section"
    assert "IC Analysis" in html, "Missing 'IC Analysis' section"

    # At least 4 embedded images
    n_images = html.count("data:image/png;base64,")
    assert n_images >= 4, f"Expected >=4 base64 images, got {n_images}"

    # Return dict
    assert result["n_figures"] >= 4, f"n_figures={result['n_figures']} < 4"
    assert Path(result["path"]) == output
    assert isinstance(result["summary"], dict)
    assert "sharpe" in result["summary"]


# ---------------------------------------------------------------------------
# Test 2: contributions dict + scorecard → extra HTML sections
# ---------------------------------------------------------------------------

def test_report_with_contributions_and_scorecard(tmp_path):
    """Contributions + ValidationReport → 'Score Decomposition' + 'Validation Scorecard' in HTML."""
    from country_rotation.reporting.report import build_report
    from country_rotation.validation.scorecard import compute_validation

    prices = _make_prices(400)
    scores = _make_scores(prices)
    cfg = _make_cfg()

    contributions = _make_contributions(scores)

    # Run a tiny validation
    grid = {"relative_selection_score": (3, 5)}
    vr = compute_validation(
        scores=scores,
        prices=prices,
        cfg=cfg,
        grid=grid,
        n_boot=50,
        n_mc=5,
        n_folds=3,
        seed=0,
    )

    output = tmp_path / "report_full.html"
    result = build_report(
        scores=scores,
        prices=prices,
        cfg=cfg,
        output_path=str(output),
        validation_report=vr,
        contributions=contributions,
        ic_methods=("absolute",),
    )

    html = output.read_text(encoding="utf-8")

    assert "Score Decomposition" in html, "Missing 'Score Decomposition' section"
    assert "Validation Scorecard" in html, "Missing 'Validation Scorecard' section"

    # At least one PASS or FAIL cell
    assert "PASS" in html or "FAIL" in html, "No PASS/FAIL in scorecard table"


# ---------------------------------------------------------------------------
# Test 3: script --help
# ---------------------------------------------------------------------------

def test_script_help():
    """scripts/build_report.py --help exits 0 and mentions --scores."""
    repo_root = Path(__file__).parent.parent
    script = repo_root / "scripts" / "build_report.py"
    env = {"PYTHONIOENCODING": "utf-8", "PATH": subprocess.os.environ.get("PATH", "")}
    # inherit full env but ensure encoding
    import os
    full_env = dict(os.environ)
    full_env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=30,
    )
    assert proc.returncode == 0, f"--help failed:\n{proc.stderr}"
    assert "--scores" in proc.stdout, f"'--scores' not found in --help output:\n{proc.stdout}"
