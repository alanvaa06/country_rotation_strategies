"""Tests for country_rotation/reporting/dashboard.py + scripts/build_dashboard.py.

TDD — written before dashboard.py exists.  All fixtures are synthetic
(no real / gitignored data required).

Tests
-----
1. test_build_strategy_pane_smoke     — pane HTML with >=4 base64 figures,
                                        verdict banner, stat cards, scorecard.
2. test_build_strategy_pane_cap_tilt  — Cap_Tilt cfg + base_weights wiring.
3. test_render_dashboard_structure    — 9 pane ids, tabs, toggle JS, default
                                        pane visible (display logic markers).
4. test_strategy_lineup_cfgs          — script's strategy lineup builds the
                                        correct engine configs per key.
5. test_script_help                   — scripts/build_dashboard.py --help.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from country_rotation.config import BacktestConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
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


def _make_base_weights(scores: pd.DataFrame) -> pd.DataFrame:
    n = len(scores.columns)
    return pd.DataFrame(
        1.0 / n, index=scores.index, columns=scores.columns
    )


def _make_contributions(scores: pd.DataFrame) -> dict:
    rng = np.random.default_rng(99)
    return {
        cat: pd.DataFrame(
            rng.uniform(-0.1, 0.1, scores.shape),
            index=scores.index, columns=scores.columns,
        )
        for cat in ("Valuation", "Momentum")
    }


def _make_cfg(**overrides) -> BacktestConfig:
    base = dict(
        selection_criteria="relative",
        relative_selection_score=3,
        weighting_method="Equal",
        bmk="World",
        bmk_weight=0.0,
        mode="active",
        periodicity=21,
        transaction_cost_bps=2.0,
        active_share=0.30,
    )
    base.update(overrides)
    return BacktestConfig(**base)


def _fake_verdict(overall: bool = False) -> dict:
    """Minimal verdict dict mirroring research_run.build_verdict_payload."""
    return {
        "segment": "World",
        "track": "prior",
        "basis": "active",
        "periodicity": 63,
        "construction": "cap_tilt",
        "active_share": 0.3,
        "verdict": {
            "checks": {
                "no_overfitting": False,
                "param_stable": True,
                "statistically_significant": False,
            },
            "overall": overall,
            "notes": ["synthetic-note-alpha", "synthetic-note-beta"],
        },
        "stats": {
            "sharpe_ann": -0.097,
            "sharpe_t_stat": -0.392,
            "psr": 0.347,
            "dsr": 0.208,
            "mc_p_value": 0.0297,
            "wf_efficiency": 2.478,
            "frac_oos_positive": 1.0,
            "bootstrap_ci": [-0.029, 0.015],
            "nw_t_vs_eqw": 3.0995,
            "stability_frac_positive": 0.0,
            "stability_default_zscore": 0.784,
        },
        "mandate_stats": {
            "ann_return": 0.0782,
            "ann_vol": 0.1368,
            "sharpe": 0.572,
            "max_drawdown": -0.327,
            "win_rate": 0.546,
            "beta": 0.933,
            "up_capture": 0.946,
            "down_capture": 0.944,
            "tracking_error": 0.0322,
            "information_ratio": -0.0785,
        },
        "composite_ic": {
            "absolute": {
                "mean_ic": 0.0148, "median_ic": -0.0264, "std_ic": 0.239,
                "t_stat": 0.4998, "icir": 0.062, "hit_rate": 0.4615,
            },
            "relative": {
                "mean_ic": 0.0618, "median_ic": 0.0591, "std_ic": 0.216,
                "t_stat": 2.2902, "icir": 0.286, "hit_rate": 0.625,
            },
        },
    }


# ---------------------------------------------------------------------------
# Test 1: build_strategy_pane smoke (Equal weighting, no base_weights)
# ---------------------------------------------------------------------------

def test_build_strategy_pane_smoke():
    from country_rotation.reporting.dashboard import build_strategy_pane

    prices = _make_prices(400)
    scores = _make_scores(prices)
    cfg = _make_cfg()
    verdict = _fake_verdict(overall=False)
    contributions = _make_contributions(scores)

    html = build_strategy_pane(
        scores, prices, cfg, None, verdict, contributions
    )

    assert isinstance(html, str) and len(html) > 1000

    # Figures embedded inline
    n_images = html.count("data:image/png;base64,")
    assert n_images >= 4, f"Expected >=4 base64 images, got {n_images}"

    # Verdict banner: overall FAIL + the three named checks
    assert "FAIL" in html
    for check in ("no_overfitting", "param_stable", "statistically_significant"):
        assert check in html, f"check '{check}' missing from banner"

    # Stat cards carry key validation numbers
    assert "PSR" in html and "DSR" in html
    assert "0.347" in html       # psr
    assert "0.208" in html       # dsr

    # Scorecard-style table thresholds
    assert "2.0" in html         # sharpe-t threshold
    assert "0.95" in html        # PSR/DSR threshold

    # Per-year + risk tables present
    assert "Per-Year" in html or "Year" in html
    assert "Tracking Error" in html

    # Notes propagated
    assert "synthetic-note-alpha" in html


# ---------------------------------------------------------------------------
# Test 2: Cap_Tilt pane — base_weights threaded into the Engine
# ---------------------------------------------------------------------------

def test_build_strategy_pane_cap_tilt():
    from country_rotation.reporting.dashboard import build_strategy_pane

    prices = _make_prices(400)
    scores = _make_scores(prices)
    base_weights = _make_base_weights(scores)
    cfg = _make_cfg(weighting_method="Cap_Tilt")
    verdict = _fake_verdict(overall=True)

    html = build_strategy_pane(
        scores, prices, cfg, base_weights, verdict, None
    )

    assert isinstance(html, str) and len(html) > 1000
    assert html.count("data:image/png;base64,") >= 4
    assert "PASS" in html


# ---------------------------------------------------------------------------
# Test 3: render_dashboard — tabs, toggle JS, pane visibility
# ---------------------------------------------------------------------------

def test_render_dashboard_structure():
    from country_rotation.reporting.dashboard import render_dashboard

    segs = ("World", "DM", "EM")
    keys = ("cap_tilt", "eqw_active", "blend50")
    labels = {
        "cap_tilt": "Benchmark-Aware (Cap-Tilt)",
        "eqw_active": "Active Equal-Weight Top-5",
        "blend50": "Core-Satellite 50/50",
    }
    segments = {
        seg: {
            key: (labels[key], f"<p>pane-{seg}-{key}</p>") for key in keys
        }
        for seg in segs
    }

    html = render_dashboard("Test Dashboard", segments, default_strategy="cap_tilt")

    # Full standalone document
    assert html.count("<html") == 1 and html.count("</html>") == 1
    assert "Test Dashboard" in html

    # All 9 pane ids present
    for seg in segs:
        for key in keys:
            assert f'id="{seg}-{key}"' in html, f"missing pane id {seg}-{key}"
            assert f"pane-{seg}-{key}" in html  # pane content embedded

    # Toggle JS (vanilla, no external deps)
    assert "selectStrat" in html and "selectSeg" in html
    assert "currentStrat" in html
    assert "<script" in html and "src=" not in html.split("<script")[1][:80]

    # Strategy labels on toggle buttons
    for label in labels.values():
        assert label in html

    # Display logic: default pane (first segment + default strategy) visible,
    # at least one other pane hidden.
    assert 'id="World-cap_tilt" class="pane" style="display:block"' in html
    assert 'id="World-eqw_active" class="pane" style="display:none"' in html
    assert 'id="DM-cap_tilt" class="pane" style="display:none"' in html


def test_render_dashboard_default_strategy_override():
    from country_rotation.reporting.dashboard import render_dashboard

    segments = {
        "World": {
            "cap_tilt": ("A", "<p>a</p>"),
            "blend50": ("B", "<p>b</p>"),
        }
    }
    html = render_dashboard("T", segments, default_strategy="blend50")
    assert 'id="World-blend50" class="pane" style="display:block"' in html
    assert 'id="World-cap_tilt" class="pane" style="display:none"' in html


# ---------------------------------------------------------------------------
# Test 4: script strategy lineup builds correct configs
# ---------------------------------------------------------------------------

def _import_build_dashboard():
    path = REPO_ROOT / "scripts" / "build_dashboard.py"
    spec = importlib.util.spec_from_file_location("build_dashboard", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses resolves string annotations
    # (PEP 563) via sys.modules[cls.__module__].
    sys.modules["build_dashboard"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_strategy_lineup_cfgs():
    mod = _import_build_dashboard()

    keys = [s.key for s in mod.STRATEGIES]
    assert keys == ["cap_tilt", "eqw_active", "blend50"]
    assert mod.DEFAULT_STRATEGY == "cap_tilt"

    base = BacktestConfig()
    by_key = {s.key: s for s in mod.STRATEGIES}

    cap = mod.strategy_cfg(base, by_key["cap_tilt"], segment="DM", periodicity=63)
    assert cap.mode == "active"
    assert cap.weighting_method == "Cap_Tilt"
    assert cap.bmk_weight == 0.0
    assert cap.bmk == "DM"
    assert cap.periodicity == 63
    assert cap.selection_criteria == "relative"
    assert cap.relative_selection_score == 5

    eqw = mod.strategy_cfg(base, by_key["eqw_active"], segment="EM", periodicity=63)
    assert eqw.mode == "active"
    assert eqw.weighting_method == "Equal"
    assert eqw.bmk_weight == 0.0
    assert eqw.bmk == "EM"

    blend = mod.strategy_cfg(base, by_key["blend50"], segment="World", periodicity=63)
    assert blend.mode == "blend"
    assert blend.weighting_method == "Equal"
    assert blend.bmk_weight == 0.5

    # Verdict file naming matches the existing research outputs
    assert by_key["cap_tilt"].verdict_name("DM") == (
        "verdict_DM_prior_vm_p63_active_captilt_capbmk.json"
    )
    assert by_key["eqw_active"].verdict_name("World") == (
        "verdict_World_prior_vm_p63_active_capbmk.json"
    )
    assert by_key["blend50"].verdict_name("EM") == (
        "verdict_EM_prior_vm_p63_active_capbmk_blend50.json"
    )


# ---------------------------------------------------------------------------
# Test 5: script --help
# ---------------------------------------------------------------------------

def test_script_help():
    script = REPO_ROOT / "scripts" / "build_dashboard.py"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, f"--help failed:\n{proc.stderr}"
    assert "--segments" in proc.stdout
    assert "--output" in proc.stdout
