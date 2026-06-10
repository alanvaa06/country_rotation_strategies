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
6. test_evidence_grade_tiers          — pure tier rules (4 tiers + edge cases).
7. test_verdict_banner_*              — grade line + per-gate chips markup.
8. test_pane_sections_collapsible     — sec-toggle buttons, default collapsed.
9. test_render_dashboard_has_no_identifiers_bar — feature removed by request.
10. test_render_dashboard_viewport    — 100vh / overflow:hidden shell.
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


def _em_cap_tilt_stats() -> dict:
    """Real EM cap_tilt numbers (verdict_EM_prior_vm_p63_active_captilt_capbmk):
    MC + walk-forward + stability pass, only the power gates (DSR/PSR/t/CI)
    fail — the canonical 'power-limited' case certified by overfit forensics."""
    return {
        "sharpe_ann": 0.2917,
        "sharpe_t_stat": 1.1759,
        "psr": 0.8818,
        "dsr": 0.7617,
        "mc_p_value": 0.0297,
        "wf_efficiency": 1.4489,
        "frac_oos_positive": 0.8,
        "bootstrap_ci": [-0.0078, 0.0434],
        "nw_t_vs_eqw": 0.542,
        "stability_frac_positive": 1.0,
        "stability_default_zscore": 0.554,
    }


# ---------------------------------------------------------------------------
# Evidence grade: pure tier rules
# ---------------------------------------------------------------------------

def _grade_verdict(**stat_overrides) -> dict:
    base = {
        "sharpe_ann": 0.8, "sharpe_t_stat": 2.5, "psr": 0.97, "dsr": 0.96,
        "mc_p_value": 0.01, "wf_efficiency": 0.9, "frac_oos_positive": 0.8,
        "bootstrap_ci": [0.01, 0.05], "stability_frac_positive": 0.9,
        "stability_default_zscore": 0.5,
    }
    base.update(stat_overrides)
    return {"stats": base, "verdict": {"checks": {}, "overall": False}}


def test_evidence_grade_tiers():
    from country_rotation.reporting.dashboard import evidence_grade

    # 1. CERTIFIED: every gate passes
    g = evidence_grade(_grade_verdict())
    assert g["tier"] == "certified"
    assert g["label"] == "CERTIFIED"

    # 2. POWER-LIMITED: MC + WF + OOS + stability pass, IR > 0, only the
    #    power gates (DSR / PSR / t / CI) fail — real EM cap_tilt numbers.
    g = evidence_grade({"stats": _em_cap_tilt_stats(), "verdict": {}})
    assert g["tier"] == "power_limited"
    assert "power-limited" in g["label"]
    assert "NOT YET CERTIFIED" in g["label"]
    assert "overfitting" in g["note"]

    # 3. WEAK EVIDENCE: MC fails (IR still positive)
    g = evidence_grade(_grade_verdict(mc_p_value=0.30, dsr=0.5))
    assert g["tier"] == "weak"
    # ... or walk-forward fails
    g = evidence_grade(_grade_verdict(wf_efficiency=0.2))
    assert g["tier"] == "weak"
    # ... or positive IR with MC fail (not negative tier)
    g = evidence_grade(_grade_verdict(sharpe_ann=0.3, mc_p_value=0.4, dsr=0.1))
    assert g["tier"] == "weak"

    # 4. NEGATIVE: IR <= 0 and MC fail
    g = evidence_grade(_grade_verdict(sharpe_ann=-0.1, mc_p_value=0.4))
    assert g["tier"] == "negative"
    assert g["label"] == "NEGATIVE"


def test_evidence_grade_edge_cases():
    from country_rotation.reporting.dashboard import evidence_grade

    # Missing stats entirely -> weak (cannot certify), no crash
    g = evidence_grade({"stats": None, "verdict": {}})
    assert g["tier"] == "weak"
    assert g["gates"] == []

    # NaN gate values fail their gate but never crash
    g = evidence_grade(_grade_verdict(dsr=float("nan")))
    assert g["tier"] == "power_limited"
    dsr_gate = [x for x in g["gates"] if x["key"] == "dsr"][0]
    assert dsr_gate["passed"] is False

    # MC+WF pass but stability fails -> not power-limited
    g = evidence_grade(_grade_verdict(stability_frac_positive=0.2, dsr=0.5))
    assert g["tier"] == "weak"


def test_evidence_grade_gates_structure():
    from country_rotation.reporting.dashboard import evidence_grade

    g = evidence_grade({"stats": _em_cap_tilt_stats(), "verdict": {}})
    keys = [x["key"] for x in g["gates"]]
    assert keys == [
        "mc_p", "wf_eff", "oos_folds", "dsr", "psr", "active_t",
        "ci_low", "stability",
    ]
    by_key = {x["key"]: x for x in g["gates"]}
    assert by_key["mc_p"]["passed"] is True
    assert by_key["wf_eff"]["passed"] is True
    assert by_key["oos_folds"]["passed"] is True
    assert by_key["stability"]["passed"] is True
    for k in ("dsr", "psr", "active_t", "ci_low"):
        assert by_key[k]["passed"] is False
        assert by_key[k]["state"] == "amber"   # power-limited fails -> amber
    assert by_key["mc_p"]["state"] == "pass"


# ---------------------------------------------------------------------------
# Verdict banner markup
# ---------------------------------------------------------------------------

def test_verdict_banner_power_limited_markup():
    from country_rotation.reporting.dashboard import verdict_banner

    html = verdict_banner({"stats": _em_cap_tilt_stats(), "verdict": {}})
    assert "NOT YET CERTIFIED" in html and "power-limited" in html
    assert "Overall Verdict" not in html
    assert html.count('class="gate-chip ') == 8
    assert "gate-amber" in html and "gate-pass" in html
    # Chips show value + threshold
    assert "0.030" in html        # MC p value
    assert "0.05" in html         # MC threshold
    assert "0.95" in html         # DSR/PSR threshold
    # Plain-language tier note
    assert "no overfitting signatures detected" in html


def test_verdict_banner_negative_markup():
    from country_rotation.reporting.dashboard import verdict_banner

    html = verdict_banner(_grade_verdict(sharpe_ann=-0.2, mc_p_value=0.5))
    assert "NEGATIVE" in html
    assert "gate-fail" in html
    assert "Overall Verdict" not in html


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

    # Evidence-grade banner replaces the old FAIL-wall
    assert "Overall Verdict: FAIL" not in html
    assert 'class="gate-chip' in html
    # Scorecard detail still carries FAIL cells + the three named checks
    assert "FAIL" in html
    for check in ("no_overfitting", "param_stable", "statistically_significant"):
        assert check in html, f"check '{check}' missing from scorecard"

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
# Collapsible sections inside a pane
# ---------------------------------------------------------------------------

def test_pane_sections_collapsible():
    from country_rotation.reporting.dashboard import build_strategy_pane

    prices = _make_prices(400)
    scores = _make_scores(prices)
    cfg = _make_cfg()
    verdict = _fake_verdict(overall=False)
    contributions = _make_contributions(scores)

    html = build_strategy_pane(scores, prices, cfg, None, verdict, contributions)

    # 7 collapsible sections with contributions: Performance, Per-Year,
    # Risk, Weights, IC, Score Decomposition, Scorecard.
    n_toggles = html.count('<button class="sec-toggle"')
    assert n_toggles == 7, f"expected 7 sec-toggle buttons, got {n_toggles}"
    assert html.count('class="sec-body"') == n_toggles
    # Default: ALL collapsed (display:none + aria-expanded=false)
    assert html.count('aria-expanded="false"') == n_toggles
    assert html.count('class="sec-body" style="display:none"') == n_toggles
    # Banner + stat cards live OUTSIDE any collapsed body (always visible):
    pre_sections = html.split('<button class="sec-toggle"')[0]
    assert "banner" in pre_sections and 'class="cards"' in pre_sections

    # Without contributions: 6 sections
    html6 = build_strategy_pane(scores, prices, cfg, None, verdict, None)
    assert html6.count('<button class="sec-toggle"') == 6


def _tiny_segments() -> dict:
    return {"World": {"cap_tilt": ("Cap-Tilt", "<p>pane</p>")}}


def test_render_dashboard_has_no_identifiers_bar():
    """Identifiers/reference-universe feature was removed by request."""
    from country_rotation.reporting.dashboard import render_dashboard

    html = render_dashboard("T", _tiny_segments())

    assert "Identifiers:" not in html
    assert "toggleIdx" not in html
    assert "idx-panel" not in html


# ---------------------------------------------------------------------------
# Viewport-fit shell
# ---------------------------------------------------------------------------

def test_render_dashboard_viewport_shell():
    from country_rotation.reporting.dashboard import render_dashboard

    html = render_dashboard("T", _tiny_segments())

    # Page never scrolls; only the pane content area does.
    assert "100vh" in html
    assert "overflow:hidden" in html
    assert "overflow-y:auto" in html
    assert 'class="dash-header"' in html
    assert 'class="dash-content"' in html
    # Section toggle JS shipped with the shell
    assert "toggleSec" in html


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
