"""Tests for country_rotation/reporting/production_dashboard.py +
scripts/build_production_dashboard.py.

TDD — written before production_dashboard.py exists.  Synthetic artifacts
mirroring the scripts/production_run.py schemas are written to tmp_path
(no real / gitignored data required).

Tests
-----
1. test_build_dashboard_structure   — 2 panes + tabs JS + header meta.
2. test_latest_signal_default_open  — "Latest Signal" section starts open,
                                      other sections collapsed; openSecs
                                      persistence JS shipped.
3. test_evolution_payload_and_chips — per-strategy JSON payload embedded,
                                      country chips + All/None + SVG-draw JS.
4. test_allocation_and_signal_tables— top-10 signal table + full allocation
                                      table rows (weight/base/tilt).
5. test_metrics_detail_tables       — mandate/period/IC definition tables.
6. test_no_nan_literals             — no NaN/None leaking into visible text.
7. test_script_help                 — CLI --help contract.
8. test_script_end_to_end           — script main() on the tmp run dir
                                      writes the HTML file (wiring smoke).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_STRATEGY_IDS = ("EM_captilt_vsEM", "DM_captilt_vsACWI")


# ---------------------------------------------------------------------------
# Synthetic run-dir fixture (schemas mirror scripts/production_run.py)
# ---------------------------------------------------------------------------

def _countries_for(sid: str) -> list:
    prefix = "EM" if sid.startswith("EM") else "DM"
    return [f"{prefix}_Land{i}" for i in range(6)]


def _write_strategy_artifacts(sdir: Path, sid: str, entry: dict) -> None:
    rng = np.random.default_rng(7 if sid.startswith("EM") else 11)
    countries = _countries_for(sid)
    sdir.mkdir(parents=True)

    # allocations.csv — rebalance dates x countries
    reb_dates = pd.bdate_range("2024-01-02", periods=8, freq="63B")
    raw = rng.random((len(reb_dates), len(countries)))
    alloc = pd.DataFrame(raw, index=reb_dates, columns=countries)
    alloc = alloc.div(alloc.sum(axis=1), axis=0)
    alloc.to_csv(sdir / "allocations.csv", index_label="date",
                 lineterminator="\n")

    # allocations_latest.json
    latest = {c: float(w) for c, w in alloc.iloc[-1].items()}
    (sdir / "allocations_latest.json").write_text(json.dumps({
        "strategy_id": sid,
        "rebalance_date": "2025-11-14",
        "next_rebalance_date": "2026-02-11",
        "weights": latest,
    }), encoding="utf-8")

    # metrics.json
    (sdir / "metrics.json").write_text(json.dumps({
        "strategy": entry,
        "data_end": "2025-11-14",
        "git_commit": "abc123def4567890",
        "benchmark_column": entry.get("bmk_index") or entry["segment"],
        "n_countries": len(countries),
        "universe": countries,
        "factor_set": ["Momentum_12_1", "PE"],
        "category_weights": {"Momentum": 0.5, "Valuation": 0.5},
        "n_rebalances": len(reb_dates),
        "mandate_stats": {
            "ann_return": 0.0368, "ann_vol": 0.1659, "sharpe": 0.222,
            "max_drawdown": -0.4226, "win_rate": 0.5212, "beta": 1.0081,
            "up_capture": 1.0074, "down_capture": 0.9937,
            "tracking_error": 0.0429, "information_ratio": 0.2836,
        },
        "period_stats": {
            "ann_return": 0.0336, "ann_vol": 0.1470, "sharpe": 0.2289,
            "max_drawdown": -0.3026, "win_rate": 0.5077, "beta": 0.9284,
            "up_capture": 0.9748, "down_capture": 0.8947,
            "tracking_error": 0.0362, "information_ratio": 0.2628,
        },
        "turnover_ann": 1.2235,
        "composite_ic": {
            "absolute": {"mean_ic": 0.0260, "median_ic": 0.0036,
                         "std_ic": 0.3142, "t_stat": 0.6668,
                         "icir": 0.0827, "hit_rate": 0.5077},
            "relative": {"mean_ic": 0.0342, "median_ic": -0.0089,
                         "std_ic": 0.3120, "t_stat": 0.8757,
                         "icir": 0.1095, "hit_rate": 0.4688},
        },
        "last_252d_active": {"window_days": 252,
                             "active_return_cum": -0.0250,
                             "ir_ann": -0.3542},
        "equity_curve_last": 1.7648,
        "validation": None,
    }), encoding="utf-8")

    # signal_latest.json
    scores = np.linspace(1.0, 0.0, len(countries))
    rng.shuffle(scores)
    payload = {}
    for c, s in zip(countries, scores):
        payload[c] = {
            "score": float(s),
            "score_change": float(s - 0.5),
            "weight": float(alloc.iloc[-1][c]),
            "base_weight": 1.0 / len(countries),
            "contributions": {"Momentum": float(s * 0.6),
                              "Valuation": float(s * 0.4)},
        }
    (sdir / "signal_latest.json").write_text(json.dumps({
        "strategy_id": sid, "date": "2025-11-14", "periodicity": 63,
        "countries": payload,
    }), encoding="utf-8")

    # signal_history_monthly.csv
    months = pd.bdate_range("2024-01-31", periods=10, freq="BME")
    hist = pd.DataFrame(rng.random((len(months), len(countries))),
                        index=months, columns=countries)
    hist.to_csv(sdir / "signal_history_monthly.csv", index_label="date",
                lineterminator="\n")

    # contributions_latest.csv
    contrib = pd.DataFrame(
        {c: [payload[c]["contributions"]["Momentum"],
             payload[c]["contributions"]["Valuation"]] for c in countries},
        index=["Momentum", "Valuation"],
    )
    contrib.to_csv(sdir / "contributions_latest.csv",
                   index_label="category", lineterminator="\n")

    # ic_series.csv — per-period Spearman IC series (both methods)
    ic_dates = pd.bdate_range("2024-01-02", periods=20, freq="63B")
    ic_df = pd.DataFrame({
        "ic_absolute": rng.normal(0.03, 0.25, len(ic_dates)),
        "ic_relative": rng.normal(0.05, 0.22, len(ic_dates)),
    }, index=ic_dates)
    ic_df.to_csv(sdir / "ic_series.csv", index_label="date",
                 lineterminator="\n")

    # tca.json — TCA summary (production_run schema)
    (sdir / "tca.json").write_text(json.dumps({
        "strategy_id": sid,
        "segment": entry["segment"],
        "cost_model": {
            "as_of": "2026-06-10",
            "commission_bps": 1.0,
            "expense_ratio_bps": 65.0 if entry["segment"] == "EM" else 50.0,
            "mgmt_fee_scenarios_bps": [0.0, 50.0],
            "universe_one_way_bps": {c: 13.0 for c in countries},
        },
        "turnover": {
            "ann_one_way": 1.1842, "avg_per_rebalance": 0.2951,
            "max_per_rebalance": 1.0, "total_name_changes": 24,
        },
        "country_avg_traded": {
            c: round(0.08 - 0.01 * i, 4) for i, c in enumerate(countries)
        },
        "cost_layers_ann_bps": {
            "spread": 31.1, "commission": 2.6, "expense": 65.0,
            "mgmt": {"0bps": 0.0, "50bps": 50.2},
        },
        "layer_irs": {
            "gross": 0.2917, "net_spread": 0.2214,
            "net_spread_expense": 0.0712, "net_mgmt_0bps": 0.0712,
            "net_mgmt_50bps": -0.0481,
        },
        "net_of_fee": {
            "gross_ir": 0.2917, "net_spread_ir": 0.2214,
            "net_spread_expense_ir": 0.0712, "net_mgmt_50_ir": -0.0481,
        },
        "breakeven_bps": 57.3,
    }), encoding="utf-8")

    # turnover.csv — per-rebalance turnover + TCA cost columns
    turnover_df = pd.DataFrame({
        "turnover": np.concatenate([[1.0],
                                    rng.uniform(0.1, 0.5, len(reb_dates) - 1)]),
        "spread_cost": rng.uniform(0.0001, 0.0006, len(reb_dates)),
        "commission_cost": rng.uniform(0.00001, 0.00005, len(reb_dates)),
        "expense_drag": np.full(len(reb_dates), 0.0016),
        "mgmt_drag_0bps": np.zeros(len(reb_dates)),
        "mgmt_drag_50bps": np.full(len(reb_dates), 0.0012),
        "active_return": rng.normal(0.002, 0.01, len(reb_dates)),
    }, index=reb_dates)
    turnover_df.to_csv(sdir / "turnover.csv", index_label="date",
                       lineterminator="\n")


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    registry = {
        "as_of_note": "test registry",
        "strategies": [
            {
                "id": "EM_captilt_vsEM",
                "label": "EM Cap-Tilt vs MSCI EM",
                "segment": "EM", "prior_set": "vm", "periodicity": 63,
                "construction": "cap_tilt", "active_share": 0.30,
                "bmk_index": None,
                "note": "Benchmark = vendor EM index. MC p 0.030.",
            },
            {
                "id": "DM_captilt_vsACWI",
                "label": "DM Cap-Tilt vs ACWI",
                "segment": "DM", "prior_set": "vm", "periodicity": 63,
                "construction": "cap_tilt", "active_share": 0.30,
                "bmk_index": "World",
                "note": "Benchmark = vendor World index.",
            },
        ],
    }
    rdir = tmp_path / "run_20251114"
    rdir.mkdir()
    (tmp_path / "production.json").write_text(json.dumps(registry),
                                              encoding="utf-8")
    (rdir / "manifest.json").write_text(json.dumps({
        "run_timestamp": "2026-06-10T00:00:00+00:00",
        "data_end": "2025-11-14",
        "git_commit": "abc123def4567890",
        "strategies": {},
    }), encoding="utf-8")
    for entry in registry["strategies"]:
        _write_strategy_artifacts(rdir / entry["id"], entry["id"], entry)
    return rdir


def _build(run_dir: Path) -> str:
    from country_rotation.reporting.production_dashboard import (
        build_dashboard,
    )

    return build_dashboard(run_dir, run_dir.parent / "production.json")


# ---------------------------------------------------------------------------
# 1. Document structure: header meta, 2 panes, tab JS
# ---------------------------------------------------------------------------

def test_build_dashboard_structure(run_dir: Path):
    html = _build(run_dir)

    assert html.count("<html") == 1 and html.count("</html>") == 1
    assert "Country Rotation — Production Dashboard" in html

    # Header meta: data end, next rebalance, git hash (manifest line)
    assert "2025-11-14" in html       # data end
    assert "2026-02-11" in html       # next rebalance
    assert "abc123def4" in html       # git hash (truncated ok)

    # Two strategy tab buttons + two panes
    for sid in _STRATEGY_IDS:
        assert f'data-tab="{sid}"' in html
        assert f'id="pane-{sid}"' in html
    assert "EM Cap-Tilt vs MSCI EM" in html
    assert "DM Cap-Tilt vs ACWI" in html

    # Tab JS (vanilla, no external scripts)
    assert "selectTab" in html and "currentTab" in html
    assert 'src="http' not in html

    # First pane visible, second hidden
    assert 'id="pane-EM_captilt_vsEM" class="pane" style="display:block"' in html
    assert 'id="pane-DM_captilt_vsACWI" class="pane" style="display:none"' in html

    # Benchmark identity badges
    assert "MSCI Emerging Markets equivalent" in html
    assert "MSCI ACWI equivalent" in html
    # Registry notes pulled through
    assert "MC p 0.030" in html

    # Viewport-fit shell reused
    assert "100vh" in html and "overflow:hidden" in html
    assert "overflow-y:auto" in html

    # >= 7 base64 PNGs per pane (ranking + allocation history + 2 IC
    # distribution + 3 TCA figures)
    assert html.count("data:image/png;base64,") >= 14


# ---------------------------------------------------------------------------
# 2. Latest Signal default-open + openSecs persistence
# ---------------------------------------------------------------------------

def test_latest_signal_default_open(run_dir: Path):
    html = _build(run_dir)

    # 6 collapsible sections per pane x 2 panes (IC Analysis + TCA added)
    assert html.count('<button class="sec-toggle"') == 12
    # Latest Signal open in both panes; the other 5 sections collapsed
    assert html.count('aria-expanded="true"') == 2
    assert html.count('aria-expanded="false"') == 10
    assert html.count('class="sec-body" style="display:block"') == 2
    assert html.count('class="sec-body" style="display:none"') == 10
    for title in ("Latest Signal", "Signal Evolution", "Allocations",
                  "Metrics Detail", "IC Analysis",
                  "Transaction Costs &amp; Turnover"):
        assert html.count(f'<span class="sec-title">{title}</span>') == 2

    # openSecs persistence pre-seeded with the open section
    assert '"latest signal": true' in html
    assert "function applySecState(btn, open)" in html
    assert "openSecs[title] = !openSecs[title];" in html


# ---------------------------------------------------------------------------
# 3. Evolution payload + chips + SVG-draw JS
# ---------------------------------------------------------------------------

def test_evolution_payload_and_chips(run_dir: Path):
    html = _build(run_dir)

    for sid in _STRATEGY_IDS:
        # Embedded JSON payload per strategy
        assert f'id="evo-data-{sid}"' in html
        assert f'id="evo-svg-{sid}"' in html
        assert f'id="evo-chips-{sid}"' in html
        # Payload carries the strategy's own universe
        block = html.split(f'id="evo-data-{sid}"')[1]
        payload_json = block.split("</script>")[0].split(">", 1)[1]
        payload = json.loads(payload_json)
        assert set(payload["countries"]) == set(_countries_for(sid))
        assert len(payload["dates"]) == 10

    # Country chips + All/None controls
    assert html.count('class="evo-chip') >= 12   # 6 countries x 2 panes
    assert "evoSetAll" in html and ">All<" in html and ">None<" in html
    # First 5 pre-selected per pane -> 5 active chips x 2
    assert html.count("evo-chip on") == 10

    # Hand-rolled SVG renderer (no external libs)
    assert "function drawEvo(" in html
    assert "polyline" in html
    assert "evoToggle" in html
    assert "EVO_COLORS" in html


# ---------------------------------------------------------------------------
# 3b. IC Analysis section
# ---------------------------------------------------------------------------

def test_ic_analysis_section_present_with_artifacts(run_dir: Path):
    """IC Analysis section renders with 2 distribution PNGs per pane when
    ic_series.csv is present."""
    html = _build(run_dir)

    # Section heading present twice (once per pane)
    assert html.count('<span class="sec-title">IC Analysis</span>') == 2
    # At minimum 2 additional base64 PNGs beyond the 4 existing (ranking +
    # allocation) — one per IC method per pane -> at least 8 total
    assert html.count("data:image/png;base64,") >= 8


def test_ic_analysis_section_absent_without_artifacts(run_dir: Path, tmp_path):
    """IC Analysis section is omitted (no crash) when ic_series.csv is absent."""
    # Remove ic_series.csv from both strategy dirs
    for sid in _STRATEGY_IDS:
        ic_path = run_dir / sid / "ic_series.csv"
        if ic_path.exists():
            ic_path.unlink()

    html = _build(run_dir)
    assert "IC Analysis" not in html
    assert "Country Rotation — Production Dashboard" in html  # no crash


# ---------------------------------------------------------------------------
# 3c. Transaction Costs & Turnover section
# ---------------------------------------------------------------------------

def test_tca_section_present_with_artifacts(run_dir: Path):
    """TCA section renders 3 figures + layer table + top-traded table +
    breakeven chip per pane when tca.json/turnover.csv are present."""
    html = _build(run_dir)

    assert html.count(
        '<span class="sec-title">Transaction Costs &amp; Turnover</span>'
    ) == 2
    # Breakeven callout chip with the fixture value
    assert html.count('class="breakeven-chip"') == 2
    assert "57 bps one-way" in html
    # Layer table rows + headers
    for label in ("Gross active", "Spread + commission", "ETF expense",
                  "Mgmt fee 50"):
        assert label in html, f"missing layer row '{label}'"
    assert "Ann. cost (bps)" in html and "Cumulative IR" in html
    # Top-8 turnover contributors (6 fixture countries -> top 6)
    assert "Turnover Contributors" in html
    # 3 TCA figures per pane on top of the 4 existing -> >= 7 each
    assert html.count("data:image/png;base64,") >= 14


def test_tca_section_absent_without_artifacts(run_dir: Path):
    """TCA section is omitted (no crash) when tca.json/turnover.csv absent."""
    for sid in _STRATEGY_IDS:
        for name in ("tca.json", "turnover.csv"):
            path = run_dir / sid / name
            if path.exists():
                path.unlink()

    html = _build(run_dir)
    assert "Transaction Costs" not in html
    assert "Country Rotation — Production Dashboard" in html  # no crash


# ---------------------------------------------------------------------------
# 4. Signal + allocation tables
# ---------------------------------------------------------------------------

def test_allocation_and_signal_tables(run_dir: Path):
    html = _build(run_dir)

    # Top-10 signal table columns
    for col in ("Country", "Score", "Δ63d", "Weight", "Base", "Tilt"):
        assert col in html

    # All countries appear in the allocation table of their pane
    em_pane = html.split('id="pane-EM_captilt_vsEM"')[1].split(
        'id="pane-DM_captilt_vsACWI"')[0]
    for c in _countries_for("EM_captilt_vsEM"):
        assert c in em_pane
    # Stat cards present
    for label in ("IR (active, ann.)", "Tracking Error", "Beta",
                  "Ann Return", "Turnover", "IC rel (mean)", "IC rel (t)",
                  "Last 252d Active", "Latest Rebalance", "Next Rebalance",
                  "Top-5 Concentration"):
        assert label in html, f"missing stat card '{label}'"


# ---------------------------------------------------------------------------
# 5. Metrics detail definition tables
# ---------------------------------------------------------------------------

def test_metrics_detail_tables(run_dir: Path):
    html = _build(run_dir)

    for heading in ("Mandate (daily, ann.)", "Period (per rebalance)",
                    "Composite IC — absolute", "Composite IC — relative"):
        assert heading in html, f"missing metrics-detail block '{heading}'"
    # A few formatted values flow through
    assert "4.29%" in html        # mandate tracking_error
    assert "0.0342" in html       # relative mean IC (4dp)


# ---------------------------------------------------------------------------
# 6. No NaN/None literals in visible text
# ---------------------------------------------------------------------------

def test_no_nan_literals(run_dir: Path):
    html = _build(run_dir)
    visible = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    visible = re.sub(r"<style.*?</style>", "", visible, flags=re.S)
    visible = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", visible)
    # The evolution chart's All/None selector buttons are legitimate UI text.
    visible = re.sub(r"<button[^>]*>(All|None)</button>", "", visible)
    text = re.sub(r"<[^>]+>", " ", visible)
    assert not re.search(r"\bNaN\b", text)
    assert not re.search(r"\bNone\b", text)
    assert not re.search(r"\bnan\b", text)


# ---------------------------------------------------------------------------
# 7+8. CLI contract + end-to-end wiring smoke
# ---------------------------------------------------------------------------

def _script_env() -> dict:
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def test_script_help():
    script = REPO_ROOT / "scripts" / "build_production_dashboard.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, env=_script_env(), timeout=60,
    )
    assert proc.returncode == 0, f"--help failed:\n{proc.stderr}"
    assert "--run-dir" in proc.stdout
    assert "--registry" in proc.stdout
    assert "--output" in proc.stdout


def test_script_end_to_end(run_dir: Path, tmp_path: Path):
    """Full main() path on synthetic artifacts (lesson: --help alone does
    not catch wiring bugs)."""
    script = REPO_ROOT / "scripts" / "build_production_dashboard.py"
    out = tmp_path / "dash.html"
    proc = subprocess.run(
        [sys.executable, str(script),
         "--run-dir", str(run_dir),
         "--registry", str(run_dir.parent / "production.json"),
         "--output", str(out)],
        capture_output=True, text=True, env=_script_env(), timeout=300,
    )
    assert proc.returncode == 0, f"script failed:\n{proc.stderr}"
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Country Rotation — Production Dashboard" in html
    for sid in _STRATEGY_IDS:
        assert f'id="pane-{sid}"' in html
