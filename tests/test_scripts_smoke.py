"""End-to-end smoke tests for the CLI scripts (scripts/build_scores.py and
scripts/run_backtest.py) on tiny synthetic xlsx fixtures.

These are the gate that catches CLI wiring regressions (wrong kwargs, missing
positional args) that import-only / --help tests cannot see.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

REPO_ROOT = Path(__file__).resolve().parents[1]

# Scripts print non-ASCII characters; on Windows a piped stdout defaults to
# cp1252 which cannot encode them — force UTF-8 in the child process.
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

_N_DAYS = 400
_COUNTRIES = [f"C{i}" for i in range(10)]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=_ENV,
    )


def _synthetic_prices(index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drifts = np.linspace(-0.0003, 0.0009, len(_COUNTRIES))
    cols = {
        name: 100.0 * np.cumprod(1.0 + rng.normal(drifts[k], 0.012, len(index)))
        for k, name in enumerate(_COUNTRIES)
    }
    return pd.DataFrame(cols, index=index)


def _write_inputs_xlsx(df: pd.DataFrame, path: Path) -> None:
    """Write *df* in the layout ingestion.read_inputs expects (skiprows=2).

    Rows 1-2 are dummy header lines, row 3 is the real header (Date + country
    names), data rows follow with datetime values in the first column.
    """
    wb = Workbook()
    ws = wb.active
    ws.append(["Source: synthetic test fixture"])  # row 1 — skipped
    ws.append(["Units: synthetic"])                # row 2 — skipped
    ws.append(["Date", *df.columns.tolist()])      # row 3 — header
    for ts, row in df.iterrows():
        ws.append([ts.to_pydatetime(), *row.tolist()])
    wb.save(path)


# ---------------------------------------------------------------------------
# run_backtest.py — full path: scores + prices xlsx → results xlsx
# ---------------------------------------------------------------------------

def test_run_backtest_smoke(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=_N_DAYS)

    prices = _synthetic_prices(idx, seed=0)
    prices["World"] = prices[_COUNTRIES].mean(axis=1)

    rng = np.random.default_rng(1)
    scores = pd.DataFrame(
        rng.random((_N_DAYS, len(_COUNTRIES))), index=idx, columns=_COUNTRIES
    )

    scores_path = tmp_path / "scores.xlsx"
    prices_path = tmp_path / "prices.xlsx"
    scores.to_excel(scores_path)   # script reads with index_col=0, parse_dates=True
    prices.to_excel(prices_path)

    out = _run(
        [
            "scripts/run_backtest.py",
            "--config", str(REPO_ROOT / "configs" / "default.json"),
            "--scores", str(scores_path),
            "--prices", str(prices_path),
            "--output-dir", str(tmp_path),
        ]
    )

    assert out.returncode == 0, f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
    result_path = tmp_path / "backtest_results.xlsx"
    assert result_path.exists()

    # Output must contain non-empty period results
    period_results = pd.read_excel(
        result_path, sheet_name="period_results", index_col=0
    )
    assert len(period_results) > 0
    assert "portfolio_return_gross" in period_results.columns


# ---------------------------------------------------------------------------
# build_scores.py — full path: Inputs/ + Classification.xlsx → scores xlsx
# ---------------------------------------------------------------------------

def _write_classification(path: Path) -> None:
    """Classification fixture incl. Region pseudo-rows (mirrors real sheet)."""
    classification = pd.DataFrame(
        {
            "Segment": ["DM"] * 5 + ["EM"] * 5 + ["Region"] * 2,
            "Region": (["Europe", "Europe", "Asia", "Asia", "LatAm"] * 2)
            + ["Region"] * 2,
            "Type": ["Country"] * 10 + ["Region"] * 2,
        },
        index=pd.Index([*_COUNTRIES, "World", "EM"], name="Country"),
    )
    classification.to_excel(path, sheet_name="regiones")


def test_research_run_smoke(tmp_path):
    """Full research pipeline on engineered fixtures: a persistent value
    signal (low PE <-> high future drift) must survive screening so the
    composite -> validation -> report -> verdict path is exercised."""
    n_days = 600
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rng = np.random.default_rng(7)

    inputs_dir = tmp_path / "Inputs"
    inputs_dir.mkdir()

    # Prices: persistent, well-separated drifts; low noise.
    drifts = np.linspace(-0.003, 0.003, len(_COUNTRIES))
    prices = pd.DataFrame(
        {
            name: 100.0 * np.cumprod(1.0 + rng.normal(drifts[k], 0.006, n_days))
            for k, name in enumerate(_COUNTRIES)
        },
        index=idx,
    )
    _write_inputs_xlsx(prices, inputs_dir / "Price.xlsx")

    # PE inversely tied to drift: best-drift country has the lowest PE,
    # so the (direction = -1) PE factor is strongly predictive.
    pe_levels = 30.0 - 2.0 * np.argsort(np.argsort(drifts))
    pe = pd.DataFrame(
        pe_levels + rng.normal(0.0, 0.3, (n_days, len(_COUNTRIES))),
        index=idx, columns=_COUNTRIES,
    )
    _write_inputs_xlsx(pe, inputs_dir / "PE.xlsx")

    # Ten_Year: flat 2% everywhere (adds the spread factor).
    ten_year = pd.DataFrame(
        2.0 + rng.normal(0.0, 0.05, (n_days, len(_COUNTRIES))),
        index=idx, columns=_COUNTRIES,
    )
    _write_inputs_xlsx(ten_year, inputs_dir / "Ten_Year.xlsx")

    cls_path = tmp_path / "Classification.xlsx"
    _write_classification(cls_path)

    cfg = json.loads(
        (REPO_ROOT / "configs" / "default.json").read_text(encoding="utf-8")
    )
    cfg["data"]["inputs_folder"] = str(inputs_dir)
    cfg["data"]["classification_file"] = str(cls_path)
    cfg["data"]["target_date"] = "2020-01-01"
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    out_dir = tmp_path / "research"
    out = _run(
        [
            "scripts/research_run.py",
            "--segment", "World",
            "--config", str(cfg_path),
            "--output-dir", str(out_dir),
            "--periodicity", "21",
            "--quick",
        ]
    )

    assert out.returncode == 0, f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"

    # Screening evidence + verdict are always written
    assert (out_dir / "screening_World.xlsx").exists()
    verdict = json.loads(
        (out_dir / "verdict_World.json").read_text(encoding="utf-8")
    )
    assert verdict["segment"] == "World"
    assert verdict["n_countries"] == 10  # Region pseudo-rows excluded
    assert isinstance(verdict["verdict"]["overall"], bool)

    # The engineered value signal must survive -> full pipeline ran
    kept = verdict["screening"]["kept"]
    assert "PE" in kept, f"engineered PE signal not kept; kept={kept}"
    assert verdict["stats"] is not None
    assert verdict["stats"]["sharpe_ann"] is not None
    assert (out_dir / "report_World.html").exists()


def test_build_scores_smoke(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=_N_DAYS)
    rng = np.random.default_rng(2)

    inputs_dir = tmp_path / "Inputs"
    inputs_dir.mkdir()

    # Price: positive random walks
    _write_inputs_xlsx(_synthetic_prices(idx, seed=3), inputs_dir / "Price.xlsx")

    # PE: positive multiples ~ U(8, 25)
    pe = pd.DataFrame(
        rng.uniform(8.0, 25.0, (_N_DAYS, len(_COUNTRIES))),
        index=idx, columns=_COUNTRIES,
    )
    _write_inputs_xlsx(pe, inputs_dir / "PE.xlsx")

    # Ten_Year: bond yields in percent ~ U(1, 9)
    ten_year = pd.DataFrame(
        rng.uniform(1.0, 9.0, (_N_DAYS, len(_COUNTRIES))),
        index=idx, columns=_COUNTRIES,
    )
    _write_inputs_xlsx(ten_year, inputs_dir / "Ten_Year.xlsx")

    # Classification: sheet 'regiones', index=country, Segment/Region/Type
    classification = pd.DataFrame(
        {
            "Segment": ["DM"] * 5 + ["EM"] * 5,
            "Region": ["Europe", "Europe", "Asia", "Asia", "LatAm"] * 2,
            "Type": ["Country"] * 10,
        },
        index=pd.Index(_COUNTRIES, name="Country"),
    )
    cls_path = tmp_path / "Classification.xlsx"
    classification.to_excel(cls_path, sheet_name="regiones")

    # Config: copy default, point data paths at the tmp fixtures
    cfg = json.loads((REPO_ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    cfg["data"]["inputs_folder"] = str(inputs_dir)
    cfg["data"]["classification_file"] = str(cls_path)
    cfg["data"]["target_date"] = "2020-01-01"
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    out = _run(
        [
            "scripts/build_scores.py",
            "--config", str(cfg_path),
            "--market", "World",
            "--output-dir", str(out_dir),
        ]
    )

    assert out.returncode == 0, f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
    result_path = out_dir / "NormalizedScores_World_custom.xlsx"
    assert result_path.exists()

    # Scores must be in [0, 1] and cover the synthetic countries
    normalized = pd.read_excel(result_path, index_col=0, parse_dates=True)
    assert set(_COUNTRIES).issubset(normalized.columns)
    values = normalized[_COUNTRIES].to_numpy()
    finite = values[~np.isnan(values)]
    assert len(finite) > 0
    assert (finite >= 0.0).all() and (finite <= 1.0).all()
