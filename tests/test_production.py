"""Tests for scripts/production_run.py — the deployed-strategy production
pipeline (registry -> per-strategy artifacts -> manifest).

The heavy ingest path is avoided by testing the pure-ish artifact builder
``produce_strategy_artifacts`` on a tiny synthetic ``processed`` dict (and,
for the main() wiring test, by monkeypatching ``research_run.ingest``).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import production_run  # noqa: E402

_N_DAYS = 400
_COUNTRIES = [f"C{i}" for i in range(10)]
_PERIODICITY = 21

_ARTIFACTS = (
    "allocations.csv",
    "allocations_latest.json",
    "metrics.json",
    "signal_latest.json",
    "signal_history.csv",
    "signal_history_monthly.csv",
    "contributions_latest.csv",
    "ic_series.csv",
    "tca.json",
    "turnover.csv",
)


# ---------------------------------------------------------------------------
# Synthetic fixtures (processed dict mirrors run_processing output keys)
# ---------------------------------------------------------------------------

def _synthetic_processed() -> tuple[dict, pd.DataFrame]:
    """Tiny processed dict + classification: 10 EM countries, vendor 'EM'
    and 'World' index columns in Price, vm priors PE + Momentum_12_1,
    Market_Cap for the Cap_Tilt base."""
    idx = pd.bdate_range("2020-01-01", periods=_N_DAYS)
    rng = np.random.default_rng(7)

    drifts = np.linspace(-0.002, 0.002, len(_COUNTRIES))
    prices = pd.DataFrame(
        {
            name: 100.0 * np.cumprod(1.0 + rng.normal(drifts[k], 0.01, _N_DAYS))
            for k, name in enumerate(_COUNTRIES)
        },
        index=idx,
    )
    price_input = prices.copy()
    price_input["EM"] = prices.mean(axis=1)            # vendor EM index
    price_input["World"] = prices.mean(axis=1) * 1.01  # vendor ACWI proxy

    pe = pd.DataFrame(
        rng.uniform(8.0, 25.0, (_N_DAYS, len(_COUNTRIES))),
        index=idx, columns=_COUNTRIES,
    )
    momentum = pd.DataFrame(
        rng.normal(0.0, 1.0, (_N_DAYS, len(_COUNTRIES))).cumsum(axis=0),
        index=idx, columns=_COUNTRIES,
    )
    mcap = pd.DataFrame(
        np.outer(np.ones(_N_DAYS), np.linspace(100.0, 1000.0, len(_COUNTRIES)))
        * (1.0 + rng.normal(0.0, 0.01, (_N_DAYS, len(_COUNTRIES)))),
        index=idx, columns=_COUNTRIES,
    )

    processed = {
        "Price": price_input,
        "PE": pe,
        "Momentum_12_1": momentum,
        "Market_Cap": mcap,
    }
    classification = pd.DataFrame(
        {
            "Segment": ["EM"] * len(_COUNTRIES) + ["Region"] * 2,
            "Region": ["Asia"] * len(_COUNTRIES) + ["Region"] * 2,
            "Type": ["Country"] * len(_COUNTRIES) + ["Region"] * 2,
        },
        index=pd.Index([*_COUNTRIES, "World", "EM"], name="Country"),
    )
    return processed, classification


def _strategy_cfg(strategy_id: str = "TEST_em", bmk_index=None) -> dict:
    return {
        "id": strategy_id,
        "label": "Synthetic test strategy",
        "segment": "EM",
        "prior_set": "vm",
        "periodicity": _PERIODICITY,
        "construction": "cap_tilt",
        "active_share": 0.30,
        "bmk_index": bmk_index,
        "note": "synthetic",
    }


# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------

def test_registry_loads_and_schema():
    registry = production_run.load_registry(
        REPO_ROOT / "configs" / "production.json"
    )
    ids = [s["id"] for s in registry["strategies"]]
    assert ids == ["EM_captilt_vsEM", "DM_captilt_vsACWI"]

    for entry in registry["strategies"]:
        for field in production_run._REQUIRED_FIELDS:
            assert field in entry, f"{entry['id']} missing '{field}'"
        assert entry["construction"] == "cap_tilt"
        assert entry["prior_set"] == "vm"
        assert entry["periodicity"] == 63
        assert entry["active_share"] == pytest.approx(0.30)

    em, dm = registry["strategies"]
    assert em["segment"] == "EM" and em["bmk_index"] is None
    assert dm["segment"] == "DM" and dm["bmk_index"] == "World"


def test_registry_rejects_missing_fields(tmp_path):
    bad = {"strategies": [{"id": "X", "segment": "EM"}]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing required"):
        production_run.load_registry(path)


# ---------------------------------------------------------------------------
# 2. End-to-end artifact production on synthetic data
# ---------------------------------------------------------------------------

def test_produce_strategy_artifacts_end_to_end(tmp_path):
    processed, classification = _synthetic_processed()
    out_dir = tmp_path / "TEST_em"

    info = production_run.produce_strategy_artifacts(
        processed, classification, _strategy_cfg(), str(out_dir), quick=True
    )

    # All artifact files exist and are in the manifest inventory.
    for name in _ARTIFACTS:
        assert (out_dir / name).exists(), f"missing artifact {name}"
        assert name in info["files"]

    # allocations.csv — Cap_Tilt active book: every rebalance row sums to 1.
    hw = pd.read_csv(out_dir / "allocations.csv", index_col=0, parse_dates=True)
    assert len(hw) > 1
    assert np.allclose(hw.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    assert info["files"]["allocations.csv"] == len(hw)

    # allocations_latest.json — latest weights + rebalance grid dates.
    latest = json.loads(
        (out_dir / "allocations_latest.json").read_text(encoding="utf-8")
    )
    assert latest["strategy_id"] == "TEST_em"
    assert latest["rebalance_date"] == str(hw.index[-1].date())
    expected_next = hw.index[-1] + pd.offsets.BDay(_PERIODICITY)
    assert latest["next_rebalance_date"] == str(expected_next.date())
    assert sum(latest["weights"].values()) == pytest.approx(1.0, abs=1e-9)

    # signal_latest.json — countries == universe, full per-country block.
    sig = json.loads((out_dir / "signal_latest.json").read_text(encoding="utf-8"))
    assert set(sig["countries"]) == set(_COUNTRIES)
    assert sig["periodicity"] == _PERIODICITY
    for block in sig["countries"].values():
        assert set(block) == {
            "score", "score_change", "weight", "base_weight", "contributions"
        }
        assert 0.0 <= block["score"] <= 1.0
        assert block["base_weight"] is not None  # cap_tilt -> cap base known
        assert set(block["contributions"]) == {"Momentum", "Valuation"}

    # signal_history.csv — full daily history, dates == score dates.
    hist = pd.read_csv(
        out_dir / "signal_history.csv", index_col=0, parse_dates=True
    )
    assert len(hist) == _N_DAYS
    assert set(hist.columns) == set(_COUNTRIES)
    assert info["files"]["signal_history.csv"] == len(hist)

    # signal_history_monthly.csv — one row per calendar month.
    monthly = pd.read_csv(
        out_dir / "signal_history_monthly.csv", index_col=0, parse_dates=True
    )
    expected_months = hist.index.to_period("M").nunique()
    assert len(monthly) == expected_months
    assert info["files"]["signal_history_monthly.csv"] == len(monthly)

    # contributions_latest.csv — category x country at the last date.
    contrib = pd.read_csv(out_dir / "contributions_latest.csv", index_col=0)
    assert set(contrib.index) == {"Momentum", "Valuation"}
    assert set(contrib.columns) == set(_COUNTRIES)

    # metrics.json — registry echo + provenance, NO timestamp anywhere.
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["strategy"]["id"] == "TEST_em"
    assert metrics["data_end"] == str(hist.index[-1].date())
    assert metrics["git_commit"]
    assert metrics["benchmark_column"] == "EM"
    assert metrics["n_countries"] == len(_COUNTRIES)
    assert metrics["factor_set"] == ["Momentum_12_1", "PE"]
    assert metrics["n_rebalances"] == len(hw)
    assert metrics["validation"] is None  # quick path skips the scorecard
    assert metrics["mandate_stats"]["tracking_error"] is not None
    assert metrics["turnover_ann"] is not None
    assert set(metrics["composite_ic"]) == {"absolute", "relative"}
    assert "timestamp" not in json.dumps(metrics).lower()

    # ic_series.csv — per-period Spearman IC series, both methods.
    ic_series = pd.read_csv(
        out_dir / "ic_series.csv", index_col=0, parse_dates=True
    )
    assert "ic_absolute" in ic_series.columns
    assert "ic_relative" in ic_series.columns
    assert len(ic_series) >= 1
    assert info["files"]["ic_series.csv"] == len(ic_series)

    # metrics.json net_of_fee — layer IRs ordered gross >= net_spread >=
    # +expense >= +mgmt50 (each layer adds a cumulative drag).
    nof = metrics["net_of_fee"]
    assert set(nof) == {
        "gross_ir", "net_spread_ir", "net_spread_expense_ir", "net_mgmt_50_ir"
    }
    assert nof["gross_ir"] >= nof["net_spread_ir"]
    assert nof["net_spread_ir"] >= nof["net_spread_expense_ir"]
    assert nof["net_spread_expense_ir"] >= nof["net_mgmt_50_ir"]

    # tca.json — turnover summary, layers, breakeven, cost-model echo.
    tca_payload = json.loads((out_dir / "tca.json").read_text(encoding="utf-8"))
    assert tca_payload["strategy_id"] == "TEST_em"
    assert tca_payload["segment"] == "EM"
    assert set(tca_payload["cost_model"]["universe_one_way_bps"]) == set(
        _COUNTRIES
    )
    assert tca_payload["turnover"]["ann_one_way"] > 0
    assert tca_payload["turnover"]["max_per_rebalance"] >= (
        tca_payload["turnover"]["avg_per_rebalance"]
    )
    assert set(tca_payload["country_avg_traded"]) == set(_COUNTRIES)
    layers_bps = tca_payload["cost_layers_ann_bps"]
    assert layers_bps["spread"] > 0
    assert layers_bps["commission"] > 0
    assert layers_bps["expense"] == pytest.approx(65.0, abs=3.0)  # EM ratio
    assert set(tca_payload["layer_irs"]) >= {
        "gross", "net_spread", "net_spread_expense", "net_mgmt_50bps"
    }
    assert tca_payload["net_of_fee"] == nof
    assert tca_payload["breakeven_bps"] is None or isinstance(
        tca_payload["breakeven_bps"], float
    )

    # turnover.csv — per-rebalance turnover + TCA cost columns, one row
    # per rebalance, turnover column matches the engine convention.
    turnover_df = pd.read_csv(
        out_dir / "turnover.csv", index_col=0, parse_dates=True
    )
    assert len(turnover_df) == len(hw)
    for col in ("turnover", "spread_cost", "commission_cost",
                "expense_drag", "active_return"):
        assert col in turnover_df.columns, f"turnover.csv missing '{col}'"
    assert turnover_df["turnover"].iloc[0] == pytest.approx(1.0)  # deployment
    assert (turnover_df["spread_cost"] >= 0).all()
    assert info["files"]["turnover.csv"] == len(turnover_df)


def test_bmk_index_override_uses_vendor_world_column(tmp_path):
    """bmk_index='World' pins the cross-segment vendor column (ACWI proxy)
    while the book stays the segment universe."""
    processed, classification = _synthetic_processed()
    out_dir = tmp_path / "TEST_em_vsWorld"

    production_run.produce_strategy_artifacts(
        processed, classification,
        _strategy_cfg("TEST_em_vsWorld", bmk_index="World"),
        str(out_dir), quick=True,
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["benchmark_column"] == "World"
    sig = json.loads((out_dir / "signal_latest.json").read_text(encoding="utf-8"))
    assert "World" not in sig["countries"]  # benchmark never enters the book
    hw = pd.read_csv(out_dir / "allocations.csv", index_col=0)
    assert "World" not in hw.columns
    assert np.allclose(hw.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_missing_vendor_benchmark_column_is_fatal(tmp_path):
    processed, classification = _synthetic_processed()
    processed["Price"] = processed["Price"].drop(columns=["EM"])
    with pytest.raises(SystemExit, match="vendor index column 'EM'"):
        production_run.produce_strategy_artifacts(
            processed, classification, _strategy_cfg(), str(tmp_path), quick=True
        )


# ---------------------------------------------------------------------------
# 3. Determinism: same inputs -> identical artifact bytes
# ---------------------------------------------------------------------------

def test_determinism_identical_artifact_bytes(tmp_path):
    processed, classification = _synthetic_processed()
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    production_run.produce_strategy_artifacts(
        processed, classification, _strategy_cfg(), str(dir_a), quick=True
    )
    production_run.produce_strategy_artifacts(
        processed, classification, _strategy_cfg(), str(dir_b), quick=True
    )

    for name in _ARTIFACTS:
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes(), (
            f"artifact {name} is not byte-identical across runs"
        )


# ---------------------------------------------------------------------------
# 4. main() wiring (monkeypatched ingest) + CLI --help smoke
# ---------------------------------------------------------------------------

def test_main_end_to_end_with_monkeypatched_ingest(tmp_path, monkeypatch):
    import research_run

    processed, classification = _synthetic_processed()
    monkeypatch.setattr(
        research_run, "ingest", lambda cfg: (processed, classification)
    )

    registry = {
        "as_of_note": "synthetic",
        "strategies": [
            _strategy_cfg("TEST_em"),
            _strategy_cfg("TEST_em_vsWorld", bmk_index="World"),
        ],
    }
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(registry), encoding="utf-8")
    out_root = tmp_path / "prod"

    monkeypatch.setattr(sys, "argv", [
        "production_run.py",
        "--config", str(REPO_ROOT / "configs" / "default.json"),
        "--registry", str(reg_path),
        "--output-dir", str(out_root),
        "--quick",
    ])
    production_run.main()

    data_end = processed["Price"].index.max()
    run_dir = out_root / f"run_{data_end:%Y%m%d}"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["data_end"] == str(data_end.date())
    assert manifest["quick"] is True
    assert manifest["run_timestamp"]
    assert manifest["git_commit"]
    assert manifest["package_version"]
    assert set(manifest["strategies"]) == {"TEST_em", "TEST_em_vsWorld"}

    # Manifest row counts match the files on disk.
    for sid, info in manifest["strategies"].items():
        assert set(info["files"]) == set(_ARTIFACTS)
        assert len(info["top5_weights"]) == 5
        for name, rows in info["files"].items():
            path = run_dir / sid / name
            assert path.exists()
            if name.endswith(".csv"):
                df = pd.read_csv(path, index_col=0)
                assert rows == len(df), f"{sid}/{name} row count mismatch"


def test_main_strategy_filter(tmp_path, monkeypatch):
    import research_run

    processed, classification = _synthetic_processed()
    monkeypatch.setattr(
        research_run, "ingest", lambda cfg: (processed, classification)
    )
    registry = {
        "as_of_note": "synthetic",
        "strategies": [
            _strategy_cfg("TEST_em"),
            _strategy_cfg("TEST_other", bmk_index="World"),
        ],
    }
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(registry), encoding="utf-8")
    out_root = tmp_path / "prod"

    monkeypatch.setattr(sys, "argv", [
        "production_run.py",
        "--config", str(REPO_ROOT / "configs" / "default.json"),
        "--registry", str(reg_path),
        "--output-dir", str(out_root),
        "--strategy", "TEST_em",
        "--quick",
    ])
    production_run.main()

    data_end = processed["Price"].index.max()
    run_dir = out_root / f"run_{data_end:%Y%m%d}"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["strategies"]) == {"TEST_em"}
    assert not (run_dir / "TEST_other").exists()


def test_cli_help_smoke():
    out = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "production_run.py"), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert out.returncode == 0
    for flag in ("--config", "--registry", "--output-dir", "--strategy", "--quick"):
        assert flag in out.stdout
