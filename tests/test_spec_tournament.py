"""Tests for scripts/spec_tournament.py — the pre-registered 6-spec
signal tournament harness.

1. test_spec_registry_pre_registered — the spec family is exactly the six
   pre-registered specs with the documented selection / IC-method pairing.
2. test_evaluate_segment_engineered_winner — on synthetic data where only
   the 'mom' composite is predictive (constant cross-sectional ranking
   aligned with persistent price drifts), S4_mom_level must win with a
   significant q and a positive lockbox IC; every spec carries the full
   {mean_ic, t, n, p, q} record.
3. test_spec_tournament_smoke — end-to-end subprocess run of main() on tiny
   synthetic xlsx fixtures (the CLI wiring gate; see lessons.md).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

_COUNTRIES = [f"C{i}" for i in range(10)]


def _import_spec_tournament():
    path = REPO_ROOT / "scripts" / "spec_tournament.py"
    spec = importlib.util.spec_from_file_location("spec_tournament", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["spec_tournament"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Pre-registered family contract
# ---------------------------------------------------------------------------

def test_spec_registry_pre_registered():
    mod = _import_spec_tournament()

    names = [s.name for s in mod.SPECS]
    assert names == [
        "S1_blend_change",
        "S2_amp_ey_level",
        "S3_amp_bp_level",
        "S4_mom_level",
        "S5_amp_ey_change",
        "S6_blend_level",
    ]

    by_name = {s.name: s for s in mod.SPECS}
    # Selection rule per spec (level = canonical AMP rank-of-level).
    assert by_name["S1_blend_change"].selection == "relative"
    assert by_name["S2_amp_ey_level"].selection == "top_n_level"
    assert by_name["S3_amp_bp_level"].selection == "top_n_level"
    assert by_name["S4_mom_level"].selection == "top_n_level"
    assert by_name["S5_amp_ey_change"].selection == "relative"
    assert by_name["S6_blend_level"].selection == "top_n_level"

    # IC method must measure what the spec trades.
    for s in mod.SPECS:
        expected = "absolute" if s.selection == "top_n_level" else "relative"
        assert s.ic_method == expected, s.name

    # Composite sharing: S2/S5 share amp_ey; S1/S6 share the blend.
    assert by_name["S2_amp_ey_level"].composite == by_name["S5_amp_ey_change"].composite
    assert by_name["S1_blend_change"].composite == by_name["S6_blend_level"].composite


# ---------------------------------------------------------------------------
# 2. evaluate_segment: engineered winner
# ---------------------------------------------------------------------------

def test_evaluate_segment_engineered_winner():
    mod = _import_spec_tournament()
    rng = np.random.default_rng(11)

    n = 800
    idx = pd.bdate_range("2015-01-02", periods=n)
    drifts = np.linspace(-0.003, 0.003, len(_COUNTRIES))
    prices = pd.DataFrame(
        {
            c: 100.0 * np.cumprod(1.0 + rng.normal(drifts[k], 0.006, n))
            for k, c in enumerate(_COUNTRIES)
        },
        index=idx,
    )

    # 'mom' composite: constant cross-sectional ranking aligned with drift.
    rank = np.argsort(np.argsort(drifts)) / (len(_COUNTRIES) - 1)
    mom = pd.DataFrame(
        np.tile(rank, (n, 1)), index=idx, columns=_COUNTRIES
    )
    # Other composites: pure noise.
    def _noise(seed):
        r = np.random.default_rng(seed)
        return pd.DataFrame(
            r.random((n, len(_COUNTRIES))), index=idx, columns=_COUNTRIES
        )

    composite_scores = {
        "blend": _noise(1),
        "amp_ey": _noise(2),
        "amp_bp": _noise(3),
        "mom": mom,
    }

    seg = mod.evaluate_segment(composite_scores, prices, periodicity=21, fdr_q=0.10)

    # Every spec has the full record.
    assert set(seg["results"]) == {s.name for s in mod.SPECS}
    for record in seg["results"].values():
        assert set(record) == {"mean_ic", "t", "n", "p", "q"}

    # The engineered momentum-level spec wins decisively.
    assert seg["winner"] == "S4_mom_level"
    win = seg["results"]["S4_mom_level"]
    assert win["mean_ic"] > 0.3
    assert win["q"] <= 0.10
    assert win["n"] >= 10

    # One-shot lockbox confirmation present and positive.
    assert set(seg["lockbox"]) == {"mean_ic", "t", "n"}
    assert seg["lockbox"]["mean_ic"] > 0.0
    assert seg["winner_selection"] == "top_n_level"

    # Windows are disjoint and ordered: screen strictly before lockbox.
    assert seg["screen_window"][1] < seg["lockbox_window"][0]


# ---------------------------------------------------------------------------
# 3. End-to-end CLI smoke (synthetic Inputs/)
# ---------------------------------------------------------------------------

def _write_inputs_xlsx(df: pd.DataFrame, path: Path) -> None:
    """Layout ingestion.read_inputs expects (2 dummy rows, then header)."""
    wb = Workbook()
    ws = wb.active
    ws.append(["Source: synthetic test fixture"])
    ws.append(["Units: synthetic"])
    ws.append(["Date", *df.columns.tolist()])
    for ts, row in df.iterrows():
        ws.append([ts.to_pydatetime(), *row.tolist()])
    wb.save(path)


def test_spec_tournament_smoke(tmp_path):
    n_days = 700
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rng = np.random.default_rng(7)

    inputs_dir = tmp_path / "Inputs"
    inputs_dir.mkdir()

    drifts = np.linspace(-0.003, 0.003, len(_COUNTRIES))
    prices = pd.DataFrame(
        {
            c: 100.0 * np.cumprod(1.0 + rng.normal(drifts[k], 0.006, n_days))
            for k, c in enumerate(_COUNTRIES)
        },
        index=idx,
    )
    _write_inputs_xlsx(prices, inputs_dir / "Price.xlsx")

    pe = pd.DataFrame(
        rng.uniform(8.0, 25.0, (n_days, len(_COUNTRIES))),
        index=idx, columns=_COUNTRIES,
    )
    _write_inputs_xlsx(pe, inputs_dir / "PE.xlsx")

    pb = pd.DataFrame(
        rng.uniform(0.8, 4.0, (n_days, len(_COUNTRIES))),
        index=idx, columns=_COUNTRIES,
    )
    _write_inputs_xlsx(pb, inputs_dir / "PB.xlsx")

    classification = pd.DataFrame(
        {
            "Segment": ["DM"] * 5 + ["EM"] * 5 + ["Region"] * 2,
            "Region": (["Europe", "Europe", "Asia", "Asia", "LatAm"] * 2)
            + ["Region"] * 2,
            "Type": ["Country"] * 10 + ["Region"] * 2,
        },
        index=pd.Index([*_COUNTRIES, "World", "EM"], name="Country"),
    )
    cls_path = tmp_path / "Classification.xlsx"
    classification.to_excel(cls_path, sheet_name="regiones")

    cfg = json.loads(
        (REPO_ROOT / "configs" / "default.json").read_text(encoding="utf-8")
    )
    cfg["data"]["inputs_folder"] = str(inputs_dir)
    cfg["data"]["classification_file"] = str(cls_path)
    cfg["data"]["target_date"] = "2020-01-01"
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    out_path = tmp_path / "spec_tournament.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/spec_tournament.py",
            "--segments", "World",
            "--periodicity", "21",
            "--config", str(cfg_path),
            "--output", str(out_path),
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_ENV,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(payload["specs"]) == {
        "S1_blend_change", "S2_amp_ey_level", "S3_amp_bp_level",
        "S4_mom_level", "S5_amp_ey_change", "S6_blend_level",
    }
    seg = payload["segments"]["World"]
    assert seg["n_countries"] == 10  # Region pseudo-rows excluded
    assert set(seg["results"]) == set(payload["specs"])
    for record in seg["results"].values():
        assert set(record) == {"mean_ic", "t", "n", "p", "q"}
    assert seg["winner"] in payload["specs"]
    assert set(seg["lockbox"]) == {"mean_ic", "t", "n"}
    # Winner's table is printed
    assert "<- winner" in proc.stdout
