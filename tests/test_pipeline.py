"""Tests for scripts/pipeline.py — the quarterly orchestrator.

The orchestrator's job is command derivation + sequencing, so the end-to-end
tests monkeypatch ``subprocess.run`` to record the exact argv sequence main()
would execute (no engines run). Registry validation and the benchmark-flag
mapping (bmk_index pinned vs segment cap index) are covered explicitly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pipeline  # noqa: E402


def _registry(tmp_path: Path, strategies: list[dict]) -> Path:
    path = tmp_path / "production.json"
    path.write_text(json.dumps({"strategies": strategies}), encoding="utf-8")
    return path


_EM_ENTRY = {
    "id": "EM_captilt_vsEM", "label": "EM", "segment": "EM",
    "prior_set": "vm", "periodicity": 63, "construction": "cap_tilt",
    "active_share": 0.30, "bmk_index": None,
}
_DM_ENTRY = {
    "id": "DM_captilt_vsACWI", "label": "DM", "segment": "DM",
    "prior_set": "vm", "periodicity": 63, "construction": "cap_tilt",
    "active_share": 0.30, "bmk_index": "World",
}


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------

def test_load_registry_round_trip(tmp_path):
    path = _registry(tmp_path, [_EM_ENTRY, _DM_ENTRY])
    strategies = pipeline.load_registry(str(path))
    assert [s["id"] for s in strategies] == ["EM_captilt_vsEM", "DM_captilt_vsACWI"]


def test_load_registry_missing_file():
    with pytest.raises(SystemExit, match="registry not found"):
        pipeline.load_registry("nope/does_not_exist.json")


def test_load_registry_missing_field(tmp_path):
    bad = {k: v for k, v in _EM_ENTRY.items() if k != "periodicity"}
    path = _registry(tmp_path, [bad])
    with pytest.raises(SystemExit, match="missing fields.*periodicity"):
        pipeline.load_registry(str(path))


def test_load_registry_empty(tmp_path):
    path = _registry(tmp_path, [])
    with pytest.raises(SystemExit, match="no strategies"):
        pipeline.load_registry(str(path))


def test_check_inputs_missing(tmp_path):
    with pytest.raises(SystemExit, match="required data missing"):
        pipeline.check_inputs(str(tmp_path))


# ---------------------------------------------------------------------------
# Command derivation
# ---------------------------------------------------------------------------

def test_recert_command_segment_cap_index():
    """bmk_index null -> the segment's own vendor cap index (_capbmk tag)."""
    cmd = pipeline.recert_command(_EM_ENTRY, quick=False, costs="configs/costs.json")
    joined = " ".join(cmd)
    assert "--segment EM" in joined
    assert "--bmk-source index" in joined
    assert "--bmk-index" not in joined
    assert "--costs configs/costs.json" in joined
    assert "--construction cap_tilt" in joined
    assert "--prior-set vm" in joined
    assert "--periodicity 63" in joined
    assert "--quick" not in joined


def test_recert_command_pinned_bmk_index():
    """bmk_index set -> pinned vendor column (_vs{NAME} tag), no --bmk-source."""
    cmd = pipeline.recert_command(_DM_ENTRY, quick=True, costs=None)
    joined = " ".join(cmd)
    assert "--bmk-index World" in joined
    assert "--bmk-source" not in joined
    assert "--costs" not in joined
    assert "--quick" in joined


def test_build_steps_quarterly_order(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "dashboard_commands",
        lambda repo_root=None: [[sys.executable, "build_production_dashboard.py"]],
    )
    steps = pipeline.build_steps(
        "quarterly", [_EM_ENTRY, _DM_ENTRY], quick=True, costs=None
    )
    labels = [label for label, _ in steps]
    assert labels == [
        "recert:EM_captilt_vsEM",
        "recert:DM_captilt_vsACWI",
        "production",
        "dashboard:build_production_dashboard",
    ]


def test_dashboard_commands_skip_research_without_verdicts(tmp_path):
    cmds = pipeline.dashboard_commands(str(tmp_path))
    names = [Path(c[1]).name for c in cmds]
    assert names == ["build_production_dashboard.py"]


def test_dashboard_commands_include_research_with_verdicts(tmp_path):
    verdict_dir = tmp_path / "outputs" / "research"
    verdict_dir.mkdir(parents=True)
    (verdict_dir / "verdict_EM_x.json").write_text("{}", encoding="utf-8")
    cmds = pipeline.dashboard_commands(str(tmp_path))
    names = [Path(c[1]).name for c in cmds]
    assert names == ["build_production_dashboard.py", "build_dashboard.py"]


# ---------------------------------------------------------------------------
# End-to-end main() (subprocess.run monkeypatched — wiring, order, failure)
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self, fail_at: int | None = None):
        self.calls: list[list[str]] = []
        self.fail_at = fail_at

    def __call__(self, cmd, cwd=None):
        self.calls.append(list(cmd))
        rc = 1 if self.fail_at == len(self.calls) else 0
        return type("R", (), {"returncode": rc})()


def _run_main(monkeypatch, tmp_path, argv, recorder):
    registry = _registry(tmp_path, [_EM_ENTRY, _DM_ENTRY])
    monkeypatch.setattr(pipeline, "check_inputs", lambda repo_root=None: None)
    monkeypatch.setattr(pipeline.subprocess, "run", recorder)
    monkeypatch.setattr(
        sys, "argv", ["pipeline.py", *argv, "--registry", str(registry)]
    )
    pipeline.main()


def test_main_quarterly_executes_in_order(monkeypatch, tmp_path):
    recorder = _Recorder()
    monkeypatch.setattr(
        pipeline, "dashboard_commands",
        lambda repo_root=None: [[sys.executable, "build_production_dashboard.py"]],
    )
    _run_main(monkeypatch, tmp_path, ["quarterly", "--quick"], recorder)
    scripts = [Path(c[1]).name for c in recorder.calls]
    assert scripts == [
        "research_run.py", "research_run.py",
        "production_run.py", "build_production_dashboard.py",
    ]
    # recert runs are net-of-cost by default; production honors --quick
    assert "--costs" in recorder.calls[0]
    assert "--quick" in recorder.calls[2]


def test_main_recert_no_costs(monkeypatch, tmp_path):
    recorder = _Recorder()
    _run_main(monkeypatch, tmp_path, ["recert", "--no-costs"], recorder)
    assert len(recorder.calls) == 2
    assert all("--costs" not in c for c in recorder.calls)


def test_main_dry_run_executes_nothing(monkeypatch, tmp_path, capsys):
    recorder = _Recorder()
    _run_main(monkeypatch, tmp_path, ["quarterly", "--dry-run"], recorder)
    assert recorder.calls == []
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "recert:EM_captilt_vsEM" in out


def test_main_stops_at_first_failure(monkeypatch, tmp_path):
    recorder = _Recorder(fail_at=1)
    with pytest.raises(SystemExit, match="FAILED at 'recert:EM_captilt_vsEM'"):
        _run_main(monkeypatch, tmp_path, ["quarterly"], recorder)
    assert len(recorder.calls) == 1  # nothing after the failure


# ---------------------------------------------------------------------------
# --strategy selector + rebalance calendar
# ---------------------------------------------------------------------------

def test_main_strategy_filter_recert(monkeypatch, tmp_path):
    recorder = _Recorder()
    _run_main(
        monkeypatch, tmp_path,
        ["recert", "--strategy", "DM_captilt_vsACWI"], recorder,
    )
    assert len(recorder.calls) == 1
    joined = " ".join(recorder.calls[0])
    assert "--segment DM" in joined and "--bmk-index World" in joined


def test_main_strategy_filter_production_passthrough(monkeypatch, tmp_path):
    recorder = _Recorder()
    _run_main(
        monkeypatch, tmp_path,
        ["production", "--strategy", "EM_captilt_vsEM"], recorder,
    )
    assert len(recorder.calls) == 1
    assert "--strategy" in recorder.calls[0]
    assert "EM_captilt_vsEM" in recorder.calls[0]


def test_main_strategy_filter_unknown(monkeypatch, tmp_path):
    recorder = _Recorder()
    with pytest.raises(SystemExit, match="not in the registry"):
        _run_main(monkeypatch, tmp_path, ["recert", "--strategy", "nope"], recorder)


def test_rebalance_schedule_matches_engine_convention():
    """First step from 2025-11-14 must be 2026-02-11 — the value the engine
    itself emitted in allocations_latest.json (anchors the convention)."""
    dates = pipeline.rebalance_schedule("2025-11-14", 63, 3)
    assert dates[0] == "2026-02-11"
    assert len(dates) == 3
    assert dates == sorted(dates)


def test_calendar_prints_schedule(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "outputs" / "production" / "run_20251114"
    for entry in (_EM_ENTRY, _DM_ENTRY):
        d = run_dir / entry["id"]
        d.mkdir(parents=True)
        (d / "allocations_latest.json").write_text(
            json.dumps({"rebalance_date": "2025-11-14"}), encoding="utf-8"
        )
    pipeline.print_calendar([_EM_ENTRY, _DM_ENTRY], periods=2,
                            repo_root=str(tmp_path))
    out = capsys.readouterr().out
    assert "EM_captilt_vsEM" in out and "DM_captilt_vsACWI" in out
    assert "2026-02-11" in out


def test_calendar_no_run_dir(tmp_path):
    with pytest.raises(SystemExit, match="no production run"):
        pipeline.print_calendar([_EM_ENTRY], periods=2, repo_root=str(tmp_path))
