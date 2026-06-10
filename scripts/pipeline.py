"""Quarterly pipeline orchestrator: ONE command for the full production cycle.

The registry (``configs/production.json``) is the single source of truth:
every stage derives its command line from the deployed-strategy entries, so
adding/removing a strategy never requires touching this script.

Stages
------
recert      Re-certification research runs for every deployed strategy
            (``research_run.py`` with flags derived from the registry entry,
            net-of-cost by default) -> verdict JSONs + HTML reports.
production  Periodic artifact production for all deployed strategies
            (``production_run.py``) -> allocations / signals / TCA under
            ``outputs/production/run_{data_end}/``.
dashboards  Self-contained HTML dashboards: production (latest run) and,
            when verdict JSONs are present, the research strategy dashboard.
quarterly   recert -> production -> dashboards (the full quarterly cycle).

Usage
-----
    python scripts/pipeline.py quarterly [--quick] [--dry-run]
    python scripts/pipeline.py production [--quick] [--dry-run]
    python scripts/pipeline.py recert [--quick] [--no-costs] [--dry-run]
    python scripts/pipeline.py dashboards [--dry-run]

``--dry-run`` validates the registry + required inputs and prints the exact
commands without executing anything. Stages run sequentially and stop at the
first failure, reporting which steps completed (each step is independently
re-runnable).

See docs/research/RUNBOOK_quarterly_recert.md for the decision protocol
around these commands (gate reading, registry updates, kill-switch criteria).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)

_DEFAULT_REGISTRY = os.path.join("configs", "production.json")
_DEFAULT_COSTS = os.path.join("configs", "costs.json")

#: Registry fields every deployed strategy must declare (mirrors
#: production_run._REQUIRED_FIELDS — validated here so --dry-run catches
#: registry problems before any engine spins up).
_REQUIRED_FIELDS = (
    "id", "label", "segment", "prior_set", "periodicity",
    "construction", "active_share", "bmk_index",
)

#: Data the pipeline cannot run without (gitignored vendor inputs).
_REQUIRED_INPUTS = ("Inputs", "Classification.xlsx")


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Registry / environment validation
# ---------------------------------------------------------------------------

def load_registry(path: str) -> list[dict]:
    """Load and validate the deployed-strategy registry.

    Raises ``SystemExit`` with an actionable message on malformed JSON or
    missing required fields — before any expensive stage starts.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"[pipeline] ERROR: registry not found: '{path}'")
    try:
        with open(path, encoding="utf-8") as fh:
            registry = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[pipeline] ERROR: registry is not valid JSON: {exc}")
    strategies = registry.get("strategies", [])
    if not strategies:
        raise SystemExit("[pipeline] ERROR: registry declares no strategies.")
    for entry in strategies:
        missing = [f for f in _REQUIRED_FIELDS if f not in entry]
        if missing:
            raise SystemExit(
                f"[pipeline] ERROR: strategy "
                f"'{entry.get('id', '<no id>')}' missing fields: {missing}"
            )
    return strategies


def check_inputs(repo_root: str | None = None) -> None:
    """Fail fast when the gitignored vendor data is absent."""
    root = repo_root if repo_root is not None else _REPO_ROOT
    missing = [
        p for p in _REQUIRED_INPUTS
        if not os.path.exists(os.path.join(root, p))
    ]
    if missing:
        raise SystemExit(
            f"[pipeline] ERROR: required data missing: {missing}. "
            "Inputs/ and Classification.xlsx are gitignored vendor data — "
            "present on the research machine only."
        )


# ---------------------------------------------------------------------------
# Stage command builders (pure: registry entry -> argv)
# ---------------------------------------------------------------------------

def recert_command(entry: dict, quick: bool, costs: str | None) -> list[str]:
    """research_run.py argv reproducing the strategy's certified verdict tag.

    Benchmark mapping mirrors research_run's tag logic: an explicit
    ``bmk_index`` pins one vendor index column (tag ``_vs{NAME}``);
    otherwise the segment's own vendor cap index is used (``--bmk-source
    index``, tag ``_capbmk``).
    """
    cmd = [
        sys.executable, os.path.join(_SCRIPTS_DIR, "research_run.py"),
        "--segment", str(entry["segment"]),
        "--track", "prior",
        "--prior-set", str(entry["prior_set"]),
        "--periodicity", str(entry["periodicity"]),
        "--basis", "active",
        "--construction", str(entry["construction"]),
    ]
    if entry["bmk_index"]:
        cmd += ["--bmk-index", str(entry["bmk_index"])]
    else:
        cmd += ["--bmk-source", "index"]
    if costs:
        cmd += ["--costs", costs]
    if quick:
        cmd.append("--quick")
    return cmd


def production_command(quick: bool) -> list[str]:
    cmd = [sys.executable, os.path.join(_SCRIPTS_DIR, "production_run.py")]
    if quick:
        cmd.append("--quick")
    return cmd


def dashboard_commands(repo_root: str | None = None) -> list[list[str]]:
    """Production dashboard always; research dashboard only when verdicts
    exist (it hard-exits otherwise)."""
    root = repo_root if repo_root is not None else _REPO_ROOT
    cmds = [
        [sys.executable,
         os.path.join(_SCRIPTS_DIR, "build_production_dashboard.py")],
    ]
    verdict_dir = os.path.join(root, "outputs", "research")
    has_verdicts = os.path.isdir(verdict_dir) and any(
        name.startswith("verdict_") and name.endswith(".json")
        for name in os.listdir(verdict_dir)
    )
    if has_verdicts:
        cmds.append(
            [sys.executable, os.path.join(_SCRIPTS_DIR, "build_dashboard.py")]
        )
    else:
        _log("no verdict JSONs in outputs/research — research dashboard "
             "skipped (run 'recert' first).")
    return cmds


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_steps(steps: list[tuple[str, list[str]]], dry_run: bool) -> None:
    """Run labeled commands sequentially; stop at first failure.

    Every completed step's artifacts are already on disk, so a failed cycle
    resumes by re-running only the failed stage.
    """
    width = max(len(label) for label, _ in steps)
    _log(f"plan: {len(steps)} step(s)")
    for label, cmd in steps:
        _log(f"  {label.ljust(width)}  $ {' '.join(cmd)}")
    if dry_run:
        _log("dry run — nothing executed.")
        return
    done: list[str] = []
    for label, cmd in steps:
        _log(f"=== {label} ===")
        result = subprocess.run(cmd, cwd=_REPO_ROOT)
        if result.returncode != 0:
            raise SystemExit(
                f"[pipeline] FAILED at '{label}' (exit {result.returncode}). "
                f"Completed: {done or 'none'}. Each stage is re-runnable "
                "independently — fix and re-run this stage."
            )
        done.append(label)
    _log(f"done: {len(done)} step(s) completed.")


def build_steps(
    stage: str,
    strategies: list[dict],
    quick: bool,
    costs: str | None,
) -> list[tuple[str, list[str]]]:
    """Assemble the (label, argv) sequence for a stage."""
    steps: list[tuple[str, list[str]]] = []
    if stage in ("recert", "quarterly"):
        for entry in strategies:
            steps.append(
                (f"recert:{entry['id']}", recert_command(entry, quick, costs))
            )
    if stage in ("production", "quarterly"):
        steps.append(("production", production_command(quick)))
    if stage in ("dashboards", "quarterly"):
        for cmd in dashboard_commands():
            name = os.path.splitext(os.path.basename(cmd[1]))[0]
            steps.append((f"dashboard:{name}", cmd))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quarterly pipeline orchestrator over the deployed-"
                    "strategy registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "stage", choices=("quarterly", "recert", "production", "dashboards"),
        help="quarterly = recert -> production -> dashboards.",
    )
    parser.add_argument(
        "--registry", default=_DEFAULT_REGISTRY,
        help="Deployed-strategy registry JSON.",
    )
    parser.add_argument(
        "--quick", action="store_true", default=False,
        help="Smoke mode for the underlying runs (smaller validation suite; "
             "production skips the scorecard).",
    )
    parser.add_argument(
        "--no-costs", action="store_true", default=False,
        help="recert: skip the cost model (gross-of-cost verdicts, no _tca "
             "tag).",
    )
    parser.add_argument(
        "--costs", default=_DEFAULT_COSTS,
        help="recert: cost-model JSON for net-of-cost verdicts.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False, dest="dry_run",
        help="Validate registry + inputs and print the command plan only.",
    )
    args = parser.parse_args()

    strategies = load_registry(os.path.join(_REPO_ROOT, args.registry)
                               if not os.path.isabs(args.registry)
                               else args.registry)
    if args.stage != "dashboards":
        check_inputs()
    costs = None if args.no_costs else args.costs
    steps = build_steps(args.stage, strategies, args.quick, costs)
    run_steps(steps, args.dry_run)


if __name__ == "__main__":
    main()
