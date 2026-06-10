"""Build the production dashboard HTML from the latest production run.

One self-contained HTML file (inline base64 PNGs + vanilla JS, no external
deps) over the artifact folder written by ``scripts/production_run.py``:
strategy tabs from the deployed registry, latest signal ranking (default
open), interactive signal-evolution SVG chart, allocation history and the
full metrics detail.

Usage
-----
    python scripts/build_production_dashboard.py
        [--run-dir outputs/production/run_YYYYMMDD]
        [--registry configs/production.json]
        [--output outputs/production/production_dashboard.html]

``--run-dir`` defaults to the LATEST ``run_*`` folder under
``outputs/production`` (lexicographic = chronological, date-stamped names).
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)

_DEFAULT_REGISTRY = "configs/production.json"
_DEFAULT_OUTPUT_ROOT = "outputs/production"
_DEFAULT_OUTPUT = os.path.join(
    _DEFAULT_OUTPUT_ROOT, "production_dashboard.html"
)


def _log(msg: str) -> None:
    print(f"[build_production_dashboard] {msg}", flush=True)


def latest_run_dir(output_root: str) -> str:
    """Latest ``run_*`` folder under *output_root* (date-stamped names)."""
    candidates = sorted(
        entry
        for entry in os.listdir(output_root)
        if entry.startswith("run_")
        and os.path.isdir(os.path.join(output_root, entry))
    )
    if not candidates:
        raise SystemExit(
            f"[build_production_dashboard] ERROR: no run_* folder under "
            f"'{output_root}' — run scripts/production_run.py first."
        )
    return os.path.join(output_root, candidates[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render the production dashboard HTML (strategy tabs, latest "
            "signal ranking, signal evolution, allocations, metrics) from "
            "a production_run artifact folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", default=None, dest="run_dir",
        help="Production run folder (default: latest run_* under "
             f"{_DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--registry", default=_DEFAULT_REGISTRY,
        help="Deployed-strategy registry JSON.",
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT,
        help="Output HTML path.",
    )
    args = parser.parse_args()

    # Lazy import: keep --help fast and matplotlib-free.
    from country_rotation.reporting.production_dashboard import (
        build_dashboard,
    )

    run_dir = args.run_dir or latest_run_dir(_DEFAULT_OUTPUT_ROOT)
    _log(f"Run dir   : {run_dir}")
    _log(f"Registry  : {args.registry}")

    html = build_dashboard(run_dir, args.registry)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(args.output) / 1e6
    n_figs = html.count("data:image/png;base64,")
    _log(f"Dashboard -> {args.output} ({size_mb:.2f} MB, {n_figs} figures)")


if __name__ == "__main__":
    main()
