"""Thin CLI: build a comprehensive HTML backtest report from scores + prices xlsx.

Usage
-----
    python scripts/build_report.py --scores NormalizedScores.xlsx [options]

Outputs an HTML report with embedded PNG charts to --output (default:
outputs/reports/report.html) and prints a one-line summary to stdout.

Note
----
Local data files (Inputs/, ProcessedInputs/) are gitignored.  The --scores
and --prices arguments must point to files present on the local machine.
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DEFAULT_CONFIG = "configs/default.json"
_DEFAULT_PRICES = "ProcessedInputs/Price.xlsx"
_DEFAULT_OUTPUT = "outputs/reports/report.html"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a comprehensive HTML backtest report (performance, risk, IC, scorecard).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Path to JSON platform config file.",
    )
    parser.add_argument(
        "--scores",
        required=True,
        help="Path to NormalizedScores xlsx (rows=dates, cols=countries).",
    )
    parser.add_argument(
        "--prices",
        default=_DEFAULT_PRICES,
        help="Path to Price xlsx (rows=dates, cols=countries + benchmark).",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help="Path to write the HTML report.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help=(
            "Run full validation suite (DSR, PSR, walk-forward, Monte-Carlo) and include "
            "Validation Scorecard section. WARNING: slow — expect several minutes."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Lazy imports (keeps --help fast and import-side-effect free)
    # ------------------------------------------------------------------
    import pandas as pd
    from country_rotation.config import load_config
    from country_rotation.reporting.report import build_report

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg_platform = load_config(args.config)
    cfg = cfg_platform.backtest

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print(f"[build_report] Reading scores from '{args.scores}' ...")
    scores = pd.read_excel(args.scores, index_col=0, parse_dates=True)

    print(f"[build_report] Reading prices from '{args.prices}' ...")
    prices = pd.read_excel(args.prices, index_col=0, parse_dates=True)

    # ------------------------------------------------------------------
    # Optional: run validation suite
    # ------------------------------------------------------------------
    validation_report = None
    if args.validate:
        print(
            "[build_report] Running validation suite (this may take several minutes) ..."
        )
        from country_rotation.validation.scorecard import compute_validation

        grid = {
            "relative_selection_score": (3, 5, 7),
            "periodicity": (21, 63),
        }
        validation_report = compute_validation(
            scores=scores,
            prices=prices,
            cfg=cfg,
            grid=grid,
        )
        overall = validation_report.verdict.overall
        print(f"[build_report] Validation overall verdict: {'PASS' if overall else 'FAIL'}")

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    print("[build_report] Building report ...")
    result = build_report(
        scores=scores,
        prices=prices,
        cfg=cfg,
        output_path=args.output,
        validation_report=validation_report,
        contributions=None,
        ic_methods=("absolute", "relative"),
    )

    # ------------------------------------------------------------------
    # Summary to stdout
    # ------------------------------------------------------------------
    print(f"[build_report] Report written -> {result['path']}")
    print(f"[build_report] Figures: {result['n_figures']}")
    s = result["summary"]
    if s:
        print(
            f"[build_report] Summary — "
            f"Ann.Ret: {s.get('ann_return', float('nan')):.2%} | "
            f"Sharpe: {s.get('sharpe', float('nan')):.2f} | "
            f"MaxDD: {s.get('max_drawdown', float('nan')):.2%}"
        )


if __name__ == "__main__":
    main()
