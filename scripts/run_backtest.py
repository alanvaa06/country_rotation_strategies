"""Thin CLI: load pre-built scores + prices, run backtest, write results xlsx.

NOTE: This script requires local data files (scores xlsx, prices xlsx) which
are gitignored (data-only, not committed to the repo).  Inputs/ and
ProcessedInputs/ must be present locally to supply the data.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_DEFAULT_CONFIG = "configs/default.json"
_DEFAULT_PRICES = "ProcessedInputs/Price.xlsx"
_DEFAULT_OUTPUT_DIR = "outputs/backtest_results"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the backtest engine on pre-built normalized scores.",
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
        help="Path to Price xlsx (rows=dates, cols=countries).",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        dest="output_dir",
        help="Directory where backtest results xlsx is written.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Lazy imports (keeps --help fast and import-side-effect free)
    # ------------------------------------------------------------------
    from country_rotation.config import load_config
    from country_rotation.backtest.engine import Engine
    from country_rotation.backtest.metrics import summary
    from country_rotation.backtest.ic import information_coefficient, ic_stats

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print(f"[run_backtest] Reading scores from '{args.scores}' …")
    scores = pd.read_excel(args.scores, index_col=0, parse_dates=True)

    print(f"[run_backtest] Reading prices from '{args.prices}' …")
    prices = pd.read_excel(args.prices, index_col=0, parse_dates=True)

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------
    print("[run_backtest] Running Engine …")
    engine = Engine(scores=scores, prices=prices, cfg=cfg.backtest)
    result = engine.run()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    periods_per_year = 252 / cfg.backtest.periodicity
    port_returns = result.period_results["portfolio_return_gross"]
    bmk_returns = result.period_results["bmk_return"]

    perf = summary(port_returns, bmk_returns, periods_per_year)

    print("[run_backtest] IC analysis …")
    ic_df = information_coefficient(
        scores=scores,
        prices=prices,
        periodicity=cfg.backtest.periodicity,
        method="absolute",
    )
    ic_summary = ic_stats(ic_df["IC"] if "IC" in ic_df.columns else pd.Series(dtype=float))

    # Print combined summary
    combined = {**perf, **{f"ic_{k}": v for k, v in ic_summary.items()}}
    print("\n[run_backtest] Summary:")
    for k, v in combined.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "backtest_results.xlsx")
    print(f"\n[run_backtest] Writing results → {out_path}")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        result.period_results.to_excel(writer, sheet_name="period_results")
        result.historical_weights.to_excel(writer, sheet_name="historical_weights")

    print("[run_backtest] Done.")


if __name__ == "__main__":
    main()
