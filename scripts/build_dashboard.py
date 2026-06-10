"""Build the multi-strategy dashboard: ONE self-contained HTML file with
segment tabs (World / DM / EM) and a per-segment strategy toggle.

Strategy lineup (3 per segment; verdict JSONs from prior research runs):

==============  ==================================  =============================
key             label                               engine config
==============  ==================================  =============================
cap_tilt        Benchmark-Aware (Cap-Tilt)          active, Cap_Tilt, cap base
(DEFAULT)                                           weights from Market_Cap
eqw_active      Active Equal-Weight Top-5           active, Equal
blend50         Core-Satellite 50/50                blend (bmk_weight=0.5), Equal
==============  ==================================  =============================

All strategies: vm prior factor set, periodicity 63, vendor cap index
benchmark (``--bmk-source index`` equivalent), active certification basis.
The per-segment research pipeline (ingest -> universe -> factor scores ->
vm composite) runs ONCE per segment and is shared across the 3 strategies —
only the engine configuration differs.

Usage
-----
    python scripts/build_dashboard.py [--segments World DM EM]
        [--output outputs/research/strategy_dashboard.html]

Note: requires local data (``Inputs/``, ``Classification.xlsx``) and the
verdict JSONs in ``outputs/research/`` — present on the research machine only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

# Allow running from repo root without installing the package; the scripts
# dir itself is added so research_run can be imported as a module.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

from country_rotation.config import BacktestConfig  # noqa: E402

_DEFAULT_CONFIG = "configs/default.json"
_DEFAULT_OUTPUT = "outputs/research/strategy_dashboard.html"
_DEFAULT_VERDICT_DIR = "outputs/research"
_DEFAULT_COSTS = "configs/costs.json"

_MIN_COUNTRIES = 8

DEFAULT_STRATEGY = "cap_tilt"


# ---------------------------------------------------------------------------
# Strategy lineup
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StrategySpec:
    """One toggleable strategy: label, engine config knobs, verdict file."""

    key: str
    label: str
    mode: str                  # 'active' | 'blend'
    weighting_method: str      # 'Equal' | 'Cap_Tilt'
    bmk_weight: float
    uses_base_weights: bool
    verdict_tag: str           # suffix after 'verdict_{seg}_prior_vm_p{p}_'
    acwi_verdict_tag: str | None = None  # ACWI-relative run tag (None = no run)

    def verdict_name(self, segment: str, periodicity: int = 63) -> str:
        """Verdict JSON filename for *segment* (matches research_run tags)."""
        return f"verdict_{segment}_prior_vm_p{periodicity}_{self.verdict_tag}.json"

    def acwi_verdict_name(
        self, segment: str, periodicity: int = 63
    ) -> str | None:
        """ACWI-relative verdict JSON filename, or None if no ACWI run exists
        for this strategy (e.g. blend50 — no ACWI blend runs)."""
        if self.acwi_verdict_tag is None:
            return None
        return (
            f"verdict_{segment}_prior_vm_p{periodicity}_"
            f"{self.acwi_verdict_tag}.json"
        )


#: Verified identity of each segment's vendor benchmark index column.
#: World: regression vs DM+EM gives 0.887/0.111 weights, residual TE 0.14%/yr
#: (MSCI ACWI's ~88/12 DM/EM split). DM: corr 0.96 with US, cap-weighted
#: developed-markets composite (MSCI World type). EM: corr 0.83 with China vs
#: 0.39 with US, cap-weighted emerging composite (MSCI EM type).
BMK_IDENTITY = {
    "World": (
        "Vendor <em>World</em> index — <strong>MSCI ACWI equivalent</strong> "
        "(All-Country: 88/12 DM/EM cap-weight, verified by regression, "
        "residual TE 0.14%/yr)"
    ),
    "DM": (
        "Vendor <em>DM</em> index — <strong>MSCI World equivalent</strong> "
        "(developed markets only, US-dominated cap-weight: corr 0.96 with US)"
    ),
    "EM": (
        "Vendor <em>EM</em> index — <strong>MSCI Emerging Markets "
        "equivalent</strong> (China-heavy cap-weight: corr 0.83 with China, "
        "0.39 with US)"
    ),
}


STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        key="cap_tilt",
        label="Benchmark-Aware (Cap-Tilt)",
        mode="active",
        weighting_method="Cap_Tilt",
        bmk_weight=0.0,
        uses_base_weights=True,
        verdict_tag="active_captilt_capbmk",
        acwi_verdict_tag="active_captilt_vsWorld",
    ),
    StrategySpec(
        key="eqw_active",
        label="Active Equal-Weight Top-5",
        mode="active",
        weighting_method="Equal",
        bmk_weight=0.0,
        uses_base_weights=False,
        verdict_tag="active_capbmk",
        acwi_verdict_tag="active_vsWorld",
    ),
    StrategySpec(
        key="blend50",
        label="Core-Satellite 50/50",
        mode="blend",
        weighting_method="Equal",
        bmk_weight=0.5,
        uses_base_weights=False,
        verdict_tag="active_capbmk_blend50",
    ),
)


def strategy_cfg(
    base: BacktestConfig,
    spec: StrategySpec,
    segment: str,
    periodicity: int = 63,
) -> BacktestConfig:
    """Engine config for one strategy: shared mandate knobs + spec overrides."""
    return dataclasses.replace(
        base,
        bmk=segment,
        periodicity=periodicity,
        selection_criteria="relative",
        relative_selection_score=5,
        mode=spec.mode,
        bmk_weight=spec.bmk_weight,
        weighting_method=spec.weighting_method,
    )


def _log(msg: str) -> None:
    print(f"[build_dashboard] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Per-segment pipeline (research_run reuse)
# ---------------------------------------------------------------------------

def build_segment_inputs(processed, classification, segment: str) -> dict:
    """Shared per-segment inputs: vm composite scores, prices with the vendor
    cap-index benchmark column, Market_Cap base weights, contributions."""
    import research_run

    universe_all = research_run.extract_universe(classification, segment)
    price_df = processed["Price"]
    universe = [c for c in universe_all if c in price_df.columns]
    if len(universe) < _MIN_COUNTRIES:
        raise SystemExit(
            f"[build_dashboard] ERROR: segment '{segment}' has only "
            f"{len(universe)} countries with prices (< {_MIN_COUNTRIES})."
        )
    _log(f"[{segment}] universe: {len(universe)} countries.")

    # Vendor cap-weighted segment index column = mandate benchmark.
    if segment not in price_df.columns:
        raise SystemExit(
            f"[build_dashboard] ERROR: no '{segment}' index column in the "
            f"Price input (vendor cap benchmark required)."
        )
    prices_with_bmk = price_df[universe].copy()
    prices_with_bmk[segment] = price_df[segment]

    # Cap_Tilt base weights from Market_Cap (research_run wiring).
    if "Market_Cap" not in processed:
        raise SystemExit(
            "[build_dashboard] ERROR: 'Market_Cap' metric missing — required "
            "for the Cap_Tilt base weights."
        )
    mcap_df = processed["Market_Cap"]
    mcap_cols = [c for c in universe if c in mcap_df.columns]
    if len(mcap_cols) < _MIN_COUNTRIES:
        raise SystemExit(
            f"[build_dashboard] ERROR: Market_Cap covers only "
            f"{len(mcap_cols)} universe countries (< {_MIN_COUNTRIES})."
        )
    mcap = mcap_df[mcap_cols].ffill()
    mcap = mcap[mcap.sum(axis=1) > 0]
    base_weights = mcap.div(mcap.sum(axis=1), axis=0)

    # vm prior factor set -> composite (normalized scores + contributions).
    factor_scores, _ = research_run.build_factor_scores(processed, universe)
    factor_set = research_run.select_prior_factors(factor_scores, "vm")
    if not factor_set:
        raise SystemExit(
            f"[build_dashboard] ERROR: no vm prior factors available for "
            f"segment '{segment}'."
        )
    _log(f"[{segment}] vm prior factor set ({len(factor_set)}): {factor_set}")
    normalized, contributions, _ = research_run.build_composite(
        factor_scores, tuple(factor_set)
    )

    return {
        "scores": normalized,
        "prices_with_bmk": prices_with_bmk,
        "base_weights": base_weights,
        "contributions": contributions,
    }


def load_verdict(verdict_dir: str, spec: StrategySpec, segment: str,
                 periodicity: int) -> dict:
    path = os.path.join(verdict_dir, spec.verdict_name(segment, periodicity))
    if not os.path.exists(path):
        raise SystemExit(
            f"[build_dashboard] ERROR: verdict JSON missing: {path} — run "
            f"scripts/research_run.py for this strategy first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_acwi_verdict(verdict_dir: str, spec: StrategySpec, segment: str,
                      periodicity: int) -> dict | None:
    """ACWI-relative verdict for *spec*, or None.

    None when the strategy has no ACWI run (blend50) or the file is missing
    (the per-pane \"vs ACWI\" section is optional decoration — a missing file
    must not break the build)."""
    name = spec.acwi_verdict_name(segment, periodicity)
    if name is None:
        return None
    path = os.path.join(verdict_dir, name)
    if not os.path.exists(path):
        _log(f"WARNING: ACWI verdict missing: {path} — "
             f"'vs ACWI' section omitted for [{segment}] '{spec.key}'.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the multi-strategy dashboard (segment tabs + strategy "
            "toggle; default strategy: benchmark-aware Cap_Tilt)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=_DEFAULT_CONFIG, help="Path to JSON platform config."
    )
    parser.add_argument(
        "--segments", nargs="+", default=["World", "DM", "EM"],
        choices=("World", "DM", "EM"),
        help="Market segments (one tab each).",
    )
    parser.add_argument(
        "--periodicity", type=int, default=63,
        help="Rebalance / IC grid step in trading days.",
    )
    parser.add_argument(
        "--verdict-dir", default=_DEFAULT_VERDICT_DIR, dest="verdict_dir",
        help="Directory holding the per-strategy verdict JSONs.",
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT,
        help="Path to write the dashboard HTML.",
    )
    parser.add_argument(
        "--costs", default=_DEFAULT_COSTS,
        help="Cost-model JSON for the per-pane Transaction Costs & Turnover "
             "section (set to '' to omit the section).",
    )
    args = parser.parse_args()

    # Lazy imports: keep --help fast and import-side-effect free.
    import research_run

    from country_rotation.config import load_config
    from country_rotation.reporting.dashboard import (
        build_strategy_pane,
        render_dashboard,
    )

    cfg_platform = load_config(args.config)

    # Cost model loaded ONCE — passed to every pane for the live TCA section.
    cost_model = None
    if args.costs:
        from country_rotation.backtest.tca import load_cost_model

        cost_model = load_cost_model(args.costs)
        _log(f"Cost model '{args.costs}' (as_of {cost_model.as_of}).")

    # Ingest ONCE — shared across segments and strategies.
    processed, classification = research_run.ingest(cfg_platform)

    segments: dict = {}
    for segment in args.segments:
        seg_inputs = build_segment_inputs(processed, classification, segment)

        panes: dict = {}
        for spec in STRATEGIES:
            cfg_bt = strategy_cfg(
                cfg_platform.backtest, spec, segment, args.periodicity
            )
            verdict = load_verdict(
                args.verdict_dir, spec, segment, args.periodicity
            )
            acwi_verdict = load_acwi_verdict(
                args.verdict_dir, spec, segment, args.periodicity
            )
            bw = seg_inputs["base_weights"] if spec.uses_base_weights else None
            _log(f"[{segment}] building pane '{spec.key}' ({spec.label}) ...")
            panes[spec.key] = (
                spec.label,
                build_strategy_pane(
                    seg_inputs["scores"],
                    seg_inputs["prices_with_bmk"],
                    cfg_bt,
                    bw,
                    verdict,
                    seg_inputs["contributions"],
                    acwi_verdict=acwi_verdict,
                    bmk_label=BMK_IDENTITY.get(segment),
                    cost_model=cost_model,
                ),
            )
        segments[segment] = panes

    html = render_dashboard(
        "Country Rotation — Multi-Strategy Dashboard",
        segments,
        default_strategy=DEFAULT_STRATEGY,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(args.output) / 1e6
    n_images = html.count("data:image/png;base64,")
    _log(f"Dashboard -> {args.output} ({size_mb:.2f} MB, {n_images} figures, "
         f"{sum(len(p) for p in segments.values())} panes)")


if __name__ == "__main__":
    main()
