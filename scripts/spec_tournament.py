"""Pre-registered signal-spec tournament: 6 specs x 3 segments, one winner each.

Motivation
----------
The composite *signal construction* and the *selection rule* (level vs change)
are joint hypotheses.  This harness pre-registers a small spec family BEFORE
looking at any out-of-sample data, evaluates every spec's IC on the screening
window only (first 80% of common dates), corrects for the family size with
Benjamini-Hochberg FDR, and confirms the single per-segment winner ONCE on the
terminal 20% lockbox (same split convention as ``selection.walkforward``).

Pre-registered spec family (fixed; literature anchor per spec)
--------------------------------------------------------------
S1_blend_change   Current baseline: vm composite — equal-weight 4-metric blend
                  (expanding zscore / absolute pct / relative rank / 63d delta
                  pct) over the 6 vm prior factors (Momentum_12_1, Momentum_6_1,
                  EarningsYieldTTM, EarningsYieldFWD, PB, PE); selection signal
                  = 63d score CHANGE.  [Asness-Moskowitz-Pedersen 2013 50/50
                  value+momentum country combo for the blend; the change-based
                  selection is this repo's legacy 'relative' rule.]
S2_amp_ey_level   0.5 * relative_rank(Momentum_12_1) + 0.5 *
                  relative_rank(EarningsYieldTTM); LEVEL selection.
                  [AMP 2013: country momentum (12-1) + country value measured
                  as E/P, equal combo, ranked on signal level.]
S3_amp_bp_level   0.5 * relative_rank(Momentum_12_1) + 0.5 * relative_rank(BP)
                  where BP = book-to-price = the PB factor's direction-applied
                  relative rank (PB direction is -1 in the catalog, so the
                  rank is inverted: high BP = cheap = high score); LEVEL
                  selection.  [AMP 2013 exact BE/ME country-value analog.]
S4_mom_level      relative_rank(Momentum_12_1) alone; LEVEL selection.
                  [Balvers-Wu 2006 momentum across national markets;
                  AMP 2013 country momentum.]
S5_amp_ey_change  S2 composite, 63d score CHANGE selection.  [Hybrid: AMP
                  composite + this repo's change-based selection rule.]
S6_blend_level    S1 composite, LEVEL selection.  [Our 4-metric blend + the
                  canonical AMP rank-of-level selection.]

All spec scores are normalized cross-sectionally to [0, 1] per date via
``signals.composite.normalize_cross_section`` for comparability.

Evaluation protocol (per segment: World / DM / EM)
--------------------------------------------------
1. Dates split: screening window = first 80% of the common (scores ∩ prices)
   dates, lockbox = last 20% — identical boundary arithmetic to
   ``selection.walkforward._split_dates`` (lockbox = last
   ``floor(0.2 * n)`` dates).
2. Per spec: Spearman IC on the SCREEN window only, via
   ``backtest.ic.information_coefficient(periodicity=63)``.  The IC *method*
   matches what the spec trades: ``relative`` (signal = 63d score change) for
   change-selection specs, ``absolute`` (signal = score level) for
   level-selection specs.  ``ic_stats`` -> mean IC, t-stat, n.
3. BH-FDR (``selection.walkforward.benjamini_hochberg``) across the 6 specs
   per segment, on one-sided p-values from the IC-series t-stat:
   ``p = scipy.stats.t.sf(t, df=n-1)``.
4. Winner per segment = best (lowest) q-value; tie-break = higher mean IC.
   Lockbox confirmation: the winner's IC on the lockbox window (mean + t),
   one-shot — re-running after changing the family is lockbox burn.
5. Output: ``outputs/research/spec_tournament.json`` with per-segment
   per-spec {mean_ic, t, n, p, q}, the winner, and the lockbox confirmation;
   a clean table is printed per segment.

Usage
-----
    python scripts/spec_tournament.py
        [--segments World DM EM] [--periodicity 63] [--fdr-q 0.10]
        [--config configs/default.json]
        [--output outputs/research/spec_tournament.json]

Note: requires local data (``Inputs/``, ``Classification.xlsx``) which is
gitignored — present on the research machine only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings

# Allow running from repo root without installing the package; the scripts
# dir itself is added so research_run can be imported as a module.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

_DEFAULT_CONFIG = "configs/default.json"
_DEFAULT_OUTPUT = "outputs/research/spec_tournament.json"

_MIN_COUNTRIES = 8

#: Screening window fraction (lockbox = 1 - this), walkforward convention.
_SCREEN_FRAC = 0.8


# ---------------------------------------------------------------------------
# Pre-registered spec family
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SignalSpec:
    """One pre-registered signal spec.

    ``selection`` is the engine ``selection_criteria`` the spec trades
    ('top_n_level' = rank of signal level, 'relative' = 63d score change);
    ``ic_method`` is the matching IC signal ('absolute' for level,
    'relative' for change) — the IC must measure what the spec trades.
    """

    name: str
    composite: str        # 'blend' | 'amp_ey' | 'amp_bp' | 'mom'
    selection: str        # 'top_n_level' | 'relative'
    anchor: str

    @property
    def ic_method(self) -> str:
        return "absolute" if self.selection == "top_n_level" else "relative"


SPECS: tuple = (
    SignalSpec("S1_blend_change", "blend", "relative",
               "AMP-2013 V+M blend + legacy change selection (baseline)"),
    SignalSpec("S2_amp_ey_level", "amp_ey", "top_n_level",
               "AMP 2013: 12-1 momentum + E/P value, level rank"),
    SignalSpec("S3_amp_bp_level", "amp_bp", "top_n_level",
               "AMP 2013: 12-1 momentum + BE/ME (book-to-price) analog"),
    SignalSpec("S4_mom_level", "mom", "top_n_level",
               "Balvers-Wu 2006 / AMP 2013 country momentum alone"),
    SignalSpec("S5_amp_ey_change", "amp_ey", "relative",
               "Hybrid: AMP E/P composite + change selection"),
    SignalSpec("S6_blend_level", "blend", "top_n_level",
               "Our 4-metric blend + canonical AMP level selection"),
)


def _log(msg: str) -> None:
    print(f"[spec_tournament] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Spec score construction (research_run reuse)
# ---------------------------------------------------------------------------

def _rank_score(processed: dict, universe: list, factor_name: str):
    """Direction-applied cross-sectional relative rank for one factor.

    Exactly the 'relative_rank' metric that feeds
    ``research_run.build_factor_scores`` (``transform_factor`` applies the
    catalog direction, so PB with direction -1 comes out as the inverted
    rank = the book-to-price / BP rank).
    """
    from country_rotation.factors.catalog import get_spec
    from country_rotation.factors.transforms import transform_factor

    if factor_name not in processed:
        raise SystemExit(
            f"[spec_tournament] ERROR: factor '{factor_name}' missing from "
            f"the processed inputs — cannot build the spec family."
        )
    spec = get_spec(factor_name)
    df = processed[factor_name]
    cols = [c for c in universe if c in df.columns]
    if len(cols) < 3:
        raise SystemExit(
            f"[spec_tournament] ERROR: factor '{factor_name}' covers only "
            f"{len(cols)} universe columns (< 3)."
        )
    return transform_factor(df[cols], spec.direction)["relative_rank"]


def _half_half(a, b):
    """0.5*a + 0.5*b on the common columns (NaN where either is NaN)."""
    common = a.columns.intersection(b.columns)
    return 0.5 * a[common] + 0.5 * b[common]


def build_spec_scores(processed: dict, universe: list) -> dict:
    """Composite score DataFrame per UNIQUE composite, normalized to [0, 1].

    Returns {composite_key: scores}; specs sharing a composite (S2/S5,
    S1/S6) share the same DataFrame.  The 'blend' composite reuses the
    research_run vm pipeline verbatim (build_factor_scores ->
    select_prior_factors('vm') -> build_composite); per-factor scores are
    independent, so ``processed`` is restricted to the vm factors purely
    for speed.
    """
    import research_run
    from country_rotation.signals.composite import normalize_cross_section

    vm_names = research_run._PRIOR_SETS["vm"]
    vm_processed = {k: processed[k] for k in vm_names if k in processed}
    factor_scores, _ = research_run.build_factor_scores(vm_processed, universe)
    vm_set = research_run.select_prior_factors(factor_scores, "vm")
    if not vm_set:
        raise SystemExit(
            "[spec_tournament] ERROR: no vm prior factors available — "
            "cannot build the S1/S6 blend composite."
        )
    blend, _, _ = research_run.build_composite(factor_scores, tuple(vm_set))
    # build_composite already applies normalize_cross_section.

    mom = _rank_score(processed, universe, "Momentum_12_1")
    ey = _rank_score(processed, universe, "EarningsYieldTTM")
    bp = _rank_score(processed, universe, "PB")  # direction -1 -> BP rank

    return {
        "blend": blend,
        "amp_ey": normalize_cross_section(_half_half(mom, ey)),
        "amp_bp": normalize_cross_section(_half_half(mom, bp)),
        "mom": normalize_cross_section(mom),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _one_sided_p(t_stat: float, n: int) -> float:
    """One-sided p from the IC-series t-stat: ``scipy.stats.t.sf(t, df=n-1)``.

    Degenerate (n < 2, NaN/inf t) -> p = 1.0 (no evidence)."""
    import numpy as np
    from scipy import stats

    if n < 2 or t_stat is None or not np.isfinite(t_stat):
        return 1.0
    return float(stats.t.sf(t_stat, df=n - 1))


def _ic_summary(scores, prices, dates, periodicity: int, method: str) -> dict:
    """Mean IC / t / n of one spec on one date slice (screen or lockbox)."""
    import pandas as pd
    from scipy.stats import ConstantInputWarning

    from country_rotation.backtest import ic as ic_module

    with warnings.catch_warnings():
        # Warm-up rows aggregate to constant cross-sections (documented
        # weighted_metric_average behaviour) -> degenerate Spearman calls;
        # those ICs are NaN and dropped, the warning is noise here.
        warnings.simplefilter("ignore", ConstantInputWarning)
        ic_df = ic_module.information_coefficient(
            scores.loc[dates], prices.loc[dates], periodicity, method
        )
    series = (
        ic_df["IC"].dropna() if "IC" in ic_df.columns
        else pd.Series(dtype=float)
    )
    stats_d = ic_module.ic_stats(series)
    return {
        "mean_ic": stats_d["mean_ic"],
        "t": stats_d["t_stat"],
        "n": int(len(series)),
    }


def evaluate_segment(
    composite_scores: dict,
    prices,
    periodicity: int,
    fdr_q: float,
) -> dict:
    """Tournament for one segment: screen-window ICs, BH-FDR, winner,
    one-shot lockbox confirmation of the winner only."""
    import numpy as np

    from country_rotation.selection.walkforward import (
        _split_dates,
        benjamini_hochberg,
    )

    # Same split convention as selection.walkforward: lockbox = LAST
    # floor((1 - _SCREEN_FRAC) * n) common dates.
    screen_dates, lockbox_dates = _split_dates(
        composite_scores, prices, lockbox_frac=1.0 - _SCREEN_FRAC
    )

    results: dict = {}
    pvalues: dict = {}
    for spec in SPECS:
        scores = composite_scores[spec.composite]
        summary = _ic_summary(
            scores, prices, screen_dates, periodicity, spec.ic_method
        )
        summary["p"] = _one_sided_p(summary["t"], summary["n"])
        results[spec.name] = summary
        pvalues[spec.name] = summary["p"]

    qvalues = benjamini_hochberg(pvalues, fdr_q)
    for name, q in qvalues.items():
        results[name]["q"] = q

    def _winner_key(name: str):
        mean_ic = results[name]["mean_ic"]
        if mean_ic is None or not np.isfinite(mean_ic):
            mean_ic = -np.inf
        return (results[name]["q"], -mean_ic)

    winner = min((s.name for s in SPECS), key=_winner_key)
    winner_spec = next(s for s in SPECS if s.name == winner)

    lockbox = _ic_summary(
        composite_scores[winner_spec.composite],
        prices,
        lockbox_dates,
        periodicity,
        winner_spec.ic_method,
    )

    return {
        "screen_window": [str(screen_dates[0].date()), str(screen_dates[-1].date())],
        "lockbox_window": [str(lockbox_dates[0].date()), str(lockbox_dates[-1].date())],
        "results": results,
        "winner": winner,
        "winner_selection": winner_spec.selection,
        "lockbox": lockbox,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_segment_table(segment: str, n_countries: int, seg: dict) -> None:
    def _f(x, fmt="{:+.4f}"):
        import numpy as np
        return fmt.format(x) if x is not None and np.isfinite(x) else "   nan"

    _log("=" * 78)
    _log(f"SEGMENT {segment}  ({n_countries} countries; "
         f"screen {seg['screen_window'][0]}..{seg['screen_window'][1]}, "
         f"lockbox {seg['lockbox_window'][0]}..{seg['lockbox_window'][1]})")
    _log(f"{'spec':<18}{'selection':<13}{'ic_method':<11}"
         f"{'mean_ic':>9}{'t':>8}{'n':>5}{'p':>9}{'q':>9}")
    for spec in SPECS:
        r = seg["results"][spec.name]
        marker = " <- winner" if spec.name == seg["winner"] else ""
        _log(
            f"{spec.name:<18}{spec.selection:<13}{spec.ic_method:<11}"
            f"{_f(r['mean_ic']):>9}{_f(r['t'], '{:+.2f}'):>8}{r['n']:>5}"
            f"{_f(r['p'], '{:.4f}'):>9}{_f(r['q'], '{:.4f}'):>9}{marker}"
        )
    lb = seg["lockbox"]
    _log(f"lockbox confirm [{seg['winner']}]: "
         f"mean_ic={_f(lb['mean_ic'])} t={_f(lb['t'], '{:+.2f}')} n={lb['n']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-registered 6-spec signal tournament per segment "
                    "(screen-window IC + BH-FDR; one-shot lockbox confirm).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--segments", nargs="+", default=("World", "DM", "EM"),
        choices=("World", "DM", "EM"), help="Market segment universes.",
    )
    parser.add_argument(
        "--periodicity", type=int, default=63,
        help="IC grid step in trading days (matches the rebalance grid).",
    )
    parser.add_argument(
        "--fdr-q", type=float, default=0.10, dest="fdr_q",
        help="BH false-discovery-rate level across the 6 specs per segment.",
    )
    parser.add_argument(
        "--config", default=_DEFAULT_CONFIG, help="Path to JSON platform config."
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT,
        help="Output JSON path for the tournament results.",
    )
    args = parser.parse_args()

    t0 = time.time()

    import research_run
    from country_rotation.config import load_config

    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    processed, classification = research_run.ingest(cfg)
    if "Price" not in processed:
        raise SystemExit("[spec_tournament] ERROR: 'Price' metric missing.")
    price_df = processed["Price"]

    payload: dict = {
        "periodicity": args.periodicity,
        "fdr_q": args.fdr_q,
        "screen_frac": _SCREEN_FRAC,
        "specs": {
            s.name: {
                "composite": s.composite,
                "selection": s.selection,
                "ic_method": s.ic_method,
                "anchor": s.anchor,
            }
            for s in SPECS
        },
        "segments": {},
    }

    for segment in args.segments:
        universe_all = research_run.extract_universe(classification, segment)
        universe = [c for c in universe_all if c in price_df.columns]
        if len(universe) < _MIN_COUNTRIES:
            raise SystemExit(
                f"[spec_tournament] ERROR: segment '{segment}' has only "
                f"{len(universe)} countries with prices (< {_MIN_COUNTRIES})."
            )
        prices = price_df[universe].copy()

        _log(f"[{segment}] building spec composites "
             f"({len(universe)} countries) ...")
        composite_scores = build_spec_scores(processed, universe)

        _log(f"[{segment}] evaluating {len(SPECS)} specs ...")
        seg = evaluate_segment(
            composite_scores, prices, args.periodicity, args.fdr_q
        )
        payload["segments"][segment] = {"n_countries": len(universe), **seg}
        print_segment_table(segment, len(universe), seg)

    payload["runtime_seconds"] = round(time.time() - t0, 1)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(research_run._json_safe(payload), f, indent=2)
    _log("=" * 78)
    _log(f"Results -> {args.output}")
    _log(f"Done in {payload['runtime_seconds']}s.")


if __name__ == "__main__":
    main()
