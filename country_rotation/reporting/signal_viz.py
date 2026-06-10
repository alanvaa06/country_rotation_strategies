"""Production signal visualizations (latest ranking, allocation history).

Replica of the legacy "Normalized Equity Ranking" chart plus the
allocation-history stacked area and the JS-embeddable signal-history
payload consumed by the production dashboard.

All figure functions take the production artifacts of
``scripts/production_run.py`` (``signal_latest.json`` dicts,
``allocations.csv`` / ``signal_history_monthly.csv`` frames), use the
non-interactive Agg backend and return base64 PNG strings — or ``None``
when the input is empty/unusable (same contract as
:mod:`country_rotation.reporting.report`).
"""
from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # MUST be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

__all__ = [
    "fig_allocation_history",
    "fig_signal_ranking",
    "signal_history_payload",
]

#: Legacy category palette (Normalized Equity Ranking chart).
CATEGORY_COLORS = {
    "Momentum": "#1f3a5f",       # dark navy
    "Profitability": "#40e0d0",  # turquoise
    "Quality": "#c8b560",        # khaki
    "Valuation": "#1e90ff",      # blue
}
_FALLBACK_COLOR = "#9ca3af"      # grey for unknown categories / total change

#: 12-colour palette for line/area charts (mirrors the dashboard JS palette).
PALETTE12 = [
    "#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#7C3AED", "#0EA5E9",
    "#EC4899", "#84CC16", "#F97316", "#14B8A6", "#1e3a5f", "#A855F7",
]

#: Meta columns appended after the per-category contribution columns.
_META_COLUMNS = ("score", "score_change", "weight", "base_weight")


def _category_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, _FALLBACK_COLOR)


def _to_float(value) -> float:
    """None / non-numeric -> NaN; everything else -> float."""
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Data prep (testable core — bar ordering can't be asserted on a PNG)
# ---------------------------------------------------------------------------

def _ranking_frame(signal_latest: Optional[dict]) -> pd.DataFrame:
    """Sorted ranking frame from a ``signal_latest.json`` payload.

    Parameters
    ----------
    signal_latest:
        ``{"countries": {name: {score, score_change, weight, base_weight,
        contributions: {category: value}}}}`` (production_run schema).
        ``change_63d`` is accepted as an alias of ``score_change``.

    Returns
    -------
    DataFrame indexed by country, sorted by total normalized score DESC.
    Columns: one per contribution category (alphabetical) followed by
    ``score``, ``score_change``, ``weight``, ``base_weight``.  Empty
    DataFrame when the payload has no countries.
    """
    countries = (signal_latest or {}).get("countries") or {}
    if not countries:
        return pd.DataFrame()

    categories = sorted({
        cat
        for payload in countries.values()
        for cat in (payload.get("contributions") or {})
    })

    rows = {}
    for name, payload in countries.items():
        contribs = payload.get("contributions") or {}
        change = payload.get("score_change", payload.get("change_63d"))
        row = {cat: _to_float(contribs.get(cat)) for cat in categories}
        row["score"] = _to_float(payload.get("score"))
        row["score_change"] = _to_float(change)
        row["weight"] = _to_float(payload.get("weight"))
        row["base_weight"] = _to_float(payload.get("base_weight"))
        rows[name] = row

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame = frame[categories + list(_META_COLUMNS)]
    return frame.sort_values("score", ascending=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure) -> str:
    """Render *fig* to a PNG and return a base64 string; closes fig."""
    import base64
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _stacked_bars(ax, x: np.ndarray, data: pd.DataFrame, labeled: bool) -> None:
    """Sign-aware stacked bars: one block per column of *data*."""
    bottoms_pos = np.zeros(len(data))
    bottoms_neg = np.zeros(len(data))
    for column in data.columns:
        vals = data[column].fillna(0.0).to_numpy(dtype=float)
        pos = np.where(vals >= 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        color = _category_color(column)
        ax.bar(x, pos, bottom=bottoms_pos, color=color, alpha=0.9,
               label=column if labeled else None)
        ax.bar(x, neg, bottom=bottoms_neg, color=color, alpha=0.9)
        bottoms_pos += pos
        bottoms_neg += neg


def fig_signal_ranking(
    signal_latest: Optional[dict], title_date: str
) -> Optional[str]:
    """Legacy 'Normalized Equity Ranking' replica (2-row stacked-bar figure).

    Top row: stacked per-category contribution bars per country, sorted by
    total normalized score DESC (legacy palette).  Bottom row: stacked +/-
    bars of contribution CHANGES — per-category decomposition when every
    country carries a ``contribution_changes`` dict, else the total
    ``score_change`` as a single grey bar.

    Returns
    -------
    base64 PNG string or None on empty/unusable input.
    """
    try:
        frame = _ranking_frame(signal_latest)
        if frame.empty:
            return None
        categories = [c for c in frame.columns if c not in _META_COLUMNS]

        # Optional per-category change decomposition (single grey total else).
        countries = signal_latest["countries"]
        change_decomp = all(
            isinstance(countries[name].get("contribution_changes"), dict)
            for name in frame.index
        )
        if change_decomp:
            changes = pd.DataFrame.from_dict(
                {
                    name: {
                        cat: _to_float(v)
                        for cat, v in countries[name][
                            "contribution_changes"
                        ].items()
                    }
                    for name in frame.index
                },
                orient="index",
            ).reindex(frame.index)
        else:
            changes = frame[["score_change"]].rename(
                columns={"score_change": "Total Δ"}
            )

        n = len(frame)
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(max(9.0, 0.62 * n), 6.8), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0]},
        )
        x = np.arange(n)

        _stacked_bars(ax_top, x, frame[categories], labeled=True)
        ax_top.set_title(
            f"Normalized Equity Ranking — {title_date}",
            fontsize=13, fontweight="bold",
        )
        ax_top.set_ylabel("Normalized Score")
        ax_top.legend(fontsize=9, loc="upper right")
        ax_top.grid(True, alpha=0.25, axis="y")
        ax_top.margins(x=0.01)

        _stacked_bars(ax_bot, x, changes, labeled=not change_decomp)
        ax_bot.axhline(0, color="black", lw=0.8)
        ax_bot.set_ylabel("Contribution Changes")
        if not change_decomp:
            ax_bot.legend(fontsize=8, loc="upper right")
        ax_bot.grid(True, alpha=0.25, axis="y")

        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(frame.index, rotation=60, ha="right",
                               fontsize=9)
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return None


def fig_allocation_history(
    allocations: Optional[pd.DataFrame], top_n: int = 8
) -> Optional[str]:
    """Stacked area of portfolio weights over time (top N by mean + 'Other').

    Parameters
    ----------
    allocations:
        Rebalance-date x country weight frame (``allocations.csv``).
    top_n:
        Countries shown individually; the rest are aggregated as 'Other'.

    Returns
    -------
    base64 PNG string or None on empty/unusable input.
    """
    try:
        if allocations is None or allocations.empty:
            return None
        weights = allocations.fillna(0.0)
        top = weights.mean().sort_values(ascending=False).head(top_n).index
        plot_data = weights[list(top)].copy()
        rest = [c for c in weights.columns if c not in set(top)]
        if rest:
            plot_data["Other"] = weights[rest].sum(axis=1)
        plot_data = plot_data.loc[:, plot_data.abs().sum() > 1e-9]
        if plot_data.empty:
            return None

        labels = list(plot_data.columns)
        colors = [
            _FALLBACK_COLOR if lbl == "Other"
            else PALETTE12[i % len(PALETTE12)]
            for i, lbl in enumerate(labels)
        ]
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.stackplot(
            plot_data.index,
            [plot_data[c].to_numpy(dtype=float) for c in labels],
            labels=labels, colors=colors, alpha=0.85,
        )
        ax.set_title("Allocation History (every rebalance)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Weight")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.0%}")
        )
        ax.set_ylim(0, None)
        ax.margins(x=0.0)
        ax.legend(fontsize=7.5, loc="upper left", ncol=3, framealpha=0.85)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JS-embeddable signal history
# ---------------------------------------------------------------------------

def signal_history_payload(
    history_monthly: Optional[pd.DataFrame],
) -> dict:
    """JSON-safe payload for the dashboard's SVG evolution chart.

    Parameters
    ----------
    history_monthly:
        Month-end normalized score frame (``signal_history_monthly.csv``),
        dates x countries.

    Returns
    -------
    ``{"dates": [ISO strings], "countries": {name: [4dp floats | None]}}``
    — NaN becomes None so ``json.dumps`` emits ``null`` (never ``NaN``).
    """
    if history_monthly is None or history_monthly.empty:
        return {"dates": [], "countries": {}}

    dates = [pd.Timestamp(d).date().isoformat() for d in history_monthly.index]
    countries = {
        str(col): [
            None if pd.isna(v) else round(float(v), 4)
            for v in history_monthly[col].tolist()
        ]
        for col in history_monthly.columns
    }
    return {"dates": dates, "countries": countries}
