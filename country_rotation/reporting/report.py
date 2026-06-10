"""HTML report builder for country rotation strategy backtests (Plan C, §7).

Produces a self-contained HTML file with inline base64 PNG figures and tables
covering: Performance, Risk, IC Analysis, Score Decomposition, Validation
Scorecard.

Backend
-------
Uses the non-interactive Agg backend to avoid display requirements.
``matplotlib.use("Agg")`` is called before any pyplot import.
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # MUST be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from country_rotation.backtest.benchmarks import equal_weight_buy_hold
from country_rotation.backtest.engine import Engine, EngineResult
from country_rotation.backtest.ic import information_coefficient, ic_stats
from country_rotation.backtest.metrics import summary
from country_rotation.config import BacktestConfig

__all__ = [
    "build_report",
    "render_html",
    "fig_cumulative",
    "fig_drawdown",
    "fig_rolling_12m",
    "fig_ic",
    "fig_contributions",
    "fig_contributions_latest",
    "fig_weights",
]

# ---------------------------------------------------------------------------
# Colour palette (muted finance style)
# ---------------------------------------------------------------------------
_C_PORT = "#2563EB"   # portfolio — blue
_C_BMK = "#DC2626"    # benchmark — red
_C_EQW = "#16A34A"    # equal-weight — green
_C_ACTIVE = "#7C3AED"  # active — purple
_C_GREY = "#6B7280"

_STACKED_PALETTE = [
    "#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#7C3AED",
    "#EC4899", "#0EA5E9", "#84CC16", "#F97316", "#14B8A6",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure) -> str:
    """Render *fig* to a PNG and return a base64-encoded string; closes fig."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown from a returns series."""
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    return (cum - running_max) / running_max


def _cumulative_equity(returns: pd.Series) -> pd.Series:
    """Compound returns into equity curve starting at 1."""
    return (1.0 + returns).cumprod()


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def fig_cumulative(
    engine_result: EngineResult,
    eqw_equity: Optional[pd.Series],
) -> Optional[str]:
    """Cumulative net equity vs benchmark vs equal-weight null.

    Parameters
    ----------
    engine_result:
        Backtest output from :class:`Engine`.
    eqw_equity:
        Equity curve from equal-weight buy-and-hold (may be None).

    Returns
    -------
    base64 PNG string or None if inputs are unusable.
    """
    try:
        dr = engine_result.daily_returns
        db = engine_result.daily_bmk_returns
        if dr is None or len(dr) == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 4))

        port_eq = _cumulative_equity(dr)
        bmk_eq = _cumulative_equity(db) if db is not None and len(db) > 0 else None

        ax.plot(port_eq.index, port_eq.values, color=_C_PORT, lw=1.5, label="Portfolio (net)")
        if bmk_eq is not None:
            ax.plot(bmk_eq.index, bmk_eq.values, color=_C_BMK, lw=1.2, ls="--", label="Benchmark")
        if eqw_equity is not None and len(eqw_equity) > 0:
            # Rebase eqw to same start as portfolio
            eqw_r = eqw_equity / eqw_equity.iloc[0]
            ax.plot(eqw_r.index, eqw_r.values, color=_C_EQW, lw=1.0, ls=":", label="Equal-Weight B&H")

        ax.set_title("Cumulative Return", fontsize=12, fontweight="bold")
        ax.set_ylabel("Equity (start = 1)")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return None


def fig_drawdown(engine_result: EngineResult) -> Optional[str]:
    """Portfolio drawdown + benchmark drawdown lines.

    Returns
    -------
    base64 PNG string or None.
    """
    try:
        dr = engine_result.daily_returns
        db = engine_result.daily_bmk_returns
        if dr is None or len(dr) == 0:
            return None

        port_dd = _drawdown_series(dr)
        fig, ax = plt.subplots(figsize=(10, 3))

        ax.fill_between(port_dd.index, port_dd.values, 0, color=_C_PORT, alpha=0.3, label="Portfolio DD")
        ax.plot(port_dd.index, port_dd.values, color=_C_PORT, lw=1.0)

        if db is not None and len(db) > 0:
            bmk_dd = _drawdown_series(db)
            ax.plot(bmk_dd.index, bmk_dd.values, color=_C_BMK, lw=1.0, ls="--", label="Benchmark DD")

        ax.set_title("Drawdown", fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return None


def fig_rolling_12m(engine_result: EngineResult) -> Optional[str]:
    """Rolling 252-day return of portfolio and benchmark.

    Returns
    -------
    base64 PNG string or None.
    """
    try:
        dr = engine_result.daily_returns
        db = engine_result.daily_bmk_returns
        if dr is None or len(dr) < 30:
            return None

        window = 252
        port_roll = (1 + dr).rolling(window).apply(lambda x: x.prod(), raw=True) - 1
        fig, ax = plt.subplots(figsize=(10, 3))

        ax.plot(port_roll.index, port_roll.values, color=_C_PORT, lw=1.3, label="Portfolio 12m return")
        if db is not None and len(db) > 0:
            bmk_roll = (1 + db).rolling(window).apply(lambda x: x.prod(), raw=True) - 1
            ax.plot(bmk_roll.index, bmk_roll.values, color=_C_BMK, lw=1.0, ls="--", label="Benchmark 12m return")

        ax.axhline(0, color=_C_GREY, lw=0.8, ls="-")
        ax.set_title("Rolling 12-Month Return", fontsize=12, fontweight="bold")
        ax.set_ylabel("Return")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return None


def fig_ic(ic_df: pd.DataFrame) -> Optional[str]:
    """Two-panel IC chart: time series + histogram.

    Parameters
    ----------
    ic_df:
        DataFrame with "IC" column, indexed by date.

    Returns
    -------
    base64 PNG string or None.
    """
    try:
        if ic_df is None or ic_df.empty or "IC" not in ic_df.columns:
            return None
        ic = ic_df["IC"].dropna()
        if len(ic) < 3:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

        # Time series
        ax1.bar(ic.index, ic.values, color=np.where(ic.values >= 0, _C_PORT, _C_BMK), alpha=0.7, width=20)
        mean_ic = ic.mean()
        ax1.axhline(mean_ic, color=_C_GREY, lw=1.5, ls="--", label=f"Mean IC = {mean_ic:.3f}")
        ax1.axhline(0, color="black", lw=0.6)
        ax1.set_title("IC Time Series", fontsize=11, fontweight="bold")
        ax1.set_ylabel("IC")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Histogram
        ax2.hist(ic.values, bins=min(30, max(5, len(ic) // 3)), color=_C_PORT, alpha=0.7, edgecolor="white")
        ax2.axvline(0, color="black", lw=0.8)
        ax2.axvline(mean_ic, color=_C_GREY, lw=1.5, ls="--", label=f"Mean = {mean_ic:.3f}")
        ax2.set_title("IC Distribution", fontsize=11, fontweight="bold")
        ax2.set_xlabel("IC")
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return None


def fig_contributions(
    contributions: dict,
    country: str,
) -> Optional[str]:
    """Stacked area of category contributions over time for one country.

    Parameters
    ----------
    contributions:
        {category: DataFrame(dates x countries)}.
    country:
        Country column to plot.

    Returns
    -------
    base64 PNG string or None.
    """
    try:
        if not contributions:
            return None
        # Build a DataFrame: dates x categories
        frames = {}
        for cat, df in contributions.items():
            if country in df.columns:
                frames[cat] = df[country]
        if not frames:
            return None
        data = pd.DataFrame(frames).dropna(how="all")
        if data.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 3.5))
        cats = list(data.columns)
        colors = _STACKED_PALETTE[: len(cats)]
        ax.stackplot(data.index, [data[c].fillna(0).values for c in cats],
                     labels=cats, colors=colors, alpha=0.75)
        ax.set_title(f"Score Contributions — {country}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Contribution")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return None


def fig_contributions_latest(contributions: dict) -> Optional[str]:
    """Stacked bar of category contributions per country at latest date.

    Parameters
    ----------
    contributions:
        {category: DataFrame(dates x countries)}.

    Returns
    -------
    base64 PNG string or None.
    """
    try:
        if not contributions:
            return None
        # Get latest row per category
        frames = {}
        for cat, df in contributions.items():
            last = df.dropna(how="all").iloc[-1] if not df.dropna(how="all").empty else None
            if last is not None:
                frames[cat] = last
        if not frames:
            return None
        data = pd.DataFrame(frames)  # countries x categories
        data = data.dropna(how="all")
        if data.empty:
            return None

        cats = list(data.columns)
        countries = list(data.index)
        colors = _STACKED_PALETTE[: len(cats)]

        fig, ax = plt.subplots(figsize=(max(8, len(countries) * 0.6), 4))
        bottoms_pos = np.zeros(len(countries))
        bottoms_neg = np.zeros(len(countries))
        x = np.arange(len(countries))

        for i, cat in enumerate(cats):
            vals = data[cat].fillna(0).values
            pos = np.where(vals >= 0, vals, 0)
            neg = np.where(vals < 0, vals, 0)
            ax.bar(x, pos, bottom=bottoms_pos, color=colors[i], label=cat, alpha=0.8)
            ax.bar(x, neg, bottom=bottoms_neg, color=colors[i], alpha=0.8)
            bottoms_pos += pos
            bottoms_neg += neg

        ax.set_xticks(x)
        ax.set_xticklabels(countries, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Latest Score Contributions by Country", fontsize=11, fontweight="bold")
        ax.set_ylabel("Contribution")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return None


def fig_weights(engine_result: EngineResult, top_n: int = 10) -> Optional[str]:
    """Stacked area of portfolio weights over time (top N + 'Other').

    Parameters
    ----------
    engine_result:
        Backtest output.
    top_n:
        Number of top-average-weight countries to show individually.

    Returns
    -------
    base64 PNG string or None.
    """
    try:
        hw = engine_result.historical_weights
        if hw is None or hw.empty:
            return None

        # Exclude benchmark column
        country_cols = [c for c in hw.columns if c not in ("World", "Benchmark")]
        if not country_cols:
            country_cols = list(hw.columns)

        weights = hw[country_cols].fillna(0.0)
        if weights.empty:
            return None

        mean_w = weights.mean().sort_values(ascending=False)
        top_countries = mean_w.head(top_n).index.tolist()
        other_cols = [c for c in country_cols if c not in top_countries]

        plot_data = weights[top_countries].copy()
        if other_cols:
            plot_data["Other"] = weights[other_cols].sum(axis=1)

        # Drop near-zero columns
        plot_data = plot_data.loc[:, plot_data.sum() > 1e-9]
        if plot_data.empty:
            return None

        labels = list(plot_data.columns)
        colors = _STACKED_PALETTE[: len(labels)]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.stackplot(
            plot_data.index,
            [plot_data[c].values for c in labels],
            labels=labels,
            colors=colors,
            alpha=0.8,
        )
        ax.set_title("Portfolio Weights Over Time", fontsize=12, fontweight="bold")
        ax.set_ylabel("Weight")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Table builders (return HTML strings)
# ---------------------------------------------------------------------------

_PASS_STYLE = "background:#d1fae5;color:#065f46;font-weight:bold;padding:2px 6px;border-radius:3px;"
_FAIL_STYLE = "background:#fee2e2;color:#991b1b;font-weight:bold;padding:2px 6px;border-radius:3px;"


def _fmt(v, pct: bool = False, decimals: int = 2) -> str:
    """Format a float value; NaN -> '—'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if pct:
        return f"{v:.{decimals}%}"
    return f"{v:.{decimals}f}"


def table_per_year(
    daily_returns: pd.Series,
    daily_bmk_returns: pd.Series,
) -> str:
    """Per-year performance table: year, port return, bmk return, active, vol, maxDD."""
    try:
        if daily_returns is None or len(daily_returns) == 0:
            return "<p><em>No daily returns available.</em></p>"

        port = daily_returns.dropna()
        bmk = daily_bmk_returns.dropna() if daily_bmk_returns is not None else pd.Series(dtype=float)

        years = port.index.year.unique()
        rows = []
        for yr in sorted(years):
            p = port[port.index.year == yr]
            b = bmk[bmk.index.year == yr] if len(bmk) > 0 else pd.Series(dtype=float)
            p_ret = float((1 + p).prod() - 1)
            b_ret = float((1 + b).prod() - 1) if len(b) > 0 else np.nan
            active = p_ret - b_ret if not np.isnan(b_ret) else np.nan
            vol = float(p.std() * np.sqrt(252)) if len(p) > 1 else np.nan
            # Max drawdown
            cum = (1 + p).cumprod()
            mdd = float(((cum - cum.cummax()) / cum.cummax()).min()) if len(p) > 0 else np.nan
            rows.append((yr, p_ret, b_ret, active, vol, mdd))

        html = (
            "<table><thead><tr>"
            "<th>Year</th><th>Port</th><th>Benchmark</th>"
            "<th>Active</th><th>Vol</th><th>Max DD</th>"
            "</tr></thead><tbody>"
        )
        for yr, p, b, act, vol, mdd in rows:
            html += (
                f"<tr><td>{yr}</td>"
                f"<td>{_fmt(p, pct=True)}</td>"
                f"<td>{_fmt(b, pct=True)}</td>"
                f"<td>{_fmt(act, pct=True)}</td>"
                f"<td>{_fmt(vol, pct=True)}</td>"
                f"<td>{_fmt(mdd, pct=True)}</td>"
                f"</tr>"
            )
        html += "</tbody></table>"
        return html
    except Exception as exc:
        return f"<p><em>Table error: {exc}</em></p>"


def table_risk(daily_metrics: dict, period_metrics: dict) -> str:
    """Combined risk table: daily + period metrics."""
    try:
        rows = [
            ("Ann. Return (daily)", daily_metrics.get("ann_return"), True),
            ("Ann. Vol (daily)", daily_metrics.get("ann_vol"), True),
            ("Sharpe (daily)", daily_metrics.get("sharpe"), False),
            ("Max Drawdown (daily)", daily_metrics.get("max_drawdown"), True),
            ("Win Rate (daily)", daily_metrics.get("win_rate"), True),
            ("Beta", daily_metrics.get("beta"), False),
            ("Up Capture", daily_metrics.get("up_capture"), False),
            ("Down Capture", daily_metrics.get("down_capture"), False),
            ("Tracking Error (daily)", daily_metrics.get("tracking_error"), True),
            ("Info Ratio (daily)", daily_metrics.get("information_ratio"), False),
            ("— Period metrics —", None, False),
            ("Ann. Return (period)", period_metrics.get("ann_return"), True),
            ("Sharpe (period)", period_metrics.get("sharpe"), False),
            ("Max Drawdown (period)", period_metrics.get("max_drawdown"), True),
        ]

        html = (
            "<table><thead><tr>"
            "<th>Metric</th><th>Value</th>"
            "</tr></thead><tbody>"
        )
        for label, val, pct in rows:
            if val is None and label.startswith("—"):
                html += f"<tr><td colspan='2'><strong>{label}</strong></td></tr>"
            else:
                html += f"<tr><td>{label}</td><td>{_fmt(val, pct=pct)}</td></tr>"
        html += "</tbody></table>"
        return html
    except Exception as exc:
        return f"<p><em>Table error: {exc}</em></p>"


def table_ic_stats(ic_stats_dict: dict, method: str) -> str:
    """IC stats table for one method."""
    try:
        keys = ["mean_ic", "median_ic", "std_ic", "t_stat", "icir", "hit_rate"]
        labels = ["Mean IC", "Median IC", "Std IC", "t-stat", "IC/IR", "Hit Rate"]
        html = (
            f"<table><thead><tr>"
            f"<th>Metric ({method})</th><th>Value</th>"
            f"</tr></thead><tbody>"
        )
        for key, label in zip(keys, labels):
            val = ic_stats_dict.get(key)
            pct = key == "hit_rate"
            html += f"<tr><td>{label}</td><td>{_fmt(val, pct=pct)}</td></tr>"
        html += "</tbody></table>"
        return html
    except Exception as exc:
        return f"<p><em>Table error: {exc}</em></p>"


def table_scorecard(validation_report) -> str:
    """Scorecard table: each sub-check with value, threshold, PASS/FAIL."""
    try:
        vr = validation_report

        def _pass_cell(passed: bool) -> str:
            style = _PASS_STYLE if passed else _FAIL_STYLE
            label = "PASS" if passed else "FAIL"
            return f'<span style="{style}">{label}</span>'

        rows = []

        # Sharpe t-stat
        t = vr.sharpe.t_stat
        rows.append(("Sharpe t-stat", _fmt(t), "≥ 2.0", _pass_cell(not np.isnan(t) and t >= 2.0)))

        # PSR
        psr = vr.psr.psr
        rows.append(("PSR", _fmt(psr), "≥ 0.95", _pass_cell(not np.isnan(psr) and psr >= 0.95)))

        # DSR
        dsr = vr.dsr.dsr
        rows.append(("DSR", _fmt(dsr), "≥ 0.95", _pass_cell(not np.isnan(dsr) and dsr >= 0.95)))

        # Bootstrap CI
        ci_low = vr.bootstrap.ci_low
        ci_high = vr.bootstrap.ci_high
        rows.append((
            "Bootstrap CI",
            f"[{_fmt(ci_low)}, {_fmt(ci_high)}]",
            "CI low > 0",
            _pass_cell(not np.isnan(ci_low) and ci_low > 0),
        ))

        # MC p-value
        mc_p = vr.mc.p_value
        rows.append(("MC p-value", _fmt(mc_p), "≤ 0.05", _pass_cell(not np.isnan(mc_p) and mc_p <= 0.05)))

        # Walk-forward efficiency
        wfe = vr.walkforward.wf_efficiency
        rows.append(("WF Efficiency", _fmt(wfe), "≥ 0.5", _pass_cell(not np.isnan(wfe) and wfe >= 0.5)))

        # WF frac OOS positive
        frac = vr.walkforward.frac_folds_oos_positive
        rows.append(("Frac OOS Positive", _fmt(frac), "≥ 0.5", _pass_cell(not np.isnan(frac) and frac >= 0.5)))

        # Stability
        stab_frac = vr.stability.frac_positive
        rows.append(("Stability (frac+)", _fmt(stab_frac), "≥ 0.7",
                      _pass_cell(not np.isnan(stab_frac) and stab_frac >= 0.7)))

        stab_z = vr.stability.default_zscore
        rows.append(("Stability (|z-score|)", _fmt(abs(stab_z) if not np.isnan(stab_z) else np.nan),
                      "≤ 1.5",
                      _pass_cell(not np.isnan(stab_z) and abs(stab_z) <= 1.5)))

        # Overall
        overall = vr.verdict.overall
        overall_style = _PASS_STYLE if overall else _FAIL_STYLE
        overall_label = "PASS" if overall else "FAIL"

        html = (
            "<table><thead><tr>"
            "<th>Check</th><th>Value</th><th>Threshold</th><th>Result</th>"
            "</tr></thead><tbody>"
        )
        for chk, val, thr, result_cell in rows:
            html += f"<tr><td>{chk}</td><td>{val}</td><td>{thr}</td><td>{result_cell}</td></tr>"

        html += (
            f"<tr style='border-top:2px solid #374151;'>"
            f"<td colspan='3'><strong>Overall Verdict</strong></td>"
            f"<td><span style='{overall_style}'>{overall_label}</span></td>"
            f"</tr>"
        )
        if vr.verdict.notes:
            for note in vr.verdict.notes:
                html += f"<tr><td colspan='4' style='color:#6B7280;font-size:0.85em;'>{note}</td></tr>"

        html += "</tbody></table>"
        return html
    except Exception as exc:
        return f"<p><em>Scorecard table error: {exc}</em></p>"


def table_weights_latest(engine_result: EngineResult) -> str:
    """Latest-period weights table."""
    try:
        hw = engine_result.historical_weights
        if hw is None or hw.empty:
            return "<p><em>No weights available.</em></p>"
        last = hw.iloc[-1].sort_values(ascending=False)
        last = last[last > 1e-6]

        html = (
            "<table><thead><tr>"
            "<th>Country</th><th>Weight</th>"
            "</tr></thead><tbody>"
        )
        for country, w in last.items():
            html += f"<tr><td>{country}</td><td>{_fmt(w, pct=True)}</td></tr>"
        html += "</tbody></table>"
        return html
    except Exception as exc:
        return f"<p><em>Table error: {exc}</em></p>"


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f9fafb; color: #111827; padding: 24px 32px; }
h1 { font-size: 1.6rem; font-weight: 700; color: #1e3a5f;
     border-bottom: 3px solid #2563EB; padding-bottom: 8px; margin-bottom: 24px; }
h2 { font-size: 1.15rem; font-weight: 600; color: #1f2937;
     background: #e5e7eb; padding: 6px 12px; border-radius: 4px;
     margin: 32px 0 12px 0; }
img { max-width: 100%; height: auto; border-radius: 6px; margin-bottom: 12px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.12); }
table { border-collapse: collapse; width: 100%; margin-bottom: 16px;
        font-size: 0.88rem; }
th { background: #374151; color: #f9fafb; padding: 6px 10px; text-align: left; }
td { padding: 5px 10px; border-bottom: 1px solid #e5e7eb; }
tr:nth-child(even) td { background: #f3f4f6; }
p { margin: 8px 0; }
.meta { font-size: 0.82rem; color: #6b7280; margin-bottom: 20px; }
"""


def render_html(title: str, sections: list[tuple[str, str]]) -> str:
    """Assemble a minimal self-contained HTML page.

    Parameters
    ----------
    title:
        Page h1 title.
    sections:
        List of (section_title, html_body) tuples.

    Returns
    -------
    Full HTML string.
    """
    body_parts = [f"<h1>{title}</h1>"]
    for sec_title, sec_body in sections:
        body_parts.append(f"<h2>{sec_title}</h2>\n{sec_body}")

    body = "\n".join(body_parts)
    return (
        "<!DOCTYPE html>"
        "<html lang='en'>"
        "<head>"
        "<meta charset='UTF-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title>"
        f"<style>{_CSS}</style>"
        "</head>"
        f"<body>{body}</body>"
        "</html>"
    )


def _img_tag(b64: Optional[str], alt: str = "") -> str:
    if b64 is None:
        return ""
    return f"<img src='data:image/png;base64,{b64}' alt='{alt}'/>"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_report(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    output_path: str,
    validation_report=None,
    contributions: Optional[dict] = None,
    ic_methods: tuple = ("absolute", "relative"),
    base_weights: Optional[pd.DataFrame] = None,
    cost_bps: Optional[pd.Series] = None,
) -> dict:
    """Build a comprehensive HTML backtest report and write it to *output_path*.

    Parameters
    ----------
    scores:
        Normalized country scores (dates x countries, no benchmark column).
    prices:
        Price levels (dates x countries + benchmark).
    cfg:
        Backtest configuration.
    output_path:
        File path for the output HTML.
    validation_report:
        Optional :class:`ValidationReport` from ``compute_validation``.
        When provided, the "Validation Scorecard" section is included.
    contributions:
        Optional ``{category: DataFrame(dates x countries)}`` from composite.
        When provided, the "Score Decomposition" section is included.
    ic_methods:
        IC methods to compute and display (tuple of 'absolute'/'relative').
    base_weights:
        Optional cap-weight base (dates x countries) threaded to the Engine —
        required when ``cfg.weighting_method == 'Cap_Tilt'``.
    cost_bps:
        Optional per-country one-way trading cost in bps (index = country
        names) threaded to the Engine.  ``None`` -> legacy flat
        ``cfg.transaction_cost_bps``.

    Returns
    -------
    dict with keys: ``path`` (str), ``n_figures`` (int), ``summary`` (dict).
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Run engine
    # ------------------------------------------------------------------
    engine_result = Engine(
        scores, prices, cfg, base_weights=base_weights, cost_bps=cost_bps
    ).run()

    # ------------------------------------------------------------------
    # 2. Equal-weight null (country columns only)
    # ------------------------------------------------------------------
    country_cols = [c for c in scores.columns if c in prices.columns and c != cfg.bmk]
    try:
        eqw_equity = equal_weight_buy_hold(prices[country_cols])
    except Exception:
        eqw_equity = None

    # ------------------------------------------------------------------
    # 3. Metrics summary (daily, ppy=252)
    # ------------------------------------------------------------------
    dr = engine_result.daily_returns
    db = engine_result.daily_bmk_returns

    daily_summary: dict = {}
    period_summary: dict = {}

    if dr is not None and len(dr) > 1:
        daily_summary = summary(dr, db if db is not None else pd.Series(0, index=dr.index), 252)

    # Period-level summary
    pr = engine_result.period_results
    if len(pr) > 1:
        ppy_period = 252 / cfg.periodicity
        period_summary = summary(
            pr["portfolio_return_net"],
            pr["bmk_return"],
            ppy_period,
        )

    # ------------------------------------------------------------------
    # 4. IC analysis
    # ------------------------------------------------------------------
    ic_results: dict = {}  # method -> ic_df
    ic_stats_results: dict = {}  # method -> stats dict
    for method in ic_methods:
        try:
            ic_df = information_coefficient(scores, prices, cfg.periodicity, method)
            ic_results[method] = ic_df
            if "IC" in ic_df.columns:
                ic_stats_results[method] = ic_stats(ic_df["IC"])
            else:
                ic_stats_results[method] = {}
        except Exception:
            ic_results[method] = pd.DataFrame(columns=["IC", "n_countries"])
            ic_stats_results[method] = {}

    # ------------------------------------------------------------------
    # 5. Build figures
    # ------------------------------------------------------------------
    n_figures = 0
    sections: list[tuple[str, str]] = []

    # --- Performance section ---
    perf_parts = []

    b64_cum = fig_cumulative(engine_result, eqw_equity)
    if b64_cum:
        perf_parts.append(_img_tag(b64_cum, "Cumulative Return"))
        n_figures += 1

    b64_dd = fig_drawdown(engine_result)
    if b64_dd:
        perf_parts.append(_img_tag(b64_dd, "Drawdown"))
        n_figures += 1

    b64_roll = fig_rolling_12m(engine_result)
    if b64_roll:
        perf_parts.append(_img_tag(b64_roll, "Rolling 12m"))
        n_figures += 1

    b64_wts = fig_weights(engine_result)
    if b64_wts:
        perf_parts.append(_img_tag(b64_wts, "Weights"))
        n_figures += 1

    perf_parts.append("<h3 style='margin:12px 0 6px'>Per-Year Performance</h3>")
    perf_parts.append(table_per_year(dr, db))
    perf_parts.append("<h3 style='margin:12px 0 6px'>Latest Weights</h3>")
    perf_parts.append(table_weights_latest(engine_result))

    sections.append(("Performance", "\n".join(perf_parts)))

    # --- Risk section ---
    risk_parts = [table_risk(daily_summary, period_summary)]
    sections.append(("Risk", "\n".join(risk_parts)))

    # --- IC Analysis section ---
    ic_parts = []
    for method in ic_methods:
        ic_df = ic_results.get(method, pd.DataFrame())
        b64_ic = fig_ic(ic_df)
        if b64_ic:
            ic_parts.append(f"<p><strong>IC ({method})</strong></p>")
            ic_parts.append(_img_tag(b64_ic, f"IC {method}"))
            n_figures += 1
        if method in ic_stats_results:
            ic_parts.append(table_ic_stats(ic_stats_results[method], method))

    if not ic_parts:
        ic_parts.append("<p><em>No IC data available.</em></p>")
    sections.append(("IC Analysis", "\n".join(ic_parts)))

    # --- Score Decomposition (optional) ---
    if contributions is not None:
        decomp_parts = []
        b64_latest = fig_contributions_latest(contributions)
        if b64_latest:
            decomp_parts.append("<p><strong>Latest Cross-Section</strong></p>")
            decomp_parts.append(_img_tag(b64_latest, "Contributions Latest"))
            n_figures += 1

        # Show time series for each country (cap at 6 for readability)
        countries_to_show = list(scores.columns)[:6]
        for country in countries_to_show:
            b64_c = fig_contributions(contributions, country)
            if b64_c:
                decomp_parts.append(_img_tag(b64_c, f"Contributions {country}"))
                n_figures += 1

        if not decomp_parts:
            decomp_parts.append("<p><em>No contribution data available.</em></p>")
        sections.append(("Score Decomposition", "\n".join(decomp_parts)))

    # --- Validation Scorecard (optional) ---
    if validation_report is not None:
        sc_parts = [table_scorecard(validation_report)]
        sections.append(("Validation Scorecard", "\n".join(sc_parts)))

    # ------------------------------------------------------------------
    # 6. Render HTML
    # ------------------------------------------------------------------
    title = "Country Rotation — Backtest Report"
    html = render_html(title, sections)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return {
        "path": output_path,
        "n_figures": n_figures,
        "summary": daily_summary if daily_summary else period_summary,
    }
