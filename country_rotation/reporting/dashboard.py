"""Multi-strategy HTML dashboard builder.

One self-contained HTML file with segment tabs (World / DM / EM) and a
per-segment strategy toggle.  Each (segment, strategy) pane is a complete
mini-report: verdict banner, stat cards, performance / risk figures and
tables, IC analysis, score building-block decomposition and a
verdict-derived validation scorecard.

All figures and tables are REUSED from :mod:`country_rotation.reporting.report`
(matplotlib Agg backend, inline base64 PNGs) — this module only orchestrates
panes and renders the tab/toggle shell with vanilla JS (no external deps).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from country_rotation.backtest.benchmarks import equal_weight_buy_hold
from country_rotation.backtest.engine import Engine
from country_rotation.backtest.ic import ic_stats, information_coefficient
from country_rotation.backtest.metrics import summary
from country_rotation.config import BacktestConfig
from country_rotation.reporting.report import (
    _CSS,
    _FAIL_STYLE,
    _PASS_STYLE,
    _fmt,
    _img_tag,
    fig_contributions,
    fig_contributions_latest,
    fig_cumulative,
    fig_drawdown,
    fig_ic,
    fig_rolling_12m,
    fig_weights,
    table_ic_stats,
    table_per_year,
    table_risk,
    table_weights_latest,
)

__all__ = ["build_strategy_pane", "render_dashboard"]


# ---------------------------------------------------------------------------
# Verdict-derived components (banner, stat cards, scorecard table)
# ---------------------------------------------------------------------------

_BANNER_PASS = (
    "background:#d1fae5;border:1px solid #10b981;color:#065f46;"
)
_BANNER_FAIL = (
    "background:#fee2e2;border:1px solid #ef4444;color:#991b1b;"
)
_CHECK_LABELS = {
    "no_overfitting": "No Overfitting",
    "param_stable": "Param Stable",
    "statistically_significant": "Statistically Significant",
}


def _check_pill(name: str, passed: bool) -> str:
    style = _PASS_STYLE if passed else _FAIL_STYLE
    label = _CHECK_LABELS.get(name, name)
    word = "PASS" if passed else "FAIL"
    return (
        f'<span style="{style}margin-right:8px;" title="{name}">'
        f"{label}: {word}</span>"
    )


def verdict_banner(verdict: dict) -> str:
    """Banner: overall verdict + the three scorecard checks, colored."""
    v = verdict.get("verdict", {}) or {}
    overall = bool(v.get("overall", False))
    checks = v.get("checks", {}) or {}

    style = _BANNER_PASS if overall else _BANNER_FAIL
    word = "PASS" if overall else "FAIL"
    pills = "".join(
        _check_pill(name, bool(passed)) for name, passed in sorted(checks.items())
    )
    return (
        f'<div class="banner" style="{style}padding:10px 14px;border-radius:6px;'
        f'margin:8px 0 16px 0;">'
        f'<strong style="font-size:1.05rem;">Overall Verdict: {word}</strong>'
        f'<span style="margin-left:16px;">{pills}</span>'
        f"</div>"
    )


def _card(label: str, value: str) -> str:
    return (
        '<div class="card" style="background:#fff;border:1px solid #e5e7eb;'
        'border-radius:6px;padding:8px 12px;min-width:130px;'
        'box-shadow:0 1px 2px rgba(0,0,0,0.06);">'
        f'<div style="font-size:0.72rem;color:#6b7280;">{label}</div>'
        f'<div style="font-size:1.05rem;font-weight:600;color:#111827;">{value}</div>'
        "</div>"
    )


def stat_cards(verdict: dict) -> str:
    """Headline stat cards from the verdict JSON (stats / mandate / IC)."""
    stats = verdict.get("stats") or {}
    mandate = verdict.get("mandate_stats") or {}
    ic_rel = (verdict.get("composite_ic") or {}).get("relative") or {}

    ci = stats.get("bootstrap_ci") or [None, None]
    ci_txt = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"

    cards = [
        _card("IR (active, ann.)", _fmt(stats.get("sharpe_ann"))),
        _card("Active t-stat", _fmt(stats.get("sharpe_t_stat"))),
        _card("PSR", _fmt(stats.get("psr"), decimals=3)),
        _card("DSR", _fmt(stats.get("dsr"), decimals=3)),
        _card("MC p-value", _fmt(stats.get("mc_p_value"), decimals=3)),
        _card("Bootstrap CI", ci_txt),
        _card("NW-t vs EqW", _fmt(stats.get("nw_t_vs_eqw"))),
        _card("Tracking Error", _fmt(mandate.get("tracking_error"), pct=True)),
        _card("Beta", _fmt(mandate.get("beta"))),
        _card("IC rel (mean)", _fmt(ic_rel.get("mean_ic"), decimals=4)),
        _card("IC rel (t)", _fmt(ic_rel.get("t_stat"))),
    ]
    return (
        '<div class="cards" style="display:flex;flex-wrap:wrap;gap:10px;'
        'margin-bottom:16px;">' + "".join(cards) + "</div>"
    )


def _pass_cell(passed: Optional[bool]) -> str:
    if passed is None:
        return "<span style='color:#6b7280;'>n/a</span>"
    style = _PASS_STYLE if passed else _FAIL_STYLE
    return f'<span style="{style}">{"PASS" if passed else "FAIL"}</span>'


def _safe(stats: dict, key: str) -> float:
    v = stats.get(key)
    return float(v) if v is not None else float("nan")


def scorecard_from_verdict(verdict: dict) -> str:
    """Scorecard-style table rebuilt from verdict['stats'] + checks + notes.

    Mirrors the thresholds of ``report.table_scorecard`` but consumes the
    serialized verdict JSON instead of a live ``ValidationReport``.
    """
    v = verdict.get("verdict", {}) or {}
    stats = verdict.get("stats")
    if not stats:
        return "<p><em>No validation stats in verdict.</em></p>"

    t = _safe(stats, "sharpe_t_stat")
    psr = _safe(stats, "psr")
    dsr = _safe(stats, "dsr")
    mc_p = _safe(stats, "mc_p_value")
    wfe = _safe(stats, "wf_efficiency")
    frac = _safe(stats, "frac_oos_positive")
    ci = stats.get("bootstrap_ci") or [None, None]
    ci_low = ci[0] if ci[0] is not None else float("nan")
    ci_high = ci[1] if ci[1] is not None else float("nan")
    stab_frac = _safe(stats, "stability_frac_positive")
    stab_z = _safe(stats, "stability_default_zscore")

    rows = [
        ("Sharpe t-stat (active)", _fmt(t), "≥ 2.0",
         _pass_cell(not np.isnan(t) and t >= 2.0)),
        ("PSR", _fmt(psr, decimals=3), "≥ 0.95",
         _pass_cell(not np.isnan(psr) and psr >= 0.95)),
        ("DSR", _fmt(dsr, decimals=3), "≥ 0.95",
         _pass_cell(not np.isnan(dsr) and dsr >= 0.95)),
        ("Bootstrap CI", f"[{_fmt(ci_low)}, {_fmt(ci_high)}]", "CI low > 0",
         _pass_cell(not np.isnan(ci_low) and ci_low > 0)),
        ("MC p-value", _fmt(mc_p, decimals=3), "≤ 0.05",
         _pass_cell(not np.isnan(mc_p) and mc_p <= 0.05)),
        ("WF Efficiency", _fmt(wfe), "≥ 0.5",
         _pass_cell(not np.isnan(wfe) and wfe >= 0.5)),
        ("Frac OOS Positive", _fmt(frac), "≥ 0.5",
         _pass_cell(not np.isnan(frac) and frac >= 0.5)),
        ("Stability (frac+)", _fmt(stab_frac), "≥ 0.7",
         _pass_cell(not np.isnan(stab_frac) and stab_frac >= 0.7)),
        ("Stability (|z-score|)",
         _fmt(abs(stab_z) if not np.isnan(stab_z) else float("nan")), "≤ 1.5",
         _pass_cell(not np.isnan(stab_z) and abs(stab_z) <= 1.5)),
    ]

    html = (
        "<table><thead><tr>"
        "<th>Check</th><th>Value</th><th>Threshold</th><th>Result</th>"
        "</tr></thead><tbody>"
    )
    for chk, val, thr, cell in rows:
        html += f"<tr><td>{chk}</td><td>{val}</td><td>{thr}</td><td>{cell}</td></tr>"

    for name, passed in sorted((v.get("checks") or {}).items()):
        label = _CHECK_LABELS.get(name, name)
        html += (
            f"<tr><td><strong>{label}</strong> ({name})</td>"
            f"<td colspan='2'>aggregate check</td>"
            f"<td>{_pass_cell(bool(passed))}</td></tr>"
        )

    overall = bool(v.get("overall", False))
    html += (
        "<tr style='border-top:2px solid #374151;'>"
        "<td colspan='3'><strong>Overall Verdict</strong></td>"
        f"<td>{_pass_cell(overall)}</td></tr>"
    )
    for note in v.get("notes") or []:
        html += (
            f"<tr><td colspan='4' style='color:#6B7280;font-size:0.85em;'>"
            f"{note}</td></tr>"
        )
    html += "</tbody></table>"
    return html


# ---------------------------------------------------------------------------
# Strategy pane
# ---------------------------------------------------------------------------

def _h3(txt: str) -> str:
    return f"<h3 style='margin:18px 0 6px;color:#1f2937;'>{txt}</h3>"


def build_strategy_pane(
    scores: pd.DataFrame,
    prices_with_bmk: pd.DataFrame,
    cfg: BacktestConfig,
    base_weights: Optional[pd.DataFrame],
    verdict: dict,
    contributions: Optional[dict],
) -> str:
    """Build the full HTML fragment for one (segment, strategy) pane.

    Parameters
    ----------
    scores:
        Normalized composite scores (dates x countries, no benchmark column).
    prices_with_bmk:
        Price levels including the ``cfg.bmk`` benchmark column (vendor cap
        index for the mandate comparisons).
    cfg:
        Strategy-specific backtest configuration.
    base_weights:
        Cap-weight base (dates x countries) — required when
        ``cfg.weighting_method == 'Cap_Tilt'``, else None.
    verdict:
        Parsed verdict JSON for this strategy (research_run payload).
    contributions:
        Optional ``{category: DataFrame(dates x countries)}`` score
        building-block decomposition.

    Returns
    -------
    HTML fragment string (no <html>/<body> wrapper).
    """
    # ------------------------------------------------------------------
    # 1. Engine run + equal-weight null
    # ------------------------------------------------------------------
    engine_result = Engine(
        scores, prices_with_bmk, cfg, base_weights=base_weights
    ).run()

    country_cols = [
        c for c in scores.columns if c in prices_with_bmk.columns and c != cfg.bmk
    ]
    try:
        eqw_equity = equal_weight_buy_hold(prices_with_bmk[country_cols])
    except Exception:
        eqw_equity = None

    # ------------------------------------------------------------------
    # 2. Metrics summaries (daily + period)
    # ------------------------------------------------------------------
    dr = engine_result.daily_returns
    db = engine_result.daily_bmk_returns

    daily_summary: dict = {}
    period_summary: dict = {}
    if dr is not None and len(dr) > 1:
        daily_summary = summary(
            dr, db if db is not None else pd.Series(0, index=dr.index), 252
        )
    pr = engine_result.period_results
    if len(pr) > 1:
        period_summary = summary(
            pr["portfolio_return_net"], pr["bmk_return"], 252 / cfg.periodicity
        )

    # ------------------------------------------------------------------
    # 3. IC analysis (absolute + relative)
    # ------------------------------------------------------------------
    ic_results: dict = {}
    ic_stats_results: dict = {}
    for method in ("absolute", "relative"):
        try:
            ic_df = information_coefficient(
                scores, prices_with_bmk, cfg.periodicity, method
            )
            ic_results[method] = ic_df
            ic_stats_results[method] = (
                ic_stats(ic_df["IC"].dropna()) if "IC" in ic_df.columns else {}
            )
        except Exception:
            ic_results[method] = pd.DataFrame(columns=["IC", "n_countries"])
            ic_stats_results[method] = {}

    # ------------------------------------------------------------------
    # 4. Assemble pane HTML
    # ------------------------------------------------------------------
    parts: list[str] = []

    parts.append(verdict_banner(verdict))
    parts.append(stat_cards(verdict))

    # --- Performance (absolute + relative) ---
    parts.append(_h3("Performance"))
    for b64, alt in (
        (fig_cumulative(engine_result, eqw_equity), "Cumulative Return"),
        (fig_drawdown(engine_result), "Drawdown"),
        (fig_rolling_12m(engine_result), "Rolling 12m Return"),
    ):
        if b64:
            parts.append(_img_tag(b64, alt))

    parts.append(_h3("Per-Year Performance"))
    parts.append(table_per_year(dr, db))

    # --- Risk ---
    parts.append(_h3("Risk (absolute + relative)"))
    parts.append(table_risk(daily_summary, period_summary))

    # --- Weights ---
    parts.append(_h3("Portfolio Weights"))
    b64_w = fig_weights(engine_result)
    if b64_w:
        parts.append(_img_tag(b64_w, "Weights"))
    parts.append("<h4 style='margin:8px 0 4px;'>Latest Weights</h4>")
    parts.append(table_weights_latest(engine_result))

    # --- IC analysis ---
    parts.append(_h3("IC Analysis"))
    for method in ("absolute", "relative"):
        b64_ic = fig_ic(ic_results.get(method, pd.DataFrame()))
        if b64_ic:
            parts.append(f"<p><strong>IC ({method})</strong></p>")
            parts.append(_img_tag(b64_ic, f"IC {method}"))
        if ic_stats_results.get(method):
            parts.append(table_ic_stats(ic_stats_results[method], method))

    # --- Score building-block decomposition ---
    if contributions:
        parts.append(_h3("Score Building-Block Decomposition"))
        b64_latest = fig_contributions_latest(contributions)
        if b64_latest:
            parts.append("<p><strong>Latest Cross-Section</strong></p>")
            parts.append(_img_tag(b64_latest, "Contributions Latest"))

        top_country = _top_country(engine_result, scores)
        if top_country is not None:
            b64_c = fig_contributions(contributions, top_country)
            if b64_c:
                parts.append(
                    f"<p><strong>Through Time — Top Holding "
                    f"({top_country})</strong></p>"
                )
                parts.append(_img_tag(b64_c, f"Contributions {top_country}"))

    # --- Validation scorecard ---
    parts.append(_h3("Validation Scorecard"))
    parts.append(scorecard_from_verdict(verdict))

    return "\n".join(parts)


def _top_country(engine_result, scores: pd.DataFrame) -> Optional[str]:
    """Country with the highest latest portfolio weight (fallback: first score col)."""
    try:
        hw = engine_result.historical_weights
        candidates = [c for c in hw.columns if c in scores.columns]
        if candidates:
            return hw[candidates].iloc[-1].idxmax()
    except Exception:
        pass
    return scores.columns[0] if len(scores.columns) else None


# ---------------------------------------------------------------------------
# Dashboard shell: segment tabs + strategy toggle (vanilla JS)
# ---------------------------------------------------------------------------

_DASH_CSS = _CSS + """
.seg-tabs { display:flex; gap:6px; margin:16px 0 8px 0; }
.seg-tab { font-size:1rem; font-weight:600; padding:8px 22px; cursor:pointer;
           border:1px solid #d1d5db; border-radius:6px 6px 0 0;
           background:#e5e7eb; color:#374151; }
.seg-tab.active { background:#2563EB; color:#fff; border-color:#2563EB; }
.strat-toggle { display:flex; gap:6px; margin:0 0 16px 0; }
.strat-btn { font-size:0.88rem; padding:6px 16px; cursor:pointer;
             border:1px solid #d1d5db; border-radius:999px;
             background:#f3f4f6; color:#374151; }
.strat-btn.active { background:#1e3a5f; color:#fff; border-color:#1e3a5f; }
.pane { background:#fff; border:1px solid #e5e7eb; border-radius:0 6px 6px 6px;
        padding:16px 20px; }
"""

_DASH_JS = """
var currentSeg = "%(default_seg)s";
var currentStrat = "%(default_strat)s";

function _setActive(selector, attr, value) {
  var btns = document.querySelectorAll(selector);
  for (var i = 0; i < btns.length; i++) {
    if (btns[i].getAttribute(attr) === value) {
      btns[i].classList.add("active");
    } else {
      btns[i].classList.remove("active");
    }
  }
}

function showPane() {
  var panes = document.querySelectorAll(".pane");
  for (var i = 0; i < panes.length; i++) {
    panes[i].style.display = "none";
  }
  var target = document.getElementById(currentSeg + "-" + currentStrat);
  if (target) { target.style.display = "block"; }
  _setActive(".seg-tab", "data-seg", currentSeg);
  _setActive(".strat-btn", "data-strat", currentStrat);
}

function selectSeg(seg) {
  // Strategy choice is remembered (currentStrat untouched).
  currentSeg = seg;
  showPane();
}

function selectStrat(strat) {
  currentStrat = strat;
  showPane();
}
"""


def render_dashboard(
    title: str,
    segments: dict,
    default_strategy: str = "cap_tilt",
) -> str:
    """Render the full multi-strategy dashboard HTML document.

    Parameters
    ----------
    title:
        Page title / h1.
    segments:
        ``{segment: {strategy_key: (label, pane_html)}}`` — insertion order
        defines tab/toggle order; the first segment is the default tab.
    default_strategy:
        Strategy key selected on load (falls back to the first key of the
        first segment if absent).

    Returns
    -------
    Self-contained HTML string (inline CSS + vanilla JS, no external deps).
    """
    if not segments:
        raise ValueError("segments must contain at least one segment")

    seg_names = list(segments)
    default_seg = seg_names[0]
    first_keys = list(segments[default_seg])
    default_strat = (
        default_strategy if default_strategy in first_keys else first_keys[0]
    )

    # Strategy toggle labels: union across segments, first-seen label wins.
    strat_labels: dict = {}
    for strat_map in segments.values():
        for key, (label, _) in strat_map.items():
            strat_labels.setdefault(key, label)

    seg_tabs = "".join(
        f'<button class="seg-tab{" active" if seg == default_seg else ""}" '
        f'data-seg="{seg}" onclick="selectSeg(\'{seg}\')">{seg}</button>'
        for seg in seg_names
    )
    strat_btns = "".join(
        f'<button class="strat-btn{" active" if key == default_strat else ""}" '
        f'data-strat="{key}" onclick="selectStrat(\'{key}\')">{label}</button>'
        for key, label in strat_labels.items()
    )

    panes = []
    for seg in seg_names:
        for key, (_, pane_html) in segments[seg].items():
            visible = seg == default_seg and key == default_strat
            display = "block" if visible else "none"
            panes.append(
                f'<div id="{seg}-{key}" class="pane" '
                f'style="display:{display}">\n{pane_html}\n</div>'
            )

    js = _DASH_JS % {"default_seg": default_seg, "default_strat": default_strat}

    body = (
        f"<h1>{title}</h1>"
        f'<div class="seg-tabs">{seg_tabs}</div>'
        f'<div class="strat-toggle">{strat_btns}</div>'
        + "\n".join(panes)
        + f"<script>{js}</script>"
    )
    return (
        "<!DOCTYPE html>"
        "<html lang='en'>"
        "<head>"
        "<meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title>"
        f"<style>{_DASH_CSS}</style>"
        "</head>"
        f"<body>{body}</body>"
        "</html>"
    )
