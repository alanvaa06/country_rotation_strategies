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

__all__ = [
    "acwi_section_body",
    "build_strategy_pane",
    "evidence_grade",
    "render_dashboard",
    "stat_cards",
    "verdict_banner",
]


# ---------------------------------------------------------------------------
# Verdict-derived components (evidence grade, banner, stat cards, scorecard)
# ---------------------------------------------------------------------------

_CHECK_LABELS = {
    "no_overfitting": "No Overfitting",
    "param_stable": "Param Stable",
    "statistically_significant": "Statistically Significant",
}

# Gate table: (key, label, stats key, threshold text, predicate, kind, decimals)
# kind 'core'  = selection-skill / robustness evidence (MC, WF, OOS, stability)
# kind 'power' = sample-size-sensitive certification gates (DSR, PSR, t, CI)
_GATE_SPECS = (
    ("mc_p", "MC p", "mc_p_value", "≤ 0.05",
     lambda x: x <= 0.05, "core", 3),
    ("wf_eff", "WF eff", "wf_efficiency", "≥ 0.5",
     lambda x: x >= 0.5, "core", 2),
    ("oos_folds", "OOS folds", "frac_oos_positive", "≥ 0.5",
     lambda x: x >= 0.5, "core", 2),
    ("dsr", "DSR", "dsr", "≥ 0.95",
     lambda x: x >= 0.95, "power", 3),
    ("psr", "PSR", "psr", "≥ 0.95",
     lambda x: x >= 0.95, "power", 3),
    ("active_t", "Active t", "sharpe_t_stat", "≥ 2.0",
     lambda x: x >= 2.0, "power", 2),
    ("ci_low", "Boot CI low", "__ci_low__", "> 0",
     lambda x: x > 0, "power", 3),
    ("stability", "Stability", "stability_frac_positive", "≥ 0.7",
     lambda x: x >= 0.7, "core", 2),
)

_TIER_LABELS = {
    "certified": "CERTIFIED",
    "power_limited": "NOT YET CERTIFIED — power-limited",
    "weak": "WEAK EVIDENCE",
    "negative": "NEGATIVE",
}

_TIER_NOTES = {
    "certified": (
        "All statistical gates pass — selection skill and realized "
        "performance certified at the pre-registered thresholds."
    ),
    "power_limited": (
        "Selection skill statistically significant; sample too short to "
        "certify the realized IR — no overfitting signatures detected "
        "(see forensics)."
    ),
    "weak": (
        "Core evidence incomplete — selection skill is not yet "
        "distinguishable from chance on this sample."
    ),
    "negative": (
        "No demonstrated selection skill — realized active performance "
        "is non-positive and the Monte-Carlo null is not rejected."
    ),
}


def _gate_value(stats: dict, stats_key: str) -> float:
    if stats_key == "__ci_low__":
        ci = stats.get("bootstrap_ci") or [None, None]
        v = ci[0]
    else:
        v = stats.get(stats_key)
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def evidence_grade(verdict: dict) -> dict:
    """Evidence grade for one strategy verdict (neutral, power-aware tiers).

    Tier rules (evaluated on ``verdict['stats']``; NaN/missing values fail
    their gate):

    1. ``certified`` (\"CERTIFIED\") — every gate passes: MC p ≤ 0.05,
       WF efficiency ≥ 0.5, frac OOS positive ≥ 0.5, DSR ≥ 0.95,
       PSR ≥ 0.95, active Sharpe t ≥ 2.0, bootstrap CI low > 0,
       stability frac-positive ≥ 0.7.
    2. ``negative`` (\"NEGATIVE\") — annualized active IR
       (``sharpe_ann``) ≤ 0 AND the Monte-Carlo null is not rejected
       (MC p > 0.05): no evidence of skill.
    3. ``power_limited`` (\"NOT YET CERTIFIED — power-limited\") — the
       selection-skill core passes (MC p ≤ 0.05, WF eff ≥ 0.5,
       frac OOS ≥ 0.5, stability ≥ 0.7) and IR > 0, but one or more
       sample-size-sensitive power gates fail (DSR / PSR / active t /
       bootstrap CI low). The signal is real at MC/WF standards; the sample
       is too short to certify the realized IR.
    4. ``weak`` (\"WEAK EVIDENCE\") — everything else (MC or WF fail with
       non-negative evidence elsewhere, stability break, missing stats).

    Returns
    -------
    dict with keys ``tier``, ``label``, ``note`` and ``gates`` — a list of
    per-gate dicts (``key``, ``label``, ``value``, ``threshold``, ``passed``,
    ``kind``, ``state``). Gate ``state`` is ``pass`` / ``fail`` / ``amber``;
    amber marks power-gate failures in the power-limited tier.
    """
    stats = verdict.get("stats") or {}
    if not stats:
        return {
            "tier": "weak",
            "label": _TIER_LABELS["weak"],
            "note": "No validation statistics available for this strategy.",
            "gates": [],
        }

    gates = []
    for key, label, stats_key, threshold, pred, kind, decimals in _GATE_SPECS:
        val = _gate_value(stats, stats_key)
        passed = bool(not np.isnan(val) and pred(val))
        gates.append({
            "key": key,
            "label": label,
            "value": _fmt(val, decimals=decimals),
            "threshold": threshold,
            "passed": passed,
            "kind": kind,
            "state": "pass" if passed else "fail",
        })
    by_key = {g["key"]: g for g in gates}

    sharpe_ann = _gate_value(stats, "sharpe_ann")
    core_pass = all(g["passed"] for g in gates if g["kind"] == "core")
    all_pass = all(g["passed"] for g in gates)
    mc_pass = by_key["mc_p"]["passed"]

    if all_pass:
        tier = "certified"
    elif (np.isnan(sharpe_ann) or sharpe_ann <= 0) and not mc_pass:
        tier = "negative"
    elif core_pass and sharpe_ann > 0:
        tier = "power_limited"
        for g in gates:  # power-gate fails are statistical power, not skill
            if g["kind"] == "power" and not g["passed"]:
                g["state"] = "amber"
    else:
        tier = "weak"

    return {
        "tier": tier,
        "label": _TIER_LABELS[tier],
        "note": _TIER_NOTES[tier],
        "gates": gates,
    }


def _gate_chips(gates: list) -> str:
    """Per-gate chip row (value + threshold + PASS/fail) for a gate list."""
    return "".join(
        f'<span class="gate-chip gate-{g["state"]}" '
        f'title="{g["label"]}: {g["value"]} (threshold {g["threshold"]})">'
        f'{g["label"]} {g["value"]} <small>{g["threshold"]}</small> '
        f'{"PASS" if g["passed"] else "fail"}</span>'
        for g in gates
    )


def verdict_banner(verdict: dict) -> str:
    """Evidence-grade banner: tier line + per-gate chips + plain-language note.

    Replaces the old binary \"Overall Verdict: PASS/FAIL\" wall — sub-evidence
    is shown per gate so a power-limited certification (MC + walk-forward
    pass, DSR short on sample size) is not misread as overfitting.
    """
    grade = evidence_grade(verdict)
    chips = _gate_chips(grade["gates"])
    chips_html = f'<div class="gate-chips">{chips}</div>' if chips else ""
    return (
        f'<div class="banner grade-{grade["tier"]}">'
        f'<div class="grade-line"><strong>{grade["label"]}</strong></div>'
        f"{chips_html}"
        f'<p class="grade-note">{grade["note"]}</p>'
        f"</div>"
    )


def _card(label: str, value: str) -> str:
    return (
        '<div class="card">'
        f'<div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>'
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
    return '<div class="cards">' + "".join(cards) + "</div>"


def acwi_section_body(acwi_verdict: dict) -> str:
    """Body HTML for the \"vs ACWI (sole benchmark)\" collapsible section.

    Re-certifies the SAME strategy against the vendor World index
    (ACWI-equivalent global cap-weighted benchmark) instead of the segment
    benchmark: compact evidence-grade line + per-gate chips (reusing
    :func:`evidence_grade`), a stat-card row from the ACWI verdict's
    ``stats`` / ``mandate_stats``, and a benchmark-provenance note.
    No figures — keeps the dashboard build fast.
    """
    grade = evidence_grade(acwi_verdict)
    stats = acwi_verdict.get("stats") or {}
    mandate = acwi_verdict.get("mandate_stats") or {}

    chips = _gate_chips(grade["gates"])
    chips_html = f'<div class="gate-chips">{chips}</div>' if chips else ""
    grade_html = (
        f'<div class="banner acwi-grade grade-{grade["tier"]}">'
        f'<div class="grade-line"><strong>{grade["label"]}</strong> '
        '<small style="font-weight:400;">vs ACWI</small></div>'
        f"{chips_html}"
        "</div>"
    )

    ci = stats.get("bootstrap_ci") or [None, None]
    ci_txt = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"
    cards = [
        _card("IR (active, ann.)", _fmt(stats.get("sharpe_ann"))),
        _card("Active t-stat", _fmt(stats.get("sharpe_t_stat"))),
        _card("MC p-value", _fmt(stats.get("mc_p_value"), decimals=3)),
        _card("DSR", _fmt(stats.get("dsr"), decimals=3)),
        _card("Bootstrap CI", ci_txt),
        _card("Tracking Error", _fmt(mandate.get("tracking_error"), pct=True)),
        _card("Beta", _fmt(mandate.get("beta"))),
    ]
    cards_html = '<div class="cards">' + "".join(cards) + "</div>"

    note = (
        '<p class="grade-note acwi-note">Benchmark for this section is the '
        "vendor World index — ACWI-equivalent (88/12 DM/EM composition "
        "verified) — used as the sole benchmark for selection, mandate "
        "comparison and certification.</p>"
    )
    return grade_html + cards_html + note


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
        "<td colspan='3'><strong>All-Gates Aggregate</strong> "
        "(see evidence grade above for the power-aware read)</td>"
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

def _section(title: str, body_html: str, open: bool = False) -> str:
    """Collapsible section: <button> header (keyboard accessible) + hidden body.

    Sections start COLLAPSED by default (display:none, aria-expanded=false);
    pass ``open=True`` for a section that should start expanded (the shell's
    ``openSecs`` map must then be pre-seeded with its normalized title). The
    shell's ``toggleSec`` JS flips visibility and the ▸/▾ arrow. Banner and
    stat cards stay outside sections so they are always visible.

    The title lives in its own ``<span class="sec-title">`` so the shell JS
    can key open/closed state by title and persist it across pane switches.
    """
    expanded = "true" if open else "false"
    arrow = "▾" if open else "▸"
    display = "block" if open else "none"
    return (
        '<section class="sec">'
        f'<button class="sec-toggle" type="button" aria-expanded="{expanded}" '
        'onclick="toggleSec(this)">'
        f'<span class="sec-arrow">{arrow}</span> '
        '<span class="sec-title">' + title + "</span></button>"
        f'<div class="sec-body" style="display:{display}">' + body_html
        + "</div></section>"
    )


def build_strategy_pane(
    scores: pd.DataFrame,
    prices_with_bmk: pd.DataFrame,
    cfg: BacktestConfig,
    base_weights: Optional[pd.DataFrame],
    verdict: dict,
    contributions: Optional[dict],
    acwi_verdict: Optional[dict] = None,
    bmk_label: Optional[str] = None,
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
    acwi_verdict:
        Optional ACWI-relative verdict JSON (same strategy certified against
        the vendor World index as sole benchmark). When provided, a
        collapsible \"vs ACWI (sole benchmark)\" section is appended.
    bmk_label:
        Human-readable identity of the pane's PRIMARY benchmark (the one all
        headline stats are measured against), e.g. \"Vendor EM index — MSCI
        Emerging Markets equivalent\". Rendered as an always-visible badge
        under the verdict banner.

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

    # Always visible: evidence-grade banner + benchmark identity + stat cards.
    parts.append(verdict_banner(verdict))
    if bmk_label:
        parts.append(
            '<div class="bmk-badge"><strong>Benchmark:</strong> '
            f"{bmk_label}</div>"
        )
    parts.append(stat_cards(verdict))

    # --- Performance (absolute + relative) ---
    perf: list[str] = []
    for b64, alt in (
        (fig_cumulative(engine_result, eqw_equity), "Cumulative Return"),
        (fig_drawdown(engine_result), "Drawdown"),
        (fig_rolling_12m(engine_result), "Rolling 12m Return"),
    ):
        if b64:
            perf.append(_img_tag(b64, alt))
    parts.append(_section("Performance", "\n".join(perf)))

    parts.append(_section("Per-Year Performance", table_per_year(dr, db)))

    # --- Risk ---
    parts.append(_section(
        "Risk (absolute + relative)",
        table_risk(daily_summary, period_summary),
    ))

    # --- Weights ---
    wts: list[str] = []
    b64_w = fig_weights(engine_result)
    if b64_w:
        wts.append(_img_tag(b64_w, "Weights"))
    wts.append("<h4 style='margin:8px 0 4px;'>Latest Weights</h4>")
    wts.append(table_weights_latest(engine_result))
    parts.append(_section("Portfolio Weights", "\n".join(wts)))

    # --- IC analysis ---
    ic_parts: list[str] = []
    for method in ("absolute", "relative"):
        b64_ic = fig_ic(ic_results.get(method, pd.DataFrame()))
        if b64_ic:
            ic_parts.append(f"<p><strong>IC ({method})</strong></p>")
            ic_parts.append(_img_tag(b64_ic, f"IC {method}"))
        if ic_stats_results.get(method):
            ic_parts.append(table_ic_stats(ic_stats_results[method], method))
    parts.append(_section("IC Analysis", "\n".join(ic_parts)))

    # --- Score building-block decomposition ---
    if contributions:
        decomp: list[str] = []
        b64_latest = fig_contributions_latest(contributions)
        if b64_latest:
            decomp.append("<p><strong>Latest Cross-Section</strong></p>")
            decomp.append(_img_tag(b64_latest, "Contributions Latest"))

        top_country = _top_country(engine_result, scores)
        if top_country is not None:
            b64_c = fig_contributions(contributions, top_country)
            if b64_c:
                decomp.append(
                    f"<p><strong>Through Time — Top Holding "
                    f"({top_country})</strong></p>"
                )
                decomp.append(_img_tag(b64_c, f"Contributions {top_country}"))
        parts.append(_section(
            "Score Building-Block Decomposition", "\n".join(decomp)
        ))

    # --- Validation scorecard ---
    parts.append(_section("Validation Scorecard", scorecard_from_verdict(verdict)))

    # --- ACWI-relative certification (no figures; verdict-only) ---
    if acwi_verdict is not None:
        parts.append(_section(
            "vs ACWI (sole benchmark)", acwi_section_body(acwi_verdict)
        ))

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
# Dashboard shell: viewport-fit app frame, segment tabs + strategy toggle,
# identifier (reference-universe) bar, collapsible-section JS (vanilla JS)
# ---------------------------------------------------------------------------

_DASH_CSS = _CSS + """
/* --- Viewport-fit app shell: the PAGE never scrolls, only .dash-content --- */
html, body { height:100vh; overflow:hidden; }
body { display:flex; flex-direction:column; padding:10px 24px 0 24px; }
.dash-header { flex:0 0 auto; }
.dash-content { flex:1 1 auto; min-height:0; overflow-y:auto;
                padding:0 2px 24px 0; }
h1 { font-size:1.25rem; margin-bottom:8px; padding-bottom:6px; }
/* --- Segment tabs + strategy toggle --- */
.seg-tabs { display:flex; gap:6px; margin:4px 0 6px 0; }
.seg-tab { font-size:0.95rem; font-weight:600; padding:6px 20px; cursor:pointer;
           border:1px solid #d1d5db; border-radius:6px 6px 0 0;
           background:#e5e7eb; color:#374151; }
.seg-tab.active { background:#2563EB; color:#fff; border-color:#2563EB; }
.strat-toggle { display:flex; gap:6px; margin:0 0 8px 0; }
.strat-btn { font-size:0.85rem; padding:5px 14px; cursor:pointer;
             border:1px solid #d1d5db; border-radius:999px;
             background:#f3f4f6; color:#374151; }
.strat-btn.active { background:#1e3a5f; color:#fff; border-color:#1e3a5f; }
.pane { background:#fff; border:1px solid #e5e7eb; border-radius:0 6px 6px 6px;
        padding:12px 20px 16px; }
/* --- Evidence-grade banner + per-gate chips --- */
.banner { padding:8px 14px; border-radius:6px; margin:2px 0 10px 0; }
.bmk-badge { font-size:0.8rem; color:#1e3a5f; background:#eef4fb;
             border:1px solid #bfdbfe; border-radius:4px;
             padding:5px 12px; margin:0 0 10px 0; }
.grade-certified { background:#d1fae5; border:1px solid #10b981; }
.grade-power_limited { background:#fef3c7; border:1px solid #f59e0b; }
.grade-weak { background:#f3f4f6; border:1px solid #9ca3af; }
.grade-negative { background:#fee2e2; border:1px solid #ef4444; }
.grade-line { font-size:1rem; font-weight:700; color:#111827; }
.grade-note { font-size:0.78rem; color:#374151; margin:5px 0 0 0; }
.gate-chips { display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }
.gate-chip { font-size:0.7rem; font-weight:600; padding:2px 8px;
             border-radius:999px; white-space:nowrap; }
.gate-chip small { font-weight:400; opacity:0.75; }
.gate-pass { background:#d1fae5; color:#065f46; border:1px solid #10b981; }
.gate-fail { background:#fee2e2; color:#991b1b; border:1px solid #ef4444; }
.gate-amber { background:#fef3c7; color:#92400e; border:1px solid #f59e0b; }
/* --- Dense stat-card grid (<= 2 rows at 1080p) --- */
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(118px,1fr));
         gap:6px; margin:0 0 10px 0; }
.card { background:#fff; border:1px solid #e5e7eb; border-radius:5px;
        padding:4px 8px; }
.card-label { font-size:0.62rem; color:#6b7280; white-space:nowrap; }
.card-value { font-size:0.86rem; font-weight:600; color:#111827;
              white-space:nowrap; }
/* --- Collapsible sections (default collapsed) --- */
.sec-toggle { display:block; width:100%; text-align:left; font-size:0.95rem;
              font-weight:600; color:#1f2937; background:#eef2f7;
              border:1px solid #d1d5db; border-radius:4px; padding:7px 12px;
              margin:8px 0 4px 0; cursor:pointer; }
.sec-toggle:hover { background:#e2e8f0; }
.sec-toggle:focus-visible { outline:2px solid #2563EB; outline-offset:1px; }
.sec-arrow { display:inline-block; width:1em; }
.sec-body { padding:2px 4px 8px 4px; }
/* --- ACWI-relative certification section --- */
.acwi-grade { margin:2px 0 8px 0; }
.acwi-note { margin-top:2px; }
"""

_DASH_JS = """
var currentSeg = "%(default_seg)s";
var currentStrat = "%(default_strat)s";
// Section open/closed state keyed by normalized section TITLE — persists
// across strategy/segment switches (an open "Performance" stays open).
var openSecs = {};

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

function secTitle(btn) {
  // Normalized state key for a section header (lowercase trimmed title).
  var span = btn.querySelector(".sec-title");
  var txt = span ? span.textContent : btn.textContent;
  return txt.replace(/[\\u25B8\\u25BE]/g, "").trim().toLowerCase();
}

function applySecState(btn, open) {
  // Render one section header + body to the given open/closed state.
  var body = btn.nextElementSibling;
  if (!body) { return; }
  body.style.display = open ? "block" : "none";
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  var arrow = btn.querySelector(".sec-arrow");
  if (arrow) { arrow.textContent = open ? "\\u25BE" : "\\u25B8"; }
}

function showPane() {
  var panes = document.querySelectorAll(".pane");
  for (var i = 0; i < panes.length; i++) {
    panes[i].style.display = "none";
  }
  var target = document.getElementById(currentSeg + "-" + currentStrat);
  if (target) {
    target.style.display = "block";
    // Re-apply remembered section state to THIS pane's toggles.
    var toggles = target.querySelectorAll(".sec-toggle");
    for (var j = 0; j < toggles.length; j++) {
      applySecState(toggles[j], !!openSecs[secTitle(toggles[j])]);
    }
  }
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

function toggleSec(btn) {
  // Flip the remembered state for this section title, then render.
  var title = secTitle(btn);
  openSecs[title] = !openSecs[title];
  applySecState(btn, openSecs[title]);
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
        '<header class="dash-header">'
        f"<h1>{title}</h1>"
        f'<div class="seg-tabs">{seg_tabs}</div>'
        f'<div class="strat-toggle">{strat_btns}</div>'
        "</header>"
        '<main class="dash-content">'
        + "\n".join(panes)
        + "</main>"
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
