"""Production dashboard builder (deployed-strategy artifacts -> one HTML).

Consumes one ``outputs/production/run_{YYYYMMDD}/`` artifact folder produced
by ``scripts/production_run.py`` plus the deployed-strategy registry
(``configs/production.json``) and renders a single self-contained HTML file:

* fixed header (data end, next rebalance, run manifest line) over a
  viewport-fit scrollable content area (reuses the
  :mod:`country_rotation.reporting.dashboard` shell patterns);
* one TAB per registry strategy — each pane shows the always-visible
  strategy label + benchmark-identity badge + stat cards, then four
  collapsible sections: **Latest Signal** (default OPEN — the centerpiece),
  **Signal Evolution** (country chips + hand-rolled SVG line chart, vanilla
  JS, no external libs), **Allocations** and **Metrics Detail**;
* section open/closed state persists across tab switches (``openSecs``
  pattern shared with the research dashboard).

Figures come from :mod:`country_rotation.reporting.signal_viz` (Agg backend,
inline base64 PNGs); the evolution chart is drawn client-side from an
embedded JSON payload (``signal_history_payload``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from country_rotation.reporting.dashboard import _DASH_CSS, _card, _section
from country_rotation.reporting.report import _img_tag
from country_rotation.reporting.signal_viz import (
    PALETTE12,
    _ranking_frame,
    fig_allocation_history,
    fig_signal_ranking,
    signal_history_payload,
)

__all__ = [
    "benchmark_identity",
    "build_dashboard",
    "build_strategy_pane",
    "production_stat_cards",
    "render_production_dashboard",
]

_TITLE = "Country Rotation — Production Dashboard"

#: Human-readable identity of each vendor benchmark column.
_BMK_IDENTITY = {
    "World": "Vendor World index — MSCI ACWI equivalent",
    "DM": "Vendor DM index — MSCI World (developed markets) equivalent",
    "EM": "Vendor EM index — MSCI Emerging Markets equivalent",
}


# ---------------------------------------------------------------------------
# Formatting helpers (None/NaN -> em-dash, never a leaked literal)
# ---------------------------------------------------------------------------

def _is_na(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _num(v, decimals: int = 2, signed: bool = False) -> str:
    if _is_na(v):
        return "—"
    sign = "+" if signed else ""
    return f"{float(v):{sign}.{decimals}f}"


def _pct(v, decimals: int = 1, signed: bool = False) -> str:
    if _is_na(v):
        return "—"
    sign = "+" if signed else ""
    return f"{float(v):{sign}.{decimals}%}"


def _text(v) -> str:
    return "—" if v is None else str(v)


def _table(headers: list, rows: list, css_class: str = "") -> str:
    cls = f' class="{css_class}"' if css_class else ""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f"<table{cls}><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Benchmark identity badge
# ---------------------------------------------------------------------------

def benchmark_identity(entry: dict) -> str:
    """Human-readable identity of a registry strategy's benchmark column."""
    column = entry.get("bmk_index") or entry.get("segment") or ""
    return _BMK_IDENTITY.get(column, f"Vendor {column} index")


def _bmk_badge(entry: dict) -> str:
    note = entry.get("note")
    note_html = f' <span class="bmk-note">· {note}</span>' if note else ""
    return (
        '<div class="bmk-badge"><strong>Benchmark:</strong> '
        f"{benchmark_identity(entry)}{note_html}</div>"
    )


# ---------------------------------------------------------------------------
# Stat cards
# ---------------------------------------------------------------------------

def production_stat_cards(metrics: dict, allocations_latest: dict) -> str:
    """Headline stat-card row for one strategy pane.

    Parameters
    ----------
    metrics:
        Parsed ``metrics.json`` (mandate/period stats, IC, turnover,
        last-252d active block).
    allocations_latest:
        Parsed ``allocations_latest.json`` (latest weights + rebalance
        dates) — also feeds the top-5 concentration card (every country
        holds weight under cap_tilt, so #holdings is uninformative).
    """
    mandate = metrics.get("mandate_stats") or {}
    ic_rel = (metrics.get("composite_ic") or {}).get("relative") or {}
    last252 = metrics.get("last_252d_active") or {}
    weights = allocations_latest.get("weights") or {}
    top5 = sum(sorted((float(w) for w in weights.values()), reverse=True)[:5])

    cards = [
        _card("IR (active, ann.)", _num(mandate.get("information_ratio"))),
        _card("Tracking Error", _pct(mandate.get("tracking_error"),
                                     decimals=2)),
        _card("Beta", _num(mandate.get("beta"))),
        _card("Ann Return", _pct(mandate.get("ann_return"))),
        _card("Turnover (ann.)", _pct(metrics.get("turnover_ann"),
                                      decimals=0)),
        _card("IC rel (mean)", _num(ic_rel.get("mean_ic"), decimals=4)),
        _card("IC rel (t)", _num(ic_rel.get("t_stat"))),
        _card("Last 252d Active", _pct(last252.get("active_return_cum"),
                                       signed=True)),
        _card("Latest Rebalance", _text(allocations_latest.get(
            "rebalance_date"))),
        _card("Next Rebalance", _text(allocations_latest.get(
            "next_rebalance_date"))),
        _card("Top-5 Concentration", _pct(top5 if weights else None)),
    ]
    return '<div class="cards">' + "".join(cards) + "</div>"


# ---------------------------------------------------------------------------
# Section bodies
# ---------------------------------------------------------------------------

def _signal_section_body(signal_latest: dict) -> str:
    """fig_signal_ranking + compact top-10 table by normalized score."""
    parts = []
    b64 = fig_signal_ranking(signal_latest, signal_latest.get("date", ""))
    if b64:
        parts.append(_img_tag(b64, "Normalized Equity Ranking"))

    frame = _ranking_frame(signal_latest)
    if not frame.empty:
        rows = []
        for country, r in frame.head(10).iterrows():
            tilt = (
                r["weight"] - r["base_weight"]
                if not (_is_na(r["weight"]) or _is_na(r["base_weight"]))
                else None
            )
            rows.append([
                country,
                _num(r["score"], decimals=3),
                _num(r["score_change"], decimals=3, signed=True),
                _pct(r["weight"]),
                _pct(r["base_weight"]),
                _pct(tilt, signed=True),
            ])
        parts.append("<h4 class='tbl-title'>Top 10 by Score</h4>")
        parts.append(_table(
            ["Country", "Score", "Δ63d", "Weight", "Base", "Tilt"],
            rows, css_class="sig-table",
        ))
    return "\n".join(parts)


def _evolution_section_body(strategy_id: str,
                            history_monthly: pd.DataFrame) -> str:
    """Country chips + All/None controls + SVG canvas + embedded payload."""
    payload = signal_history_payload(history_monthly)
    countries = list(payload["countries"])
    chips = []
    for i, name in enumerate(countries):
        color = PALETTE12[i % len(PALETTE12)]
        on = " on" if i < 5 else ""
        chips.append(
            f'<button class="evo-chip{on}" data-country="{name}" '
            f'style="--chip:{color}" '
            f"onclick=\"evoToggle('{strategy_id}', this)\">{name}</button>"
        )
    controls = (
        f'<button class="evo-ctl" '
        f"onclick=\"evoSetAll('{strategy_id}', true)\">All</button>"
        f'<button class="evo-ctl" '
        f"onclick=\"evoSetAll('{strategy_id}', false)\">None</button>"
    )
    return (
        f'<div class="evo-toolbar" id="evo-chips-{strategy_id}">'
        f"{controls}{''.join(chips)}</div>"
        f'<svg class="evo-svg" viewBox="0 0 960 360" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'role="img" aria-label="Signal evolution" '
        f'id="evo-svg-{strategy_id}"></svg>'
        '<p class="evo-note">Month-end normalized composite score '
        "(0–1, cross-sectional). Toggle countries to compare.</p>"
        f'<script type="application/json" class="evo-data" '
        f'data-sid="{strategy_id}" id="evo-data-{strategy_id}">'
        f"{json.dumps(payload)}</script>"
    )


def _allocations_section_body(signal_latest: dict,
                              allocations: pd.DataFrame) -> str:
    """fig_allocation_history + full latest allocation table (weight desc)."""
    parts = []
    b64 = fig_allocation_history(allocations)
    if b64:
        parts.append(_img_tag(b64, "Allocation History"))

    frame = _ranking_frame(signal_latest)
    if not frame.empty:
        frame = frame.sort_values("weight", ascending=False)
        rows = []
        for country, r in frame.iterrows():
            tilt = (
                r["weight"] - r["base_weight"]
                if not (_is_na(r["weight"]) or _is_na(r["base_weight"]))
                else None
            )
            rows.append([
                country,
                _pct(r["weight"], decimals=2),
                _pct(r["base_weight"], decimals=2),
                _pct(tilt, decimals=2, signed=True),
            ])
        parts.append("<h4 class='tbl-title'>Latest Allocation "
                     "(full book)</h4>")
        parts.append(_table(["Country", "Weight", "Base", "Tilt"], rows,
                            css_class="alloc-table"))
    return "\n".join(parts)


#: metrics.json key -> (label, formatter) for the definition tables.
_STAT_FORMATS = {
    "ann_return": ("Ann Return", lambda v: _pct(v, decimals=2)),
    "ann_vol": ("Ann Volatility", lambda v: _pct(v, decimals=2)),
    "sharpe": ("Sharpe", _num),
    "max_drawdown": ("Max Drawdown", lambda v: _pct(v, decimals=2)),
    "win_rate": ("Win Rate", _pct),
    "beta": ("Beta", _num),
    "up_capture": ("Up Capture", _num),
    "down_capture": ("Down Capture", _num),
    "tracking_error": ("Tracking Error", lambda v: _pct(v, decimals=2)),
    "information_ratio": ("Information Ratio", _num),
    "mean_ic": ("Mean IC", lambda v: _num(v, decimals=4)),
    "median_ic": ("Median IC", lambda v: _num(v, decimals=4)),
    "std_ic": ("IC Std", lambda v: _num(v, decimals=4)),
    "t_stat": ("t-stat", _num),
    "icir": ("ICIR", _num),
    "hit_rate": ("Hit Rate", _pct),
    "window_days": ("Window (days)", _text),
    "active_return_cum": ("Active Return (cum)",
                          lambda v: _pct(v, decimals=2, signed=True)),
    "ir_ann": ("IR (ann.)", _num),
}


def _definition_table(title: str, stats: Optional[dict]) -> str:
    if not stats:
        return ""
    rows = []
    for key, value in stats.items():
        label, formatter = _STAT_FORMATS.get(
            key, (key.replace("_", " ").title(), _num)
        )
        rows.append([label, formatter(value)])
    return (
        f"<div class='def-block'><h4 class='tbl-title'>{title}</h4>"
        + _table(["Metric", "Value"], rows, css_class="def-table")
        + "</div>"
    )


def _metrics_section_body(metrics: dict) -> str:
    """Definition tables: mandate, period, IC abs/rel + run provenance."""
    ic = metrics.get("composite_ic") or {}
    category_weights = metrics.get("category_weights") or {}
    provenance = {
        "Benchmark Column": _text(metrics.get("benchmark_column")),
        "Universe Size": _text(metrics.get("n_countries")),
        "Rebalances": _text(metrics.get("n_rebalances")),
        "Factor Set": ", ".join(metrics.get("factor_set") or []) or "—",
        "Category Weights": " · ".join(
            f"{cat} {_pct(w, decimals=0)}"
            for cat, w in category_weights.items()
        ) or "—",
        "Data End": _text(metrics.get("data_end")),
        "Git Commit": _text(metrics.get("git_commit"))[:12],
    }
    blocks = [
        _definition_table("Mandate (daily, ann.)",
                          metrics.get("mandate_stats")),
        _definition_table("Period (per rebalance)",
                          metrics.get("period_stats")),
        _definition_table("Composite IC — absolute", ic.get("absolute")),
        _definition_table("Composite IC — relative", ic.get("relative")),
        _definition_table("Last 252d Active",
                          metrics.get("last_252d_active")),
        "<div class='def-block'><h4 class='tbl-title'>Run Provenance</h4>"
        + _table(["Field", "Value"],
                 [[k, v] for k, v in provenance.items()],
                 css_class="def-table")
        + "</div>",
    ]
    return '<div class="def-grid">' + "".join(b for b in blocks if b) + "</div>"


# ---------------------------------------------------------------------------
# Strategy pane
# ---------------------------------------------------------------------------

def build_strategy_pane(
    entry: dict,
    metrics: dict,
    signal_latest: dict,
    allocations_latest: dict,
    allocations: pd.DataFrame,
    history_monthly: pd.DataFrame,
) -> str:
    """HTML fragment for one deployed strategy (no shell wrapper).

    Parameters
    ----------
    entry:
        Registry entry (``configs/production.json`` strategies item).
    metrics / signal_latest / allocations_latest:
        Parsed per-strategy JSON artifacts.
    allocations:
        ``allocations.csv`` frame (rebalance dates x countries).
    history_monthly:
        ``signal_history_monthly.csv`` frame (month-end dates x countries).
    """
    parts = [
        f'<h2 class="strat-label">{entry.get("label", entry["id"])}</h2>',
        _bmk_badge(entry),
        production_stat_cards(metrics, allocations_latest),
        _section("Latest Signal", _signal_section_body(signal_latest),
                 open=True),
        _section("Signal Evolution",
                 _evolution_section_body(entry["id"], history_monthly)),
        _section("Allocations",
                 _allocations_section_body(signal_latest, allocations)),
        _section("Metrics Detail", _metrics_section_body(metrics)),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Shell: CSS + JS (tabs, openSecs persistence, SVG evolution chart)
# ---------------------------------------------------------------------------

_PROD_CSS = _DASH_CSS + """
/* --- Production dashboard additions --- */
.meta-line { font-size:0.8rem; color:#6b7280; margin:2px 0 8px 0; }
.meta-line strong { color:#1f2937; }
.strat-label { font-size:1.02rem; font-weight:700; color:#1e3a5f;
               margin:0 0 8px 0; }
.bmk-note { color:#6b7280; font-weight:400; }
.tbl-title { margin:10px 0 4px; font-size:0.9rem; color:#1f2937; }
.sig-table, .alloc-table { max-width:760px; }
/* Evolution chart: legend chips double as toggles */
.evo-toolbar { display:flex; flex-wrap:wrap; gap:4px; margin:4px 0 8px 0; }
.evo-ctl { font-size:0.72rem; font-weight:600; padding:3px 10px;
           border-radius:4px; border:1px solid #9ca3af; background:#fff;
           color:#374151; cursor:pointer; }
.evo-ctl:hover { background:#f3f4f6; }
.evo-chip { font-size:0.72rem; font-weight:600; padding:3px 10px;
            border-radius:999px; border:1px solid #d1d5db;
            background:#f3f4f6; color:#374151; cursor:pointer; }
.evo-chip.on { background:var(--chip,#2563EB);
               border-color:var(--chip,#2563EB); color:#fff; }
.evo-svg { width:100%; max-width:1100px; height:auto; background:#fff;
           border:1px solid #e5e7eb; border-radius:6px; }
.evo-note { font-size:0.74rem; color:#6b7280; margin:4px 0 0 0; }
/* Metrics-detail definition tables: dense responsive grid */
.def-grid { display:grid; gap:0 18px;
            grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); }
.def-table { font-size:0.8rem; }
.def-table td:last-child { text-align:right; font-variant-numeric:tabular-nums; }
/* Print-safe-ish: let the page flow and expand every section */
@media print {
  html, body { height:auto; overflow:visible; }
  .dash-content { overflow:visible; }
  .sec-body { display:block !important; }
  .pane { display:block !important; }
}
"""

# NOTE: plain-string JS with __TOKEN__ substitution ("%"-formatting would
# collide with the modulo operator).
_PROD_JS = """
var currentTab = "__DEFAULT_TAB__";
// Section open/closed state keyed by normalized section TITLE — persists
// across strategy tabs. "Latest Signal" is the centerpiece: starts OPEN.
var openSecs = {"latest signal": true};

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
  var span = btn.querySelector(".sec-title");
  var txt = span ? span.textContent : btn.textContent;
  return txt.replace(/[\\u25B8\\u25BE]/g, "").trim().toLowerCase();
}

function applySecState(btn, open) {
  var body = btn.nextElementSibling;
  if (!body) { return; }
  body.style.display = open ? "block" : "none";
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  var arrow = btn.querySelector(".sec-arrow");
  if (arrow) { arrow.textContent = open ? "\\u25BE" : "\\u25B8"; }
}

function toggleSec(btn) {
  var title = secTitle(btn);
  openSecs[title] = !openSecs[title];
  applySecState(btn, openSecs[title]);
}

function showTab() {
  var panes = document.querySelectorAll(".pane");
  for (var i = 0; i < panes.length; i++) {
    panes[i].style.display = "none";
  }
  var target = document.getElementById("pane-" + currentTab);
  if (target) {
    target.style.display = "block";
    var toggles = target.querySelectorAll(".sec-toggle");
    for (var j = 0; j < toggles.length; j++) {
      applySecState(toggles[j], !!openSecs[secTitle(toggles[j])]);
    }
  }
  _setActive(".seg-tab", "data-tab", currentTab);
}

function selectTab(tab) {
  currentTab = tab;
  showTab();
}

// ---------------------------------------------------------------------
// Signal-evolution SVG line chart (hand-rolled, no external libs)
// ---------------------------------------------------------------------
var EVO_COLORS = __EVO_COLORS__;
var EVO_DATA = {};
var EVO_SEL = {};

function evoInit() {
  var blocks = document.querySelectorAll("script.evo-data");
  for (var i = 0; i < blocks.length; i++) {
    var sid = blocks[i].getAttribute("data-sid");
    EVO_DATA[sid] = JSON.parse(blocks[i].textContent);
    EVO_SEL[sid] = {};
    var chips = document.querySelectorAll(
      "#evo-chips-" + sid + " .evo-chip");
    for (var j = 0; j < chips.length; j++) {
      EVO_SEL[sid][chips[j].getAttribute("data-country")] =
        chips[j].classList.contains("on");
    }
    drawEvo(sid);
  }
}

function evoToggle(sid, btn) {
  var name = btn.getAttribute("data-country");
  EVO_SEL[sid][name] = !EVO_SEL[sid][name];
  btn.classList.toggle("on", EVO_SEL[sid][name]);
  drawEvo(sid);
}

function evoSetAll(sid, on) {
  var chips = document.querySelectorAll("#evo-chips-" + sid + " .evo-chip");
  for (var j = 0; j < chips.length; j++) {
    EVO_SEL[sid][chips[j].getAttribute("data-country")] = on;
    chips[j].classList.toggle("on", on);
  }
  drawEvo(sid);
}

function drawEvo(sid) {
  var data = EVO_DATA[sid];
  var svg = document.getElementById("evo-svg-" + sid);
  if (!data || !svg) { return; }
  var W = 960, H = 360, padL = 46, padR = 14, padT = 12, padB = 36;
  var plotW = W - padL - padR, plotH = H - padT - padB;
  var n = data.dates.length;
  var parts = [];

  // y grid + labels (normalized score, fixed 0-1 scale)
  for (var g = 0; g <= 4; g++) {
    var yv = g / 4;
    var y = padT + plotH * (1 - yv);
    parts.push('<line x1="' + padL + '" y1="' + y + '" x2="' + (W - padR) +
      '" y2="' + y + '" stroke="#e5e7eb" stroke-width="1"/>');
    parts.push('<text x="' + (padL - 8) + '" y="' + (y + 4) +
      '" text-anchor="end" font-size="11" fill="#6b7280">' +
      yv.toFixed(2) + '</text>');
  }
  // x ticks: ~6 labeled month-end dates
  if (n > 0) {
    var step = Math.max(1, Math.round((n - 1) / 5));
    for (var t = 0; t < n; t += step) {
      var tx = padL + (n === 1 ? 0 : plotW * t / (n - 1));
      parts.push('<line x1="' + tx + '" y1="' + (H - padB) + '" x2="' + tx +
        '" y2="' + (H - padB + 4) + '" stroke="#9ca3af" stroke-width="1"/>');
      parts.push('<text x="' + tx + '" y="' + (H - padB + 18) +
        '" text-anchor="middle" font-size="10" fill="#6b7280">' +
        data.dates[t] + '</text>');
    }
  }
  // one polyline per selected country (colors match the legend chips)
  var names = Object.keys(data.countries);
  for (var k = 0; k < names.length; k++) {
    if (!EVO_SEL[sid][names[k]]) { continue; }
    var series = data.countries[names[k]];
    var pts = [];
    for (var p = 0; p < n; p++) {
      var v = series[p];
      if (v === null || v === undefined) { continue; }
      var px = padL + (n === 1 ? 0 : plotW * p / (n - 1));
      var py = padT + plotH * (1 - Math.max(0, Math.min(1, v)));
      pts.push(px.toFixed(1) + "," + py.toFixed(1));
    }
    if (pts.length > 1) {
      parts.push('<polyline fill="none" stroke="' +
        EVO_COLORS[k % EVO_COLORS.length] +
        '" stroke-width="1.8" stroke-linejoin="round" points="' +
        pts.join(" ") + '"/>');
    }
  }
  svg.innerHTML = parts.join("");
}

showTab();
evoInit();
"""


def render_production_dashboard(
    panes: list,
    data_end: str,
    next_rebalance: str,
    run_name: str,
    git_commit: str,
) -> str:
    """Render the full production dashboard HTML document.

    Parameters
    ----------
    panes:
        ``[(strategy_id, label, pane_html), ...]`` — the first entry is the
        default tab.
    data_end / next_rebalance / run_name / git_commit:
        Header manifest line content.
    """
    if not panes:
        raise ValueError("panes must contain at least one strategy")
    default_tab = panes[0][0]

    tabs = "".join(
        f'<button class="seg-tab{" active" if sid == default_tab else ""}" '
        f'data-tab="{sid}" onclick="selectTab(\'{sid}\')">{label}</button>'
        for sid, label, _ in panes
    )
    pane_divs = "\n".join(
        f'<div id="pane-{sid}" class="pane" '
        f'style="display:{"block" if sid == default_tab else "none"}">\n'
        f"{html}\n</div>"
        for sid, _, html in panes
    )
    meta = (
        f'<div class="meta-line">Data end <strong>{data_end}</strong>'
        f" · Next rebalance <strong>{next_rebalance}</strong>"
        f" · {run_name} · commit {git_commit[:10]}</div>"
    )
    js = (
        _PROD_JS
        .replace("__DEFAULT_TAB__", default_tab)
        .replace("__EVO_COLORS__", json.dumps(PALETTE12))
    )
    body = (
        '<header class="dash-header">'
        f"<h1>{_TITLE}</h1>{meta}"
        f'<div class="seg-tabs">{tabs}</div>'
        "</header>"
        f'<main class="dash-content">{pane_divs}</main>'
        f"<script>{js}</script>"
    )
    return (
        "<!DOCTYPE html>"
        "<html lang='en'>"
        "<head>"
        "<meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_TITLE}</title>"
        f"<style>{_PROD_CSS}</style>"
        "</head>"
        f"<body>{body}</body>"
        "</html>"
    )


# ---------------------------------------------------------------------------
# Orchestrator: run dir + registry -> HTML
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True)


def build_dashboard(
    run_dir: Union[str, Path], registry_path: Union[str, Path]
) -> str:
    """Build the production dashboard HTML from one artifact run folder.

    Parameters
    ----------
    run_dir:
        ``outputs/production/run_{YYYYMMDD}/`` folder containing one
        subfolder per registry strategy (production_run artifact set).
    registry_path:
        Deployed-strategy registry JSON (``configs/production.json``).

    Returns
    -------
    Self-contained HTML string. Raises ``FileNotFoundError`` when a registry
    strategy's artifacts are missing — production builds fail loudly.
    """
    run_dir = Path(run_dir)
    registry = _read_json(Path(registry_path))

    panes = []
    data_end = "—"
    next_rebalance = "—"
    git_commit = "unknown"
    for entry in registry["strategies"]:
        sdir = run_dir / entry["id"]
        metrics = _read_json(sdir / "metrics.json")
        signal_latest = _read_json(sdir / "signal_latest.json")
        allocations_latest = _read_json(sdir / "allocations_latest.json")
        allocations = _read_frame(sdir / "allocations.csv")
        history_monthly = _read_frame(sdir / "signal_history_monthly.csv")

        panes.append((
            entry["id"],
            entry.get("label", entry["id"]),
            build_strategy_pane(entry, metrics, signal_latest,
                                allocations_latest, allocations,
                                history_monthly),
        ))
        data_end = metrics.get("data_end", data_end)
        next_rebalance = allocations_latest.get(
            "next_rebalance_date", next_rebalance)
        git_commit = metrics.get("git_commit", git_commit)

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        data_end = manifest.get("data_end", data_end)
        git_commit = manifest.get("git_commit", git_commit)

    return render_production_dashboard(
        panes, data_end, next_rebalance, run_dir.name, git_commit
    )
