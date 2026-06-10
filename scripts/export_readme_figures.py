"""Export curated dashboard figures to ``docs/figures/`` for the README.

The dashboards embed every figure as a base64 PNG inside a strategy pane
(``<div id="..." class="pane">``).  This script decodes a curated subset to
standalone PNGs so the README can tell the construction/validation story
with the exact images the dashboards show — single source of truth, no
separately-maintained plotting code.

Usage
-----
    python scripts/export_readme_figures.py            # export curated set
    python scripts/export_readme_figures.py --list     # inventory only

Re-run after rebuilding the dashboards (``python scripts/pipeline.py
dashboards``) to refresh the committed figures.
"""
from __future__ import annotations

import argparse
import base64
import os
import re

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)

_PRODUCTION_DASH = os.path.join("outputs", "production", "production_dashboard.html")
_RESEARCH_DASH = os.path.join("outputs", "research", "strategy_dashboard.html")
_DEFAULT_OUT = os.path.join("docs", "figures")

_PANE_RE = re.compile(r'<div id="([^"]+)" class="pane"')
_IMG_RE = re.compile(r"<img src='data:image/png;base64,([^']+)' alt='([^']*)'/>")

#: (dashboard, pane id, alt text, occurrence within pane+alt) -> output name.
#: TCA figures share one alt; their stable order is turnover bars (0),
#: cost-layer waterfall (1), net cumulative lines (2) — see
#: reporting.dashboard.tca_section_body.
_EXPORTS: dict[tuple[str, str, str, int], str] = {
    # --- production dashboard: signal, IC, allocations, TCA ---------------
    (_PRODUCTION_DASH, "pane-EM_captilt_vsEM", "Normalized Equity Ranking", 0):
        "em_signal_ranking.png",
    (_PRODUCTION_DASH, "pane-DM_captilt_vsACWI", "Normalized Equity Ranking", 0):
        "dm_signal_ranking.png",
    (_PRODUCTION_DASH, "pane-EM_captilt_vsEM",
     "IC Distribution — Relative (63d score change)", 0):
        "em_ic_distribution.png",
    (_PRODUCTION_DASH, "pane-EM_captilt_vsEM", "Allocation History", 0):
        "em_allocation_history.png",
    (_PRODUCTION_DASH, "pane-DM_captilt_vsACWI", "Allocation History", 0):
        "dm_allocation_history.png",
    (_PRODUCTION_DASH, "pane-EM_captilt_vsEM", "TCA figure", 1):
        "em_tca_cost_layers.png",
    (_PRODUCTION_DASH, "pane-EM_captilt_vsEM", "TCA figure", 2):
        "em_tca_net_cumulative.png",
    (_PRODUCTION_DASH, "pane-DM_captilt_vsACWI", "TCA figure", 1):
        "dm_tca_cost_layers.png",
    # --- research dashboard: backtest curves + signal decomposition -------
    (_RESEARCH_DASH, "EM-cap_tilt", "Cumulative Return", 0):
        "em_captilt_cumulative_return.png",
    (_RESEARCH_DASH, "EM-cap_tilt", "Drawdown", 0):
        "em_captilt_drawdown.png",
    (_RESEARCH_DASH, "DM-cap_tilt", "Cumulative Return", 0):
        "dm_captilt_cumulative_return.png",
    (_RESEARCH_DASH, "EM-cap_tilt", "Contributions Latest", 0):
        "em_captilt_contributions.png",
}


def parse_figures(html: str) -> list[tuple[str, str, int, str]]:
    """All embedded figures in document order.

    Returns ``(pane id, alt text, occurrence within pane+alt, base64)``
    tuples; figures before the first pane div get pane id ``<root>``.
    """
    events: list[tuple[int, str, tuple]] = []
    for m in _PANE_RE.finditer(html):
        events.append((m.start(), "pane", (m.group(1),)))
    for m in _IMG_RE.finditer(html):
        events.append((m.start(), "img", (m.group(1), m.group(2))))
    events.sort(key=lambda e: e[0])

    pane = "<root>"
    counts: dict[tuple[str, str], int] = {}
    out: list[tuple[str, str, int, str]] = []
    for _, kind, payload in events:
        if kind == "pane":
            pane = payload[0]
            continue
        b64, alt = payload
        key = (pane, alt)
        occurrence = counts.get(key, 0)
        counts[key] = occurrence + 1
        out.append((pane, alt, occurrence, b64))
    return out


def export(out_dir: str, repo_root: str | None = None) -> list[str]:
    """Decode the curated figure set into *out_dir*; returns written paths."""
    root = repo_root if repo_root is not None else _REPO_ROOT
    by_dash: dict[str, dict[tuple[str, str, int], str]] = {}
    for (dash, pane, alt, occ), name in _EXPORTS.items():
        by_dash.setdefault(dash, {})[(pane, alt, occ)] = name

    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    for dash, wanted in by_dash.items():
        path = os.path.join(root, dash)
        if not os.path.isfile(path):
            raise SystemExit(
                f"[export_readme_figures] ERROR: '{dash}' not found — build "
                "the dashboards first (python scripts/pipeline.py dashboards)."
            )
        with open(path, encoding="utf-8") as fh:
            figures = parse_figures(fh.read())
        found = {(pane, alt, occ): b64 for pane, alt, occ, b64 in figures}
        for key, name in sorted(wanted.items(), key=lambda kv: kv[1]):
            if key not in found:
                raise SystemExit(
                    f"[export_readme_figures] ERROR: figure {key} missing "
                    f"from '{dash}' — dashboard layout changed? Run --list."
                )
            target = os.path.join(out_dir, name)
            with open(target, "wb") as fh:
                fh.write(base64.b64decode(found[key]))
            written.append(target)
            print(f"[export_readme_figures] {name} <- {key[0]} | {key[1]} "
                  f"#{key[2]}")
    return written


def list_inventory(repo_root: str | None = None) -> None:
    root = repo_root if repo_root is not None else _REPO_ROOT
    for dash in (_PRODUCTION_DASH, _RESEARCH_DASH):
        path = os.path.join(root, dash)
        if not os.path.isfile(path):
            print(f"[export_readme_figures] (missing) {dash}")
            continue
        with open(path, encoding="utf-8") as fh:
            figures = parse_figures(fh.read())
        print(f"===== {dash} ({len(figures)} figures)")
        for pane, alt, occ, _ in figures:
            print(f"  {pane:30s} | {alt:50s} #{occ}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export curated dashboard figures for the README.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out", default=os.path.join(_REPO_ROOT, _DEFAULT_OUT),
        help="Output directory for the PNGs.",
    )
    parser.add_argument(
        "--list", action="store_true", default=False, dest="list_only",
        help="Print the figure inventory of both dashboards and exit.",
    )
    args = parser.parse_args()
    if args.list_only:
        list_inventory()
        return
    written = export(args.out)
    print(f"[export_readme_figures] {len(written)} figure(s) -> {args.out}")


if __name__ == "__main__":
    main()
