"""Tests for scripts/export_readme_figures.py — pane-aware figure extraction."""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_readme_figures as erf  # noqa: E402

_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()


def _img(alt: str) -> str:
    return f"<img src='data:image/png;base64,{_PNG}' alt='{alt}'/>"


def test_parse_figures_pane_and_occurrence():
    html = (
        '<html><div id="pane-A" class="pane" data-x="1">'
        + _img("Ranking") + _img("TCA figure") + _img("TCA figure")
        + '</div><div id="pane-B" class="pane">'
        + _img("Ranking")
        + "</div></html>"
    )
    figs = erf.parse_figures(html)
    keys = [(p, a, o) for p, a, o, _ in figs]
    assert keys == [
        ("pane-A", "Ranking", 0),
        ("pane-A", "TCA figure", 0),
        ("pane-A", "TCA figure", 1),
        ("pane-B", "Ranking", 0),
    ]


def test_parse_figures_root_pane():
    figs = erf.parse_figures(_img("Loose"))
    assert figs[0][:3] == ("<root>", "Loose", 0)


def test_export_writes_curated_pngs(tmp_path, monkeypatch):
    html = (
        '<div id="pane-X" class="pane">' + _img("Ranking") + "</div>"
    )
    dash_rel = "dash.html"
    (tmp_path / dash_rel).write_text(html, encoding="utf-8")
    monkeypatch.setattr(
        erf, "_EXPORTS", {(dash_rel, "pane-X", "Ranking", 0): "x_ranking.png"}
    )
    out_dir = tmp_path / "figures"
    written = erf.export(str(out_dir), repo_root=str(tmp_path))
    assert [Path(w).name for w in written] == ["x_ranking.png"]
    assert (out_dir / "x_ranking.png").read_bytes().startswith(b"\x89PNG")


def test_export_missing_dashboard_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        erf, "_EXPORTS", {("nope.html", "p", "a", 0): "f.png"}
    )
    with pytest.raises(SystemExit, match="build .*dashboards"):
        erf.export(str(tmp_path / "figures"), repo_root=str(tmp_path))


def test_export_missing_figure_actionable(tmp_path, monkeypatch):
    (tmp_path / "dash.html").write_text(
        '<div id="pane-X" class="pane">' + _img("Other") + "</div>",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        erf, "_EXPORTS", {("dash.html", "pane-X", "Ranking", 0): "f.png"}
    )
    with pytest.raises(SystemExit, match="missing"):
        erf.export(str(tmp_path / "figures"), repo_root=str(tmp_path))


def test_real_curated_set_matches_inventory_when_dashboards_exist():
    """On machines with built dashboards, every curated key must resolve
    (guards against silent dashboard-layout drift)."""
    dashboards = {dash for dash, _, _, _ in erf._EXPORTS}
    for dash in dashboards:
        path = REPO_ROOT / dash
        if not path.is_file():
            pytest.skip("dashboards not built on this machine")
    out = []
    for dash in dashboards:
        html = (REPO_ROOT / dash).read_text(encoding="utf-8")
        found = {(p, a, o) for p, a, o, _ in erf.parse_figures(html)}
        for key in erf._EXPORTS:
            if key[0] == dash and key[1:] not in found:
                out.append(key)
    assert not out, f"curated figures missing from dashboards: {out}"
