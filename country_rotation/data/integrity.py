# country_rotation/data/integrity.py
"""Leakage guards and coverage diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LookaheadReport:
    clean: bool
    dirty_outputs: tuple


def lookahead_check(pipeline, dfs, cutoff, perturb_scale=100.0):
    """Run pipeline on original and on future-perturbed inputs; any difference
    at dates <= cutoff is look-ahead leakage."""
    base = pipeline({k: v.copy() for k, v in dfs.items()})
    perturbed_in = {}
    for k, v in dfs.items():
        p = v.copy()
        p.loc[p.index > cutoff] = p.loc[p.index > cutoff].fillna(0.0) * perturb_scale + perturb_scale
        perturbed_in[k] = p
    pert = pipeline(perturbed_in)
    dirty = []
    for name, df in base.items():
        if name not in pert:
            continue
        a = df.loc[df.index <= cutoff]
        b = pert[name].loc[pert[name].index <= cutoff]
        cols = sorted(set(a.columns) & set(b.columns))
        if set(a.columns) != set(b.columns):
            dirty.append(name)
            continue
        a = a[cols]; b = b[cols]
        if not np.allclose(a.fillna(-9e9).to_numpy(), b.fillna(-9e9).to_numpy(), equal_nan=False):
            dirty.append(name)
    return LookaheadReport(clean=not dirty, dirty_outputs=tuple(dirty))


def coverage_matrix(dfs):
    """Return a DataFrame summarising date range, country count, and pct missing per dataset."""
    rows = {}
    for name, df in dfs.items():
        valid = df.notna()
        rows[name] = {
            "n_countries": int((valid.sum() > 0).sum()),
            "first_date": df.index[0] if len(df) else pd.NaT,
            "last_date": df.index[-1] if len(df) else pd.NaT,
            "pct_missing": float(df.isna().mean().mean()),
        }
    return pd.DataFrame(rows).T
