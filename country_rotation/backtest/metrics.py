"""
Performance metrics consolidation.

Matches legacy math in backtest.py lines ~919-1111 exactly:
  - Annualized return: geometric  total_return^(365/days) - 1
  - Annualized vol:    std * sqrt(periods_per_year)
  - Sharpe:            ann_return / ann_vol  (no risk-free rate, matching legacy)
  - Beta:              cov(port, bmk) / var(bmk)
  - Tracking error:    std(active) * sqrt(periods_per_year)
  - Capture ratios:    mean(port | bmk>0) / mean(bmk | bmk>0), idem down
  - IR:                active_ann_return / tracking_error
  - Win rate:          (returns > 0).mean()
  - Max drawdown:      min((cum - running_max) / running_max)  on cumulative wealth

All degenerate inputs return np.nan values; never raise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def summary(
    returns: pd.Series,
    bmk_returns: pd.Series,
    periods_per_year: float,
) -> dict:
    """
    Compute full performance summary matching legacy backtest.py.

    Parameters
    ----------
    returns : pd.Series
        Portfolio period returns.
    bmk_returns : pd.Series
        Benchmark period returns (same index).
    periods_per_year : float
        Annualisation factor (252 for daily, 52 for weekly, …).

    Returns
    -------
    dict with keys:
        ann_return, ann_vol, sharpe, max_drawdown, win_rate,
        beta, up_capture, down_capture, tracking_error, information_ratio
    """
    # Align
    aligned = pd.DataFrame({"port": returns, "bmk": bmk_returns}).dropna()
    port = aligned["port"]
    bmk = aligned["bmk"]
    active = port - bmk

    # ------------------------------------------------------------------ #
    # Annualised return — geometric, calendar-day scaling (legacy ~:973)
    # ------------------------------------------------------------------ #
    def _ann_return(r: pd.Series) -> float:
        if len(r) == 0:
            return np.nan
        if r.index.dtype == "datetime64[ns]" or hasattr(r.index[0], "date"):
            days = (r.index[-1] - r.index[0]).days
        else:
            days = 0
        if days <= 0:
            return np.nan
        total = float((1.0 + r).prod())
        if total <= 0:
            return np.nan
        return total ** (365.0 / days) - 1.0

    # ------------------------------------------------------------------ #
    # Annualised vol — std * sqrt(ppy)  (legacy ~:980)
    # ------------------------------------------------------------------ #
    def _ann_vol(r: pd.Series) -> float:
        if len(r) < 2:
            return np.nan
        return float(r.std() * np.sqrt(periods_per_year))

    # ------------------------------------------------------------------ #
    # Sharpe — ann_return / ann_vol, no risk-free  (legacy ~:987)
    # ------------------------------------------------------------------ #
    def _sharpe(ann_ret: float, ann_v: float) -> float:
        if np.isnan(ann_ret) or np.isnan(ann_v) or ann_v == 0.0:
            return np.nan
        return ann_ret / ann_v

    # ------------------------------------------------------------------ #
    # Beta — cov / var  (legacy ~:998-1002)
    # ------------------------------------------------------------------ #
    def _beta(p: pd.Series, b: pd.Series) -> float:
        if len(p) < 2:
            return np.nan
        var_b = float(b.var())
        if var_b == 0.0:
            return np.nan
        return float(p.cov(b) / var_b)

    # ------------------------------------------------------------------ #
    # Tracking error — std(active) * sqrt(ppy)  (legacy ~:1009)
    # ------------------------------------------------------------------ #
    def _tracking_error(act: pd.Series) -> float:
        if len(act) < 2:
            return np.nan
        return float(act.std() * np.sqrt(periods_per_year))

    # ------------------------------------------------------------------ #
    # Capture ratios  (legacy ~:1018-1032)
    # ------------------------------------------------------------------ #
    def _capture(p: pd.Series, b: pd.Series):
        up_mask = b > 0
        down_mask = b < 0
        if up_mask.sum() > 0 and b[up_mask].mean() != 0.0:
            up = float(p[up_mask].mean() / b[up_mask].mean())
        else:
            up = np.nan
        if down_mask.sum() > 0 and b[down_mask].mean() != 0.0:
            down = float(p[down_mask].mean() / b[down_mask].mean())
        else:
            down = np.nan
        return up, down

    # ------------------------------------------------------------------ #
    # Max drawdown — on cumulative wealth  (legacy ~:1108-1110)
    # ------------------------------------------------------------------ #
    def _max_drawdown(r: pd.Series) -> float:
        if len(r) == 0:
            return np.nan
        cum = (1.0 + r).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        return float(dd.min())

    # ------------------------------------------------------------------ #
    # Win rate  (legacy ~:1059)
    # ------------------------------------------------------------------ #
    def _win_rate(r: pd.Series) -> float:
        if len(r) == 0:
            return np.nan
        return float((r > 0).sum() / len(r))

    # --- compute ---
    port_ann_ret = _ann_return(port)
    bmk_ann_ret = _ann_return(bmk)
    port_ann_vol = _ann_vol(port)

    # win_rate: degenerate (all zero vol) — count exact zeroes as NOT wins
    win_rate = _win_rate(port)

    # tracking error and IR use active series  (legacy ~:1049)
    te = _tracking_error(active)
    active_ann_ret = (
        port_ann_ret - bmk_ann_ret
        if not (np.isnan(port_ann_ret) or np.isnan(bmk_ann_ret))
        else np.nan
    )
    ir = _sharpe(active_ann_ret, te)

    up_cap, dn_cap = _capture(port, bmk)

    # _ann_vol already returns 0.0 for constant series (std == 0) and NaN
    # only for truly degenerate inputs (len < 2) — use it directly.
    ann_vol_out = port_ann_vol

    return {
        "ann_return": port_ann_ret,
        "ann_vol": ann_vol_out,
        "sharpe": _sharpe(port_ann_ret, ann_vol_out),
        "max_drawdown": _max_drawdown(port),
        "win_rate": win_rate,
        "beta": _beta(port, bmk),
        "up_capture": up_cap,
        "down_capture": dn_cap,
        "tracking_error": te,
        "information_ratio": ir,
    }
