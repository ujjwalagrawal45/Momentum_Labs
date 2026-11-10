"""Momentum signals. Every function returns a (date x symbol) frame that at row t
uses only data up to and including t's close -- no look-ahead.

Day-count conventions: 21 trading days ~ 1 month, 252 ~ 1 year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel

M, Y = 21, 252


def total_return(close: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Price return from t-lookback to t-skip. skip=21 gives the classic '12-1' momentum
    (skipping the last month avoids the short-term reversal effect)."""
    return close.shift(skip) / close.shift(lookback) - 1


def daily_vol(close: pd.DataFrame, window: int = Y) -> pd.DataFrame:
    return np.log(close).diff().rolling(window, min_periods=int(window * 0.8)).std() * np.sqrt(Y)


def mom_12_1(p: Panel) -> pd.DataFrame:
    return total_return(p.close, Y, M)


def mom_6_1(p: Panel) -> pd.DataFrame:
    return total_return(p.close, 6 * M, M)


def mom_3(p: Panel) -> pd.DataFrame:
    return total_return(p.close, 3 * M)


def risk_adj_12_1(p: Panel) -> pd.DataFrame:
    """12-1 return / annualised vol (Sharpe-style momentum; MSCI Momentum uses this idea)."""
    return mom_12_1(p) / daily_vol(p.close)


def nse_momentum(p: Panel) -> pd.DataFrame:
    """NSE momentum-index style score (Nifty200 Momentum 30 methodology):
    6m and 12m returns each divided by 1y annualised vol, cross-sectionally z-scored,
    averaged 50/50, then mapped to a positive 'normalised momentum score'."""
    vol = daily_vol(p.close)
    z6 = zscore(total_return(p.close, 6 * M) / vol)
    z12 = zscore(total_return(p.close, Y) / vol)
    z = zscore(0.5 * z6 + 0.5 * z12)
    return pd.DataFrame(np.where(z >= 0, 1 + z, 1 / (1 - z)), index=z.index, columns=z.columns)


def high_52w(p: Panel) -> pd.DataFrame:
    """Proximity to 52-week high (George & Hwang 2004). 1.0 = at the high."""
    return p.close / p.high.rolling(Y, min_periods=int(Y * 0.8)).max()


def frog_in_pan(p: Panel) -> pd.DataFrame:
    """Information discreteness (Da, Gurun & Warachka 2014), sign-flipped so higher = better.
    Momentum built from many small up-days ('smooth') persists more than one built from a
    few big jumps. ID = sign(ret_12_1) * (%neg_days - %pos_days); we return -ID."""
    r = p.close.pct_change(fill_method=None).shift(M)
    pos = (r > 0).rolling(Y - M).mean()
    neg = (r < 0).rolling(Y - M).mean()
    ID = np.sign(mom_12_1(p)) * (neg - pos)
    return -ID


def residual_momentum(p: Panel) -> pd.DataFrame:
    """Idiosyncratic momentum (Blitz, Huij & Martens 2011): strip out market beta, then take
    the 12-1 cumulative residual return scaled by residual vol. Less crash-prone than raw
    momentum because it doesn't load up on high-beta names after a rally."""
    r = p.close.pct_change(fill_method=None)
    m = p.benchmark.pct_change(fill_method=None).reindex(r.index)
    win = Y
    cov = r.rolling(win, min_periods=200).cov(m)
    beta = cov.div(m.rolling(win, min_periods=200).var(), axis=0)
    resid = r - beta.mul(m, axis=0)
    cum = resid.shift(M).rolling(Y - M, min_periods=180).sum()
    sd = resid.shift(M).rolling(Y - M, min_periods=180).std()
    return cum / sd


def zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score per date, winsorised at +/-clip."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip)


SIGNALS = {
    "mom_12_1": mom_12_1,
    "mom_6_1": mom_6_1,
    "mom_3": mom_3,
    "risk_adj_12_1": risk_adj_12_1,
    "nse_momentum": nse_momentum,
    "high_52w": high_52w,
    "frog_in_pan": frog_in_pan,
    "residual_momentum": residual_momentum,
}


def composite(p: Panel, weights: dict[str, float]) -> pd.DataFrame:
    """Weighted sum of cross-sectionally z-scored signals. A single signal with weight 1
    is just that signal's z-score (rank order unchanged)."""
    unknown = set(weights) - set(SIGNALS)
    if unknown:
        raise ValueError(f"Unknown signals {unknown}; choose from {list(SIGNALS)}")
    total = None
    for name, w in weights.items():
        z = zscore(SIGNALS[name](p)) * w
        total = z if total is None else total + z  # NaN in any component -> not ranked
    return total
