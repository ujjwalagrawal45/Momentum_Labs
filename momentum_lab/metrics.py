"""Performance and risk statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult

Y = 252


def cagr(nav: pd.Series) -> float:
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1


def summary(res: BacktestResult) -> pd.DataFrame:
    rf_d = (1 + res.config.rf_rate) ** (1 / Y) - 1
    rows = {}
    for label, nav in [("Strategy", res.nav), ("Benchmark", res.benchmark)]:
        r = nav.pct_change().dropna()
        ex = r - rf_d
        downside = ex[ex < 0].std() * np.sqrt(Y)
        dd = drawdown(nav)
        rows[label] = {
            "CAGR": cagr(nav),
            "Volatility": r.std() * np.sqrt(Y),
            "Sharpe": ex.mean() / r.std() * np.sqrt(Y),
            "Sortino": ex.mean() * Y / downside if downside else np.nan,
            "Max drawdown": dd.min(),
            "Calmar": cagr(nav) / -dd.min() if dd.min() < 0 else np.nan,
            "Best month": nav.resample("ME").last().pct_change().max(),
            "Worst month": nav.resample("ME").last().pct_change().min(),
            "% positive months": (nav.resample("ME").last().pct_change().dropna() > 0).mean(),
        }
    s, b = res.nav.pct_change().dropna(), res.benchmark.pct_change().dropna()
    beta = s.cov(b) / b.var()
    active = s - b
    rows["Strategy"].update({
        "Beta": beta,
        "Alpha (ann.)": (s.mean() - beta * b.mean()) * Y,
        "Tracking error": active.std() * np.sqrt(Y),
        "Information ratio": active.mean() / active.std() * np.sqrt(Y),
        "Correlation": s.corr(b),
        "Annual turnover": annual_turnover(res),
        "Cost drag (ann.)": res.costs.sum() / res.nav.mean() / ((res.nav.index[-1] - res.nav.index[0]).days / 365.25),
        "Avg holdings": (res.weights > 0).sum(axis=1).mean(),
        "Avg exposure": res.exposure.mean(),
        "Final value (Rs)": res.nav.iloc[-1],
    })
    rows["Benchmark"]["Final value (Rs)"] = res.benchmark.iloc[-1]
    return pd.DataFrame(rows)


def annual_turnover(res: BacktestResult) -> float:
    """One-way turnover: sum(|traded notional|) / 2 / avg NAV per year."""
    if res.trades.empty:
        return 0.0
    yrs = (res.nav.index[-1] - res.nav.index[0]).days / 365.25
    return res.trades["notional"].abs().sum() / 2 / res.nav.mean() / yrs


def calendar_returns(nav: pd.Series) -> pd.DataFrame:
    """Year x month return table, with full-year column."""
    m = nav.resample("ME").last()
    m = pd.concat([nav.iloc[:1], m]).pct_change().dropna()
    tbl = m.groupby([m.index.year, m.index.month]).first().unstack()
    tbl.columns = [pd.Timestamp(2000, c, 1).strftime("%b") for c in tbl.columns]
    yearly = nav.resample("YE").last()
    yearly = pd.concat([nav.iloc[:1], yearly]).pct_change().dropna()
    tbl["Year"] = yearly.values[: len(tbl)]
    return tbl


def deflated_sharpe(sr_annual: float, n_trials: int, n_obs: int, skew: float = 0, kurt: float = 3) -> float:
    """Bailey & Lopez de Prado (2014): probability the observed Sharpe is genuine after
    trying `n_trials` parameter combos. Use this when you've swept many configs."""
    from scipy.stats import norm
    sr = sr_annual / np.sqrt(Y)
    emc = 0.5772156649
    sr0 = np.sqrt(1 / n_obs) * ((1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(1 - 1 / (n_trials * np.e))) if n_trials > 1 else 0
    z = (sr - sr0) * np.sqrt(n_obs - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(norm.cdf(z))
