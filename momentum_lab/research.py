"""Signal research tools: quantile spreads, information coefficient, parameter sweeps.

Quant workflow: (1) prove the signal ranks stocks monotonically (quantile test / IC),
(2) only then build a portfolio, (3) sweep parameters on an in-sample window and confirm on
an untouched out-of-sample window.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import signals as sig
from .backtest import eligibility, rebalance_dates, run_backtest
from .config import StrategyConfig
from .data import Panel
from .metrics import cagr, drawdown


def quantile_analysis(p: Panel, scores: pd.DataFrame, cfg: StrategyConfig,
                      n_q: int = 5, horizon: int = 21) -> tuple[pd.DataFrame, pd.Series]:
    """Each rebalance date: bucket eligible stocks by score into n_q quantiles, measure the
    forward `horizon`-day return (entered next day open, to avoid look-ahead).
    Returns (per-period quantile returns, rank-IC series)."""
    elig = eligibility(p, cfg.replace(abs_momentum=False))  # test the signal on the full cross-section
    fwd = p.close.shift(-horizon) / p.open.shift(-1) - 1
    dates = rebalance_dates(p.close.loc[cfg.start:cfg.end].index, cfg.rebalance)[:-1]
    q_rows, ic = {}, {}
    for d in dates:
        s = scores.loc[d].where(elig.loc[d]).dropna()
        f = fwd.loc[d, s.index].dropna()
        s = s[f.index]
        if len(s) < n_q * 5:
            continue
        q = pd.qcut(s.rank(method="first"), n_q, labels=[f"Q{i}" for i in range(1, n_q + 1)])
        q_rows[d] = f.groupby(q, observed=True).mean()
        ic[d] = s.corr(f, method="spearman")
    qr = pd.DataFrame(q_rows).T
    qr[f"Q{n_q}-Q1"] = qr[f"Q{n_q}"] - qr["Q1"]
    return qr, pd.Series(ic, name="IC")


def quantile_report(qr: pd.DataFrame, ic: pd.Series, periods_per_year: int = 12) -> pd.DataFrame:
    ann = (1 + qr).prod() ** (periods_per_year / len(qr)) - 1
    out = pd.DataFrame({"Ann. return": ann, "Hit rate": (qr > 0).mean()})
    out.loc["IC mean", "Ann. return"] = ic.mean()
    out.loc["IC t-stat", "Ann. return"] = ic.mean() / ic.std() * np.sqrt(len(ic))
    return out


def signal_scoreboard(p: Panel, cfg: StrategyConfig) -> pd.DataFrame:
    """Quantile test for every built-in signal side by side."""
    rows = {}
    for name, fn in sig.SIGNALS.items():
        qr, ic = quantile_analysis(p, fn(p), cfg)
        rep = quantile_report(qr, ic)["Ann. return"]
        rows[name] = rep
    return pd.DataFrame(rows).T


def param_sweep(p: Panel, base: StrategyConfig, grid: dict[str, list],
                split: str | None = None) -> pd.DataFrame:
    """Run every combination in `grid`. If `split` (a date) is given, metrics are reported
    separately for in-sample (< split) and out-of-sample (>= split) periods. Choose
    parameters on IS only; OOS tells you whether they generalise."""
    score_cache: dict = {}
    rows = []
    keys = list(grid)
    for combo in itertools.product(*grid.values()):
        cfg = base.replace(**dict(zip(keys, combo)))
        sk = repr(sorted(cfg.signal_weights.items()))
        if sk not in score_cache:
            score_cache[sk] = sig.composite(p, cfg.signal_weights)
        res = run_backtest(p, cfg, scores=score_cache[sk])
        row = {k: (str(v) if isinstance(v, dict) else v) for k, v in zip(keys, combo)}
        periods = {"": res.nav} if not split else {"IS ": res.nav[:split], "OOS ": res.nav[split:]}
        for tag, nav in periods.items():
            if len(nav) < 30:
                continue
            r = nav.pct_change().dropna()
            row[f"{tag}CAGR"] = cagr(nav)
            row[f"{tag}Sharpe"] = (r.mean() - ((1 + cfg.rf_rate) ** (1 / 252) - 1)) / r.std() * np.sqrt(252)
            row[f"{tag}MaxDD"] = drawdown(nav).min()
        rows.append(row)
        print("  " + ", ".join(f"{k}={v}" for k, v in row.items() if not isinstance(v, float)) +
              " | " + ", ".join(f"{k}={v:.3f}" for k, v in row.items() if isinstance(v, float)))
    return pd.DataFrame(rows)
