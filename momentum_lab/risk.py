"""Risk model, portfolio-weighting models and strategy-level exposure overlays.

Covariance: sample covariance of the picks' daily returns over cfg.cov_lookback days, shrunk towards a
single-index (market-model) target, Ledoit-Wolf style: delta = sum Var(s_ij) / sum (f_ij - s_ij)^2,
scaled to the forecast horizon.

Weighting (cfg.weighting), all long-only, fully invested within the stock sleeve, capped at max_weight:
  min_variance    : lowest-risk mix of the picks (ignores forecasts)
  erc             : equal risk contribution (Maillard, Roncalli & Teiletche 2010)
  erc_tilt        : risk budgets proportional to each pick's calibrated P(beat benchmark)
  hrp             : hierarchical risk parity (Lopez de Prado 2016)
  mean_variance   : max  w'alpha - risk_aversion/2 * w'Sigma w
  black_litterman : equal-weight prior + the alphas as views (He & Litterman), then mean-variance
  kelly           : fractional multi-asset Kelly  w = kelly_fraction * Sigma^-1 mu  (mu = equity premium + alpha);
                    if the Kelly weights add up to less than 1 the rest stays in cash (an exposure overlay)

Exposure overlays (strategy level, the smallest one wins): max_allocation, regime filter, vol target,
Kelly sum, Grossman-Zhou drawdown control. Changes smaller than exposure_band are skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from .expected_returns import Forecaster

MODEL_WEIGHTINGS = ("min_variance", "erc", "erc_tilt", "hrp", "mean_variance", "black_litterman", "kelly")
NEEDS_FORECAST = ("erc_tilt", "mean_variance", "black_litterman", "kelly")
BL_TAU = 0.05


def shrunk_cov(r: pd.DataFrame, m: pd.Series) -> tuple[np.ndarray, float]:
    """Daily covariance of `r` (T x N), shrunk towards the single-index model built on market returns `m`."""
    x = r.to_numpy()
    x = x - np.nanmean(x, axis=0)
    x = np.nan_to_num(x)
    t = len(x)
    s = x.T @ x / t
    mk = np.nan_to_num(m.to_numpy() - np.nanmean(m.to_numpy()))
    vm = mk @ mk / t
    beta = x.T @ mk / t / vm if vm > 0 else np.zeros(x.shape[1])
    f = np.outer(beta, beta) * vm
    np.fill_diagonal(f, np.diag(s))
    pi = ((x[:, :, None] * x[:, None, :] - s) ** 2).mean(axis=0).sum() / t
    gamma = ((f - s) ** 2).sum()
    d = float(np.clip(pi / gamma, 0, 1)) if gamma > 0 else 1.0
    return d * f + (1 - d) * s, d


def _box_opt(fun, n, cap, x0=None):
    x0 = np.full(n, 1 / n) if x0 is None else x0
    res = minimize(fun, x0, method="SLSQP", bounds=[(0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x if res.success or np.isfinite(res.x).all() else x0, 0, None)
    return w / w.sum()


def risk_budget(cov: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Weights whose risk contributions w_i (Sigma w)_i are proportional to b (Spinu's convex form)."""
    b = b / b.sum()
    sd = np.sqrt(np.diag(cov))
    res = minimize(lambda y: 0.5 * y @ cov @ y - b @ np.log(y), b / sd, method="L-BFGS-B",
                   jac=lambda y: cov @ y - b / y, bounds=[(1e-10, None)] * len(b))
    return res.x / res.x.sum()


def hrp(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), "single"))
    w = np.ones(len(sd))

    def cvar(ix):
        c = cov[np.ix_(ix, ix)]
        iv = 1 / np.diag(c)
        iv /= iv.sum()
        return iv @ c @ iv

    stack = [list(order)]
    while stack:
        ix = stack.pop()
        if len(ix) < 2:
            continue
        a, b = ix[:len(ix) // 2], ix[len(ix) // 2:]
        va, vb = cvar(a), cvar(b)
        alpha = 1 - va / (va + vb)
        w[a] *= alpha
        w[b] *= 1 - alpha
        stack += [a, b]
    return w / w.sum()


class PortfolioModel:
    """Everything the model-based weightings need, computed point-in-time at each signal date."""

    def __init__(self, p, scores: pd.DataFrame, elig: pd.DataFrame, cfg):
        self.p, self.cfg, self.scores, self.elig = p, cfg, scores, elig
        self.ret = p.close.pct_change(fill_method=None)
        self.mret = p.benchmark.reindex(p.close.index).ffill().pct_change()
        self.fc = Forecaster(p, scores, elig, cfg) if cfg.weighting in NEEDS_FORECAST else None
        self.last = {}  # diagnostics of the latest call (shrinkage, forecasts, Kelly sum)

    def weights(self, sd: pd.Timestamp, names: list, vol_ann: pd.Series) -> tuple[pd.Series, float | None]:
        """(sleeve weights summing to 1, Kelly exposure hint or None)."""
        cfg = self.cfg
        r = self.ret.loc[:sd].iloc[-cfg.cov_lookback:][names]
        r = r.loc[:, r.notna().mean() > 0.5]
        if r.shape[1] < 2:
            return pd.Series(1 / len(names), index=names), None
        h = self.fc.h if self.fc else {"W": 5, "M": 21, "Q": 63}[cfg.rebalance]
        cov_d, delta = shrunk_cov(r, self.mret.reindex(r.index))
        cov = cov_d * h
        ix = list(r.columns)
        n, cap = len(ix), max(cfg.max_weight, 1 / len(ix))
        self.last = {"shrinkage": delta, "names": n}
        fc = None
        if self.fc is not None:
            uni = self.scores.loc[sd].where(self.elig.loc[sd])
            fc = self.fc.forecast(sd, self.scores.loc[sd, ix], vol_ann, uni)
            if fc is None:  # not enough history yet: fall back to pure risk weighting
                self.last["fallback"] = "no forecast yet -> erc"
                return pd.Series(risk_budget(cov, np.ones(n)), index=ix).reindex(names).fillna(0), None
            self.last["forecast"] = fc
        kelly_sum = None
        if cfg.weighting == "min_variance":
            w = _box_opt(lambda w: w @ cov @ w, n, cap)
        elif cfg.weighting == "erc":
            w = risk_budget(cov, np.ones(n))
        elif cfg.weighting == "erc_tilt":
            w = risk_budget(cov, fc["p_beat"].clip(0.05).to_numpy())
        elif cfg.weighting == "hrp":
            w = hrp(cov)
        elif cfg.weighting in ("mean_variance", "black_litterman"):
            a = fc["alpha"].to_numpy()
            if cfg.weighting == "black_litterman":
                pi = cfg.risk_aversion * cov @ np.full(n, 1 / n)        # implied returns of equal weight
                tc = BL_TAU * cov
                omega = np.diag(np.diag(tc)) * (1 - cfg.view_confidence) / max(cfg.view_confidence, 1e-6)
                a = pi + tc @ np.linalg.solve(tc + omega, a)            # views: alpha above equilibrium
            lam = cfg.risk_aversion
            w = _box_opt(lambda w: -(w @ a) + lam / 2 * (w @ cov @ w), n, cap)
        elif cfg.weighting == "kelly":
            mu = cfg.equity_premium * h / 252 + fc["alpha"].to_numpy()
            raw = np.clip(cfg.kelly_fraction * np.linalg.solve(cov, mu), 0, None)
            kelly_sum = float(raw.sum())
            self.last["kelly_sum"] = kelly_sum
            w = raw / raw.sum() if raw.sum() > 0 else np.full(n, 1 / n)
        else:
            raise ValueError(cfg.weighting)
        return pd.Series(w, index=ix).reindex(names).fillna(0), kelly_sum


def exposure(cfg, below_ma: bool, nav_hist: pd.Series | None, prev_gross: float | None,
             kelly_sum: float | None) -> tuple[float, str]:
    """Strategy-level exposure: the smallest of the enabled overlays. Returns (gross, binding rule)."""
    cands = {"max allocation": cfg.max_allocation}
    if cfg.regime_filter == "bench_200dma" and below_ma:
        cands["regime filter"] = cfg.regime_exposure
    if nav_hist is not None and len(nav_hist) > cfg.vol_lookback:
        r = nav_hist.pct_change().iloc[-cfg.vol_lookback:]
        if cfg.vol_target:
            rv = r.std() * np.sqrt(252)
            if rv > 0:
                cands["vol target"] = cfg.vol_target / rv
    if kelly_sum is not None:
        cands["kelly"] = kelly_sum
    if cfg.dd_limit and nav_hist is not None and len(nav_hist):
        w, peak = float(nav_hist.iloc[-1]), float(nav_hist.max())
        cushion = (w - (1 - cfg.dd_limit) * peak) / w
        cands["drawdown control"] = max(cfg.dd_multiplier * cushion, cfg.dd_floor)
    rule = min(cands, key=cands.get)
    g = float(np.clip(cands[rule], 0, 1))
    if prev_gross is not None and cfg.exposure_band and abs(g - prev_gross) < cfg.exposure_band:
        return prev_gross, "unchanged (within exposure band)"
    return g, rule
