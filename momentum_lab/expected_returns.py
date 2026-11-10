"""Probability-backed expected returns: turn the composite score into a calibrated forecast.

For every past sample date d (month-ends) we record, for each eligible stock, its cross-sectional
z-score and its excess return over the benchmark from the next open to h trading days later
(h = the strategy's rebalance horizon). An observation only becomes usable once its outcome is
known (d + h days), so every forecast at signal date t is walk-forward.

From that history, at t:
  * rank IC          : mean Spearman correlation of score vs forward excess return
  * P(beat)          : isotonic (monotone) fit of z -> P(stock beats the benchmark over h)
  * expected alpha   : either Grinold's  IC x sigma_h x z  ('grinold'), or the isotonic fit of
                       z -> forward excess return, centred on the training-set average ('calibrated').
                       Centring removes the level drift that survivorship bias puts into today's
                       index members, so alpha is purely cross-sectional.
Alphas are multiplied by cfg.alpha_shrink (forecasts are noisy; 0.5 = halve them).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HORIZON = {"W": 5, "M": 21, "Q": 63}
MIN_DATES = 24          # sample dates needed before forecasts are trusted (else: no forecast)
MIN_NAMES = 30


def _z(s: pd.Series) -> pd.Series:
    return ((s - s.mean()) / s.std()).clip(-3, 3)


class Forecaster:
    def __init__(self, p, scores: pd.DataFrame, elig: pd.DataFrame, cfg):
        from .backtest import rebalance_dates
        self.cfg = cfg
        self.h = HORIZON[cfg.rebalance]
        dates = p.close.index
        self.pos = {d: i for i, d in enumerate(dates)}
        fwd = p.close.shift(-self.h) / p.open.shift(-1) - 1
        bench = p.benchmark.reindex(dates).ffill()
        bfwd = bench.shift(-self.h) / bench - 1
        rows = []
        for d in rebalance_dates(dates, "M"):
            if self.pos[d] + self.h >= len(dates):
                break
            s = scores.loc[d].where(elig.loc[d]).dropna()
            f = fwd.loc[d, s.index].dropna()
            if len(f) < MIN_NAMES:
                continue
            ex = f - bfwd.loc[d]
            rows.append(pd.DataFrame({"date": d, "avail": self.pos[d] + self.h, "z": _z(s[f.index]).values,
                                      "excess": ex.values}))
        self.obs = pd.concat(rows, ignore_index=True) if rows else \
            pd.DataFrame(columns=["date", "avail", "z", "excess"])
        self.ic = self.obs.groupby("date")[["z", "excess"]].apply(lambda g: g["z"].corr(g["excess"], method="spearman"))
        self.ic_avail = pd.Series([self.pos[d] + self.h for d in self.ic.index], index=self.ic.index)
        self._cache = {}

    # ------------------------------------------------------------------------------------------
    def fit(self, t: pd.Timestamp) -> dict | None:
        """Walk-forward fit using only outcomes known by signal date t."""
        if t in self._cache:
            return self._cache[t]
        k = self.pos[t]
        tr = self.obs[self.obs["avail"] <= k]
        if self.cfg.forecast_lookback_years:
            tr = tr[tr["date"] >= t - pd.DateOffset(years=self.cfg.forecast_lookback_years)]
        out = None
        if tr["date"].nunique() >= MIN_DATES:
            ic = self.ic[(self.ic_avail <= k) & self.ic.index.isin(tr["date"].unique())]
            beat = IsotonicRegression(increasing="auto", y_min=0, y_max=1, out_of_bounds="clip")
            beat.fit(tr["z"], (tr["excess"] > 0).astype(float))
            exr = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(tr["z"], tr["excess"])
            out = {"ic": float(ic.mean()), "ic_t": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
                   "n_dates": int(tr["date"].nunique()), "beat": beat, "excess": exr,
                   "mean_excess": float(tr["excess"].mean())}
        self._cache[t] = out
        return out

    def forecast(self, t: pd.Timestamp, scores: pd.Series, vol_ann: pd.Series, universe: pd.Series) -> pd.DataFrame | None:
        """Per-stock forecast for the names in `scores` at t: z, P(beat), alpha (excess vs benchmark
        over the horizon, already shrunk). `universe` = the eligible cross-section's scores (for z)."""
        f = self.fit(t)
        if f is None:
            return None
        u = universe.dropna()
        z = ((scores - u.mean()) / u.std()).clip(-3, 3)
        sig_h = vol_ann.reindex(z.index).fillna(vol_ann.median()) * np.sqrt(self.h / 252)
        if self.cfg.alpha_model == "grinold":
            a = f["ic"] * sig_h * z
        else:
            a = pd.Series(f["excess"].predict(z.values), index=z.index) - f["mean_excess"]
        return pd.DataFrame({"z": z, "p_beat": f["beat"].predict(z.values), "alpha": a * self.cfg.alpha_shrink,
                             "sigma_h": sig_h})


def calibration_table(fc: Forecaster, split: str | None = None, n_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample check: at each sample date, forecast with the walk-forward fit, then compare
    predicted P(beat) / alpha with what happened. Returns (decile table, per-date IC table)."""
    rows = []
    for d, g in fc.obs.groupby("date"):
        if split and d < pd.Timestamp(split):
            continue
        f = fc.fit(d)
        if f is None:
            continue
        rows.append(g.assign(p_beat=f["beat"].predict(g["z"].values),
                             pred_excess=f["excess"].predict(g["z"].values) - f["mean_excess"],
                             realised_rel=g["excess"] - g["excess"].mean(), ic_known=f["ic"]))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    o = pd.concat(rows, ignore_index=True)
    o["bin"] = o.groupby("date")["z"].transform(lambda s: pd.qcut(s.rank(method="first"), n_bins, labels=False) + 1)
    t = o.groupby("bin").agg(names=("z", "size"), avg_z=("z", "mean"), predicted_P_beat=("p_beat", "mean"),
                             realised_P_beat=("excess", lambda x: (x > 0).mean()),
                             predicted_rel_return=("pred_excess", "mean"),
                             realised_rel_return=("realised_rel", "mean"),
                             realised_excess_vs_bench=("excess", "mean")).reset_index()
    t = t.rename(columns={"bin": "score decile (10 = best)"})
    g = o.groupby("date")
    ic = pd.DataFrame({"ic_realised": g[["z", "excess"]].apply(lambda x: x["z"].corr(x["excess"], method="spearman")),
                       "ic_known_beforehand": g["ic_known"].first(),
                       "top_decile_excess_vs_bench": o[o["bin"] == n_bins].groupby("date")["excess"].mean()})
    return t, ic.reset_index()
