"""Forecast & risk-model report: is the strategy's score a calibrated, probability-backed forecast?
And what would each portfolio model do with today's picks?

    python -m momentum_lab forecast                         # strategies marked 'Include in rolling backtest'
    python -m momentum_lab forecast configs/quant_composite.json --split 2022-01-01

Everything is walk-forward: a forecast at date t is fitted only on outcomes known by t.
Tables (in the app's Results page):
  Summary      per strategy: IC, IC t-stat, Brier skill, how well the top decile was predicted
  Calibration  per score decile: predicted vs realised P(beat Nifty) and relative return
  IC by date   realised rank IC of every sample date vs the IC known beforehand
  Today        the latest signal date's picks: z-score, P(beat), expected alpha, and the weight each
               model would give (equal, inverse_vol, erc, erc_tilt, hrp, min_variance, mean_variance,
               black_litterman, kelly), plus correlation diagnostics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_lab.paths import CONFIGS
from momentum_lab import StrategyConfig, load_panel, load_universe, results
from momentum_lab import signals as sig
from momentum_lab.backtest import build_weights, eligibility, rebalance_dates, select
from momentum_lab.expected_returns import Forecaster, calibration_table
from momentum_lab.risk import MODEL_WEIGHTINGS, PortfolioModel, shrunk_cov
from momentum_lab.settings import globals_

G = globals_()


def today_sheet(p, cfg, scores, elig, vol) -> tuple[pd.DataFrame, dict]:
    sd = rebalance_dates(p.close.index, cfg.rebalance)[-1]
    sc = scores.loc[sd].where(elig.loc[sd]).dropna()
    names = select(sc, set(), p.meta["sector"], cfg)
    out = pd.DataFrame(index=names)
    out["name"] = p.meta["name"].reindex(names).values
    out["sector"] = p.meta["sector"].reindex(names).values
    out["score"] = sc[names]
    out["vol_1y %"] = vol.loc[sd, names]
    fc_cfg = cfg.replace(weighting="kelly")
    pm = PortfolioModel(p, scores, elig, fc_cfg)
    fc = pm.fc.forecast(sd, sc[names], vol.loc[sd], scores.loc[sd].where(elig.loc[sd]))
    if fc is not None:
        out["z"] = fc["z"]
        out["P(beat Nifty next period) %"] = fc["p_beat"]
        out["expected alpha next period %"] = fc["alpha"]
    diag = {"signal date": sd.date(), "picks": len(names)}
    for wt in ["equal", "inverse_vol", *MODEL_WEIGHTINGS]:
        c = cfg.replace(weighting=wt)
        model = PortfolioModel(p, scores, elig, c) if wt in MODEL_WEIGHTINGS else None
        if model is not None:
            model.fc = pm.fc if model.fc is not None else None  # reuse the one fitted forecaster
        out[f"w_{wt} %"] = build_weights(names, sc, vol.loc[sd], c, model, sd).reindex(names).fillna(0)
        if wt == "kelly" and model is not None and "kelly_sum" in model.last:
            diag["Kelly: share of capital it would invest"] = min(model.last["kelly_sum"], 1.0)
            diag["Kelly: raw sum of weights (before 100% cap)"] = model.last["kelly_sum"]
    r = p.close.pct_change(fill_method=None).loc[:sd].iloc[-cfg.cov_lookback:][names]
    r = r.loc[:, r.notna().mean() > 0.5]
    cov, delta = shrunk_cov(r, p.benchmark.reindex(p.close.index).ffill().pct_change().reindex(r.index))
    sd_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd_, sd_)
    ev = np.linalg.eigvalsh(corr)
    diag.update({"Average pairwise correlation of picks": corr[np.triu_indices(len(corr), 1)].mean(),
                 "Effective number of independent bets": ev.sum() ** 2 / (ev ** 2).sum(),
                 "Covariance shrinkage towards market model": delta,
                 "Equal-weight portfolio volatility (1y, ann.)": float(np.sqrt(np.full(len(sd_), 1 / len(sd_)) @ cov
                                                                            @ np.full(len(sd_), 1 / len(sd_)) * 252))})
    return out.reset_index(names="symbol"), diag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--split", help="only score forecasts made on/after this date (default: all, still walk-forward)")
    a = ap.parse_args()
    paths = a.configs or [str(CONFIGS / f"{n}.json") for n in G["rolling_strategies"]]
    cfgs = [StrategyConfig.from_json(x) for x in paths]
    print("Loading prices...")
    p = load_panel(load_universe(cfgs[0].universe_csv), cfgs[0].benchmark)
    vol = sig.daily_vol(p.close, 63)
    summary, calib, ics, today, diags = [], [], [], [], []
    for cfg in cfgs:
        print(f"[{cfg.name}] fitting walk-forward forecasts...")
        scores = sig.composite(p, cfg.signal_weights)
        elig = eligibility(p, cfg)
        fc = Forecaster(p, scores, elig, cfg)
        t, ic = calibration_table(fc, a.split)
        if t.empty:
            print("  not enough history")
            continue
        o = fc.obs[fc.obs["date"] >= pd.Timestamp(a.split)] if a.split else fc.obs
        base = (o["excess"] > 0).mean()
        brier_model = float(((t["predicted_P_beat"] - t["realised_P_beat"]) ** 2 * t["names"]).sum() / t["names"].sum())
        top = t.iloc[-1]
        summary.append({"strategy": cfg.name, "forecast horizon (days)": fc.h,
                        "sample dates scored": len(ic), "mean rank IC": ic["ic_realised"].mean(),
                        # monthly samples of a fc.h-day outcome overlap, so deflate the naive t-stat
                        "IC t-stat (overlap-adjusted)": ic["ic_realised"].mean() / ic["ic_realised"].std()
                        * np.sqrt(len(ic) / max(fc.h / 21, 1)),
                        "dates with IC > 0 %": (ic["ic_realised"] > 0).mean(),
                        "base rate: stocks beating Nifty %": base,
                        "top decile: predicted P(beat) %": top["predicted_P_beat"],
                        "top decile: realised P(beat) %": top["realised_P_beat"],
                        "top decile: predicted relative return %": top["predicted_rel_return"],
                        "top decile: realised relative return %": top["realised_rel_return"],
                        "top decile: realised excess vs Nifty %": top["realised_excess_vs_bench"],
                        "calibration error (decile mean sq. gap, P(beat))": brier_model,
                        "monotone? (realised P(beat) rises with decile)": "YES" if t["realised_P_beat"].is_monotonic_increasing
                        else f"mostly (Spearman {t['score decile (10 = best)'].corr(t['realised_P_beat'], method='spearman'):.2f})"})
        calib.append(t.assign(strategy=cfg.name))
        ics.append(ic.assign(strategy=cfg.name))
        tt, dg = today_sheet(p, cfg, scores, elig, vol)
        today.append(tt.assign(strategy=cfg.name))
        diags.append({"strategy": cfg.name, **dg})

    def front(df):
        return df[["strategy"] + [c for c in df.columns if c != "strategy"]]

    sheets = {"Summary": pd.DataFrame(summary).set_index("strategy").T.reset_index(names="Metric"),
              "Calibration": front(pd.concat(calib, ignore_index=True)),
              "IC by date": front(pd.concat(ics, ignore_index=True)),
              "Today": front(pd.concat(today, ignore_index=True)),
              "Today - risk": pd.DataFrame(diags).set_index("strategy").T.reset_index(names="Metric")}
    calib_all = sheets["Calibration"]
    chart_rows = calib_all.pivot_table(index="score decile (10 = best)", columns="strategy",
                                       values="realised_P_beat").reset_index()
    chart_rows.columns = [str(c) for c in chart_rows.columns]
    sheets["Calibration chart"] = chart_rows
    path = results.save("forecast", "Forecasts & risk models", sheets, kind="forecast",
                        params={"split": a.split},
                        notes="Walk-forward: every forecast is fitted only on outcomes known at the time. "
                              "'Today' shows the latest signal date's picks and the weight each portfolio "
                              "model would give them.",
                        charts=[{"sheet": "Calibration chart", "x": "score decile (10 = best)",
                                 "y": [c for c in chart_rows.columns if c != "score decile (10 = best)"],
                                 "kind": "bar", "pct": True,
                                 "title": "Realised chance of beating Nifty, by score decile (10 = best)"}],
                        headline=[{"label": f"{r['strategy']}: mean rank IC", "fmt": "num",
                                   "value": r["mean rank IC"]} for r in summary])
    print(f"\n{pd.DataFrame(summary)[['strategy', 'mean rank IC', 'IC t-stat (overlap-adjusted)', 'top decile: predicted P(beat) %', 'top decile: realised P(beat) %']].to_string(index=False)}")
    print(f"\nSaved {path.name}: open the app (python -m momentum_lab app) -> Results.")


if __name__ == "__main__":
    main()
