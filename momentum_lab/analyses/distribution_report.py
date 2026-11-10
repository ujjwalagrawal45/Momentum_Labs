"""Return distribution: the strategy's 1-month returns from the model backtest, against a normal (bell-curve) fit.

    python -m momentum_lab dist configs/quant_composite.json
    python -m momentum_lab dist configs/quant_composite.json --window 21

Uses the backtest NAV (after costs, cash interest, regime filter and vol target, exactly as the strategy runs).
'Rolling' = every 21-trading-day window, one starting each day. Neighbouring windows share 20 of their 21 days,
so they are not independent: the sample is about as informative as the calendar-month count, which is shown
alongside. The Nifty 500 is measured over the same windows.

Shows up in the app (python -m momentum_lab app) -> Results. Survivorship bias applies (universe = today's Nifty 500).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from momentum_lab import StrategyConfig, load_panel, load_universe, results
from momentum_lab.backtest import run_backtest



def describe(x: pd.Series) -> dict:
    """Distribution statistics of a return series (fractions)."""
    x = x.dropna()
    mu, sd = x.mean(), x.std()
    q = x.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"Windows": len(x), "Average %": mu, "Median %": q[0.5], "Standard deviation %": sd,
            "Skew": stats.skew(x), "Excess kurtosis (0 = normal)": stats.kurtosis(x),
            "Worst %": x.min(), "1st percentile %": q[0.01], "5th percentile %": q[0.05],
            "25th percentile %": q[0.25], "75th percentile %": q[0.75], "95th percentile %": q[0.95],
            "99th percentile %": q[0.99], "Best %": x.max(), "Chance of a losing month %": (x < 0).mean(),
            "Chance of losing more than 5% %": (x < -0.05).mean(),
            "Bad month (5% chance), normal fit %": mu - 1.645 * sd,
            "Bad month (5% chance), actual %": q[0.05]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--window", type=int, default=21, help="window length in trading days (21 = one month)")
    a = ap.parse_args()
    cfg = StrategyConfig.from_json(a.config)
    W = a.window
    print("Loading prices...")
    p = load_panel(load_universe(cfg.universe_csv), cfg.benchmark)
    print("Running the model backtest...")
    res = run_backtest(p, cfg)
    nav, bench = res.nav, res.benchmark
    s = nav.pct_change(W).dropna()
    b = bench.pct_change(W).reindex(s.index)
    ex = s - b
    mu, sd = s.mean(), s.std()

    # ---- summary: strategy, Nifty 500, strategy minus Nifty
    ds, db, de = describe(s), describe(b), describe(ex)
    summary = pd.DataFrame({"Metric": list(ds), "Strategy": list(ds.values()), "Nifty 500": list(db.values()),
                            "Strategy minus Nifty": list(de.values())})
    jb_s, jb_b = stats.jarque_bera(s), stats.jarque_bera(b.dropna())
    extra = pd.DataFrame([["Chance of beating Nifty %", float((ex > 0).mean()), np.nan, np.nan],
                          ["Normality test p-value (Jarque-Bera; < 0.05 = not normal)", jb_s.pvalue, jb_b.pvalue, np.nan]],
                         columns=summary.columns)
    summary = pd.concat([summary, extra], ignore_index=True)

    # ---- histogram with the normal curve it would have if returns were normal
    lo, hi = np.floor(s.min() * 100) / 100, np.ceil(s.max() * 100) / 100
    width = 0.01 if hi - lo <= 0.4 else 0.02
    edges = np.arange(lo, hi + width / 2, width)
    if len(edges) < 2:
        edges = np.array([lo, lo + width])
    cnt, _ = np.histogram(s, edges)
    bcnt, _ = np.histogram(b.dropna().clip(edges[0], edges[-1]), edges)
    norm = np.diff(stats.norm.cdf(edges, mu, sd))
    hist = pd.DataFrame({"Return range": [f"{x:+.0%} to {y:+.0%}" for x, y in zip(edges[:-1], edges[1:])],
                         "Strategy %": cnt / cnt.sum(), "Normal curve %": norm, "Nifty 500 %": bcnt / bcnt.sum()})

    # ---- how often returns land within 1, 2, 3 standard deviations (normal: 68.3 / 95.4 / 99.7%)
    z = (s - mu) / sd
    bands = pd.DataFrame([{"Range": f"within ±{k} sd ({mu - k * sd:+.1%} to {mu + k * sd:+.1%})",
                           "Normal curve expects %": stats.norm.cdf(k) - stats.norm.cdf(-k),
                           "Strategy actual %": (z.abs() <= k).mean(),
                           "Months below the range %": (z < -k).mean(), "Months above the range %": (z > k).mean()}
                          for k in (1, 2, 3)])

    # ---- calendar months (non-overlapping) and per year
    me = nav.resample("ME").last()
    bme = bench.resample("ME").last()
    months = pd.DataFrame({"Month": me.index.strftime("%Y-%m"), "Strategy %": me.pct_change().values,
                           "Nifty 500 %": bme.pct_change().values}).dropna()
    months["Strategy minus Nifty %"] = months["Strategy %"] - months["Nifty 500 %"]
    mm, msd = months["Strategy %"].mean(), months["Strategy %"].std()
    months["z-score (sd from average)"] = (months["Strategy %"] - mm) / msd
    months["Percentile among all months %"] = months["Strategy %"].rank(pct=True)
    yr = s.groupby(s.index.year)
    by_year = pd.DataFrame({"Year": yr.size().index.astype(str), "Windows": yr.size().values,
                            "Average %": yr.mean().values, "Standard deviation %": yr.std().values,
                            "Worst %": yr.min().values, "Best %": yr.max().values,
                            "Chance of a losing month %": yr.apply(lambda x: (x < 0).mean()).values,
                            "Nifty 500 average %": b.groupby(b.index.year).mean().values})

    # ---- rolling series for the chart
    roll = pd.DataFrame({"Date": s.index, "Strategy %": s.values, "Nifty 500 %": b.values})
    roll["Average %"] = mu
    roll["-2 sd %"], roll["+2 sd %"] = mu - 2 * sd, mu + 2 * sd

    last = s.iloc[-1]
    notes = (f"{cfg.name} model backtest {nav.index[0].date()} to {nav.index[-1].date()}, after costs and cash interest. "
             f"{len(s)} overlapping {W}-trading-day windows; they share most of their days, so the real sample is about "
             f"{len(months)} independent calendar months (the 'Calendar months' sheet).\n\n"
             f"Bell-curve fit: average {mu:+.1%}, standard deviation {sd:.1%}. About two months in three should land "
             f"between {mu - sd:+.1%} and {mu + sd:+.1%}, and 19 in 20 between {mu - 2 * sd:+.1%} and {mu + 2 * sd:+.1%}. "
             f"Skew {stats.skew(s):+.2f} and excess kurtosis {stats.kurtosis(s):+.2f}: "
             + ("fatter tails than a bell curve, so big months (both ways) happen more often than the normal fit says. "
                if stats.kurtosis(s) > 0.5 else "tails close to a bell curve. ")
             + f"Latest {W}-day window: {last:+.1%}, {(last - mu) / sd:+.1f} sd from the average "
             f"({(s < last).mean():.0%} of windows were lower).\n\n"
             "Survivorship bias (today's Nifty 500 list) inflates the average; treat it as an upper bound.")
    key = f"distribution_{cfg.name}_{W}d"
    results.save(
        key, f"Return distribution: {cfg.name}, {W}-day windows", kind="distribution",
        params={"strategy": cfg.name, "window": W, "from": str(nav.index[0].date()), "to": str(nav.index[-1].date())},
        notes=notes,
        headline=[{"label": "Average month", "value": mu, "fmt": "pct"},
                  {"label": "Standard deviation", "value": sd, "fmt": "pct"},
                  {"label": "Typical range (±1 sd)", "value": f"{mu - sd:+.1%} to {mu + sd:+.1%}", "fmt": "text"},
                  {"label": "Chance of a losing month", "value": float((s < 0).mean()), "fmt": "pct"},
                  {"label": "Bad month (5% chance)", "value": float(s.quantile(0.05)), "fmt": "pct"},
                  {"label": "Chance of beating Nifty", "value": float((ex > 0).mean()), "fmt": "pct"}],
        sheets={"Summary": summary, "Histogram": hist, "Standard-deviation bands": bands, "By year": by_year,
                "Calendar months": months, "Rolling 1-month return": roll},
        charts=[{"sheet": "Histogram", "x": "Return range", "y": ["Strategy %"], "overlay": ["Normal curve %"],
                 "kind": "bar", "title": f"{W}-day returns vs a normal curve (share of windows)", "pct": True},
                {"sheet": "Histogram", "x": "Return range", "y": ["Strategy %", "Nifty 500 %"], "kind": "bar",
                 "title": f"{W}-day returns: strategy vs Nifty 500 (share of windows)", "pct": True},
                {"sheet": "Rolling 1-month return", "x": "Date", "y": ["Strategy %", "Nifty 500 %", "Average %", "-2 sd %", "+2 sd %"],
                 "kind": "line", "title": f"Rolling {W}-day return", "pct": True}],
        formats={"Windows": "num", "z-score (sd from average)": "num", "Skew": "num",
                 "Excess kurtosis (0 = normal)": "num"})
    print(f"Saved -> Results: {key}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
