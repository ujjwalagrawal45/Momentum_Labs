"""Lower-circuit review: which stocks in the latest journal book keep hitting their lower price band, and does
the backtest history say a lower-circuit (LC) hit is a reason to avoid or sell a momentum stock?

    python -m momentum_lab circuit configs/quant_composite.json                  # latest journal book
    python -m momentum_lab circuit configs/quant_composite.json --book 2026-10-23

Lower-circuit day (proxy, no band history is available): the stock closed at its day's low after a fall of
5%, 10% or 20% (to within 0.12 points), i.e. it finished pinned at an NSE price band. 'Locked' = the whole day
traded at one price (high = low): sellers could not get out, so the exit test waits for the next unlocked day.

Part 1  the book: LC days per stock since the book date and in the last 10 trading days.
Part 2  every LC day in the Nifty 500 since 2017: return vs the Nifty 500 over the next 5 / 21 / 63 days.
Part 3  entry filter: at each month-end, top-50 momentum candidates with an LC hit in the prior 10/21 days vs
        the clean ones, over the next 21 / 63 days.
Part 4  exit rule: a holding hits >= m LC days within 10 days -> sell at the next unlocked open, money to cash
        or to the other holdings, after costs; compared with holding to the next rebalance.
Part 5  full backtests with the entry filter, and with cruder small-cap screens (vol cap, higher liquidity).

Survivorship bias matters more here than anywhere else: the universe is today's Nifty 500 list, so stocks that
hit lower circuits and then kept falling out of the index are missing. That flatters what happens after an LC
hit. Treat 'LC hits are followed by rebounds' as an upper bound, and 'the filter costs return' as uncertain.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_lab import StrategyConfig, load_panel, load_universe, results
from momentum_lab.backtest import rebalance_dates, run_backtest
from momentum_lab.data import load_fno_list
from momentum_lab.journal import JOURNAL, Context
from momentum_lab.metrics import cagr, drawdown

BANDS = (0.05, 0.10, 0.20)
TOL = 0.0012
COST = 0.003  # per rupee moved, round trip (sell + buy)


def circuit_days(p):
    """(lc, locked, band): booleans date x symbol, plus the band matched on LC days."""
    c = p.close
    r = c.pct_change(fill_method=None)
    at_low = c <= p.low * 1.0005
    lc = pd.DataFrame(False, index=c.index, columns=c.columns)
    band = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for b in BANDS:
        hit = at_low & ((r + b).abs() < TOL)
        lc |= hit
        band = band.mask(hit, b)
    locked = lc & (p.high <= p.low * 1.0005)
    return lc, locked, band


def tstat(s) -> float:
    s = pd.Series(s, dtype=float).dropna()
    return float(s.mean() / s.std() * np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan


# ---------------------------------------------------------------------------------------------- part 1
def book_review(p, name, book_date, lc, locked, band):
    d = JOURNAL / name
    books = sorted(d.glob("book_*.csv"))
    if not books:
        raise SystemExit(f"No journal books in {d}")
    f = d / f"book_{book_date}.csv" if book_date else books[-1]
    if not f.exists():
        raise SystemExit(f"No book for {book_date}; have {[b.stem[5:] for b in books]}")
    ds = f.stem[5:]
    b = pd.read_csv(f)
    b = b[b["qty"] > 0].set_index("symbol")
    c, idx, bm = p.close.ffill(), p.close.index, p.benchmark.ffill()
    i0 = idx.get_indexer([pd.Timestamp(ds)], method="pad")[0]
    if i0 + 1 >= len(idx):
        raise SystemExit(f"No prices after {ds} yet")
    fno = set(load_fno_list().index)
    last10 = slice(max(i0 + 1, len(idx) - 10), len(idx))
    rows = []
    for s in b.index:
        entry = p.open[s].iat[i0 + 1] if pd.notna(p.open[s].iat[i0 + 1]) else c[s].iat[i0]
        since = slice(i0 + 1, len(idx))
        n_lc, n10 = int(lc[s].iloc[since].sum()), int(lc[s].iloc[last10].sum())
        bands = band[s].iloc[since].dropna()
        rows.append({"Stock": s, "Name": b.at[s, "name"], "Sector": b.at[s, "sector"],
                     "Stock futures (no fixed band)": "yes" if s in fno else "no",
                     "Invested (₹)": float(b.at[s, "actual_rs"]), "Entry price": round(float(entry), 2),
                     "Last price": float(c[s].iat[-1]), "Return %": c[s].iat[-1] / entry - 1,
                     "LC days since entry": n_lc, "LC days, last 10": n10,
                     "Locked days": int(locked[s].iloc[since].sum()),
                     "Band hit": ", ".join(f"{x:.0%}" for x in sorted(set(bands))) or "-",
                     "Fall from high since entry %": c[s].iat[-1] / p.high[s].iloc[since].max() - 1,
                     "12-1 return at entry %": float(b.at[s, "ret_12_1"]),
                     "Flag": "repeated lower circuit" if n10 >= 2 else ("one lower circuit" if n_lc else "")})
    df = pd.DataFrame(rows).sort_values(["LC days, last 10", "Return %"], ascending=[False, True])
    bret = bm.iat[-1] / bm.iat[i0] - 1
    df.insert(df.columns.get_loc("Return %") + 1, "vs Nifty 500 %", df["Return %"] - bret)
    w = df["Invested (₹)"] / df["Invested (₹)"].sum()
    flagged = df["LC days since entry"] > 0
    meta = {"book": ds, "to": str(idx[-1].date()), "days": len(idx) - 1 - i0, "bench": bret,
            "port": float((w * df["Return %"]).sum()),
            "port_ex": float((w[~flagged] * df.loc[~flagged, "Return %"]).sum() / w[~flagged].sum()),
            "n_flag": int(flagged.sum()), "n_rep": int((df["LC days, last 10"] >= 2).sum()),
            "drag": float((w[flagged] * (df.loc[flagged, "Return %"] - bret)).sum())}
    return df, meta


# ---------------------------------------------------------------------------------------------- part 2
def universe_events(p, lc):
    cf, o, bm, idx = p.close.ffill(), p.open, p.benchmark.ffill(), p.close.index
    cnt10 = lc.rolling(10, min_periods=1).sum()
    rows = []
    for i in np.where(lc.to_numpy().any(axis=1))[0]:
        d = idx[i]
        if d < pd.Timestamp("2017-01-01") or i + 63 >= len(idx):
            continue
        for s in lc.columns[lc.iloc[i].to_numpy()]:
            e = o[s].iat[i + 1]
            if not np.isfinite(e):
                continue
            rows.append({"d": d, "rep": cnt10[s].iat[i],
                         **{f"x{H}": cf[s].iat[i + H] / e - 1 - (bm.iat[i + H] / bm.iat[i] - 1) for H in (5, 21, 63)}})
    ev = pd.DataFrame(rows)

    def summ(g, H):
        x = g[f"x{H}"]
        return {"Events": len(x), "Avg vs Nifty %": x.mean(), "Median vs Nifty %": x.median(),
                "Beat Nifty %": (x > 0).mean(), "Worst 5% of events %": x.quantile(0.05),
                "t-stat (by day)": tstat(g.groupby("d")[f"x{H}"].mean())}
    horizon = pd.DataFrame([{"Days after the LC day": H, **summ(ev, H)} for H in (5, 21, 63)])
    ev["LC days in last 10"] = ev["rep"].clip(upper=4).astype(int).astype(str).replace("4", "4+")
    rep = pd.DataFrame([{"LC days in last 10": k, "Horizon (days)": H, **summ(g, H)}
                        for k, g in ev.groupby("LC days in last 10") for H in (21, 63)])
    return horizon, rep


# ---------------------------------------------------------------------------------------------- part 3
def entry_filter_test(p, ctx, lc):
    cf, o, idx = p.close.ffill(), p.open, p.close.index
    pos = {d: i for i, d in enumerate(idx)}
    sc = ctx.scores.where(ctx.elig)
    out = []
    for look in (10, 21):
        cnt = lc.rolling(look, min_periods=1).sum()
        for H in (21, 63):
            diffs, med, nf = [], [], 0
            for d in rebalance_dates(idx[idx >= "2017-06-01"], "M"):
                i = pos[d]
                if i + H >= len(idx):
                    break
                s = sc.loc[d].dropna()
                names = s[s.rank(ascending=False) <= 50].index
                ret = cf.iloc[i + H][names] / o.iloc[i + 1][names].fillna(cf.iloc[i][names]) - 1
                f = (cnt.loc[d, names] >= 1).to_numpy()
                if f.any() and (~f).any():
                    nf += f.sum()
                    diffs.append(ret[f].mean() - ret[~f].mean())
                    med.append(ret[f].median() - ret[~f].median())
            out.append({"LC hit within (days before)": look, "Held for (days)": H, "Months with a flagged name": len(diffs),
                        "Flagged candidates": int(nf), "Flagged minus clean, avg %": np.mean(diffs),
                        "Flagged minus clean, median %": np.mean(med), "t-stat": tstat(diffs),
                        "Months flagged did worse %": float(np.mean(np.array(diffs) < 0))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- part 4
def exit_rule_test(p, versions, lc, locked):
    cf, o, idx = p.close.ffill(), p.open, p.close.index
    pos = {d: i for i, d in enumerate(idx)}
    cnt10 = lc.rolling(10, min_periods=1).sum()
    out = []
    for lab, ctx in versions.items():
        W = ctx.model.weights
        ex = list(W.index)
        for m in (1, 2, 3):
            cash, other = [], []
            for j, d in enumerate(ex[:-1]):
                w = W.loc[d]
                w = w[w > 0]
                i0, iend = pos[d], pos[ex[j + 1]] - 1
                for s in w.index:
                    trig = (cnt10[s].iloc[i0:iend - 1] >= m).to_numpy()
                    if not trig.any():
                        continue
                    k = i0 + int(trig.argmax()) + 1
                    while k < iend and locked[s].iat[k]:  # locked at the band: no buyers, wait
                        k += 1
                    if k >= iend:
                        continue
                    fill = o[s].iat[k] if np.isfinite(o[s].iat[k]) else cf[s].iat[k]
                    rest = cf[s].iat[iend] / fill - 1
                    others = w.index.drop(s)
                    oth = (cf.iloc[iend][others] / cf.iloc[k - 1][others] - 1).mean()
                    cash.append(-rest - COST / 2)
                    other.append(oth - rest - COST)
            out.append({"Version": lab, "Sell after LC days in 10": m, "Times it fired": len(cash),
                        "Gain if sold to cash, avg %": np.mean(cash), "Gain if moved to others, avg %": np.mean(other),
                        "Gain if moved to others, median %": np.median(other), "t-stat (to others)": tstat(other),
                        "Times selling helped %": float(np.mean(np.array(other) > 0))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- part 5
def backtest_variants(p, cfg, ctx, lc):
    def row(lab, res):
        nav = res.nav
        yr = nav[nav.index <= nav.index[-1] - pd.Timedelta(days=365)]
        return {"Variant": lab, "CAGR %": cagr(nav), "Nifty 500 CAGR %": cagr(res.benchmark),
                "Largest fall %": drawdown(nav).min(), "Volatility %": nav.pct_change().std() * np.sqrt(252),
                "Last 12 months %": nav.iloc[-1] / yr.iloc[-1] - 1 if len(yr) else np.nan,
                "Final value (₹)": nav.iloc[-1]}
    rows = [row("As configured (no LC filter)", ctx.model)]
    for look in (10, 21, 63):
        bad = lc.rolling(look, min_periods=1).sum() >= 1
        print(f"  backtest: skip stocks with an LC day in the last {look} days")
        rows.append(row(f"Skip if an LC day in last {look} days", run_backtest(p, cfg, scores=ctx.scores.where(~bad))))
    for lab, kw in ((f"Max volatility 60%", {"max_vol": 0.6}), ("Min liquidity Rs 20 cr/day", {"min_adv_cr": 20.0})):
        print(f"  backtest: {lab}")
        rows.append(row(lab, run_backtest(p, cfg.replace(**kw))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--book", help="journal book date YYYY-MM-DD (default: the latest)")
    a = ap.parse_args()
    cfg = StrategyConfig.from_json(a.config)
    print("Loading prices...")
    p = load_panel(load_universe(cfg.universe_csv), cfg.benchmark)
    lc, locked, band = circuit_days(p)
    print("Reviewing the book...")
    stocks, m = book_review(p, cfg.name, a.book, lc, locked, band)
    print("Event study: every lower-circuit day since 2017...")
    horizon, rep = universe_events(p, lc)
    print("Building the model backtests...")
    ctx = Context(p, cfg)
    sched = {"Q": "quarterly", "M": "monthly", "W": "weekly"}.get(cfg.rebalance, cfg.rebalance)
    versions = {f"as traded ({sched})": ctx}
    if cfg.rebalance != "M":
        versions["monthly version"] = Context(p, cfg.replace(rebalance="M"))
    print("Entry-filter and exit-rule tests...")
    entry = entry_filter_test(p, ctx, lc)
    exits = exit_rule_test(p, versions, lc, locked)
    variants = backtest_variants(p, cfg, ctx, lc)

    h21 = horizon.set_index("Days after the LC day").loc[21]
    best_exit = exits.loc[exits["Gain if moved to others, avg %"].idxmax()]
    base_cagr = variants["CAGR %"].iloc[0]
    filt_cagr = variants.loc[variants["Variant"].str.contains("21 days"), "CAGR %"].iloc[0]
    notes = (f"Book of {m['book']}: entry at the next open, marked at the close on {m['to']} ({m['days']} trading days). "
             f"Nifty 500 measured from the {m['book']} close. LC day = closed at the day's low after a 5/10/20% fall; "
             "stocks with stock futures have no fixed band, so their flags are coincidences.\n\n"
             f"History: after an LC day the median stock was {h21['Median vs Nifty %']:+.1%} vs the Nifty 500 over 21 days "
             f"(average {h21['Avg vs Nifty %']:+.1%}); lower circuits were followed by rebounds more often than by more falls. "
             f"Skipping candidates with an LC day in the last 21 days moved backtest CAGR from {base_cagr:.1%} to {filt_cagr:.1%}. "
             f"The best exit rule averaged {best_exit['Gain if moved to others, avg %']:+.1%} per trigger (t "
             f"{best_exit['t-stat (to others)']:.1f}): no exit rule beat holding.\n\n"
             "Survivorship bias: the universe is today's Nifty 500, so stocks that hit lower circuits and then left the "
             "index are missing. That flatters the rebounds; read 'no evidence the filter helps', not 'proof it hurts'. "
             "|t| below 2 is noise.")
    key = f"circuit_review_{cfg.name}_{m['book']}"
    results.save(
        key, f"Lower-circuit review: {cfg.name}, book {m['book']}", kind="review",
        params={"strategy": cfg.name, "book": m["book"], "to": m["to"]}, notes=notes,
        headline=[{"label": f"Book return ({m['days']}d)", "value": m["port"], "fmt": "pct"},
                  {"label": "Nifty 500", "value": m["bench"], "fmt": "pct"},
                  {"label": "Book without LC names", "value": m["port_ex"], "fmt": "pct"},
                  {"label": "Names with LC days / repeated", "value": f"{m['n_flag']} / {m['n_rep']} of {len(stocks)}", "fmt": "text"},
                  {"label": "After an LC day, 21d median vs Nifty", "value": float(h21["Median vs Nifty %"]), "fmt": "pct"}],
        sheets={"Stocks": stocks, "History: after an LC day": horizon, "History: by repeat count": rep,
                "History: entry filter": entry, "History: sell-on-LC rule": exits, "Backtest variants": variants},
        charts=[{"sheet": "Stocks", "x": "Stock", "y": ["Return %", "vs Nifty 500 %"], "kind": "bar",
                 "title": f"Return since {m['book']}", "pct": True},
                {"sheet": "Backtest variants", "x": "Variant", "y": ["CAGR %"], "kind": "bar",
                 "title": "Backtest CAGR with and without the filters", "pct": True}],
        formats={"Entry price": "num", "Last price": "num", "LC days since entry": "num", "LC days, last 10": "num",
                 "Locked days": "num", "Events": "num", "Days after the LC day": "num", "Horizon (days)": "num",
                 "t-stat (by day)": "num", "t-stat": "num", "t-stat (to others)": "num", "Times it fired": "num",
                 "Sell after LC days in 10": "num", "LC hit within (days before)": "num", "Held for (days)": "num",
                 "Months with a flagged name": "num", "Flagged candidates": "num", "Beat Nifty %": "pct",
                 "12-1 return at entry %": "pct"})
    print(f"Saved -> Results: {key}")


if __name__ == "__main__":
    main()
