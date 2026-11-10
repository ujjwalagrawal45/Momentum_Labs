"""Holdings review: which of a journal book's stocks led, which went flat and which lagged since the
book date, and whether you could have spotted the laggards early and cut them.

    python -m momentum_lab review configs/quant_composite.json                  # latest journal book
    python -m momentum_lab review configs/quant_composite.json --book 2026-10-23

Part 1 (the book): each stock's return from the book price (the close on the book date) to the latest close,
against the Nifty 500, grouped into Leaders / In line / Laggards, with the traits each stock had on entry.

Part 2 (the history test): the same traits, plus each holding's return in its first 2/5/10/21 trading days,
checked against how the holding did afterwards. It uses every rebalance of the model backtest, with the
strategy's own schedule and a monthly version for more samples. Then it simulates the rule "on day k, cut the
k-day bottom m names by half or fully, and move the money to cash or to the other holdings", after costs.

Shows up in the app (python -m momentum_lab app) -> Results. Survivorship bias: the universe is today's Nifty 500 list,
so past small, illiquid names that later joined the index look better than they really were.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_lab import StrategyConfig, load_panel, load_universe, results
from momentum_lab.journal import JOURNAL, Context

TRAITS = {"score": "Composite score", "r12_1": "12-1 month return", "pre1w": "Return, week before entry",
          "pre1m": "Return, month before entry", "vol63": "Volatility (63d)", "beta": "Beta to Nifty 500 (6m)",
          "off52h": "Distance from 52-week high", "above50": "Distance above 50-day average",
          "adv_cr": "Liquidity (median traded value, Rs cr)", "new": "New buy (not held before)"}
CHECKS = (2, 5, 10, 21)
COST = 0.003  # per rupee moved: sell + buy round trip at about 0.15% each (STT, stamp, slippage, impact)


def traits(p, ctx, i_sig: int, syms, held_before: set) -> pd.DataFrame:
    """Entry traits as of the signal-day close (index position i_sig)."""
    c, idx = p.close.ffill(), p.close.index
    r, br = c.pct_change(fill_method=None), p.benchmark.ffill().pct_change()
    sd = idx[i_sig]
    win = slice(i_sig - 125, i_sig + 1)
    out = {}
    for s in syms:
        cs = c[s]
        out[s] = {"score": ctx.scores.at[sd, s], "r12_1": ctx.mom.at[sd, s],
                  "pre1w": cs.iat[i_sig] / cs.iat[i_sig - 5] - 1, "pre1m": cs.iat[i_sig] / cs.iat[i_sig - 21] - 1,
                  "vol63": ctx.vol.at[sd, s],
                  "beta": r[s].iloc[win].cov(br.iloc[win]) / br.iloc[win].var(),
                  "off52h": cs.iat[i_sig] / p.high[s].iloc[i_sig - 251:i_sig + 1].max() - 1,
                  "above50": cs.iat[i_sig] / cs.iloc[i_sig - 49:i_sig + 1].mean() - 1,
                  "adv_cr": ctx.adv.at[sd, s] / 1e7, "new": float(s not in held_before)}
    return pd.DataFrame(out).T


# ---------------------------------------------------------------------------------------------- part 1
def book_review(p, ctx, name: str, book_date: str | None):
    d = JOURNAL / name
    books = sorted(d.glob("book_*.csv"))
    if not books:
        raise SystemExit(f"No journal books in {d}")
    f = d / f"book_{book_date}.csv" if book_date else books[-1]
    if not f.exists():
        raise SystemExit(f"No book for {book_date}; have {[b.stem[5:] for b in books]}")
    ds = f.stem[5:]
    prior = [b for b in books if b.stem[5:] < ds]
    held_before = set(pd.read_csv(prior[-1])["symbol"]) if prior else set()
    b = pd.read_csv(f)
    b = b[b["qty"] > 0].set_index("symbol")
    c, idx, bm = p.close.ffill(), p.close.index, p.benchmark.ffill()
    i0 = idx.get_indexer([pd.Timestamp(ds)], method="pad")[0]
    last = idx[-1]
    if last <= idx[i0]:
        raise SystemExit(f"No prices after {ds} yet")
    n_days = len(idx) - 1 - i0
    t = traits(p, ctx, i0, b.index, held_before)
    rows = []
    for s in b.index:
        e = float(b.at[s, "price"])
        row = {"Stock": s, "Name": b.at[s, "name"], "Sector": b.at[s, "sector"], "Rank": int(b.at[s, "rank"]),
               "Invested (₹)": float(b.at[s, "actual_rs"]), "Entry price": e, "Last price": float(c[s].iat[-1]),
               "Return %": c[s].iat[-1] / e - 1}
        for k in (1, 2):
            if i0 + k < len(idx):
                row[f"After {k} day{'s' * (k > 1)} %"] = c[s].iat[i0 + k] / e - 1
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Stock")
    bret = bm.iat[-1] / bm.iat[i0] - 1
    df["vs Nifty 500 %"] = df["Return %"] - bret
    df["Group"] = np.where(df["vs Nifty 500 %"] >= 0.05, "Leader",
                           np.where(df["vs Nifty 500 %"] < 0, "Laggard", "In line"))
    for k, lab in TRAITS.items():
        df[lab] = t[k].astype(float)
    df["New buy (not held before)"] = np.where(t["new"] > 0, "new", "held")
    df = df.sort_values("Return %", ascending=False)
    w = df["Invested (₹)"] / df["Invested (₹)"].sum()
    port = float((w * df["Return %"]).sum())

    num = [lab for k, lab in TRAITS.items() if k != "new"]
    g = df.groupby("Group")
    groups = pd.DataFrame({"Stocks": g.size(), "Names": g.apply(lambda x: ", ".join(x.index), include_groups=False),
                           "Avg return %": g["Return %"].mean(), "Share of invested %": g["Invested (₹)"].sum() / df["Invested (₹)"].sum(),
                           "New buys": g["New buy (not held before)"].apply(lambda x: (x == "new").sum())})
    groups = groups.join(g[num].median().add_prefix("Median: ")).reindex(["Leader", "In line", "Laggard"]).dropna(how="all")
    groups = groups.reset_index().rename(columns={"index": "Group"})

    sec = df.groupby("Sector").agg(Stocks=("Return %", "size"), **{"Avg return %": ("Return %", "mean"),
                                   "Best": ("Return %", "idxmax"), "Worst": ("Return %", "idxmin")})
    sec = sec.sort_values("Avg return %", ascending=False).reset_index()

    # within this one book: does each trait line up with the return? (25 stocks: only |corr| > ~0.4 means much)
    early = [c_ for c_ in df.columns if c_.startswith("After ")]
    corr = pd.DataFrame([{"Trait": lab, "Rank correlation with return": df[lab].corr(df["Return %"], method="spearman")}
                         for lab in num + early])
    meta = {"book": ds, "to": str(last.date()), "days": n_days, "port": port, "bench": bret}
    return df.reset_index(), groups, sec, corr, meta


# ---------------------------------------------------------------------------------------------- part 2
def holdings_history(p, cfg: StrategyConfig) -> pd.DataFrame:
    ctx = Context(p, cfg)
    W = ctx.model.weights
    c, o, idx, bm = p.close.ffill(), p.open, p.close.index, p.benchmark.ffill()
    pos = {d: i for i, d in enumerate(idx)}
    ex, rows = list(W.index), []
    for j, d in enumerate(ex[:-1]):
        w = W.loc[d]
        w = w[w > 0]
        i0, iend = pos[d], pos[ex[j + 1]] - 1  # in at the exec-day open, out at the close before the next one
        prev = W.loc[ex[j - 1]] if j else pd.Series(dtype=float)
        t = traits(p, ctx, i0 - 1, w.index, set(prev[prev > 0].index))
        for s in w.index:
            cs = c[s]
            e = o.at[d, s] if pd.notna(o.at[d, s]) else cs.iat[i0 - 1]
            row = {"exec": d, "sym": s, "w": w[s] / w.sum(), "ret": cs.iat[iend] / e - 1,
                   "bret": bm.iat[iend] / bm.iat[i0 - 1] - 1, **t.loc[s].to_dict()}
            for k in CHECKS:
                ik = i0 + k - 1
                if ik < iend:
                    row[f"r{k}"], row[f"rest{k}"] = cs.iat[ik] / e - 1, cs.iat[iend] / cs.iat[ik] - 1
            rows.append(row)
    return pd.DataFrame(rows)


def ic(df: pd.DataFrame, x: str, y: str) -> dict:
    s = pd.Series({d: g[x].astype(float).corr(g[y], method="spearman") for d, g in df.groupby("exec")}).dropna()
    return {"Avg rank correlation": s.mean(), "t-stat": s.mean() / s.std() * np.sqrt(len(s)) if len(s) > 2 else np.nan,
            "Periods positive %": (s > 0).mean(), "Periods": len(s)}


def cut_rule(df: pd.DataFrame, k: int, m: int, frac: float, dest: str) -> dict:
    diffs = []
    for _, g in df.dropna(subset=[f"r{k}"]).groupby("exec"):
        if len(g) < 2 * m:
            continue
        vk = g["w"] * (1 + g[f"r{k}"])
        base = (vk * (1 + g[f"rest{k}"])).sum()
        lose = (g[f"r{k}"].rank(method="first") <= m).to_numpy()
        cut = vk[lose] * frac
        freed = cut.sum()
        v2 = vk.copy()
        v2[lose] -= cut
        if dest == "other holdings":
            v2[~lose] += freed * vk[~lose] / vk[~lose].sum()
            new = (v2 * (1 + g[f"rest{k}"])).sum() - 2 * freed * COST
        else:
            new = (v2 * (1 + g[f"rest{k}"])).sum() + freed - freed * COST
        diffs.append(new - base)
    s = pd.Series(diffs)
    return {"Check on day": k, "Cut bottom": m, "Cut by %": frac, "Money goes to": dest,
            "Avg gain per period %": s.mean(), "t-stat": s.mean() / s.std() * np.sqrt(len(s)),
            "Periods it helped %": (s > 0).mean(), "Periods": len(s)}


def history_tests(p, cfg: StrategyConfig):
    sched = {"Q": "quarterly", "M": "monthly", "W": "weekly"}.get(cfg.rebalance, cfg.rebalance)
    runs = {f"as traded ({sched})": cfg}
    if cfg.rebalance != "M":
        runs["monthly version"] = cfg.replace(rebalance="M")
    ic_rows, early_rows, rule_rows = [], [], []
    for lab, c_ in runs.items():
        print(f"  history: {lab}")
        h = holdings_history(p, c_)
        for k, name in TRAITS.items():
            ic_rows.append({"Version": lab, "Trait at entry": name, **ic(h, k, "ret")})
        for k in CHECKS:
            if f"r{k}" in h:
                early_rows.append({"Version": lab, "Early window (trading days)": k,
                                   **ic(h.dropna(subset=[f"r{k}"]), f"r{k}", f"rest{k}")})
        for k in (5, 10, 21):
            for m in (3, 5):
                for frac in (0.5, 1.0):
                    for dest in ("cash", "other holdings"):
                        rule_rows.append({"Version": lab, **cut_rule(h, k, m, frac, dest)})
    return pd.DataFrame(ic_rows), pd.DataFrame(early_rows), pd.DataFrame(rule_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--book", help="journal book date YYYY-MM-DD (default: the latest)")
    a = ap.parse_args()
    cfg = StrategyConfig.from_json(a.config)
    print("Loading prices...")
    p = load_panel(load_universe(cfg.universe_csv), cfg.benchmark)
    print("Reviewing the book...")
    ctx = Context(p, cfg)
    stocks, groups, sec, corr, m = book_review(p, ctx, cfg.name, a.book)
    print("Testing the traits and the cut-laggards rule on the backtest history...")
    ics, early, rules = history_tests(p, cfg)

    traded = rules[rules["Version"] == rules["Version"].iloc[0]]
    best = traded.loc[traded["Avg gain per period %"].idxmax()]
    lead, lag = (stocks["Group"] == "Leader").sum(), (stocks["Group"] == "Laggard").sum()
    notes = (f"Book of {m['book']}, measured from the book price (close on {m['book']}) to the close on {m['to']} "
             f"({m['days']} trading days). Leader = beat the Nifty 500 by 5 points or more; Laggard = trailed it. "
             "\n\nThe history sheets use every rebalance of the model backtest (survivorship-biased: today's Nifty 500 "
             "list). A rank correlation of 0.05 with |t| above 2 is a real but small edge; |t| below 2 is noise. "
             "'Cut-laggards test' compares each rule with doing nothing, per holding period, after about 0.3% costs "
             "per rupee moved. Cash earns nothing in that test.")
    key = f"holding_review_{cfg.name}_{m['book']}"
    results.save(
        key, f"Holdings review: {cfg.name}, book {m['book']}", kind="review",
        params={"strategy": cfg.name, "book": m["book"], "to": m["to"]},
        notes=notes,
        headline=[{"label": f"Book return ({m['days']}d)", "value": m["port"], "fmt": "pct"},
                  {"label": "Nifty 500", "value": m["bench"], "fmt": "pct"},
                  {"label": "Ahead of Nifty", "value": m["port"] - m["bench"], "fmt": "pct"},
                  {"label": "Leaders / laggards", "value": f"{lead} / {lag} of {len(stocks)}", "fmt": "text"},
                  {"label": "Best cut rule, per period", "value": float(best["Avg gain per period %"]), "fmt": "pct"}],
        sheets={"Stocks": stocks, "Leaders vs laggards": groups, "By sector": sec,
                "This book: traits vs return": corr, "History: traits at entry": ics,
                "History: early return vs rest": early, "History: cut-laggards test": rules},
        charts=[{"sheet": "Stocks", "x": "Stock", "y": ["Return %", "vs Nifty 500 %"], "kind": "bar",
                 "title": f"Return since {m['book']}", "pct": True},
                {"sheet": "By sector", "x": "Sector", "y": ["Avg return %"], "kind": "bar",
                 "title": "Average return by sector", "pct": True}],
        formats={"Rank": "num", "Entry price": "num", "Last price": "num", "t-stat": "num", "Periods": "num",
                 "Stocks": "num", "New buys": "num", "Cut bottom": "num", "Check on day": "num",
                 "Early window (trading days)": "num", "Avg rank correlation": "num",
                 "Rank correlation with return": "num", "Cut by %": "pct", "Periods it helped %": "pct",
                 "Periods positive %": "pct", "Share of invested %": "pct",
                 **{lab: ("num" if k in ("score", "beta", "adv_cr") else "pct")
                    for k, lab in TRAITS.items() if k != "new"},
                 **{f"Median: {lab}": ("num" if k in ("score", "beta", "adv_cr") else "pct")
                    for k, lab in TRAITS.items() if k != "new"}})
    print(f"Saved -> Results: {key}")


if __name__ == "__main__":
    main()
