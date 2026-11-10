"""Month-by-month track record: what was BUY / HOLD / SELL at each rebalance, the money in
each position, and how the picks did against the Nifty 500 (alpha, accuracy, Sharpe, exit timing).

    python -m momentum_lab track configs/quant_composite.json --from 2025-01-01 --capital 1000000
    python -m momentum_lab track configs/quant_composite.json --journal     # your live journal runs instead

Default mode replays the month-end process (journal.month_end, the same logic as the app) in a scratch
folder, so journal/ is untouched. --journal reads journal/<name>/ as it is.
Shows up in the app (python -m momentum_lab app) -> Results.

Conventions (same as the journal): orders from a rebalance fill at the next day's open; a period's
return runs from that open to the next rebalance close. Alpha = stock or strategy return minus the
Nifty 500 return over the same window.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_lab import StrategyConfig, journal, load_panel, load_universe, results
from momentum_lab.backtest import decision_dates



def replay(cfg: StrategyConfig, p, start: str, end: str | None, capital: float, scratch: Path) -> Path:
    """Run month_end on every rebalance date in [start, end] into scratch/<name>/."""
    journal.JOURNAL = scratch  # strategy_dir() reads this global, so all writes land in scratch
    ctx = journal.Context(p, cfg)
    idx = p.close.index
    sim = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end or idx[-1]))]
    dates = sorted(set(decision_dates(sim, cfg)) | {sim[-1]})
    for k, d in enumerate(dates):
        print(f"  month-end {d.date()} ({k + 1}/{len(dates)})")
        journal.month_end(ctx, str(d.date()), capital=capital if k == 0 else None)
    return scratch / cfg.name


def load_books(d: Path) -> tuple[list[dict], dict, dict, dict]:
    runs = sorted(json.loads((d / "runs.json").read_text()), key=lambda r: r["date"])
    read = lambda pre: {r["date"]: pd.read_csv(d / f"{pre}_{r['date']}.csv") for r in runs
                        if (d / f"{pre}_{r['date']}.csv").exists()}
    return runs, read("book"), read("orders"), read("review")


def next_open(p, ds: str) -> pd.Timestamp | None:
    later = p.close.index[p.close.index > pd.Timestamp(ds)]
    return later[0] if len(later) else None


def action_rows(p, runs, books, orders, reviews) -> pd.DataFrame:
    """One row per (rebalance, stock): the action taken then, and how that holding did until the next one."""
    rows = []
    for i, run in enumerate(runs):
        ds = run["date"]
        book = books[ds][books[ds]["qty"] != 0].set_index("symbol")
        o = orders.get(ds, pd.DataFrame(columns=["symbol", "type", "qty_change", "old_qty", "ref_price", "value_rs"]))
        o = o.set_index("symbol")
        nxt = runs[i + 1]["date"] if i + 1 < len(runs) else None
        rev = reviews[nxt].set_index("symbol") if nxt in reviews else pd.DataFrame()
        bench = runs[i + 1]["review"]["benchmark_return"] if nxt and runs[i + 1].get("review") else np.nan
        for sym in book.index.union(o.index[o["type"] == "exit"]):
            if sym in book.index:
                b = book.loc[sym]
                typ = o.loc[sym, "type"] if sym in o.index else "unchanged"
                action = {"new": "BUY", "exit": "SELL"}.get(typ, "HOLD")
                if b["side"] == "SHORT":
                    action = {"BUY": "SHORT", "HOLD": "HOLD SHORT"}[action]
                note = {"new": "new position", "resize": f"resized {o.loc[sym, 'qty_change']:+.0f}" if sym in o.index else "",
                        "unchanged": ""}.get(typ, "")
                r = rev.loc[sym] if sym in rev.index else None
                ret = float(r["return"]) * (1 if b["side"] == "LONG" else -1) if r is not None else np.nan
                rows.append({"rebalance": ds, "symbol": sym, "name": b["name"], "sector": b["sector"],
                             "action": action, "note": note, "rank": b.get("rank"), "score": b["score"],
                             "weight": b["weight"], "qty": b["qty"], "ref_price": b["price"],
                             "invested_rs": abs(b["actual_rs"]),
                             "entry_price": r["entry_price"] if r is not None else np.nan,
                             "mark_price": r["mark_price"] if r is not None else np.nan,
                             "period_end": nxt, "return": ret,
                             "pnl_rs": float(r["pnl_rs"]) if r is not None else np.nan,
                             "nifty500_return": bench, "alpha": ret - bench,
                             "beat_nifty": (None if np.isnan(ret) or np.isnan(bench) else ("YES" if ret > bench else "NO"))})
            else:  # exited at this rebalance: sold at the next open
                x = o.loc[sym]
                rows.append({"rebalance": ds, "symbol": sym, "name": p.meta["name"].get(sym, sym),
                             "sector": p.meta["sector"].get(sym, ""),
                             "action": "SELL" if x["old_qty"] > 0 else "COVER", "note": "exit (rank/filters)",
                             "qty": x["qty_change"], "ref_price": x["ref_price"], "invested_rs": -abs(x["value_rs"])})
    return pd.DataFrame(rows)


def positions(p, runs, books) -> pd.DataFrame:
    """Each holding from first buy to exit, priced at the actual fills (next opens)."""
    dates = [r["date"] for r in runs]
    held = {ds: set(books[ds].loc[books[ds]["qty"] != 0, "symbol"]) for ds in dates}
    side = {(ds, s): sd for ds in dates for s, sd in books[ds].loc[books[ds]["qty"] != 0, ["symbol", "side"]].values}
    close, opn, bench = p.close.ffill(), p.open, p.benchmark
    last = close.index[-1]
    rows = []
    for sym in sorted(set().union(*held.values())):
        i = 0
        while i < len(dates):
            if sym not in held[dates[i]]:
                i += 1
                continue
            j = i
            while j + 1 < len(dates) and sym in held[dates[j + 1]]:
                j += 1
            buy_day = next_open(p, dates[i])
            exit_run = dates[j + 1] if j + 1 < len(dates) else None
            sell_day = next_open(p, exit_run) if exit_run else None
            if buy_day is None:
                break
            px_in = opn.at[buy_day, sym] if not np.isnan(opn.at[buy_day, sym]) else close.at[buy_day, sym]
            if sell_day is not None:
                px_out, out_day, status = opn.at[sell_day, sym], sell_day, "CLOSED"
                if np.isnan(px_out):
                    px_out = close.at[sell_day, sym]
            else:  # still held, or sell signalled at the last run but not filled yet
                px_out, out_day, status = close.at[last, sym], last, "SELL PENDING" if exit_run else "OPEN"
            sgn = 1 if side[(dates[i], sym)] == "LONG" else -1
            path = close.loc[buy_day:out_day, sym] / px_in - 1
            ret = sgn * (px_out / px_in - 1)
            b_ret = bench.asof(out_day) / bench.asof(buy_day - pd.Timedelta(days=1)) - 1
            peak = (sgn * path).max()
            rows.append({"symbol": sym, "name": p.meta["name"].get(sym, sym), "side": side[(dates[i], sym)],
                         "status": status, "bought_on_signal": dates[i], "entry_date": buy_day.date(),
                         "exit_signal": exit_run, "exit_date": out_day.date() if status == "CLOSED" else None,
                         "months_held": j - i + 1, "days_held": (out_day - buy_day).days,
                         "entry_price": px_in, "exit_or_last_price": px_out, "return": ret,
                         "nifty500_return": b_ret, "alpha": ret - b_ret,
                         "best_return_during_hold": peak, "gave_back_from_peak": ret - peak,
                         "result": "WIN" if ret > 0 else "LOSS", "beat_nifty": "YES" if ret > b_ret else "NO"})
            i = j + 1
    return pd.DataFrame(rows).sort_values(["status", "entry_date"], ascending=[False, True]) if rows else pd.DataFrame()


def monthly(runs, acts: pd.DataFrame, rf: float) -> pd.DataFrame:
    rows = []
    for i, run in enumerate(runs[:-1]):
        nxt = runs[i + 1]
        rv = nxt.get("review") or {}
        a = acts[acts["rebalance"] == run["date"]]
        h = a[a["action"].isin(["BUY", "HOLD", "SHORT", "HOLD SHORT"])]
        rows.append({"rebalance": run["date"], "period_end": nxt["date"], "regime": run["regime"],
                     "capital_start": run["capital"], "invested_rs": run["invested_rs"],
                     "cash_rs": run["capital"] - run["invested_rs"], "exposure": run["long_exposure"],
                     "holdings": len(h), "buys": int((a["action"] == "BUY").sum()),
                     "holds": int((a["action"] == "HOLD").sum()), "sells": int((a["action"] == "SELL").sum()),
                     "costs_rs": rv.get("costs"), "net_pnl_rs": rv.get("net_pnl"),
                     "strategy_return": rv.get("return"), "nifty500_return": rv.get("benchmark_return"),
                     "alpha": rv.get("return", np.nan) - rv.get("benchmark_return", np.nan),
                     "stock_hit_rate": (h["return"] > 0).mean(),
                     "stock_accuracy_vs_nifty": (h["alpha"] > 0).mean(),
                     "avg_stock_alpha": h["alpha"].mean(),
                     "new_buys_alpha": h.loc[h["action"] == "BUY", "alpha"].mean(),
                     "holds_alpha": h.loc[h["action"] == "HOLD", "alpha"].mean()})
    m = pd.DataFrame(rows)
    if m.empty:
        return m
    m["cum_strategy"] = (1 + m["strategy_return"]).cumprod() - 1
    m["cum_nifty500"] = (1 + m["nifty500_return"]).cumprod() - 1
    m["cum_alpha"] = m["cum_strategy"] - m["cum_nifty500"]
    rf_m = (1 + rf) ** (1 / 12) - 1
    ex = m["strategy_return"] - rf_m
    m["sharpe_to_date"] = (ex.expanding(3).mean() / m["strategy_return"].expanding(3).std() * np.sqrt(12))
    m["beat_nifty"] = np.where(m["alpha"] > 0, "YES", "NO")
    return m


def exit_timing(acts: pd.DataFrame) -> pd.DataFrame:
    """Average alpha by how many months a stock had already been held -- when does the edge fade?"""
    h = acts[acts["action"].isin(["BUY", "HOLD"]) & acts["alpha"].notna()].sort_values("rebalance").copy()
    if h.empty:
        return pd.DataFrame()
    dates = sorted(acts["rebalance"].unique())
    pos = {d: k for k, d in enumerate(dates)}
    h["k"] = h["rebalance"].map(pos)
    # month number within the current holding streak
    h["month_in_trade"] = 0
    for sym, g in h.groupby("symbol"):
        n, prev = 0, None
        for ix, k in zip(g.index, g["k"]):
            n = n + 1 if prev is not None and k == prev + 1 else 1
            h.at[ix, "month_in_trade"] = n
            prev = k
    h["bucket"] = h["month_in_trade"].clip(upper=7).map(lambda n: f"{n}" if n < 7 else "7+")
    t = h.groupby("bucket").agg(observations=("alpha", "size"), avg_return=("return", "mean"),
                                avg_alpha=("alpha", "mean"), median_alpha=("alpha", "median"),
                                accuracy_vs_nifty=("alpha", lambda s: (s > 0).mean()),
                                hit_rate=("return", lambda s: (s > 0).mean()))
    t.index.name = "month_in_trade"
    return t.reset_index()


def summary(cfg, runs, m: pd.DataFrame, pos: pd.DataFrame, acts: pd.DataFrame, source: str) -> pd.DataFrame:
    rf_m = (1 + cfg.rf_rate) ** (1 / 12) - 1
    r, b = m["strategy_return"], m["nifty500_return"]
    n = len(m)
    sharpe = (r.mean() - rf_m) / r.std() * np.sqrt(12) if n > 2 else np.nan
    b_sharpe = (b.mean() - rf_m) / b.std() * np.sqrt(12) if n > 2 else np.nan
    beta = np.cov(r, b)[0, 1] / b.var() if n > 2 else np.nan
    jensen = ((r - rf_m) - beta * (b - rf_m)).mean() * 12 if n > 2 else np.nan
    te = (r - b).std() * np.sqrt(12)
    nav = (1 + r).cumprod()
    closed = pos[pos["status"] == "CLOSED"] if len(pos) else pos
    h = acts[acts["alpha"].notna()]
    items = [
        ("Strategy", cfg.name), ("Source", source),
        ("Period", f"{runs[0]['date']} -> {runs[-1]['date']}"), ("Months tracked", n),
        ("Starting capital (Rs)", runs[0]["capital"]), ("Current capital (Rs)", runs[-1]["capital"]),
        ("", ""), ("RETURNS", ""),
        ("Total return", nav.iloc[-1] - 1), ("Nifty 500 total return", (1 + b).prod() - 1),
        ("Total alpha (strategy - Nifty)", nav.iloc[-1] - (1 + b).prod()),
        ("Avg monthly return", r.mean()), ("Avg monthly alpha", (r - b).mean()),
        ("Best month", r.max()), ("Worst month", r.min()),
        ("Max drawdown (month-end)", (nav / nav.cummax() - 1).min()),
        ("", ""), ("RISK-ADJUSTED", ""),
        ("Sharpe (annualised, monthly)", sharpe), ("Nifty 500 Sharpe", b_sharpe),
        ("Beta vs Nifty 500", beta), ("Jensen's alpha (annualised)", jensen),
        ("Information ratio", (r - b).mean() * 12 / te if te else np.nan),
        ("", ""), ("ACCURACY", ""),
        ("Months beating Nifty", (r > b).mean()), ("Months positive", (r > 0).mean()),
        ("Stock-months positive", (h["return"] > 0).mean()),
        ("Stock-months beating Nifty", (h["alpha"] > 0).mean()),
        ("Closed trades", len(closed)),
        ("Closed trades won", (closed["return"] > 0).mean() if len(closed) else np.nan),
        ("Closed trades beat Nifty", (closed["alpha"] > 0).mean() if len(closed) else np.nan),
        ("Avg closed trade return", closed["return"].mean() if len(closed) else np.nan),
        ("Avg closed trade alpha", closed["alpha"].mean() if len(closed) else np.nan),
        ("Avg months held (closed)", closed["months_held"].mean() if len(closed) else np.nan),
        ("Avg give-back from peak (closed)", closed["gave_back_from_peak"].mean() if len(closed) else np.nan),
    ]
    if n < 12:
        items.append(("Note", f"Only {n} months: Sharpe and beta are noisy below ~12-24 months."))
    return pd.DataFrame(items, columns=["Metric", "Value"])


PCT = {"weight", "return", "nifty500_return", "alpha", "strategy_return", "cum_strategy", "cum_nifty500",
       "cum_alpha", "exposure", "stock_hit_rate", "stock_accuracy_vs_nifty", "avg_stock_alpha", "new_buys_alpha",
       "holds_alpha", "best_return_during_hold", "gave_back_from_peak", "avg_return", "avg_alpha",
       "median_alpha", "accuracy_vs_nifty", "hit_rate"}
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--from", dest="start", default="2025-01-01", help="first rebalance date (replay mode)")
    ap.add_argument("--to", dest="end", help="last date (default: latest data)")
    ap.add_argument("--capital", type=float, help="starting capital in Rs (default: config initial_capital)")
    ap.add_argument("--journal", action="store_true", help="use journal/<name>/ runs instead of replaying")
    a = ap.parse_args()

    cfg = StrategyConfig.from_json(a.config)
    print("Loading prices...")
    p = load_panel(load_universe(cfg.universe_csv), cfg.benchmark)
    with tempfile.TemporaryDirectory() as tmp:
        if a.journal:
            d, source = journal.JOURNAL / cfg.name, f"live journal ({journal.JOURNAL / cfg.name})"
        else:
            print("Replaying month-end runs (signals + model backtest take ~20s)...")
            d = replay(cfg, p, a.start, a.end, a.capital or cfg.initial_capital, Path(tmp))
            source = "replay of month-end process (assumes orders filled at next open)"
        runs, books, orders, reviews = load_books(d)
    if len(runs) < 2:
        raise SystemExit(f"Need at least 2 month-end runs, found {len(runs)}. Use an earlier --from.")

    acts = action_rows(p, runs, books, orders, reviews)
    m = monthly(runs, acts, cfg.rf_rate)
    pos = positions(p, runs, books)
    sheets = {"Summary": summary(cfg, runs, m, pos, acts, source), "Monthly": m,
              "Monthly actions": acts, "Positions": pos, "Exit timing": exit_timing(acts)}
    def shown(k, v):  # Summary mixes %, rupees and ratios in one column: format it here
        if not isinstance(v, float) or not isinstance(k, str) or k.isupper():
            return v
        if not any(s in k for s in ("Rs", "Sharpe", "Beta", "ratio", "months held")):
            return f"{v:.2%}"
        return f"{v:,.0f}" if "Rs" in k else f"{v:.2f}"
    view = {**sheets, "Summary": sheets["Summary"].assign(
        Value=[shown(k, v) for k, v in zip(sheets["Summary"]["Metric"], sheets["Summary"]["Value"])])}
    path = results.save(f"track_record_{cfg.name}" + ("_journal" if a.journal else ""),
                        f"Track record · {cfg.name} · {'live journal' if a.journal else 'replay'} · "
                        f"{runs[0]['date']} to {runs[-1]['date']}",
                        view, kind="track", params={"strategy": cfg.name, "from": runs[0]["date"],
                                                      "to": runs[-1]["date"], "source": source},
                        charts=[{"sheet": "Monthly", "x": "period_end", "y": ["cum_strategy", "cum_nifty500"],
                                 "pct": True, "title": "Cumulative return: strategy vs Nifty 500"},
                                {"sheet": "Monthly", "x": "period_end", "y": ["alpha"], "kind": "bar", "pct": True,
                                 "title": "Alpha vs Nifty 500 each period"}],
                        formats={**{c: "pct" for c in PCT}, "score": "num", "sharpe_to_date": "num"})
    s = sheets["Summary"].set_index("Metric")["Value"]
    print(f"\nTotal return {s['Total return']:.1%} vs Nifty 500 {s['Nifty 500 total return']:.1%} | "
          f"Sharpe {s['Sharpe (annualised, monthly)']:.2f} | months beating Nifty {s['Months beating Nifty']:.0%}")
    print(f"Saved {path.name}: open the app (python -m momentum_lab app) -> Results.")


if __name__ == "__main__":
    main()
