"""Month-by-month pick history: which stocks the model held in each calendar month, what changed vs the
month before (NEW / HELD / EXITED) and how each pick did.

Two modes:
  traded  : the book the strategy actually holds on its own schedule (a quarterly strategy changes picks
            once a quarter and holds them for the two months in between) -- straight from the model backtest.
  monthly : a fresh decision at every month-end, as if the strategy rebalanced monthly (rank buffer kept).

Conventions (same as the journal): a book decided at a month-end close is entered at the next open and held
to the next month-end close. Continuing names are measured close to close. Returns are before costs; the
'model' return in traded mode is the backtest NAV (after costs and cash interest).
"""
from __future__ import annotations

import pandas as pd

from .backtest import rebalance_dates, target_book
from .journal import Context, _month_end_day, _rebalances_at


def _book(w: pd.Series, sw: pd.Series) -> pd.Series:
    """Long weights positive, short weights negative."""
    return pd.concat([w[w > 0], -sw[sw > 0]]) if len(sw) else w[w > 0]


def history(ctx: Context, mode: str = "traded", today: pd.Timestamp | None = None) -> dict:
    p, cfg = ctx.p, ctx.cfg
    idx = p.close.index
    pos = {d: i for i, d in enumerate(idx)}
    close = p.close.ffill()
    nav, bench = ctx.model.nav, p.benchmark.ffill()
    last = idx[-1]
    today = today or pd.Timestamp.today().normalize()
    # month-end signal dates from the model start; the last one is a real month-end only if the month is over
    ends = list(rebalance_dates(idx[idx >= nav.index[0]], "M"))
    month_done = last >= _month_end_day(last) or today > _month_end_day(last)
    complete_ends = ends if month_done else ends[:-1]

    # ---- the book decided at each month-end -------------------------------------------------------
    books: dict[pd.Timestamp, pd.Series] = {}
    changed: dict[pd.Timestamp, bool] = {}
    if mode == "monthly":
        held, prev_gross = pd.Series(dtype=float), None
        for s in complete_ends:
            below = bool(p.benchmark[s] < ctx.bench_ma[s])
            w, sw, gross, _ = target_book(s, ctx.scores, ctx.elig, ctx.short_elig, ctx.vol,
                                          set(held[held > 0].index), set(held[held < 0].index),
                                          p.meta["sector"], cfg, below, nav.loc[:s], prev_gross, ctx.pm)
            held, prev_gross = _book(w, sw), gross
            books[s], changed[s] = held, True
    else:
        W = ctx.model.weights
        by_signal = {idx[pos[e] - 1]: W.loc[e][W.loc[e] != 0] for e in W.index}  # exec day -> signal day
        sig_days = sorted(by_signal)
        cur, k = pd.Series(dtype=float), 0
        for s in complete_ends:
            moved = False
            while k < len(sig_days) and sig_days[k] <= s:
                cur, k, moved = by_signal[sig_days[k]], k + 1, True
            if s == last and month_done and _rebalances_at(s, cfg):
                # the backtest stops one decision short of the latest close: make that decision here
                below = bool(p.benchmark[s] < ctx.bench_ma[s])
                w, sw, _, _ = target_book(s, ctx.scores, ctx.elig, ctx.short_elig, ctx.vol,
                                          set(cur[cur > 0].index), set(cur[cur < 0].index), p.meta["sector"],
                                          cfg, below, nav.loc[:s], None, ctx.pm)
                cur, moved = _book(w, sw), True
            books[s], changed[s] = cur, moved

    # ---- walk the months ------------------------------------------------------------------------
    def px_open(d, sym):
        i = pos[d] + 1
        if i < len(idx):
            v = p.open.iat[i, p.open.columns.get_loc(sym)]
            if pd.notna(v) and v > 0:
                return float(v)
        return float(close.at[d, sym])

    def meta(sym):
        return p.meta.loc[sym, "name"], p.meta.loc[sym, "sector"]

    months, streak = [], {}  # streak: symbol -> (first month label, cumulative growth)
    prev_book = pd.Series(dtype=float)
    for j, s in enumerate(complete_ends):
        book = books[s]
        if not months and not len(book):
            continue  # before the model's first rebalance
        end = ends[j + 1] if j + 1 < len(ends) else None  # None = upcoming month, nothing to measure yet
        hold_month = (s + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
        status = "upcoming" if end is None else ("in_progress" if end == last and not month_done else "complete")
        valid = s in ctx.scores.index
        rank = ctx.scores.loc[s].where(ctx.elig.loc[s]).rank(ascending=False) if valid else pd.Series(dtype=float)
        exited = []
        for sym, wt in prev_book.items():
            if sym in book.index and (book[sym] > 0) == (wt > 0):
                continue
            first, g = streak.pop(sym, (None, 1.0))
            out_px = px_open(s, sym)
            name, sector = meta(sym)
            exited.append({"symbol": sym, "name": name, "sector": sector, "side": "LONG" if wt > 0 else "SHORT",
                           "weight": round(float(wt), 4), "held_since": first, "ret_while_held": g - 1,
                           "exit_price": round(out_px, 2),
                           "ret_after_exit": float(close.at[last, sym]) / out_px - 1 if out_px else None})
        rows = []
        for sym, wt in book.sort_values(ascending=False).items():
            new = sym not in prev_book.index or (prev_book[sym] > 0) != (wt > 0)
            entry = px_open(s, sym) if new else float(close.at[s, sym])
            ret = float(close.at[end, sym]) / entry - 1 if end is not None and entry else None
            first, g = (hold_month, 1.0) if new or sym not in streak else streak[sym]
            if ret is not None:
                g *= 1 + ret
            streak[sym] = (first, g)
            name, sector = meta(sym)
            rows.append({"symbol": sym, "name": name, "sector": sector, "side": "LONG" if wt > 0 else "SHORT",
                         "weight": round(float(wt), 4), "status": "NEW" if new else "HELD",
                         "rank": int(rank[sym]) if wt > 0 and pd.notna(rank.get(sym)) else None,
                         "score": round(float(ctx.scores.at[s, sym]), 3) if valid and pd.notna(ctx.scores.at[s, sym]) else None,
                         "mom_12_1": round(float(ctx.mom.at[s, sym]), 3) if pd.notna(ctx.mom.at[s, sym]) else None,
                         "entry_price": round(entry, 2), "ret": ret, "held_since": first,
                         "ret_since_added": g - 1 if ret is not None or first != hold_month else None})
        port = sum(r["weight"] * r["ret"] for r in rows if r["ret"] is not None) if end is not None else None
        months.append({
            "month": hold_month, "signal_date": str(s.date()),
            "end_date": str(end.date()) if end is not None else None, "status": status,
            "rebalanced": bool(changed[s]), "n": len(rows), "n_new": sum(r["status"] == "NEW" for r in rows),
            "n_exited": len(exited), "long_exposure": round(float(book[book > 0].sum()), 4),
            "ret_picks": port,
            "ret_model": float(nav.at[end] / nav.at[s] - 1) if mode == "traded" and end is not None
            and s in nav.index and end in nav.index else None,
            "ret_bench": float(bench.at[end] / bench.at[s] - 1) if end is not None else None,
            "holdings": rows, "exited": exited})
        prev_book = book
    return {"strategy": cfg.name, "mode": mode, "rebalance": cfg.rebalance, "data_date": str(last.date()),
            "top_n": cfg.top_n, "short_n": cfg.short_n, "months": months[::-1]}
