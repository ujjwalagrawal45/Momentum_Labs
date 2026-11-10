"""Rolling 2-year walk-forward backtest: every BUY / SELL / HOLD / ADD / TRIM decision, month by month,
for each strategy, in every 2-year window from Jan 2019 to Sep 2026 -> the app's Results page.

    python -m momentum_lab rolling                                  # everything from 'Momentum Lab Settings.xlsx'
    python -m momentum_lab rolling configs/quant_composite.json --capital 1000000 --step 1   # command-line overrides

Defaults (strategies, capital, windows, cost stress, tax rates, decision checks, market phases) come from the
settings workbook: Strategies, Rolling Backtest, Tax, Decision Checks and Market Phases sheets.

Capital between windows ('Rolling Backtest' sheet, or --mode):
  compound : ONE pot. Windows run back to back (Jan 2019-Dec 2020, Jan 2021-Dec 2022, ...; the last one may be
             shorter). The first starts with Rs <capital>; each next window starts with the previous window's
             after-tax liquidation value. Tax losses carried forward and the yearly LTCG exemption carry across.
             The Nifty 500 benchmark is chained the same way. Answers: does the money compound?
  fresh    : every window is a cold start with Rs <capital>, windows overlap (rolled every --step months).
             Answers: does it work whatever month you start in?
A window starts in cash on the signal date before it (e.g. 31 Dec 2018), first orders at the next open, then the
strategy's own rebalance calendar until the window's last close.
The decision logic is backtest.target_book (identical to the app / month-end journal), sized the way you
would actually trade it: whole shares for longs (floor), futures lots for shorts (nearest lot), the
journal's 25% no-trade band on resizes, full Indian cost stack, daily mark-to-market.

Three variants of every window:
  pre_tax    : base costs, no tax                       (comparable to the research backtests)
  after_tax  : base costs, capital-gains tax paid out of the portfolio at each 31 Mar, and on exit
  stress     : 2x slippage and 2x market impact, after tax

Tax (rates from the Tax sheet, applied to all years): STCG, LTCG above the yearly exemption, + cess, FIFO lots,
ST losses set off against ST/LT gains, LT losses only against LT, unused losses carried forward.
Short futures P&L is business income: flat rate + cess on net yearly profit, losses carried forward.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_lab.paths import CONFIGS
from momentum_lab import StrategyConfig, load_panel, load_universe
from momentum_lab.backtest import decision_dates, is_scheduled, risk_trigger, target_book, trade_costs
from momentum_lab.journal import Context
from momentum_lab.settings import CHECK_DEFS, WORKBOOK, globals_
from momentum_lab import results

G = globals_()  # 'Momentum Lab Settings.xlsx'
DEFAULT_CONFIGS = G["rolling_strategies"]
TAX = G["tax"]
STRESS = G["rolling"]["stress"]
NO_TRADE_BAND = G["live"]["no_trade_band"]
VARIANTS = ("pre_tax", "after_tax", "stress")
PHASES = [tuple(p) for p in G["phases"]]  # regime breakdown (inclusive month ranges)


def fy(d: pd.Timestamp) -> int:
    return d.year if d.month >= 4 else d.year - 1


class TaxBook:
    """Indian capital-gains ledger per financial year, with set-off and loss carry-forward."""

    def __init__(self):
        self.gains = defaultdict(lambda: {"ST": 0.0, "LT": 0.0, "FNO": 0.0})
        self.cf = {"ST": 0.0, "LT": 0.0, "FNO": 0.0}
        self.exempt_left = {}  # FY -> LTCG exemption not yet used (a FY can span two chained windows)

    def add(self, year: int, kind: str, amt: float) -> None:
        self.gains[year][kind] += amt

    def settle(self, year: int, extra: dict | None = None) -> float:
        g = dict(self.gains.pop(year, {"ST": 0.0, "LT": 0.0, "FNO": 0.0}))
        for k, v in (extra or {}).items():
            g[k] += v
        st, lt, fno = g["ST"], g["LT"], g["FNO"]
        if st < 0 < lt:  # current-year ST loss against LT gain
            x = min(-st, lt)
            st, lt = st + x, lt - x
        for kind, pool in (("LT", "LT"), ("ST", "ST"), ("LT", "ST"), ("FNO", "FNO")):  # brought-forward losses
            cur = {"ST": st, "LT": lt, "FNO": fno}[kind]
            x = min(self.cf[pool], max(cur, 0))
            self.cf[pool] -= x
            if kind == "ST":
                st -= x
            elif kind == "LT":
                lt -= x
            else:
                fno -= x
        for kind, v in (("ST", st), ("LT", lt), ("FNO", fno)):
            if v < 0:
                self.cf[kind] += -v
        st, lt, fno = max(st, 0), max(lt, 0), max(fno, 0)
        ex = min(self.exempt_left.get(year, TAX["ltcg_exempt"]), lt)
        self.exempt_left[year] = self.exempt_left.get(year, TAX["ltcg_exempt"]) - ex
        return (TAX["stcg"] * st + TAX["ltcg"] * (lt - ex) + TAX["fno"] * fno) * (1 + TAX["cess"])


class Sim:
    """Share-based daily simulator over one window, using the month-end journal's sizing rules."""

    def __init__(self, ctx: Context):
        p, cfg = ctx.p, ctx.cfg
        self.ctx, self.cfg, self.p = ctx, cfg, p
        self.dates, self.syms = p.close.index, p.close.columns
        self.col = {s: j for j, s in enumerate(self.syms)}
        self.close = p.close.ffill().to_numpy()
        self.open = p.open.reindex_like(p.close).to_numpy()
        self.sigma = ctx.sigma_d.to_numpy()
        self.adv = ctx.adv.to_numpy()
        self.bench = p.benchmark.reindex(self.dates).ffill()
        self.below = (p.benchmark < ctx.bench_ma).reindex(self.dates).fillna(False)
        self.lot = ctx.fno["lot_size"].reindex(self.syms).fillna(1).to_numpy()
        self.rebal = set(decision_dates(self.dates, cfg))
        self.names, self.sectors = p.meta["name"], p.meta["sector"]
        self.buffer = cfg.buffer_rank or cfg.top_n
        self.daily_cash = (1 + cfg.cash_rate) ** (1 / 252) - 1

    # ------------------------------------------------------------------------------------------
    def run(self, s0: int, end: int, capital: float, costs, pay_tax: bool, log: bool, book: TaxBook | None = None) -> dict:
        cfg, ctx, dates, n = self.cfg, self.ctx, self.dates, len(self.syms)
        shares, cash = np.zeros(n), float(capital)
        lots = defaultdict(deque)                       # j -> FIFO [qty, price, date] for long tax lots
        short_px = {}                                    # j -> avg short entry price
        tax, tax_paid, tax_owed = book or TaxBook(), 0.0, 0.0  # a chained window inherits losses + used exemption
        sig_idx = [s0] + [k for k in range(s0 + 1, end) if dates[k] in self.rebal]
        nav = np.empty(end - s0 + 1)
        long_mv = np.zeros(end - s0 + 1)
        nav[0] = cash
        orders, actions, trips, rebs, open_trip = [], [], [], [], {}
        day_cost, day_tax = defaultdict(float), defaultdict(float)
        k_sig = 0
        for i in range(s0 + 1, end + 1):
            d = dates[i]
            base = cash + float(np.nansum(np.minimum(shares, 0) * self.close[i - 1]))  # cash net of short proceeds
            cash += max(base, 0) * self.daily_cash
            if k_sig < len(sig_idx) and sig_idx[k_sig] == i - 1 and (
                    is_scheduled(dates[i - 1], cfg) or i - 1 == s0 or
                    risk_trigger(cfg, bool(self.below[dates[i - 1]]), ctx.model.nav.loc[:dates[i - 1]],
                                 rebs[-1]["gross_exposure"] if rebs else None)):
                cash = self._rebalance(i, nav[i - 1 - s0], shares, cash, lots, short_px, tax, costs,
                                       orders, actions if log else None, trips, open_trip, rebs, day_cost)
            if k_sig < len(sig_idx) and sig_idx[k_sig] == i - 1:
                k_sig += 1
            mv = shares * self.close[i]
            nav[i - s0] = cash + float(np.nansum(mv))
            long_mv[i - s0] = float(np.nansum(np.maximum(mv, 0)))
            if i < end and fy(dates[i + 1]) != fy(d):  # 31 March: settle the financial year
                t = tax.settle(fy(d))
                tax_owed += t
                if pay_tax and t > 0:
                    cash -= t
                    tax_paid += t
                    day_tax[d] += t
                    nav[i - s0] -= t

        # ---- end of window: hypothetical liquidation at the last close --------------------------
        px = self.close[end]
        held = np.nonzero(shares)[0]
        exit_cost = float(trade_costs(shares[held] * px[held], shares[held] < 0, self.sigma[end, held],
                                      self.adv[end, held], costs).sum()) if len(held) else 0.0
        extra = {"ST": 0.0, "LT": 0.0, "FNO": 0.0}
        for j in held:
            if shares[j] > 0:
                for q, lp, ld in lots[j]:
                    extra["LT" if (dates[end] - ld).days > 365 else "ST"] += q * (px[j] - lp)
            else:
                extra["FNO"] += -shares[j] * (short_px[j] - px[j])
        final_tax = tax.settle(fy(dates[end]), extra)
        tax_owed += final_tax
        for j, t in list(open_trip.items()):
            self._close_trip(t, j, end, px[j], shares[j], "OPEN at window end", trips)
        nav_s = pd.Series(nav, index=dates[s0:end + 1])
        net_liq = nav[-1] - exit_cost - (final_tax if pay_tax else 0.0)
        return {"nav": nav_s, "long_mv": pd.Series(long_mv, index=nav_s.index), "orders": orders,
                "actions": actions, "trips": trips, "rebs": rebs, "costs": pd.Series(day_cost, dtype=float),
                "tax_days": pd.Series(day_tax, dtype=float), "tax_paid": tax_paid + (final_tax if pay_tax else 0.0),
                "tax_owed": tax_owed, "exit_cost": exit_cost, "net_liq": net_liq, "book": tax}

    # ------------------------------------------------------------------------------------------
    def _rebalance(self, i, cap, shares, cash, lots, short_px, tax, costs, orders, actions, trips, open_trip,
                   rebs, day_cost) -> float:
        cfg, ctx, s, d = self.cfg, self.ctx, i - 1, self.dates[i]
        sd = self.dates[s]
        held_long, held_short = set(self.syms[shares > 0]), set(self.syms[shares < 0])
        w, sw, gross, rule = target_book(sd, ctx.scores, ctx.elig, ctx.short_elig, ctx.vol, held_long, held_short,
                                         self.sectors, cfg, bool(self.below[sd]), ctx.model.nav.loc[:sd],
                                         rebs[-1]["gross_exposure"] if rebs else None, ctx.pm)
        px_sd = self.close[s]
        tw, target = np.zeros(len(shares)), np.zeros(len(shares))
        for sym, wt in w.items():
            j = self.col[sym]
            tw[j], target[j] = wt, math.floor(wt * cap / px_sd[j])
        for sym, wt in sw.items():
            j = self.col[sym]
            tw[j], target[j] = -wt, -math.floor(wt * cap / (px_sd[j] * self.lot[j]) + 0.5) * self.lot[j]
        band = (shares != 0) & (np.sign(shares) == np.sign(target)) & \
               (np.abs(target - shares) <= NO_TRADE_BAND * np.abs(target))
        target[band] = shares[band]
        old = shares.copy()
        delta = target - old
        fill = np.where(np.isnan(self.open[i]) | (self.open[i] <= 0), self.close[i], self.open[i])
        m = np.nonzero(delta)[0]
        c = trade_costs(delta[m] * fill[m], delta[m] > 0, self.sigma[s, m], self.adv[s, m], costs) if len(m) else []
        cost = dict(zip(m, c))
        cash -= float(np.sum(delta[m] * fill[m])) + float(np.sum(c))
        day_cost[d] += float(np.sum(c))

        for j in m:  # tax lots, futures P&L, round trips
            q_long = max(target[j], 0) - max(old[j], 0)
            q_short = min(target[j], 0) - min(old[j], 0)
            if q_long > 0:
                lots[j].append([q_long, fill[j], d])
            elif q_long < 0:
                left = -q_long
                while left > 1e-9 and lots[j]:
                    lot = lots[j][0]
                    q = min(lot[0], left)
                    tax.add(fy(d), "LT" if (d - lot[2]).days > 365 else "ST", q * (fill[j] - lot[1]))
                    lot[0] -= q
                    left -= q
                    if lot[0] <= 1e-9:
                        lots[j].popleft()
            if q_short < 0:
                prev = -min(old[j], 0)
                short_px[j] = (short_px.get(j, 0) * prev + fill[j] * -q_short) / (prev - q_short)
            elif q_short > 0:
                tax.add(fy(d), "FNO", q_short * (short_px[j] - fill[j]))
            t = open_trip.get(j)
            opened = delta[j]
            if old[j] == 0 or (t and np.sign(old[j]) != np.sign(target[j]) and target[j] != 0):
                if t:  # long <-> short flip: close the old trip at this fill, open a new one
                    self._close_trip(t, j, i, fill[j], old[j], "CLOSED", trips)
                    opened = target[j]
                open_trip[j] = t = {"j": j, "side": "LONG" if target[j] > 0 else "SHORT", "entry_sig": sd,
                                    "entry_i": i, "entry_px": fill[j], "flow": 0.0, "gross_in": 0.0,
                                    "resizes": 0, "costs": 0.0}
            elif target[j] != 0:
                t["resizes"] += 1
            t["flow"] -= opened * fill[j]
            t["gross_in"] += abs(opened * fill[j]) if np.sign(opened) == np.sign(target[j] or old[j]) else 0
            t["costs"] += cost[j]
            if target[j] == 0:
                self._close_trip(t, j, i, fill[j], 0, "CLOSED", trips)
                del open_trip[j]
        shares[:] = target
        if cfg.short_n:
            for j in list(short_px):
                if shares[j] >= 0:
                    del short_px[j]

        tgt_long = float((tw[tw > 0] * cap).sum())
        inv = float(np.nansum(np.maximum(target, 0) * px_sd))
        picked = tw > 0
        rebs.append({"signal_date": sd, "fill_date": d, "nav": cap, "regime": "below 200DMA" if self.below[sd]
                     else "above 200DMA", "gross_exposure": gross, "exposure_rule": rule, "target_long_rs": tgt_long, "invested_rs": inv,
                     "picks": int(picked.sum()), "picks_bought": int((picked & (target > 0)).sum()),
                     "shorts_wanted": int((tw < 0).sum()), "shorts_placed": int((target < 0).sum()),
                     "orders": len(m), "buys": int(((old == 0) & (target != 0)).sum()),
                     "sells": int(((old != 0) & (target == 0)).sum()),
                     "resizes": int(((old != 0) & (target != 0) & (target != old)).sum()), "costs_rs": float(np.sum(c))})
        for j in m:
            orders.append({"signal_date": sd, "fill_date": d, "j": j, "qty": delta[j], "price": fill[j],
                           "value": delta[j] * fill[j], "cost": cost[j]})
        if actions is None:
            return cash
        rank = ctx.scores.loc[sd].where(ctx.elig.loc[sd]).rank(ascending=False)
        for j in np.nonzero((old != 0) | (target != 0) | (tw != 0))[0]:
            sym, o, nq = self.syms[j], old[j], target[j]
            r = rank.get(sym)
            r = None if pd.isna(r) else int(r)
            if o == 0 and nq == 0:
                act = "SKIP"
            elif o == 0:
                act = "BUY" if nq > 0 else "SHORT"
            elif nq == 0:
                act = "SELL" if o > 0 else "COVER"
            elif nq == o:
                act = "HOLD" if nq > 0 else "HOLD SHORT"
            else:
                act = ("ADD" if abs(nq) > abs(o) else "TRIM") + ("" if nq > 0 else " SHORT")
            if act == "SKIP" or (nq == 0 and tw[j] != 0):
                why = f"target Rs {abs(tw[j]) * cap:,.0f} < 1 {'lot' if tw[j] < 0 else 'share'} " \
                      f"(Rs {px_sd[j] * (self.lot[j] if tw[j] < 0 else 1):,.0f})"
            elif act in ("SELL", "COVER"):
                if o > 0 and not ctx.elig.at[sd, sym]:
                    why = "12-1 momentum turned negative" if ctx.mom.at[sd, sym] <= 0 else "failed liquidity/price filter"
                elif o > 0 and r and r > self.buffer:
                    why = f"rank {r} fell below hold buffer {self.buffer}"
                elif o > 0:
                    why = "sector cap / displaced by higher rank"
                else:
                    why = "no longer in bottom ranks"
            elif act in ("BUY", "SHORT"):
                why = f"new pick, rank {r}" if r else "new short"
            elif act.startswith(("ADD", "TRIM")):
                why = "resize to target weight (>25% off)"
            else:
                why = "within hold buffer, size within 25% band"
            actions.append({"signal_date": sd.date(), "fill_date": d.date(), "symbol": sym,
                            "name": self.names.get(sym, sym), "sector": self.sectors.get(sym, ""), "action": act,
                            "reason": why, "rank": r, "score": round(float(ctx.scores.at[sd, sym]), 3),
                            "target_weight %": tw[j], "target (₹)": tw[j] * cap, "old_qty": o, "new_qty": nq,
                            "qty_change": nq - o, "signal_close": px_sd[j], "fill_price": fill[j],
                            "order_value (₹)": (nq - o) * fill[j], "cost (₹)": cost.get(j, 0.0),
                            "position_value (₹)": nq * fill[j], "_j": j, "_i": i})
        return cash

    def _close_trip(self, t, j, i, px, qty_left, status, trips):
        flow = t["flow"] + qty_left * px
        sgn = 1 if t["side"] == "LONG" else -1
        b0, b1 = self.bench.iloc[t["entry_i"] - 1], self.bench.iloc[i - 1 if status == "CLOSED" else i]
        price_ret = sgn * (px / t["entry_px"] - 1)
        trips.append({"symbol": self.syms[j], "name": self.names.get(self.syms[j], ""),
                      "sector": self.sectors.get(self.syms[j], ""), "side": t["side"], "status": status,
                      "entry_signal": t["entry_sig"].date(), "entry_date": self.dates[t["entry_i"]].date(),
                      "exit_date": self.dates[i].date(), "days_held": (self.dates[i] - self.dates[t["entry_i"]]).days,
                      "resizes": t["resizes"], "entry_price": t["entry_px"], "exit_price": px,
                      "price_return %": price_ret, "nifty_return %": b1 / b0 - 1,
                      "alpha %": price_ret - (b1 / b0 - 1), "invested (₹)": t["gross_in"],
                      "gross_pnl (₹)": flow, "costs (₹)": t["costs"], "net_pnl (₹)": flow - t["costs"],
                      "result": "WIN" if flow - t["costs"] > 0 else "LOSS"})


# ------------------------------------------------------------------------------------------------
def perf(nav: pd.Series, bench: pd.Series, rf: float) -> dict:
    r, b = nav.pct_change().dropna(), bench.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    rf_d = (1 + rf) ** (1 / 252) - 1
    mn, mb = nav.resample("ME").last(), bench.resample("ME").last()
    mn, mb = pd.concat([nav.iloc[:1], mn]).pct_change().dropna(), pd.concat([bench.iloc[:1], mb]).pct_change().dropna()
    beta = r.cov(b) / b.var()
    return {"total": nav.iloc[-1] / nav.iloc[0] - 1, "cagr": (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            "vol": r.std() * np.sqrt(252), "sharpe": (r.mean() - rf_d) / r.std() * np.sqrt(252) if r.std() else np.nan,
            "maxdd": (nav / nav.cummax() - 1).min(), "beta": beta,
            "months_beat": float((mn.values > mb.reindex(mn.index).values).mean()),
            "worst_month": mn.min(), "best_month": mn.max()}


def window_list(dates: pd.DatetimeIndex, first: str, last: str, years: int, step: int) -> list[tuple]:
    out, m = [], pd.Period(first, "M")
    last_end = pd.Period(last, "M")
    while True:
        e = m + 12 * years - 1
        if e > last_end:
            break
        sig = dates[dates < m.start_time]
        end = dates[dates <= e.end_time]
        if len(sig) and len(end) and end[-1] >= e.start_time:
            out.append((f"{m.strftime('%b %Y')} - {e.strftime('%b %Y')}", dates.get_loc(sig[-1]),
                        dates.get_loc(end[-1])))
        m += step
    return out


def chain_windows(dates: pd.DatetimeIndex, first: str, last: str, years: int) -> list[tuple]:
    """Back-to-back windows (compound mode): each starts where the previous ended; the last may be shorter."""
    out, m = [], pd.Period(first, "M")
    last_end = pd.Period(last, "M")
    while m <= last_end:
        e = min(m + 12 * years - 1, last_end)
        sig = dates[dates < m.start_time]
        end = dates[dates <= e.end_time]
        if len(sig) and len(end) and end[-1] >= m.start_time:
            n = (e - m).n + 1
            out.append((f"{m.strftime('%b %Y')} - {e.strftime('%b %Y')}" + (f" ({n} months)" if n < 12 * years else ""),
                        dates.get_loc(sig[-1]), dates.get_loc(end[-1])))
        m = e + 1
    return out


def forward_returns(sim: Sim, actions: list, rebs: list, end: int) -> None:
    """Stock and Nifty return from each fill to the next fill (or window end): did the decision pay?"""
    fills = [sim.dates.get_loc(pd.Timestamp(r["fill_date"])) for r in rebs] + [end]
    nxt = dict(zip(fills[:-1], fills[1:]))
    for a in actions:
        i, j = a.pop("_i"), a.pop("_j")
        k = nxt[i]
        exit_px = sim.close[k, j] if k == end else (sim.open[k, j] if not np.isnan(sim.open[k, j]) else sim.close[k, j])
        sr = exit_px / a["fill_price"] - 1
        br = sim.bench.iloc[k if k == end else k - 1] / sim.bench.iloc[i - 1] - 1
        a["next_period_end"] = sim.dates[k].date()
        a["stock_next_period %"] = sr
        a["nifty_next_period %"] = br
        a["stock_alpha_next_period %"] = sr - br
        held = a["new_qty"]
        a["position_pnl_next_period (₹)"] = held * (exit_px - a["fill_price"])


def monthly_table(res: dict, bench: pd.Series, label: dict) -> pd.DataFrame:
    nav, lm = res["nav"], res["long_mv"]
    me = nav.iloc[1:].groupby(nav.index[1:].to_period("M")).tail(1)  # iloc[0] is the start signal close
    prev = pd.concat([nav.iloc[:1], me]).shift(1).iloc[1:]
    bm = bench.reindex(me.index)
    bprev = pd.concat([bench.reindex(nav.index[:1]), bm]).shift(1).iloc[1:]
    o = pd.DataFrame(res["orders"])
    rows = []
    for d, v in me.items():
        per = d.to_period("M")
        oo = o[pd.to_datetime(o["fill_date"]).dt.to_period("M") == per] if len(o) else o
        r = v / prev[d] - 1
        br = bm[d] / bprev[d] - 1
        rows.append({**label, "month": per.strftime("%Y-%m"), "start_value (₹)": prev[d], "end_value (₹)": v,
                     "return %": r, "nifty_return %": br, "alpha %": r - br,
                     "invested %": lm[d] / v if v else np.nan, "holdings": None,
                     "orders": len(oo), "buy_value (₹)": float(oo.loc[oo["qty"] > 0, "value"].sum()) if len(oo) else 0.0,
                     "sell_value (₹)": float(-oo.loc[oo["qty"] < 0, "value"].sum()) if len(oo) else 0.0,
                     "costs (₹)": float(res["costs"][res["costs"].index.to_period("M") == per].sum()),
                     "tax_paid (₹)": float(res["tax_days"][res["tax_days"].index.to_period("M") == per].sum())
                     if len(res["tax_days"]) else 0.0})
    m = pd.DataFrame(rows)
    m["cum_return %"] = (1 + m["return %"]).cumprod() - 1
    m["cum_nifty %"] = (1 + m["nifty_return %"]).cumprod() - 1
    m["cum_alpha %"] = m["cum_return %"] - m["cum_nifty %"]
    m["drawdown %"] = m["end_value (₹)"] / m["end_value (₹)"].cummax().clip(lower=m["start_value (₹)"].iloc[0]) - 1
    return m.drop(columns="holdings")


# ------------------------------------------------------------------------------------------------
def run_strategy(cfg: StrategyConfig, p, windows, cont, capital: float, compound: bool, sim=None) -> dict:
    print(f"\n[{cfg.name}] building signals + model backtest...")
    cfg = cfg.replace(initial_capital=capital)
    if sim is None:  # callers running several modes on one config can pass a ready Sim
        sim = Sim(Context(p, cfg))
    stress_costs = replace(cfg.costs, slippage_bps=cfg.costs.slippage_bps * STRESS,
                           impact_coef=cfg.costs.impact_coef * STRESS)
    variants = {"pre_tax": (cfg.costs, False), "after_tax": (cfg.costs, True), "stress": (stress_costs, True)}
    rows, monthly, all_actions, out = [], [], [], {"win_navs": []}
    for wi, (label, s0, end) in enumerate([cont] + windows):
        is_cont = wi == 0
        print(f"  {label}{' (continuous)' if is_cont else ''}")
        if is_cont or not compound or wi == 1:  # fresh capital; in compound mode only for the first window
            start, books, b_start, b_book = dict.fromkeys(VARIANTS, capital), dict.fromkeys(VARIANTS), capital, TaxBook()
        res = {v: sim.run(s0, end, start[v], c, tax, log=(v == "after_tax"), book=books[v])
               for v, (c, tax) in variants.items()}
        bench = sim.bench.iloc[s0:end + 1]
        b_nav = bench / bench.iloc[0] * b_start
        b_book.add(fy(sim.dates[end]), "LT" if (sim.dates[end] - sim.dates[s0]).days > 365 else "ST",
                   b_nav.iloc[-1] - b_start)
        b_tax = b_book.settle(fy(sim.dates[end]))
        yrs = (sim.dates[end] - sim.dates[s0]).days / 365.25
        a = res["after_tax"]
        forward_returns(sim, a["actions"], a["rebs"], end)
        pp, pa, ps = (perf(res[v]["nav"], b_nav, cfg.rf_rate) for v in VARIANTS)
        pb = perf(b_nav, b_nav, cfg.rf_rate)
        rebs = pd.DataFrame(a["rebs"])
        trips = pd.DataFrame(a["trips"])
        acts = pd.DataFrame(a["actions"])
        closed = trips[trips["status"] == "CLOSED"] if len(trips) else trips
        held = acts[acts["action"].isin(["HOLD", "ADD", "TRIM", "BUY"])] if len(acts) else acts
        sold = acts[acts["action"] == "SELL"] if len(acts) else acts
        at_ret = a["net_liq"] / start["after_tax"] - 1
        b_at = (b_nav.iloc[-1] - b_tax) / b_start - 1
        st_ret = res["stress"]["net_liq"] / start["stress"] - 1
        cg = lambda r: (1 + r) ** (1 / yrs) - 1
        row = {"strategy": cfg.name, "window": label, "start_signal": sim.dates[s0].date(),
               "end_date": sim.dates[end].date(), "years": yrs,
               "start_value_after_tax (₹)": start["after_tax"], "nifty_start_value (₹)": b_start,
               "end_value_pre_tax (₹)": res["pre_tax"]["nav"].iloc[-1],
               "pre_tax_return %": pp["total"], "pre_tax_CAGR %": pp["cagr"],
               "after_tax_liquidation_value (₹)": a["net_liq"], "after_tax_return %": at_ret,
               "after_tax_CAGR %": cg(at_ret), "stress_after_tax_return %": st_ret,
               "stress_after_tax_CAGR %": cg(st_ret), "stress_liquidation_value (₹)": res["stress"]["net_liq"],
               "nifty_after_tax_value (₹)": b_nav.iloc[-1] - b_tax,
               "nifty_return %": pb["total"], "nifty_CAGR %": pb["cagr"],
               "nifty_after_tax_return %": b_at, "nifty_after_tax_CAGR %": cg(b_at),
               "alpha_pre_tax_CAGR %": pp["cagr"] - pb["cagr"],
               "alpha_after_tax_CAGR %": cg(at_ret) - cg(b_at),
               "alpha_stress_CAGR %": cg(st_ret) - cg(b_at),
               "beat_nifty_after_tax": "YES" if at_ret > b_at else "NO",
               "max_drawdown %": pp["maxdd"], "nifty_max_drawdown %": pb["maxdd"],
               "volatility %": pp["vol"], "sharpe": pp["sharpe"], "nifty_sharpe": pb["sharpe"], "beta": pp["beta"],
               "months_beating_nifty %": pp["months_beat"], "worst_month %": pp["worst_month"],
               "avg_invested %": float((a["long_mv"] / a["nav"]).mean()),
               "target_capital_actually_invested %": rebs["invested_rs"].sum() / rebs["target_long_rs"].sum()
               if rebs["target_long_rs"].sum() else np.nan,
               "picks_not_bought %": 1 - rebs["picks_bought"].sum() / max(rebs["picks"].sum(), 1),
               "shorts_not_placed %": (1 - rebs["shorts_placed"].sum() / rebs["shorts_wanted"].sum())
               if rebs["shorts_wanted"].sum() else np.nan,
               "avg_holdings": float(rebs["picks_bought"].mean()), "rebalances": len(rebs),
               "orders": int(rebs["orders"].sum()), "buys": int(rebs["buys"].sum()), "sells": int(rebs["sells"].sum()),
               "resizes": int(rebs["resizes"].sum()),
               "turnover_per_year %": sum(abs(o["value"]) for o in a["orders"]) / 2 / a["nav"].mean() / yrs,
               "costs (₹)": float(res["pre_tax"]["costs"].sum()),
               "costs_per_year_of_capital %": float(res["pre_tax"]["costs"].sum()) / start["pre_tax"] / yrs,
               "stress_costs (₹)": float(res["stress"]["costs"].sum()),
               "tax_paid (₹)": a["tax_paid"], "nifty_tax (₹)": b_tax,
               "tax_drag_CAGR %": pp["cagr"] - cg(at_ret),
               "closed_trades": len(closed), "trade_win_rate %": (closed["net_pnl (₹)"] > 0).mean() if len(closed) else np.nan,
               "trades_beat_nifty %": (closed["alpha %"] > 0).mean() if len(closed) else np.nan,
               "avg_days_held": closed["days_held"].mean() if len(closed) else np.nan,
               "held_stock_alpha_next_month %": held["stock_alpha_next_period %"].mean() if len(held) else np.nan,
               "sold_stock_alpha_next_month %": sold["stock_alpha_next_period %"].mean() if len(sold) else np.nan}
        rows.append(row)
        if not is_cont:
            out["win_navs"].append(a["nav"])
        if compound and not is_cont:  # roll each variant's liquidation value (and tax state) into the next window
            start = {v: res[v]["net_liq"] for v in VARIANTS}
            books = {v: res[v]["book"] for v in VARIANTS}
            b_start = b_nav.iloc[-1] - b_tax
        if is_cont:
            out["cont"] = {"res": res, "bench": b_nav, "actions": acts, "trips": trips, "rebs": rebs, "row": row}
            for v in VARIANTS:
                monthly.append(monthly_table(res[v], b_nav, {"strategy": cfg.name, "variant": v}))
        else:
            m = monthly_table(a, b_nav, {"strategy": cfg.name, "window": label})
            monthly.append(m)
            if len(acts):
                all_actions.append(acts.assign(strategy=cfg.name, window=label))
    out["rows"] = rows
    out["monthly"] = monthly
    out["all_actions"] = pd.concat(all_actions, ignore_index=True) if all_actions else pd.DataFrame()
    return out


def regimes(name: str, cont: dict, below: pd.Series) -> pd.DataFrame:
    nav, b = cont["res"]["pre_tax"]["nav"], cont["bench"]
    rows = []
    for ph, a, z in PHASES:
        sel = nav[(nav.index >= pd.Period(a, "M").start_time) & (nav.index <= pd.Period(z, "M").end_time)]
        if len(sel) < 2:
            continue
        i0 = nav.index.get_loc(sel.index[0])
        n0 = nav.iloc[i0 - 1] if i0 else nav.iloc[0]
        b0 = b.iloc[i0 - 1] if i0 else b.iloc[0]
        s = sel.iloc[-1] / n0 - 1
        br = b[sel.index[-1]] / b0 - 1
        seg = pd.concat([pd.Series([n0]), sel.reset_index(drop=True)])
        bseg = pd.concat([pd.Series([b0]), b[sel.index].reset_index(drop=True)])
        rows.append({"strategy": name, "phase": ph, "from": a, "to": z, "strategy_return %": s,
                     "nifty_return %": br, "alpha %": s - br, "strategy_max_drawdown %": (seg / seg.cummax() - 1).min(),
                     "nifty_max_drawdown %": (bseg / bseg.cummax() - 1).min(),
                     "avg_invested %": float((cont["res"]["pre_tax"]["long_mv"][sel.index] / sel).mean())})
    me = nav.groupby(nav.index.to_period("M")).tail(1)
    r = me.pct_change().dropna()
    bm = b.reindex(me.index).pct_change().dropna()
    reg = below.reindex(me.index).shift(1).dropna().astype(bool)  # regime at the start of each month
    for flag, lab in ((False, "Months starting ABOVE Nifty 200DMA"), (True, "Months starting BELOW Nifty 200DMA")):
        idx = reg[reg == flag].index.intersection(r.index)
        if len(idx):
            rows.append({"strategy": name, "phase": lab, "from": "", "to": f"{len(idx)} months",
                         "strategy_return %": (1 + r[idx]).prod() ** (12 / len(idx)) - 1,
                         "nifty_return %": (1 + bm[idx]).prod() ** (12 / len(idx)) - 1,
                         "alpha %": (1 + r[idx]).prod() ** (12 / len(idx)) - (1 + bm[idx]).prod() ** (12 / len(idx)),
                         "strategy_max_drawdown %": r[idx].min(), "nifty_max_drawdown %": bm[idx].min(),
                         "avg_invested %": np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
CHECK_FN = {  # id -> (metric on one strategy's window rows, pass test against the threshold)
    "beat_share": (lambda w, t: (w["beat_nifty_after_tax"] == "YES").mean(), lambda v, t: v >= t),
    "median_alpha": (lambda w, t: w["alpha_after_tax_CAGR %"].median(), lambda v, t: v >= t),
    "worst_alpha": (lambda w, t: w["alpha_after_tax_CAGR %"].min(), lambda v, t: v > t),
    "stress_share": (lambda w, t: (w["alpha_stress_CAGR %"] > 0).mean(), lambda v, t: v >= t),
    "survivorship": (lambda w, t: w["alpha_after_tax_CAGR %"].median() - t, lambda v, t: v > 0),
    "drawdown": (lambda w, t: (w["max_drawdown %"] - w["nifty_max_drawdown %"]).median(), lambda v, t: v >= -t),
    "invested": (lambda w, t: w["target_capital_actually_invested %"].median(), lambda v, t: v >= t),
}
LABELS = {cid: label for cid, label, _ in CHECK_DEFS}
CHECKS = [(LABELS[c["id"]].format(f"{c['threshold']:.1%}".replace(".0%", "%")), *CHECK_FN[c["id"]], c["threshold"])
          for c in G["checks"] if c["enabled"]]


def verdict(win: pd.DataFrame, cont_rows: pd.DataFrame, capital: float, compound: bool) -> pd.DataFrame:
    items = []
    for name, w in win.groupby("strategy", sort=False):
        c = cont_rows.set_index("strategy").loc[name]
        d = {"strategy": name}
        if compound:
            yrs, last = w["years"].sum(), w.iloc[-1]
            cg = lambda v: (v / capital) ** (1 / yrs) - 1
            d.update({
                f"COMPOUNDING: ONE POT ROLLED THROUGH {len(w)} BACK-TO-BACK WINDOWS": "",
                "Initial capital (₹)": capital,
                "Final value, after tax (₹)": last["after_tax_liquidation_value (₹)"],
                "Growth multiple, after tax (x)": last["after_tax_liquidation_value (₹)"] / capital,
                "Compounded CAGR, after tax %": cg(last["after_tax_liquidation_value (₹)"]),
                "Final value, 2x cost stress (₹)": last["stress_liquidation_value (₹)"],
                "Compounded CAGR, 2x cost stress %": cg(last["stress_liquidation_value (₹)"]),
                "Nifty 500 final value, after tax (₹)": last["nifty_after_tax_value (₹)"],
                "Nifty 500 compounded CAGR, after tax %": cg(last["nifty_after_tax_value (₹)"]),
                "Ahead of Nifty at the end (₹)": last["after_tax_liquidation_value (₹)"] - last["nifty_after_tax_value (₹)"],
                "Windows where the pot grew %": (w["after_tax_return %"] > 0).mean(),
                "Windows where the pot grew faster than Nifty %": (w["beat_nifty_after_tax"] == "YES").mean(),
                "Years covered": yrs})
        d.update({
            "2-YEAR WINDOWS": "", "Windows tested": len(w),
            "Median after-tax 2y return %": w["after_tax_return %"].median(),
            "Median Nifty after-tax 2y return %": w["nifty_after_tax_return %"].median(),
            "Windows beating Nifty (pre-tax) %": (w["alpha_pre_tax_CAGR %"] > 0).mean(),
            "Windows beating Nifty (after tax) %": (w["beat_nifty_after_tax"] == "YES").mean(),
            "Windows beating Nifty (2x cost stress) %": (w["alpha_stress_CAGR %"] > 0).mean(),
            "Windows with a loss (after tax) %": (w["after_tax_return %"] < 0).mean(),
            "Median after-tax alpha (CAGR pp) %": w["alpha_after_tax_CAGR %"].median(),
            "Worst after-tax alpha (CAGR pp) %": w["alpha_after_tax_CAGR %"].min(),
            "Best after-tax alpha (CAGR pp) %": w["alpha_after_tax_CAGR %"].max(),
            "Worst window": w.loc[w["alpha_after_tax_CAGR %"].idxmin(), "window"],
            "Worst 2y after-tax return %": w["after_tax_return %"].min(),
            "Median max drawdown %": w["max_drawdown %"].median(),
            "Worst max drawdown %": w["max_drawdown %"].min(),
            "Median Sharpe (pre-tax)": w["sharpe"].median(),
            "Median Nifty Sharpe": w["nifty_sharpe"].median(),
            "IMPLEMENTATION AT THIS CAPITAL": "",
            ("Starting capital of window 1 (₹)" if compound else "Capital per window (₹)"): capital,
            "Target capital actually invested %": w["target_capital_actually_invested %"].median(),
            "Average invested (rest is cash) %": w["avg_invested %"].median(),
            "Picks not bought (1 share too expensive) %": w["picks_not_bought %"].median(),
            "Shorts not placed (below 1 futures lot) %": w["shorts_not_placed %"].median(),
            "Costs per year, % of capital %": w["costs_per_year_of_capital %"].median(),
            "Tax drag (CAGR pp) %": w["tax_drag_CAGR %"].median(),
            "Turnover per year %": w["turnover_per_year %"].median(),
            "DECISION QUALITY": "",
            "Closed-trade win rate %": w["trade_win_rate %"].median(),
            "Closed trades beating Nifty %": w["trades_beat_nifty %"].median(),
            "Held stocks: avg next-month alpha %": w["held_stock_alpha_next_month %"].median(),
            "Sold stocks: avg next-month alpha %": w["sold_stock_alpha_next_month %"].median(),
            "CONTINUOUS RUN (JAN 2019 - SEP 2026)": "",
            "Pre-tax CAGR %": c["pre_tax_CAGR %"], "After-tax CAGR %": c["after_tax_CAGR %"],
            "Nifty 500 CAGR %": c["nifty_CAGR %"], "Max drawdown %": c["max_drawdown %"],
            "Nifty max drawdown %": c["nifty_max_drawdown %"],
            "Final after-tax value (₹)": c["after_tax_liquidation_value (₹)"],
            "DECISION CHECKS (thresholds: settings workbook, Decision Checks sheet)": ""})
        passed = 0
        for lab, metric, ok, thr in CHECKS:
            v = metric(w, thr)
            passed += bool(ok(v, thr))
            d[lab] = f"{'PASS' if ok(v, thr) else 'FAIL'}  ({v:.1%})"
        d["Checks passed"] = f"{passed} / {len(CHECKS)}"
        items.append(d)
    t = pd.DataFrame(items).set_index("strategy").T
    t.index.name = "Metric"
    return t.reset_index()


def survivorship(win: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, w in win.groupby("strategy", sort=False):
        for h in (0.0, 0.02, 0.04, 0.06):
            a = w["alpha_after_tax_CAGR %"] - h
            rows.append({"strategy": name, "assumed_haircut_pp_per_year %": h,
                         "median_after_tax_alpha %": a.median(), "worst_after_tax_alpha %": a.min(),
                         "windows_still_beating_nifty %": (a > 0).mean()})
    return pd.DataFrame(rows)


def readme(a, cfgs, windows, cont, dates) -> pd.DataFrame:
    c = cfgs[0].costs
    compound = a.mode == "compound"
    lines = [
        ("WHAT THIS IS", ""),
        ("Purpose", "Walk-forward test of each momentum strategy in every rolling 2-year window, to judge "
                    "whether it is worth deploying real capital."),
        ("Strategies", ", ".join(c.name for c in cfgs)),
        ("Capital", f"COMPOUND: Rs {a.capital:,.0f} once, at the start of the first window. At each window end the "
                    "book is liquidated (exit costs + tax) and that after-tax cash starts the next window; carried-"
                    "forward tax losses and the FY's used LTCG exemption carry over too. Nifty 500 is chained the "
                    "same way. See the 'Compounding' sheet and the top of 'Verdict'." if compound else
                    f"FRESH: Rs {a.capital:,.0f} cold start in cash for every window"),
        ("Windows", f"{len(windows)} back-to-back windows of up to {a.years} years: {windows[0][0]} ... {windows[-1][0]}"
                    if compound else f"{len(windows)} windows of {a.years} years, rolled every {a.step} month(s): "
                    f"{windows[0][0]} ... {windows[-1][0]}"),
        ("Continuous run", f"{cont[0]}: one unbroken path used for the trade log, positions and regimes"),
        ("Data", f"Upstox daily OHLC, split/bonus adjusted, through {dates[-1].date()}"),
        ("Settings", f"From '{WORKBOOK.name}' plus any command-line overrides."),
        ("", ""), ("HOW A MONTH WORKS", ""),
        ("Signal", "Scores, filters, regime (Nifty 500 vs 200DMA) and vol target from closes up to the signal date "
                   "(last trading day of the period). Same function as the app: backtest.target_book."),
        ("Sizing", "Longs: floor(weight x NAV / signal close) whole shares. Shorts: nearest stock-futures lot "
                   f"(today's lot sizes). Resizes under {NO_TRADE_BAND:.0%} are skipped (journal no-trade band)."),
        ("Execution", f"All orders fill at the next day's open (close if no open). Costs: Rs {c.brokerage_per_order:g} "
                      f"brokerage/order + {c.brokerage_bps:g} bp, STT {c.stt_bps:g} bp, stamp {c.stamp_duty_buy_bps:g} bp "
                      f"on buys, exchange + SEBI + {c.gst_rate:.0%} GST, Rs {c.dp_charge_per_sell:g} DP per sell, "
                      f"{c.slippage_bps:g} bp slippage, square-root market impact (coef {c.impact_coef:g})."),
        ("Between rebalances", "Positions drift with prices; NAV is marked to market daily; idle cash earns the "
                               "config's cash_rate (liquid fund)."),
        ("", ""), ("VARIANTS", ""),
        ("pre_tax", "Base costs, no tax. Comparable to the model backtest in the app."),
        ("after_tax", "Base costs; capital-gains tax paid from the portfolio every 31 March; on the last day the "
                      "book is liquidated (exit costs + tax) -> 'after-tax liquidation value'. HEADLINE CASE."),
        ("stress", f"{STRESS:.0f}x slippage and {STRESS:.0f}x market impact, after tax."),
        ("Benchmark", "Nifty 500 price index bought at the signal close and held; after-tax = LTCG on exit."),
        ("", ""), ("TAX MODEL", ""),
        ("Rates", f"Same rates for all years: STCG {TAX['stcg']:.1%}, LTCG {TAX['ltcg']:.1%} above "
                  f"Rs {TAX['ltcg_exempt']:,.0f}/FY, +{TAX['cess']:.0%} cess (no surcharge). Before 23 Jul 2024 the real "
                  "rates were 15% / 10% above Rs 1L, so with today's rates earlier years are taxed slightly harder "
                  "than they really were -- deliberately forward-looking."),
        ("Set-off", "FIFO lots. ST losses offset ST then LT gains; LT losses offset LT only; unused losses carried "
                    f"forward. Futures (shorts) profit taxed as business income at a flat {TAX['fno']:.0%} + cess."),
        ("Not modelled", "Advance-tax timing (tax is paid at FY end), cost basis excluding brokerage, slab rates, "
                         "dividends and dividend tax."),
        ("", ""), ("KNOWN BIASES -- READ BEFORE DECIDING", ""),
        ("Survivorship", "Universe is TODAY's Nifty 500 for every year. Stocks that fell out or were delisted since "
                         "2019 are missing, and today's members partly got in because they went up. This inflates "
                         "momentum returns. See the 'Survivorship' sheet for how much alpha survives a haircut."),
        ("F&O list", "Short eligibility and lot sizes use today's F&O list, not the historical one."),
        ("Dividends", "Neither the stocks nor the Nifty 500 price index include dividends (~1-1.5%/yr for the "
                      "index, less for momentum stocks), so the benchmark is slightly understated."),
        ("Overlap", "Compound mode: windows do not overlap, so each is independent -- but there are only a few, "
                    "and the window-end liquidation adds one extra sell + rebuy (costs, tax) that the continuous run "
                    "does not have." if compound else
                    "Rolled 2-year windows overlap by up to 21 months, so they are not independent "
                    "samples: 24 windows carry roughly 4 years of independent evidence."),
        ("Parameter selection", "The configs were designed with knowledge of the 2014-2026 history, so even "
                                "out-of-window results are partly in-sample."),
        ("", ""), ("TABLES", ""),
        ("Verdict", "Headline numbers per strategy + suggested pass/fail decision checks."),
        *([("Compounding", "The one pot's value at the start and at the end of every window, after tax, vs Nifty 500 "
                           "chained the same way.")] if compound else []),
        ("Windows", "One row per strategy x 2-year window, all variants."),
        ("Alpha by window", "After-tax alpha (CAGR pp) per window, strategies side by side."),
        ("Growth", "Continuous run: value of the capital each month-end, after tax, vs Nifty 500."),
        ("Monthly (continuous)", "Month-over-month P&L of the continuous run, each variant."),
        ("Monthly (windows)", "Month-over-month P&L of every window (after-tax variant)."),
        ("Actions", "Continuous run: every stock at every rebalance -- BUY / SELL / HOLD / ADD / TRIM / SKIP / "
                    "SHORT / COVER, the reason, size, fill, cost, and what the stock did over the next period."),
        ("Positions", "Continuous run: each round trip from entry to exit with P&L and alpha."),
        ("Rebalances", "Continuous run: one row per rebalance (regime, exposure, invested vs target, orders)."),
        ("Regimes", "Continuous run split by market phase and by Nifty 200DMA regime."),
        ("Survivorship", "Alpha after assumed 0/2/4/6 pp per year survivorship haircuts."),
    ]
    return pd.DataFrame(lines, columns=["Item", "Detail"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    R = G["rolling"]
    ap.add_argument("configs", nargs="*", help="config JSONs (default: 'Include in rolling backtest' = YES)")
    ap.add_argument("--capital", type=float, help=f"starting capital (default Rs {R['capital']:,.0f})")
    ap.add_argument("--mode", choices=("compound", "fresh"), default=R.get("capital_mode") or "compound",
                    help="compound = one pot rolled through back-to-back windows; fresh = new capital every window")
    ap.add_argument("--from", dest="first", default=R["first"], help="first window start month")
    ap.add_argument("--to", dest="last", default=R["last"], help="last window end month")
    ap.add_argument("--years", type=int, default=R["years"], help="window length in years")
    ap.add_argument("--step", type=int, default=R["step"], help="months between window starts")
    a = ap.parse_args()
    caps = [a.capital] if a.capital else [R["capital"]] + ([R["compare_capital"]] if R.get("compare_capital") else [])
    for cap in caps:
        a.capital = cap
        run_report(a)


def run_report(a):

    paths = a.configs or [str(CONFIGS / f"{n}.json") for n in DEFAULT_CONFIGS]
    cfgs = [StrategyConfig.from_json(x) for x in paths]
    print("Loading prices...")
    p = load_panel(load_universe(cfgs[0].universe_csv), cfgs[0].benchmark)
    dates = p.close.index
    compound = a.mode == "compound"
    windows = chain_windows(dates, a.first, a.last, a.years) if compound else \
        window_list(dates, a.first, a.last, a.years, a.step)
    if not windows:
        raise SystemExit("No complete window fits the data range.")
    s_last = dates[dates <= pd.Period(a.last, "M").end_time]
    cont = (f"{pd.Period(a.first, 'M').strftime('%b %Y')} - {s_last[-1].strftime('%b %Y')} (continuous)",
            windows[0][1], dates.get_loc(s_last[-1]))
    print(f"{len(windows)} {'back-to-back windows, capital compounding' if compound else 'windows, fresh capital each'}"
          f": {windows[0][0]} ... {windows[-1][0]}")

    results = [run_strategy(c, p, windows, cont, a.capital, compound) for c in cfgs]
    rows = pd.DataFrame([r for res in results for r in res["rows"]])
    is_cont = rows["window"] == cont[0]
    win, cont_rows = rows[~is_cont].reset_index(drop=True), rows[is_cont].reset_index(drop=True)
    alpha = win.pivot(index="window", columns="strategy", values="alpha_after_tax_CAGR %")
    alpha = alpha.loc[[w[0] for w in windows], [c.name for c in cfgs]].reset_index()
    growth = pd.DataFrame({"month": results[0]["cont"]["res"]["after_tax"]["nav"].resample("ME").last().index.strftime("%Y-%m")})
    for c, res in zip(cfgs, results):
        growth[c.name] = res["cont"]["res"]["after_tax"]["nav"].resample("ME").last().values
    growth["Nifty 500"] = results[0]["cont"]["bench"].resample("ME").last().values
    below = (p.benchmark < p.benchmark.rolling(200).mean()).reindex(dates)

    def tag(df, name):
        return df.assign(strategy=name)[["strategy"] + [x for x in df.columns if x != "strategy"]] if len(df) else df

    pot = pd.DataFrame({"point": ["Start"] + [f"End of {w[0]}" for w in windows],
                        "date": [win["start_signal"].iloc[0]] + list(win["end_date"].iloc[:len(windows)])})
    for c in cfgs:
        pot[c.name] = [a.capital] + list(win.loc[win["strategy"] == c.name, "after_tax_liquidation_value (₹)"])
    pot["Nifty 500"] = [a.capital] + list(win.loc[win["strategy"] == cfgs[0].name, "nifty_after_tax_value (₹)"])

    sheets = {
        "Read me": readme(a, cfgs, windows, cont, dates),
        "Verdict": verdict(win, cont_rows, a.capital, compound),
        **({"Compounding": pot} if compound else {}),
        "Windows": pd.concat([win, cont_rows], ignore_index=True),
        "Alpha by window": alpha,
        "Growth": growth,
        "Monthly (continuous)": pd.concat([m for res in results for m in res["monthly"] if "variant" in m],
                                          ignore_index=True),
        "Monthly (windows)": pd.concat([m for res in results for m in res["monthly"] if "window" in m],
                                       ignore_index=True),
        "Actions": pd.concat([tag(res["cont"]["actions"], c.name) for c, res in zip(cfgs, results)], ignore_index=True),
        "Positions": pd.concat([tag(res["cont"]["trips"], c.name) for c, res in zip(cfgs, results)], ignore_index=True),
        "Rebalances": pd.concat([tag(res["cont"]["rebs"], c.name) for c, res in zip(cfgs, results)], ignore_index=True),
        "Regimes": pd.concat([regimes(c.name, res["cont"], below) for c, res in zip(cfgs, results)], ignore_index=True),
        "Survivorship": survivorship(win),
    }
    for k in ("Actions", "Positions", "Rebalances"):
        for col in ("signal_date", "fill_date"):
            if col in sheets[k]:
                sheets[k][col] = pd.to_datetime(sheets[k][col]).dt.date
    v = sheets["Verdict"].set_index("Metric")
    headline = []
    if compound:
        headline = [{"label": f"{c.name}: Rs {a.capital:,.0f} grew to", "fmt": "rs",
                     "value": v.loc["Final value, after tax (₹)", c.name]} for c in cfgs]
        headline.append({"label": "Nifty 500: grew to", "fmt": "rs",
                         "value": v.loc["Nifty 500 final value, after tax (₹)", cfgs[0].name]})
    charts = [{"sheet": "Growth", "x": "month", "y": [c.name for c in cfgs] + ["Nifty 500"], "log": True,
               "title": "Continuous run: value of the capital after tax vs Nifty 500 (log scale)"},
              {"sheet": "Alpha by window", "x": "window", "y": [c.name for c in cfgs], "kind": "bar", "pct": True,
               "title": "After-tax alpha vs Nifty 500 in each window (CAGR points)"}]
    if compound:
        charts.insert(0, {"sheet": "Compounding", "x": "point", "y": [c.name for c in cfgs] + ["Nifty 500"],
                          "title": "The one pot, rolled window to window, after tax"})
    key = f"rolling_{a.years}y_{a.capital:.0f}_{a.mode}"
    path = results.save(key, f"Rolling backtest · Rs {a.capital:,.0f} · {a.mode}", sheets, kind="rolling",
                        params={"capital": a.capital, "mode": a.mode, "from": a.first, "to": a.last,
                                "years": a.years, "step": a.step},
                        charts=charts, headline=headline)

    print(f"\n{'strategy':22s} {'beat Nifty AT':>14s} {'med alpha AT':>13s} {'worst alpha':>12s} "
          f"{'invested':>9s} {'checks':>7s}")
    for c in cfgs:
        w = win[win["strategy"] == c.name]
        print(f"{c.name:22s} {(w['beat_nifty_after_tax'] == 'YES').mean():14.0%} "
              f"{w['alpha_after_tax_CAGR %'].median():13.1%} {w['alpha_after_tax_CAGR %'].min():12.1%} "
              f"{w['target_capital_actually_invested %'].median():9.0%} {v.loc['Checks passed', c.name]:>7s}")
    if compound:
        print(f"\nOne pot of Rs {a.capital:,.0f} rolled through {len(windows)} windows ({windows[0][0]} ... {windows[-1][0]}):")
        for c in cfgs:
            print(f"  {c.name:22s} -> Rs {v.loc['Final value, after tax (₹)', c.name]:>12,.0f} after tax  "
                  f"({v.loc['Growth multiple, after tax (x)', c.name]:.2f}x, "
                  f"CAGR {v.loc['Compounded CAGR, after tax %', c.name]:.1%})")
        print(f"  {'Nifty 500':22s} -> Rs {v.loc['Nifty 500 final value, after tax (₹)', cfgs[0].name]:>12,.0f} after tax  "
              f"(CAGR {v.loc['Nifty 500 compounded CAGR, after tax %', cfgs[0].name]:.1%})")
    print(f"\nSaved {path.name}: open the app (python -m momentum_lab app) -> Results to see every table.")


if __name__ == "__main__":
    main()
