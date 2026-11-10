"""Month-end journal: the live-trading loop.

Each run (on a signal date, normally the last trading day of the month):
  1. reviews the book recommended at the previous run: entry at the open after that run,
     marked at today's close -> P&L per position, costs, cash interest, vs benchmark
  2. builds the new target book with the exact backtest logic (backtest.target_book),
     using the previous book as 'held' so the rank buffer works
  3. turns it into orders (shares for longs, futures lots for shorts) vs the previous book
  4. saves everything under journal/<strategy>/ (CSVs + runs.json), shown in the app

It assumes you executed the previous orders as recommended at the next day's open.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import signals as sig
from .backtest import (eligibility, is_scheduled, make_model, risk_trigger, run_backtest, short_eligibility,
                       target_book, trade_costs)
from .config import StrategyConfig
from .data import Panel, load_fno_list
from .metrics import cagr, drawdown
from .settings import globals_ as settings

from .paths import JOURNAL, ROOT  # noqa: F401  (JOURNAL is reassigned by analyses.track_record)


class Context:
    """Signals/filters computed once per session and reused across runs (backfill)."""

    def __init__(self, p: Panel, cfg: StrategyConfig):
        self.p, self.cfg = p, cfg
        self.scores = sig.composite(p, cfg.signal_weights)
        self.elig = eligibility(p, cfg)
        self.short_elig = short_eligibility(p, cfg) if cfg.short_n else None
        self.vol = sig.daily_vol(p.close, 63)
        self.sigma_d = self.vol / np.sqrt(252)
        self.adv = p.traded_value.rolling(cfg.adv_window, min_periods=1).median()
        self.bench_ma = p.benchmark.rolling(200).mean()
        self.mom = sig.mom_12_1(p)
        self.pm = make_model(p, self.scores, self.elig, cfg)  # risk model / forecasts (model weightings only)
        # model track record (also supplies the NAV history for vol targeting / drawdown control)
        self.model = run_backtest(p, cfg, scores=self.scores, model=self.pm)
        self.fno = load_fno_list()


def strategy_dir(name: str) -> Path:
    d = JOURNAL / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_runs(name: str) -> list[dict]:
    f = strategy_dir(name) / "runs.json"
    return json.loads(f.read_text()) if f.exists() else []


def save_runs(name: str, runs: list[dict]) -> None:
    (strategy_dir(name) / "runs.json").write_text(json.dumps(sorted(runs, key=lambda r: r["date"]), indent=2))


def delete_run(name: str, date: str) -> None:
    d = strategy_dir(name)
    save_runs(name, [r for r in load_runs(name) if r["date"] != date])
    for f in d.glob(f"*_{date}.*"):
        f.unlink()


def _review(ctx: Context, prev: dict, prev_book: pd.DataFrame, asof: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    p, cfg = ctx.p, ctx.cfg
    dates = p.close.index
    prev_d = pd.Timestamp(prev["date"])
    entry_day = dates[dates > prev_d][0]
    close = p.close.ffill()
    b = prev_book[prev_book["qty"] != 0].copy()
    b["entry_date"] = entry_day.date()
    b["entry_price"] = p.open.loc[entry_day, b["symbol"]].fillna(close.loc[entry_day, b["symbol"]]).values
    b["mark_price"] = close.loc[asof, b["symbol"]].values
    b["return"] = b["mark_price"] / b["entry_price"] - 1
    b["pnl_rs"] = b["qty"] * (b["mark_price"] - b["entry_price"])
    b["contribution"] = b["pnl_rs"] / prev["capital"]
    b = b[["symbol", "name", "sector", "side", "qty", "entry_date", "entry_price", "mark_price",
           "return", "pnl_rs", "contribution"]].sort_values("pnl_rs", ascending=False)

    gross_pnl = b["pnl_rs"].sum()
    costs = prev.get("est_costs", 0.0)
    long_cost = (prev_book.loc[prev_book["qty"] > 0, "qty"] * prev_book.loc[prev_book["qty"] > 0, "price"]).sum()
    days = (asof - prev_d).days
    interest = max(prev["capital"] - long_cost, 0) * ((1 + cfg.cash_rate) ** (days / 365) - 1)
    net = gross_pnl - costs + interest
    bench_ret = p.benchmark[asof] / p.benchmark[prev_d] - 1
    stats = {
        "from": prev["date"], "to": str(asof.date()), "capital_start": prev["capital"],
        "pnl_long": float(b.loc[b["side"] == "LONG", "pnl_rs"].sum()),
        "pnl_short": float(b.loc[b["side"] == "SHORT", "pnl_rs"].sum()),
        "costs": float(costs), "cash_interest": float(interest), "net_pnl": float(net),
        "return": float(net / prev["capital"]), "benchmark_return": float(bench_ret),
        "hit_rate": float((b["pnl_rs"] > 0).mean()) if len(b) else None,
    }
    return b, stats


def _size(ctx: Context, asof: pd.Timestamp, w: pd.Series, sw: pd.Series, capital: float,
          prev_book: pd.DataFrame, paper_shorts: bool) -> pd.DataFrame:
    """Target weights -> whole shares (longs) / futures lots (shorts), with the no-trade band."""
    p = ctx.p
    rank = ctx.scores.loc[asof].where(ctx.elig.loc[asof]).rank(ascending=False)
    rows = []
    for sym, wt in list(w.items()) + [(s, -x) for s, x in sw.items()]:
        px = float(p.close.loc[asof, sym])
        side = "LONG" if wt > 0 else "SHORT"
        lot = int(ctx.fno["lot_size"].get(sym, 1)) if side == "SHORT" and not paper_shorts else 1
        units = wt * capital / (px * lot)
        units = math.floor(units) if side == "LONG" else -math.floor(-units + 0.5)  # shorts: nearest lot
        rows.append({"symbol": sym, "name": p.meta.loc[sym, "name"], "sector": p.meta.loc[sym, "sector"],
                     "side": side, "weight": round(wt, 4), "target_rs": round(wt * capital), "price": px,
                     "lot_size": lot, "lots": units if side == "SHORT" else None, "qty": units * lot,
                     "score": round(float(ctx.scores.loc[asof, sym]), 3),
                     "rank": int(rank[sym]) if side == "LONG" and not pd.isna(rank.get(sym)) else None,
                     "ret_12_1": round(float(ctx.mom.loc[asof, sym]), 3)})
    book = pd.DataFrame(rows, columns=["symbol", "name", "sector", "side", "weight", "target_rs", "price",
                                       "lot_size", "lots", "qty", "score", "rank", "ret_12_1"])
    # No-trade band (settings workbook, 'Live & Data'): leave a continuing position alone unless it is
    # more than the band away from target -- tiny resizes cost more in fees than they fix in tracking error.
    band = settings()["live"]["no_trade_band"]
    old_q = prev_book.set_index("symbol")["qty"] if len(prev_book) else pd.Series(dtype=float)
    for ix, r in book.iterrows():
        oq = old_q.get(r["symbol"], 0)
        if oq and np.sign(oq) == np.sign(r["qty"]) and abs(r["qty"] - oq) <= band * abs(r["qty"]):
            book.at[ix, "qty"] = oq
            if r["side"] == "SHORT":
                book.at[ix, "lots"] = oq / r["lot_size"]
    book["actual_rs"] = (book["qty"] * book["price"]).round(0)
    book["note"] = np.where((book["side"] == "SHORT") & (book["qty"] == 0), "below 1 futures lot: skipped", "")
    book = book.sort_values(["side", "weight"], ascending=[True, False])
    return book


def _orders(ctx: Context, asof: pd.Timestamp, book: pd.DataFrame, prev_book: pd.DataFrame) -> pd.DataFrame:
    """Orders that turn prev_book into book, with estimated costs."""
    p, cfg = ctx.p, ctx.cfg
    old = prev_book.set_index("symbol")["qty"] if len(prev_book) else pd.Series(dtype=float)
    new = book.set_index("symbol")["qty"]
    allsyms = old.index.union(new.index)
    delta = new.reindex(allsyms).fillna(0) - old.reindex(allsyms).fillna(0)
    delta = delta[delta != 0]
    o = pd.DataFrame({"symbol": delta.index, "qty_change": delta.values})
    o["old_qty"] = old.reindex(o["symbol"]).fillna(0).values
    o["new_qty"] = new.reindex(o["symbol"]).fillna(0).values
    o["action"] = [
        ("BUY" if dq > 0 else "SELL") if (oq >= 0 and nq >= 0) else ("SHORT" if dq < 0 else "COVER")
        for dq, oq, nq in zip(o["qty_change"], o["old_qty"], o["new_qty"])]
    o["type"] = np.select([o["old_qty"] == 0, o["new_qty"] == 0], ["new", "exit"], "resize")
    o["name"] = p.meta.reindex(o["symbol"])["name"].values
    o["ref_price"] = p.close.ffill().loc[asof, o["symbol"]].values
    o["value_rs"] = (o["qty_change"] * o["ref_price"]).round(0)
    o["lot_size"] = ctx.fno["lot_size"].reindex(o["symbol"]).values
    o.loc[(o["old_qty"] >= 0) & (o["new_qty"] >= 0), "lot_size"] = np.nan
    o["instrument"] = np.where(o["lot_size"].notna(), "stock futures (near month)", "cash equity (delivery)")
    i = p.close.index.get_loc(asof)
    o["est_cost_rs"] = trade_costs(o["value_rs"].to_numpy(dtype=float), o["qty_change"].to_numpy() > 0,
                                   ctx.sigma_d.iloc[i].reindex(o["symbol"]).to_numpy(),
                                   ctx.adv.iloc[i].reindex(o["symbol"]).to_numpy(), cfg.costs).round(0)
    order_rank = {"SELL": 0, "COVER": 1, "BUY": 2, "SHORT": 3}
    o = o.sort_values(["action", "value_rs"], key=lambda s: s.map(order_rank) if s.name == "action" else -s.abs())
    o = o[["action", "type", "symbol", "name", "instrument", "qty_change", "lot_size", "old_qty", "new_qty",
           "ref_price", "value_rs", "est_cost_rs"]]
    return o


def month_end(ctx: Context, asof: str, capital: float | None = None, paper_shorts: bool = False) -> dict:
    p, cfg = ctx.p, ctx.cfg
    dates = p.close.index
    req = pd.Timestamp(asof)
    if req > dates[-1]:
        raise ValueError(f"No data for {req.date()} yet (latest is {dates[-1].date()}). Refresh data first.")
    asof = dates[dates <= req][-1]
    ds = str(asof.date())
    name = cfg.name
    runs = [r for r in load_runs(name) if r["date"] != ds]
    later = [r["date"] for r in runs if r["date"] > ds]
    if later:
        raise ValueError(f"Journal already has later runs {later}. Delete them first to re-run {ds}.")
    prev = max(runs, key=lambda r: r["date"]) if runs else None
    d = strategy_dir(name)

    # ---- 1. review last month's book --------------------------------------------------------
    review, rstats = None, None
    prev_book = pd.read_csv(d / f"book_{prev['date']}.csv") if prev else pd.DataFrame(
        columns=["symbol", "side", "qty", "price", "lot_size"])
    if prev:
        review, rstats = _review(ctx, prev, prev_book, asof)
        review.to_csv(d / f"review_{ds}.csv", index=False)
    if capital is None:
        capital = prev["capital"] + rstats["net_pnl"] if prev else cfg.initial_capital

    # ---- 2. new target book (same decision function as the backtest) ------------------------
    held_long = set(prev_book.loc[prev_book["qty"] > 0, "symbol"])
    held_short = set(prev_book.loc[prev_book["qty"] < 0, "symbol"])
    nav = ctx.model.nav.loc[:asof]
    below = bool(p.benchmark[asof] < ctx.bench_ma[asof])
    prev_gross = prev.get("gross_exposure") if prev else None
    if prev and not is_scheduled(asof, cfg) and not risk_trigger(cfg, below, nav, prev_gross):
        # off-cycle risk check, exposure fine: carry the previous book forward unchanged
        px = p.close.ffill().loc[asof]
        val = prev_book.set_index("symbol")["qty"] * px.reindex(prev_book["symbol"]).values
        w = (val[val > 0] * (1 + 1e-6) / capital).astype(float)
        sw = (-val[val < 0] / capital).astype(float)
        gross, rule = prev_gross if prev_gross is not None else float(w.sum()), "risk check: no change needed"
    else:
        w, sw, gross, rule = target_book(asof, ctx.scores, ctx.elig, ctx.short_elig, ctx.vol, held_long,
                                         held_short, p.meta["sector"], cfg, below, nav, prev_gross, ctx.pm)
        if prev and not is_scheduled(asof, cfg):
            rule = f"risk check: early rebalance ({rule})"
    book = _size(ctx, asof, w, sw, capital, prev_book, paper_shorts)
    book.to_csv(d / f"book_{ds}.csv", index=False)

    # ---- 3. orders vs previous book ---------------------------------------------------------
    o = _orders(ctx, asof, book, prev_book)
    o.to_csv(d / f"orders_{ds}.csv", index=False)

    # ---- 4. record --------------------------------------------------------------------------
    run = {"date": ds, "capital": float(capital), "regime": "below 200DMA" if below else "above 200DMA",
           "long_exposure": float(w.sum()), "short_exposure": float(sw.sum()),
           "gross_exposure": float(gross), "exposure_rule": rule,
           "invested_rs": float(book.loc[book["qty"] > 0, "actual_rs"].sum()),
           "short_rs": float(-book.loc[book["qty"] < 0, "actual_rs"].sum()),
           "n_long": int((book["qty"] > 0).sum()), "n_short": int((book["qty"] < 0).sum()),
           "n_orders": int(len(o)), "est_costs": float(o["est_cost_rs"].sum()),
           "paper_shorts": paper_shorts, "review": rstats, "data_date": str(dates[-1].date())}
    runs.append(run)
    save_runs(name, runs)
    return run


def _month_end_day(d: pd.Timestamp) -> pd.Timestamp:
    """Last weekday of d's month (exchange holidays aside): the normal signal date."""
    last = d + pd.offsets.MonthEnd(0)
    return last - pd.offsets.BDay(0) if last.weekday() < 5 else last - pd.offsets.BDay(1)


def _rebalances_at(d: pd.Timestamp, cfg: StrategyConfig) -> bool:
    return cfg.rebalance in ("W", "M") or d.month % 3 == 0


def preview(ctx: Context, capital: float | None = None, fresh: bool = False, paper_shorts: bool = False) -> dict:
    """What to hold next month: the target book at the latest close, sized for `capital`.

    Read-only: the journal is not touched. Unless `fresh`, it starts from the latest journal book before
    the latest close (rank buffer and no-trade band apply, and the orders are the changes vs that book);
    `fresh` = you hold nothing yet, so the orders build the whole book. Rebalances are always forced here --
    `schedule` says whether the strategy would actually rebalance at this month's month-end run.
    """
    p, cfg = ctx.p, ctx.cfg
    asof = p.close.index[-1]
    runs = [r for r in load_runs(cfg.name) if r["date"] < str(asof.date())]
    prev = None if fresh or not runs else runs[-1]
    prev_book = pd.read_csv(strategy_dir(cfg.name) / f"book_{prev['date']}.csv") if prev else pd.DataFrame(
        columns=["symbol", "side", "qty", "price", "lot_size"])
    if capital is None:
        capital = prev["capital"] if prev else cfg.initial_capital
    nav = ctx.model.nav.loc[:asof]
    below = bool(p.benchmark[asof] < ctx.bench_ma[asof])
    prev_gross = prev.get("gross_exposure") if prev else None
    w, sw, gross, rule = target_book(asof, ctx.scores, ctx.elig, ctx.short_elig, ctx.vol,
                                     set(prev_book.loc[prev_book["qty"] > 0, "symbol"]),
                                     set(prev_book.loc[prev_book["qty"] < 0, "symbol"]),
                                     p.meta["sector"], cfg, below, nav, prev_gross, ctx.pm)
    book = _size(ctx, asof, w, sw, capital, prev_book, paper_shorts)
    o = _orders(ctx, asof, book, prev_book)
    me = _month_end_day(asof)
    nxt = me
    while not _rebalances_at(nxt, cfg):
        nxt = _month_end_day(nxt + pd.offsets.MonthBegin(1))
    invested = float(book.loc[book["qty"] > 0, "actual_rs"].sum())
    return {"strategy": cfg.name, "asof": str(asof.date()), "capital": float(capital),
            "regime": "below 200DMA" if below else "above 200DMA", "exposure_rule": rule,
            "long_exposure": float(w.sum()), "short_exposure": float(sw.sum()), "gross_exposure": float(gross),
            "invested_rs": invested, "cash_rs": float(capital) - invested,
            "n_long": int((book["qty"] > 0).sum()), "n_short": int((book["qty"] < 0).sum()),
            "n_orders": int(len(o)), "est_costs": float(o["est_cost_rs"].sum()),
            "held_from": prev["date"] if prev else None,
            "schedule": {"rebalance": cfg.rebalance, "month_end": str(me.date()),
                         "rebalances_at_month_end": _rebalances_at(me, cfg), "next_rebalance": str(nxt.date()),
                         "risk_check": cfg.risk_check or None},
            "book": book, "orders": o}


def history(name: str) -> pd.DataFrame:
    """One row per reviewed period, chained into cumulative strategy vs benchmark."""
    rows = [r["review"] for r in load_runs(name) if r.get("review")]
    if not rows:
        return pd.DataFrame()
    h = pd.DataFrame(rows)
    h["cum_strategy"] = (1 + h["return"]).cumprod() - 1
    h["cum_benchmark"] = (1 + h["benchmark_return"]).cumprod() - 1
    h["excess"] = h["return"] - h["benchmark_return"]
    return h


def model_stats(ctx: Context) -> dict:
    nav, b = ctx.model.nav, ctx.model.benchmark
    r = nav.pct_change().dropna()
    rf = (1 + ctx.cfg.rf_rate) ** (1 / 252) - 1
    ytd = nav[nav.index.year == nav.index[-1].year]
    return {"start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
            "cagr": cagr(nav), "bench_cagr": cagr(b), "sharpe": (r.mean() - rf) / r.std() * np.sqrt(252),
            "max_dd": drawdown(nav).min(), "bench_max_dd": drawdown(b).min(),
            "vol": r.std() * np.sqrt(252), "ytd": ytd.iloc[-1] / nav[nav.index < ytd.index[0]].iloc[-1] - 1,
            "bench_ytd": b.iloc[-1] / b[b.index < ytd.index[0]].iloc[-1] - 1}
