"""Event-driven daily backtester for long-only, rebalanced cross-sectional strategies.

Timeline for each rebalance (no look-ahead):
  signal date t  : close of last trading day of the period; scores, filters, regime and
                   vol-target are computed from data up to t.
  execution t+1  : trade at t+1 open (or close) with full Indian cost model.
Between rebalances positions drift with prices (share-based book, not constant weights).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import signals as sig
from .config import Costs, StrategyConfig
from .data import Panel
from .risk import MODEL_WEIGHTINGS, PortfolioModel, exposure


@dataclass
class BacktestResult:
    config: StrategyConfig
    nav: pd.Series             # strategy equity curve
    benchmark: pd.Series       # benchmark rebased to the same start capital
    weights: pd.DataFrame      # target weights at each execution date (date x symbol)
    trades: pd.DataFrame       # one row per executed order
    exposure: pd.Series        # gross exposure chosen at each execution date
    costs: pd.Series           # Rs costs paid per day
    holdings_value: pd.DataFrame  # end-of-day position values at execution dates
    legs: pd.DataFrame         # daily Rs P&L split: long, short, cash interest, costs

    @property
    def returns(self) -> pd.Series:
        return self.nav.pct_change().fillna(0)


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading day of each period."""
    s = pd.Series(index, index=index)
    key = {"W": index.to_period("W"), "M": index.to_period("M"), "Q": index.to_period("Q")}[freq]
    return pd.DatetimeIndex(s.groupby(key).last().values)


def is_scheduled(d: pd.Timestamp, cfg: StrategyConfig) -> bool:
    """True if signal date d is a regular rebalance (not just an off-cycle risk check): the risk-check
    period containing d ends at or after the end of d's rebalance period."""
    if not cfg.risk_check:
        return True
    return d.to_period(cfg.risk_check).end_time >= d.to_period(cfg.rebalance).end_time


def decision_dates(index: pd.DatetimeIndex, cfg: StrategyConfig) -> pd.DatetimeIndex:
    """Rebalance dates plus, with risk_check set, the off-cycle risk-check dates."""
    d = rebalance_dates(index, cfg.rebalance)
    return d.union(rebalance_dates(index, cfg.risk_check)) if cfg.risk_check else d


def risk_trigger(cfg: StrategyConfig, below_ma: bool, nav_hist: pd.Series | None, prev_gross: float | None) -> bool:
    """Off-cycle risk check: rebalance early only if exposure has to move by at least the exposure band
    (minimum 5 points), so ordinary wobbles in the vol estimate don't cause monthly trading."""
    if prev_gross is None:
        return False
    g, _ = exposure(cfg, below_ma, nav_hist, None, None)
    return abs(g - prev_gross) >= max(cfg.exposure_band, 0.05)


def base_eligibility(p: Panel, cfg: StrategyConfig) -> pd.DataFrame:
    """Boolean (date x symbol): history, price, liquidity and vol filters (both legs)."""
    c = p.close
    ok = c.notna() & (c >= cfg.min_price)
    ok &= c.notna().rolling(cfg.min_history_days, min_periods=1).sum() >= cfg.min_history_days
    adv = p.traded_value.rolling(cfg.adv_window, min_periods=cfg.adv_window // 2).median()
    ok &= adv >= cfg.min_adv_cr * 1e7
    if cfg.max_vol is not None:
        ok &= sig.daily_vol(c) <= cfg.max_vol
    return ok


def eligibility(p: Panel, cfg: StrategyConfig) -> pd.DataFrame:
    """Boolean (date x symbol): passes all point-in-time filters for the long book."""
    c = p.close
    ok = base_eligibility(p, cfg)
    if cfg.abs_momentum:
        ok &= sig.mom_12_1(p) > 0
    if cfg.above_ma:
        ok &= c > c.rolling(cfg.above_ma).mean()
    return ok


def short_eligibility(p: Panel, cfg: StrategyConfig) -> pd.DataFrame:
    """Short candidates: pass base filters, negative 12-1 momentum, and (by default) have
    stock futures. Note the F&O list is today's, not point-in-time."""
    ok = base_eligibility(p, cfg) & (sig.mom_12_1(p) < 0)
    if cfg.short_fno_only:
        from .data import load_fno_list
        ok &= pd.Series(p.close.columns.isin(load_fno_list().index), index=p.close.columns)
    return ok


def select(scores: pd.Series, held: set, sectors: pd.Series, cfg: StrategyConfig,
           n: int | None = None, buffer: int | None = None) -> list:
    """Rank-based selection (highest score first) with a hold buffer and per-sector name cap."""
    n = n or cfg.top_n
    ranks = scores.rank(ascending=False, method="first")
    buffer = buffer or cfg.buffer_rank or n
    keepers = sorted([s for s in held if s in ranks.index and ranks[s] <= buffer], key=ranks.get)
    others = [s for s in ranks.sort_values().index if s not in keepers]
    chosen, per_sector = [], {}
    for s in keepers + others:
        if len(chosen) >= n:
            break
        sec = sectors.get(s)
        if cfg.max_per_sector and per_sector.get(sec, 0) >= cfg.max_per_sector:
            continue
        chosen.append(s)
        per_sector[sec] = per_sector.get(sec, 0) + 1
    return chosen


def cap_weights(w: pd.Series, cap: float) -> pd.Series:
    """Iteratively clip at `cap` and redistribute excess pro-rata to uncapped names."""
    w = w / w.sum()
    if cap * len(w) < 1:  # infeasible: fall back to equal weight
        return pd.Series(1 / len(w), index=w.index)
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        w[~over] += excess * w[~over] / w[~over].sum()
    return w


def build_weights(names: list, scores: pd.Series, vol: pd.Series, cfg: StrategyConfig,
                  model: PortfolioModel | None = None, sd: pd.Timestamp | None = None) -> pd.Series:
    """Weights summing to 1. Risk-model weightings need `model` and the signal date `sd`; without a
    model (e.g. the short book) they fall back to inverse_vol."""
    if not names:
        return pd.Series(dtype=float)
    if cfg.weighting in MODEL_WEIGHTINGS:
        if model is None:
            return build_weights(names, scores, vol, cfg.replace(weighting="inverse_vol"))
        w, _ = model.weights(sd, names, vol)
        return cap_weights(w.fillna(0), max(cfg.max_weight, 1 / len(names)))
    if cfg.weighting == "equal":
        w = pd.Series(1.0, index=names)
    elif cfg.weighting == "inverse_vol":
        w = 1 / vol[names].clip(lower=0.05)
    elif cfg.weighting == "score":
        s = scores[names]
        w = s - s.min() + s.std()  # shift positive, keep dispersion
    else:
        raise ValueError(cfg.weighting)
    return cap_weights(w.fillna(w.median()), cfg.max_weight)


def trade_costs(notional: np.ndarray, is_buy: np.ndarray, sigma: np.ndarray,
                adv: np.ndarray, c: Costs) -> np.ndarray:
    """Per-order Rs cost: statutory charges + brokerage + slippage + sqrt market impact."""
    n = np.abs(notional)
    brokerage = c.brokerage_per_order + n * c.brokerage_bps / 1e4
    exch = n * (c.exchange_bps + c.sebi_bps) / 1e4
    gst = c.gst_rate * (brokerage + exch)
    stt = n * c.stt_bps / 1e4
    stamp = np.where(is_buy, n * c.stamp_duty_buy_bps / 1e4, 0)
    dp = np.where(is_buy, 0, c.dp_charge_per_sell)
    participation = np.where(adv > 0, n / adv, 0)
    impact = n * (c.slippage_bps / 1e4 + c.impact_coef * np.nan_to_num(sigma) * np.sqrt(participation))
    return brokerage + exch + gst + stt + stamp + dp + impact


def target_book(sd: pd.Timestamp, scores: pd.DataFrame, elig: pd.DataFrame,
                short_elig: pd.DataFrame | None, vol: pd.DataFrame, held_long: set, held_short: set,
                sectors: pd.Series, cfg: StrategyConfig, below_ma: bool,
                nav_hist: pd.Series | None = None, prev_gross: float | None = None,
                model: PortfolioModel | None = None) -> tuple[pd.Series, pd.Series, float, str]:
    """The portfolio decision at signal date `sd`. Returns (long weights incl. exposure scaling,
    short weights as positive fractions of NAV, long gross exposure, the exposure rule that bound).
    Shared by the backtester, the rolling backtest and the month-end journal so all make identical
    decisions. `nav_hist` = the strategy's own NAV up to sd (vol target, drawdown control)."""
    sc = scores.loc[sd].where(elig.loc[sd]).dropna()
    names = select(sc, held_long, sectors, cfg)
    if model is not None:
        model.last = {}
    w = build_weights(names, sc, vol.loc[sd], cfg, model, sd)
    sw = pd.Series(dtype=float)
    if cfg.short_n:
        # weakest scores first; never short something we're long
        ss = -scores.loc[sd].where(short_elig.loc[sd]).dropna().drop(names, errors="ignore")
        snames = select(ss, held_short, sectors, cfg, n=cfg.short_n, buffer=cfg.short_buffer_rank or cfg.short_n)
        sw = build_weights(snames, ss, vol.loc[sd], cfg) * cfg.short_exposure
    kelly = model.last.get("kelly_sum") if model is not None else None
    gross, rule = exposure(cfg, below_ma, nav_hist, prev_gross, kelly)
    if cfg.derisk == "drop" and gross < 1 and len(w):  # de-risk by holding fewer names, not trimming all
        keep = sc[w.index].sort_values(ascending=False).index[:max(1, round(len(w) * gross))]
        return w[keep] * gross / w[keep].sum(), sw, gross, rule
    return w * gross, sw, gross, rule


def make_model(p: Panel, scores: pd.DataFrame, elig: pd.DataFrame, cfg: StrategyConfig) -> PortfolioModel | None:
    return PortfolioModel(p, scores, elig, cfg) if cfg.weighting in MODEL_WEIGHTINGS else None


def run_backtest(p: Panel, cfg: StrategyConfig, scores: pd.DataFrame | None = None,
                 model: PortfolioModel | None = None) -> BacktestResult:
    if scores is None:
        scores = sig.composite(p, cfg.signal_weights)
    elig = eligibility(p, cfg)
    model = model or make_model(p, scores, elig, cfg)
    short_elig = short_eligibility(p, cfg) if cfg.short_n else None
    vol = sig.daily_vol(p.close, 63)
    sigma_d = vol / np.sqrt(252)
    adv = p.traded_value.rolling(cfg.adv_window, min_periods=1).median()
    bench_ma = p.benchmark.rolling(200).mean()

    dates = p.close.index
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end) if cfg.end else dates[-1]
    sim = dates[(dates >= start) & (dates <= end)]
    sig_dates = decision_dates(sim, cfg)[:-1] if len(sim) else []
    # map execution day -> signal day
    pos = {d: i for i, d in enumerate(dates)}
    exec_map = {dates[pos[d] + 1]: d for d in sig_dates if pos[d] + 1 < len(dates)}

    syms = p.close.columns
    sectors = p.meta["sector"]
    close_ff = p.close.ffill().to_numpy()
    px_exec = (p.open if cfg.execution == "next_open" else p.close).to_numpy()
    n = len(syms)
    shares = np.zeros(n)
    cash = cfg.initial_capital
    nav, cost_log, exposure_log, w_log, trades, hv_log = {}, {}, {}, {}, [], {}
    daily_cash = (1 + cfg.cash_rate) ** (1 / 252) - 1
    legs = {}

    for d in sim:
        i = pos[d]
        interest = cash * daily_cash
        cash += interest
        s0 = shares.copy()
        ref_px = close_ff[i - 1]       # price the day's P&L is measured from
        day_cost_long = day_cost_short = 0.0
        act = d in exec_map
        if act and not is_scheduled(exec_map[d], cfg):  # off-cycle risk check
            sd = exec_map[d]
            act = risk_trigger(cfg, p.benchmark[sd] < bench_ma[sd], pd.Series(nav),
                               exposure_log[max(exposure_log)] if exposure_log else None)
        if act:
            sd = exec_map[d]
            # ---- target portfolio from info at signal date --------------------------
            prev = exposure_log[max(exposure_log)] if exposure_log else None
            w, sw, gross, _ = target_book(sd, scores, elig, short_elig, vol, set(syms[shares > 0]),
                                          set(syms[shares < 0]), sectors, cfg,
                                          p.benchmark[sd] < bench_ma[sd], pd.Series(nav) if nav else None,
                                          prev, model)
            exposure_log[d] = gross

            # ---- trade ---------------------------------------------------------------
            px = px_exec[i]
            tradable = ~np.isnan(px) & (px > 0)
            val_px = np.where(tradable, px, close_ff[i - 1])
            equity = cash + np.nansum(shares * val_px)
            tgt_w = np.zeros(n)
            idx = syms.get_indexer(w.index)
            tgt_w[idx] = w.values
            tgt_w[syms.get_indexer(sw.index)] = -sw.values
            target = np.where(tradable, tgt_w * equity * 0.997 / np.where(tradable, px, 1), shares)
            delta = target - shares
            delta[np.abs(delta * val_px) < 1] = 0  # ignore dust
            mask = delta != 0
            if mask.any():
                notional = delta[mask] * px[mask]
                c = trade_costs(notional, delta[mask] > 0, sigma_d.iloc[i - 1].to_numpy()[mask],
                                adv.iloc[i - 1].to_numpy()[mask], cfg.costs)
                cash -= notional.sum() + c.sum()
                is_short = np.minimum(shares, target)[mask] < 0
                day_cost_short, day_cost_long = c[is_short].sum(), c[~is_short].sum()
                shares = shares + delta
                cost_log[d] = c.sum()
                trades.append(pd.DataFrame({"date": d, "symbol": syms[mask], "shares": delta[mask],
                                            "price": px[mask], "notional": notional, "cost": c}))
            w_log[d] = pd.Series(tgt_w, index=syms)[tgt_w != 0]
            ref_px = val_px
        # P&L: old book from prior close to the execution price, new book from there to close
        pnl_old = np.nan_to_num(s0 * (ref_px - close_ff[i - 1])) if act else np.zeros(n)
        pnl_new = np.nan_to_num(shares * (close_ff[i] - ref_px))
        legs[d] = {"long": pnl_old[s0 > 0].sum() + pnl_new[shares > 0].sum() - day_cost_long,
                   "short": pnl_old[s0 < 0].sum() + pnl_new[shares < 0].sum() - day_cost_short,
                   "cash": interest}
        mv = shares * close_ff[i]
        nav[d] = cash + np.nansum(mv)
        if act:
            hv_log[d] = pd.Series(mv, index=syms)[shares != 0]

    nav = pd.Series(nav)
    bench = p.benchmark.reindex(nav.index)
    bench = bench / bench.iloc[0] * cfg.initial_capital
    return BacktestResult(
        config=cfg, nav=nav, benchmark=bench,
        weights=pd.DataFrame(w_log).T.fillna(0),
        trades=pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
        exposure=pd.Series(exposure_log), costs=pd.Series(cost_log).reindex(nav.index).fillna(0),
        holdings_value=pd.DataFrame(hv_log).T.fillna(0),
        legs=pd.DataFrame(legs).T,
    )
