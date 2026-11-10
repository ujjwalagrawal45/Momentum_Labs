"""Head-to-head of portfolio-construction and exposure settings against a drawdown budget.

    python -m momentum_lab risk                      # strategies marked 'Include in rolling backtest', D = 20%
    python -m momentum_lab risk --dd 0.25 --capital 1000000 configs/quant_composite.json

Stage 1 (fast screen): every strategy x weighting x exposure rule, one continuous model backtest
          (pre-tax, fractional shares) -> CAGR, max drawdown, exposure, turnover.
Stage 2 (realistic): a shortlist through the rolling backtest's simulator at your capital (whole shares,
          full costs, tax): one compounding pot through back-to-back windows AND fresh capital in every
          overlapping window. The question: did the pot ever fall more than D, and what did it grow to?

Exposure rules tried (D from --dd):
  as is        : the strategy's own settings
  static       : max allocation = D / the strategy's worst model drawdown (sized with hindsight: optimistic)
  dd m=2/3/5   : Grossman-Zhou drawdown control at D, monthly risk check, 10-point exposure band, 25% floor
  dd m=3 drop  : as dd m=3, but cut exposure by dropping the weakest names instead of trimming all
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_lab.paths import CONFIGS
from momentum_lab import StrategyConfig, load_panel, load_universe
from momentum_lab import signals as sig
from momentum_lab.backtest import eligibility, make_model, run_backtest
from momentum_lab.journal import Context
from momentum_lab.metrics import cagr, drawdown
from momentum_lab import results
from momentum_lab.settings import globals_
from momentum_lab.analyses.rolling_backtest import Sim, chain_windows, run_strategy, window_list

warnings.filterwarnings("ignore", category=FutureWarning)
G = globals_()
WEIGHTINGS = ("equal", "erc_tilt", "black_litterman", "kelly")


def exposure_rules(dd: float, worst: float) -> dict[str, dict]:
    ctl = {"dd_limit": dd, "dd_floor": 0.25, "risk_check": "M", "exposure_band": 0.10}
    return {"as is": {},
            "static": {"max_allocation": round(min(1.0, dd / abs(worst)), 2)},
            "dd m=2": {**ctl, "dd_multiplier": 2},
            "dd m=3": {**ctl, "dd_multiplier": 3},
            "dd m=5": {**ctl, "dd_multiplier": 5},
            "dd m=3 drop": {**ctl, "dd_multiplier": 3, "derisk": "drop"}}


def stage1(p, base: StrategyConfig, dd: float) -> pd.DataFrame:
    scores = sig.composite(p, base.signal_weights)
    elig = eligibility(p, base)
    fc = None
    rows = []
    weightings = [base.weighting] + [w for w in WEIGHTINGS if w != base.weighting]
    worst = {}
    for wt in weightings:
        cfg0 = base.replace(weighting=wt)
        m0 = make_model(p, scores, elig, cfg0)
        if m0 is not None and m0.fc is not None:
            fc = fc or m0.fc
            m0.fc = fc
        r0 = run_backtest(p, cfg0, scores=scores, model=m0)
        worst[wt] = drawdown(r0.nav).min()
        for rule, kw in exposure_rules(dd, worst[wt]).items():
            cfg = cfg0.replace(**kw)
            t0 = time.time()
            if kw:  # exposure settings don't change the weighting model, so reuse it
                r = run_backtest(p, cfg, scores=scores, model=m0)
            else:
                r = r0
            nav = r.nav
            rows.append({"strategy": base.name, "weighting": wt, "exposure rule": rule, "settings": kw,
                         "CAGR %": cagr(nav), "max drawdown %": drawdown(nav).min(),
                         "within budget": "YES" if drawdown(nav).min() >= -dd else "NO",
                         "CAGR / |max DD|": cagr(nav) / abs(drawdown(nav).min()),
                         "avg exposure %": r.exposure.mean(), "rebalances": len(r.exposure),
                         "orders": len(r.trades)})
            print(f"  {base.name:20s} {wt:16s} {rule:12s} CAGR {rows[-1]['CAGR %']:6.1%}  "
                  f"maxDD {rows[-1]['max drawdown %']:6.1%}  exp {rows[-1]['avg exposure %']:4.0%}  "
                  f"({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def shortlist(s1: pd.DataFrame, dd: float, per_strategy: int) -> pd.DataFrame:
    keep = []
    for name, g in s1.groupby("strategy", sort=False):
        keep.append(g[(g["exposure rule"] == "as is") & (g.index == g.index[0])])  # the strategy as it is today
        ok = g[(g["max drawdown %"] >= -dd * 1.1) & (g["exposure rule"] != "as is")]  # 10% slack: stage 1 is idealised
        pool = ok if len(ok) else g[g["exposure rule"] != "as is"].nlargest(per_strategy * 2, "max drawdown %")
        keep.append(pool.nlargest(per_strategy, "CAGR %"))
    return pd.concat(keep).drop_duplicates(subset=["strategy", "weighting", "exposure rule"])


def stage2(p, cand: pd.DataFrame, bases: dict, capital: float, dd: float) -> pd.DataFrame:
    R = G["rolling"]
    dates = p.close.index
    chained = chain_windows(dates, R["first"], R["last"], R["years"])
    fresh = window_list(dates, R["first"], R["last"], R["years"], R["step"])
    s_last = dates[dates <= pd.Period(R["last"], "M").end_time]
    cont = (f"{R['first']} - {R['last']} (continuous)", chained[0][1], dates.get_loc(s_last[-1]))
    rows = []
    for _, c in cand.iterrows():
        cfg = bases[c["strategy"]].replace(weighting=c["weighting"], **c["settings"],
                                           name=f"{c['strategy']} | {c['weighting']} | {c['exposure rule']}")
        cfg = cfg.replace(initial_capital=capital)
        sim = Sim(Context(p, cfg))
        oc = run_strategy(cfg, p, chained, cont, capital, True, sim)
        of = run_strategy(cfg, p, fresh, cont, capital, False, sim)
        pot = pd.concat(oc["win_navs"])
        pot = pot[~pot.index.duplicated(keep="last")]
        w_c = pd.DataFrame(oc["rows"][1:])
        w_f = pd.DataFrame(of["rows"][1:])
        f_dd = pd.Series([drawdown(n).min() for n in of["win_navs"]])
        yrs = w_c["years"].sum()
        final, nifty = w_c["after_tax_liquidation_value (₹)"].iloc[-1], w_c["nifty_after_tax_value (₹)"].iloc[-1]
        rows.append({"strategy": c["strategy"], "weighting": c["weighting"], "exposure rule": c["exposure rule"],
                     "settings": c["settings"],
                     "POT: final value after tax (₹)": final, "POT: CAGR after tax %": (final / capital) ** (1 / yrs) - 1,
                     "POT: worst fall from peak %": drawdown(pot).min(),
                     "POT: within budget": "YES" if drawdown(pot).min() >= -dd else "NO",
                     "Nifty 500 pot after tax (₹)": nifty,
                     "Nifty 500 CAGR after tax %": (nifty / capital) ** (1 / yrs) - 1,
                     "FRESH: windows tested": len(w_f),
                     "FRESH: windows within budget %": float((f_dd >= -dd).mean()),
                     "FRESH: worst fall in any window %": float(f_dd.min()),
                     "FRESH: windows beating Nifty after tax %": float((w_f["beat_nifty_after_tax"] == "YES").mean()),
                     "FRESH: median after-tax alpha (CAGR pp) %": float(w_f["alpha_after_tax_CAGR %"].median()),
                     "FRESH: median avg invested %": float(w_f["avg_invested %"].median()),
                     "FRESH: median costs per year %": float(w_f["costs_per_year_of_capital %"].median()),
                     "stage 1 CAGR (idealised) %": c["CAGR %"], "stage 1 max DD %": c["max drawdown %"]})
        print(f"  -> pot Rs {final:,.0f} ({rows[-1]['POT: CAGR after tax %']:.1%}/yr), worst fall "
              f"{rows[-1]['POT: worst fall from peak %']:.1%}, fresh windows within budget "
              f"{rows[-1]['FRESH: windows within budget %']:.0%}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--dd", type=float, default=0.20, help="largest fall from peak you accept (0.20 = 20%%)")
    ap.add_argument("--capital", type=float, default=G["rolling"]["capital"])
    ap.add_argument("--per-strategy", type=int, default=2, help="shortlisted variants per strategy for stage 2")
    a = ap.parse_args()
    paths = a.configs or [str(CONFIGS / f"{n}.json") for n in G["rolling_strategies"]]
    bases = {c.name: c for c in (StrategyConfig.from_json(x) for x in paths)}
    print("Loading prices...")
    p = load_panel(load_universe(next(iter(bases.values())).universe_csv), next(iter(bases.values())).benchmark)
    print(f"\nSTAGE 1: fast screen (drawdown budget {a.dd:.0%})")
    s1 = pd.concat([stage1(p, b.replace(initial_capital=1_000_000), a.dd) for b in bases.values()], ignore_index=True)
    cand = shortlist(s1, a.dd, a.per_strategy)
    print(f"\nSTAGE 2: {len(cand)} shortlisted settings through the rolling simulator at Rs {a.capital:,.0f}")
    s2 = stage2(p, cand, bases, a.capital, a.dd)
    s2 = s2.sort_values(["POT: within budget", "POT: CAGR after tax %"], ascending=[False, False])
    for df in (s1, s2):
        df["settings"] = df["settings"].map(lambda d: ", ".join(f"{k}={v}" for k, v in d.items()) or "(as configured)")
    best = s2[s2["POT: within budget"] == "YES"]
    headline = [{"label": "Setups that stayed within the budget", "fmt": "text",
                 "value": f"{len(best)} of {len(s2)}"}]
    if len(best):
        b0 = best.iloc[0]
        headline += [{"label": f"Best within budget: {b0['strategy']} / {b0['weighting']} / {b0['exposure rule']}",
                      "fmt": "rs", "value": b0["POT: final value after tax (₹)"]},
                     {"label": "Its worst fall from peak", "fmt": "pct", "value": b0["POT: worst fall from peak %"]}]
    headline.append({"label": "Nifty 500 pot after tax", "fmt": "rs", "value": s2["Nifty 500 pot after tax (₹)"].iloc[0]})
    path = results.save(f"risk_compare_dd{a.dd * 100:.0f}_{a.capital:.0f}",
                        f"Risk comparison · fall budget {a.dd:.0%} · Rs {a.capital:,.0f}",
                        {"Stage 2 (realistic)": s2, "Stage 1 (screen)": s1}, kind="risk",
                        params={"dd": a.dd, "capital": a.capital, "per_strategy": a.per_strategy},
                        notes="Stage 1 is a quick idealised screen (pre-tax, fractional shares). Stage 2 runs the "
                              "shortlist through the rolling simulator at your capital (whole shares, full costs, "
                              "tax). 'static' caps are sized with hindsight from the worst historical drawdown, "
                              "so treat them as optimistic.",
                        headline=headline)
    print("\n" + s2[["strategy", "weighting", "exposure rule", "POT: final value after tax (₹)", "POT: CAGR after tax %",
                     "POT: worst fall from peak %", "FRESH: windows within budget %",
                     "FRESH: median after-tax alpha (CAGR pp) %"]].to_string(index=False))
    print(f"\nSaved {path.name}: open the app (python -m momentum_lab app) -> Results.")


if __name__ == "__main__":
    main()
