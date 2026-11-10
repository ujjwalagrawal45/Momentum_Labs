"""Signal research: does each momentum signal actually rank stocks? (run before building portfolios)

    python -m momentum_lab signals                       # quintile scoreboard for all built-in signals
    python -m momentum_lab signals --sweep --split 2022-01-01   # parameter sweep with IS/OOS split
"""
import argparse

import pandas as pd

from momentum_lab import StrategyConfig, load_panel, load_universe, results
from momentum_lab.metrics import deflated_sharpe
from momentum_lab.research import param_sweep, signal_scoreboard
from momentum_lab.paths import CONFIGS
from momentum_lab.settings import globals_


def main():
    R = globals_()  # defaults from 'Momentum Lab Settings.xlsx' (Research + Sweep Signal Sets sheets)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / f"{R['research']['base_strategy']}.json"))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--split", default=R["research"]["split"], help="in-sample / out-of-sample boundary")
    a = ap.parse_args()

    base = StrategyConfig.from_json(a.config)
    panel = load_panel(load_universe(base.universe_csv), base.benchmark)
    pd.set_option("display.width", 200)

    if not a.sweep:
        print("Monthly quintile portfolios (equal-weight, next-open entry, 1-month hold), annualised:")
        board = signal_scoreboard(panel, base)
        results.save("signal_scoreboard", "Signal research · quintile scoreboard",
                     {"Scoreboard": board.reset_index(names="signal")}, kind="research",
                     params={"base": base.name},
                     notes="Monthly quintile portfolios (equal weight, next-open entry, 1-month hold), annualised. "
                           "A useful signal rises steadily from the bottom quintile to the top one.")
        print(board.round(3).to_string())
    else:
        # The grid comes from the settings workbook (Research + Sweep Signal Sets sheets). Every combination
        # is one 'trial' -- the more you try, the higher the bar (see deflated Sharpe below).
        grid = {"signal_weights": R["sweep_sets"]} if R["sweep_sets"] else {}
        for key in ("top_n", "rebalance", "buffer_rank", "max_per_sector", "weighting"):
            if R["research"].get(f"sweep_{key}"):
                grid[key] = R["research"][f"sweep_{key}"]
        print(f"Sweep grid: {grid}")
        res = param_sweep(panel, base, grid, split=a.split)
        print(res.round(3).to_string())
        best = res.sort_values("IS Sharpe").iloc[-1]
        n_obs = len(panel.close.loc[base.start:a.split])
        print(f"\nBest in-sample combo: {best.drop([c for c in res.columns if 'S ' in c]).to_dict()}")
        print(f"  IS Sharpe {best['IS Sharpe']:.2f} -> OOS Sharpe {best['OOS Sharpe']:.2f}")
        print(f"  Deflated Sharpe probability (n_trials={len(res)}): "
              f"{deflated_sharpe(best['IS Sharpe'], len(res), n_obs):.2f}  (want > 0.95)")
        results.save("param_sweep", "Signal research · parameter sweep", {"Sweep": res}, kind="research",
                     params={"base": base.name, "split": a.split, "trials": len(res)},
                     notes=f"In-sample before {a.split}, out-of-sample after. Pick settings on in-sample only; the "
                           f"best in-sample combo's deflated Sharpe probability is "
                           f"{deflated_sharpe(best['IS Sharpe'], len(res), n_obs):.2f} (want > 0.95).")


if __name__ == "__main__":
    main()
