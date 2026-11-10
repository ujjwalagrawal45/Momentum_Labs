"""Download/refresh daily OHLCV for the universe + benchmarks from Upstox.

    python -m momentum_lab fetch                 # download anything not cached (from the settings workbook's 'Live & Data' start date)
    python -m momentum_lab fetch --update        # append new days only (fast; use at month-end)
    python -m momentum_lab fetch --refresh       # re-download everything
"""
import argparse
from datetime import date

from momentum_lab import download_all, load_env, load_universe
from momentum_lab.settings import globals_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="ind_nifty500list.csv", help="stock-list CSV (looked up in data/)")
    ap.add_argument("--start", default=globals_()["live"]["data_start"], help="history start (keep ~1y before backtest start for warm-up)")
    ap.add_argument("--refresh", action="store_true", help="re-download full history")
    ap.add_argument("--update", action="store_true", help="append only new days (fast)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    load_env()
    download_all(load_universe(a.universe), date.fromisoformat(a.start), workers=a.workers,
                 refresh=a.refresh, update=a.update)


if __name__ == "__main__":
    main()
