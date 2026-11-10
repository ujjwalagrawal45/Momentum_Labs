"""Smoke tests that run on a fresh checkout: no downloaded prices, no .env, no network.

    pip install -e ".[dev]"  &&  pytest
"""
import importlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from momentum_lab import StrategyConfig, paths, run_backtest
from momentum_lab.__main__ import COMMANDS
from momentum_lab.data import Panel, load_universe
from momentum_lab.metrics import summary


@pytest.mark.parametrize("module", sorted({m for m, _ in COMMANDS.values()}))
def test_every_command_imports(module):
    importlib.import_module(module)  # no side effects at import time


@pytest.mark.parametrize("cmd", ["fetch", "rolling", "track", "signals"])
def test_cli_help(cmd):
    r = subprocess.run([sys.executable, "-m", "momentum_lab", cmd, "--help"], capture_output=True, text=True,
                       cwd=paths.ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "usage:" in r.stdout


def test_cli_lists_commands():
    r = subprocess.run([sys.executable, "-m", "momentum_lab"], capture_output=True, text=True, cwd=paths.ROOT)
    assert r.returncode == 0 and all(c in r.stdout for c in COMMANDS)


def test_universe_ships_with_repo():
    u = load_universe("ind_nifty500list.csv")
    assert paths.universe_path("ind_nifty500list.csv").parent == paths.DATA_DIR
    assert len(u) > 400 and {"symbol", "sector"} <= set(u.columns)


def test_settings_workbook_is_valid():
    from momentum_lab.settings import read_workbook
    s = read_workbook()
    assert s["strategies"], "the workbook defines no strategies"


def synthetic_panel(n_stocks=40, years=5, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=252 * years)
    syms = [f"S{i:02d}" for i in range(n_stocks)]
    drift = rng.normal(0.0004, 0.0004, n_stocks)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(drift, 0.02, (len(idx), n_stocks)), axis=0)),
                         index=idx, columns=syms)
    vol = pd.DataFrame(1e6, index=idx, columns=syms)
    meta = pd.DataFrame({"name": syms, "sector": [f"Sec{i % 8}" for i in range(n_stocks)]}, index=syms)
    bench = close.mean(axis=1)
    return Panel(open=close.shift(1).fillna(close), high=close * 1.01, low=close * 0.99, close=close,
                 volume=vol, benchmark=bench, meta=meta)


def test_backtest_runs_on_synthetic_prices():
    cfg = StrategyConfig(name="smoke", start="2016-01-01", top_n=10)
    res = run_backtest(synthetic_panel(), cfg)
    assert res.nav.notna().all() and (res.nav > 0).all()
    assert not summary(res).empty
