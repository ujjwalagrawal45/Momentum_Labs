"""momentum_lab: build, test and research cross-sectional momentum strategies on NSE stocks."""
from .config import Costs, StrategyConfig
from .data import Panel, download_all, load_env, load_panel, load_universe
from .backtest import run_backtest
from .metrics import summary
from .settings import sync as _sync_settings

_sync_settings()  # apply 'Momentum Lab Settings.xlsx' if it was saved since the last run
