"""Every file and folder the lab reads or writes, defined in one place.

The project root is the folder that holds the `momentum_lab` package. Set the environment variable
MOMENTUM_LAB_HOME to keep data, settings, journal and results somewhere else (e.g. a shared drive).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MOMENTUM_LAB_HOME") or Path(__file__).resolve().parent.parent).resolve()

ENV_FILE = ROOT / ".env"                          # secrets (never shared; see .env.example)
WORKBOOK = ROOT / "Momentum Lab Settings.xlsx"    # every setting of the lab
CONFIGS = ROOT / "configs"                        # generated from the workbook
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                        # cached daily prices, one CSV per symbol
JOURNAL = ROOT / "journal"                        # your month-end books and orders
OUTPUT = ROOT / "output"
RESULTS = OUTPUT / "results"                      # analysis results shown in the app


def universe_path(name: str | Path) -> Path:
    """Resolve a stock-list CSV: an existing path as given, else data/<name>, else <root>/<name>."""
    p = Path(name)
    if p.exists():
        return p
    for base in (DATA_DIR, ROOT):
        if (base / p.name).exists():
            return base / p.name
    return DATA_DIR / p.name
