"""Single command-line entry point for the lab.

    python -m momentum_lab <command> [options]      (or `momentum-lab <command>` after `pip install -e .`)
    python -m momentum_lab <command> --help         options for one command
"""
from __future__ import annotations

import runpy
import sys

# command -> (module, one-line help). Each module parses its own options from sys.argv.
COMMANDS = {
    "app":      ("momentum_lab.app", "Start the web app at http://127.0.0.1:5000"),
    "fetch":    ("momentum_lab.fetch", "Download / update daily prices from Upstox"),
    "settings": ("momentum_lab.settings", "Validate and apply the settings workbook (--rebuild, --upgrade)"),
    "rolling":  ("momentum_lab.analyses.rolling_backtest", "Rolling 2-year-window backtest after costs and tax"),
    "risk":     ("momentum_lab.analyses.risk_compare", "Weightings x exposure rules against a drawdown budget"),
    "forecast": ("momentum_lab.analyses.forecast_report", "Forecast calibration and risk-model weightings"),
    "track":    ("momentum_lab.analyses.track_record", "Month-by-month track record of one strategy"),
    "review":   ("momentum_lab.analyses.holding_review", "Leaders vs laggards in the latest journal book"),
    "circuit":  ("momentum_lab.analyses.circuit_review", "Lower-circuit review of the latest journal book"),
    "dist":     ("momentum_lab.analyses.distribution_report", "Rolling 1-month return distribution"),
    "signals":  ("momentum_lab.analyses.research_signals", "Signal quintile scoreboard (--sweep: parameter sweep)"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    rows = "\n".join(f"  {name:<{width}}  {text}" for name, (_, text) in COMMANDS.items())
    return f"usage: python -m momentum_lab <command> [options]\n\ncommands:\n{rows}\n"


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        sys.exit(f"Unknown command '{cmd}'.\n\n{usage()}")
    module = COMMANDS[cmd][0]
    sys.argv = [f"momentum_lab {cmd}", *rest]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
