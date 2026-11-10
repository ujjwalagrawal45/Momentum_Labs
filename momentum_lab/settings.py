"""Momentum Lab settings workbook: every knob of the lab in one Excel file, "Momentum Lab Settings.xlsx".

Edit the yellow cells, save, and the next script / app run picks the changes up:
  * each strategy column becomes configs/<name>.json (what app.py and the analysis scripts read)
  * the global sheets (costs, tax, rolling backtest, checks, phases, live, research) become
    configs/.lab_settings.json, read through globals_()

    python -m momentum_lab settings              # validate the workbook and apply it now
    python -m momentum_lab settings --rebuild    # recreate the workbook from the current configs (backs up the old one)

Sync is automatic: importing momentum_lab re-applies the workbook whenever it was saved after the last sync.
If the workbook has an error, nothing is applied, the previous settings stay in force, and the error names the cell.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import asdict, fields
from datetime import date, datetime
from pathlib import Path

from .config import Costs, StrategyConfig

from .paths import CONFIGS, ROOT, WORKBOOK, universe_path  # noqa: F401
INACTIVE = CONFIGS / "inactive"
STATE = CONFIGS / ".lab_settings.json"
SLOTS = 8          # strategy columns C..J on every strategy sheet
SLOT_COL = 3       # column C = slot 1

SIGNAL_INFO = {
    "mom_12_1": "Return from 12 months ago to 1 month ago (classic momentum, skips the last month)",
    "mom_6_1": "Return from 6 months ago to 1 month ago",
    "mom_3": "Last 3 months' return (no skip, so prone to short-term reversal)",
    "risk_adj_12_1": "12-1 return divided by 1-year volatility (steadier trends rank higher)",
    "nse_momentum": "NSE Momentum-index style: 6m and 12m risk-adjusted returns, 50/50",
    "high_52w": "Closeness to the 52-week high (1 = at the high)",
    "frog_in_pan": "Smoothness: momentum from many small up-days beats a few big jumps",
    "residual_momentum": "Momentum after stripping out the market's move (beta)",
}

# (key, label, kind, help). kind: int num pct date month bool str list choice:a|b ; "?" = blank means off/none
SECTION = "§"
STRATEGY_SHEETS = {
    "Universe & Filters": [
        (SECTION, "UNIVERSE AND PERIOD"),
        ("universe_csv", "Stock list (CSV in the project folder)", "str", "NSE list with Symbol, Company Name, Industry"),
        ("benchmark", "Benchmark", "str", "Index file data/raw/_<name>.csv, e.g. NIFTY500"),
        ("start", "Backtest start date", "date", "YYYY-MM-DD. Data starts 2015, signals need 1 year of warm-up"),
        ("end", "Backtest end date", "date?", "Blank = latest data"),
        (SECTION, "STOCK FILTERS (a stock must pass all of them to be bought)"),
        ("min_history_days", "Min. trading history (days)", "int", "252 = about 1 year"),
        ("min_price", "Min. share price (₹)", "num", "Skips penny stocks"),
        ("min_adv_cr", "Min. median daily traded value (₹ crore)", "num", "Liquidity filter, e.g. 5 = ₹5 crore a day"),
        ("adv_window", "Traded-value lookback (days)", "int", "63 = about 3 months"),
        ("max_vol", "Max. annual volatility", "pct?", "Blank = no cap. e.g. 80%"),
        ("abs_momentum", "Require positive 12-1 momentum", "bool", "YES = never buy a stock whose own trend is down"),
        ("above_ma", "Require price above its N-day average", "int?", "Blank = off. e.g. 200"),
    ],
    "Portfolio": [
        (SECTION, "PORTFOLIO CONSTRUCTION"),
        ("initial_capital", "Starting capital (₹)", "num", "Default capital for the app's first run and Next month view"),
        ("rebalance", "Rebalance frequency", "choice:M|Q|W", "M = monthly, Q = quarterly, W = weekly (last trading day)"),
        ("execution", "Order fill price", "choice:next_open|next_close", "Signal on the close, trade the next day"),
        ("top_n", "Number of stocks to hold", "int", "Fewer stocks = bigger positions (helps small accounts)"),
        ("buffer_rank", "Keep a holding while its rank is within", "int?",
         "Hold buffer, e.g. 50 with 25 stocks. Blank = sell as soon as it leaves the top N. Biggest turnover lever"),
        ("weighting", "Position weighting",
         "choice:equal|inverse_vol|score|min_variance|erc|erc_tilt|hrp|mean_variance|black_litterman|kelly",
         "equal; inverse_vol = less money in volatile stocks; score = more money in higher scores. Risk-model "
         "weightings (use correlations, see 'Risk & Shorts'): min_variance; erc = equal risk from every stock; "
         "erc_tilt = more risk to stocks with a higher chance of beating Nifty; hrp = hierarchical risk parity; "
         "mean_variance / black_litterman = use the expected-return forecasts; kelly = Kelly sizing (can also "
         "leave cash uninvested)"),
        ("max_weight", "Max. weight per stock", "pct", "e.g. 8%"),
        ("max_per_sector", "Max. stocks per sector", "int?", "Blank = no sector cap"),
    ],
    "Risk & Shorts": [
        (SECTION, "RISK OVERLAYS"),
        ("regime_filter", "Market regime filter", "choice?:bench_200dma|off",
         "bench_200dma = cut exposure when the benchmark is below its 200-day average; off = always fully invested"),
        ("regime_exposure", "Exposure when the market is below its 200DMA", "pct", "e.g. 50% = half in cash"),
        ("vol_target", "Volatility target (annual)", "pct?", "Blank = off. e.g. 20%: scale down when the strategy gets volatile"),
        ("vol_lookback", "Volatility-target lookback (days)", "int", "63 = about 3 months"),
        (SECTION, "CASH"),
        ("cash_rate", "Interest on idle cash (per year)", "pct", "e.g. 6.5% for a liquid fund"),
        ("rf_rate", "Risk-free rate (for Sharpe ratio)", "pct", "e.g. 6.5%"),
        (SECTION, "SHORT BOOK (stock futures; needs several lakhs per lot)"),
        ("short_n", "Number of shorts", "int", "0 = long-only"),
        ("short_exposure", "Short book size (% of capital)", "pct", "e.g. 50%"),
        ("short_buffer_rank", "Keep a short while its bottom rank is within", "int?", "Blank = same as number of shorts"),
        ("short_fno_only", "Short only F&O stocks", "bool", "YES: in India an overnight short has to be a stock future"),
        (SECTION, "EXPOSURE: HOW MUCH OF THE CAPITAL IS AT RISK (the smallest of these wins; the rest earns cash interest)"),
        ("max_allocation", "Max. share of capital in stocks", "pct",
         "Static cap. Rule of thumb: the drawdown you can live with / the strategy's worst drawdown, e.g. 20% / 40% = 50%"),
        ("dd_limit", "Drawdown control: max. drawdown", "pct?",
         "Blank = off. Grossman-Zhou: exposure shrinks as the strategy nears this fall from its peak, e.g. 25%"),
        ("dd_multiplier", "Drawdown control: multiplier", "num",
         "Exposure = multiplier x cushion above the floor. Higher = stays invested longer, protects less. e.g. 2-4"),
        ("dd_floor", "Drawdown control: min. exposure", "pct", "Never go below this, so it can't get stuck in cash. e.g. 25%"),
        ("exposure_band", "Skip exposure changes smaller than", "pct",
         "e.g. 10% = keep the current exposure unless the new one differs by 10 points or more (saves costs)"),
        ("risk_check", "Also check exposure between rebalances", "choice?:M|W|off",
         "off = only at rebalances. M = every month-end (W = weekly): if exposure must move by the band "
         "(min. 5 points), rebalance early. Needed for a drawdown limit to work with quarterly rebalancing. "
         "Run the app at every month-end then"),
        ("derisk", "How to cut exposure", "choice:trim|drop",
         "trim = sell a slice of every holding; drop = sell the weakest-ranked names outright (fewer orders, "
         "fewer DP charges on a small account)"),
        (SECTION, "FORECASTS AND RISK MODEL (used by the risk-model weightings only)"),
        ("alpha_model", "Expected-return model", "choice:calibrated|grinold",
         "calibrated = history of how stocks with this score did vs Nifty; grinold = IC x volatility x score"),
        ("alpha_shrink", "Shrink forecasts to", "pct", "Forecasts are noisy: 50% = use half of the estimate"),
        ("forecast_lookback_years", "Forecast history (years)", "int?", "Blank = all history before the signal date"),
        ("cov_lookback", "Correlation / covariance lookback (days)", "int", "252 = 1 year of daily returns"),
        ("risk_aversion", "Risk aversion (mean-variance, Black-Litterman)", "num", "Black-Litterman: higher = smaller tilts away from equal weight (10 = moderate). Mean-variance: higher = closer to the lowest-risk mix"),
        ("view_confidence", "Black-Litterman confidence in forecasts", "pct", "0% = ignore forecasts (equal weight), 100% = trust fully"),
        ("kelly_fraction", "Kelly fraction", "pct", "Share of full Kelly. 25-50% is standard; full Kelly is far too aggressive"),
        ("equity_premium", "Assumed equity return over cash (Kelly)", "pct", "Per year, e.g. 6%"),
    ],
}

GLOBAL_SHEETS = {  # sheet -> (state key, [(key, label, kind, default, help)])
    "Costs": ("costs", [
        (SECTION, "BROKER AND EXCHANGE CHARGES (apply to every strategy)"),
        ("brokerage_per_order", "Brokerage per order (₹)", "num", 20.0, "Flat fee per executed order. Zero-brokerage delivery broker: 0"),
        ("brokerage_bps", "Brokerage as % of order (bps)", "num", 0.0, "1 bp = 0.01%. Use instead of, or on top of, the flat fee"),
        ("dp_charge_per_sell", "DP charge per stock sold (₹)", "num", 20.0, "Depository charge, per scrip per sell day"),
        ("gst_rate", "GST on brokerage and exchange fees", "pct", 0.18, "18%"),
        (SECTION, "STATUTORY (change only if the law changes)"),
        ("stt_bps", "STT, buy and sell (bps)", "num", 10.0, "10 bps = 0.1% delivery STT"),
        ("stamp_duty_buy_bps", "Stamp duty on buys (bps)", "num", 1.5, "1.5 bps = 0.015%"),
        ("exchange_bps", "NSE transaction charge (bps)", "num", 0.297, "0.297 bps = 0.00297%"),
        ("sebi_bps", "SEBI fee (bps)", "num", 0.01, "₹10 per crore"),
        (SECTION, "EXECUTION QUALITY (estimates)"),
        ("slippage_bps", "Slippage (bps)", "num", 10.0, "Half-spread + timing loss vs the open price"),
        ("impact_coef", "Market impact coefficient", "num", 0.5, "Square-root impact: coef × daily vol × √(order / daily volume)"),
    ]),
    "Tax": ("tax", [
        (SECTION, "CAPITAL-GAINS TAX (used by the rolling backtest)"),
        ("stcg", "Short-term capital gains rate (held ≤ 1 year)", "pct", 0.20, "20% since 23 Jul 2024"),
        ("ltcg", "Long-term capital gains rate (held > 1 year)", "pct", 0.125, "12.5% since 23 Jul 2024"),
        ("ltcg_exempt", "LTCG exemption per financial year (₹)", "num", 125_000.0, "₹1.25 lakh"),
        ("fno", "Tax on futures (short) profits", "pct", 0.30, "Business income: use your slab rate"),
        ("cess", "Health & education cess", "pct", 0.04, "4% on top of the tax"),
    ]),
    "Rolling Backtest": ("rolling", [
        (SECTION, "ROLLING WALK-FORWARD BACKTEST (rolling_backtest.py)"),
        ("capital", "Starting capital (₹)", "num", 50_000.0,
         "compound: invested once and rolled through every window; fresh: every window starts with this much cash"),
        ("first", "First window starts (month)", "month", "2019-01", "YYYY-MM"),
        ("last", "Last window ends (month)", "month", "2026-09", "YYYY-MM, must be inside the data"),
        ("years", "Window length (years)", "int", 2, ""),
        ("step", "Months between window starts", "int", 3, "1 = monthly roll, 3 = quarterly, 12 = yearly"),
        ("stress", "Cost stress multiple", "num", 2.0, "Stress variant multiplies slippage and market impact by this"),
        ("compare_capital", "Also run at this capital (₹)", "num?", 1_000_000.0,
         "Optional second report at another capital, to see what account size does. Blank = skip"),
        ("capital_mode", "Capital between windows", "choice:compound|fresh", "compound",
         "compound = ONE pot: windows run back to back and each starts with the previous window's after-tax value "
         "(does the money compound?). fresh = every window starts with the starting capital, windows overlap by "
         "the step above (does it work whatever month you start?)"),
    ]),
    "Live & Data": ("live", [
        (SECTION, "LIVE MONTH-END (app.py / journal) AND DATA"),
        ("no_trade_band", "Skip resizes smaller than", "pct", 0.25,
         "25% = leave a held position alone unless it is more than 25% away from its target size"),
        ("data_start", "Price history start date", "date", "2012-01-01", "The fetch command and the app download from this date"),
    ]),
    "Research": ("research", [
        (SECTION, "SIGNAL RESEARCH AND PARAMETER SWEEP (research_signals.py)"),
        ("base_strategy", "Base strategy for research", "str", "baseline_12_1", "Strategy name from the Strategies sheet"),
        ("split", "In-sample / out-of-sample split date", "date", "2022-01-01", "Parameters are chosen before this date only"),
        (SECTION, "SWEEP GRID: comma-separated values to try; blank = not swept. Signal sets are on the 'Sweep Signal Sets' sheet"),
        ("sweep_top_n", "Number of stocks", "list", "20, 30", "e.g. 15, 20, 25, 30"),
        ("sweep_rebalance", "Rebalance frequency", "list", "M, Q", "M, Q, W"),
        ("sweep_buffer_rank", "Hold buffer", "list", "", "e.g. 30, 50, 75"),
        ("sweep_max_per_sector", "Max. stocks per sector", "list", "", "e.g. 4, 6"),
        ("sweep_weighting", "Weighting", "list", "", "e.g. equal, inverse_vol"),
    ]),
}

CHECK_DEFS = [  # id, label template ({} = threshold), default threshold
    ("beat_share", "Beats Nifty 500 after tax & costs in at least {} of 2-year windows", 0.70),
    ("median_alpha", "Median after-tax alpha of at least {} per year", 0.03),
    ("worst_alpha", "Worst-window after-tax alpha better than {} per year", -0.05),
    ("stress_share", "Beats Nifty in at least {} of windows under cost stress", 0.60),
    ("survivorship", "Median after-tax alpha still positive after a {} per year survivorship haircut", 0.04),
    ("drawdown", "Median max drawdown no more than {} worse than Nifty's", 0.0),
    ("invested", "At least {} of target capital actually invested (whole-share rounding)", 0.80),
]

DEFAULT_PHASES = [("Pre-COVID drift", "2019-01", "2020-01"), ("COVID crash", "2020-02", "2020-03"),
                  ("Post-COVID bull run", "2020-04", "2021-12"), ("2022 rate-hike chop", "2022-01", "2023-03"),
                  ("2023-24 broad rally", "2023-04", "2024-09"), ("2024-25 correction", "2024-10", "2025-03"),
                  ("2025-26", "2025-04", "2026-09")]
MAX_PHASES = 20
DEFAULT_SWEEP_SETS = [{"mom_12_1": 1.0}, {"nse_momentum": 1.0}, {"residual_momentum": 1.0},
                      {"mom_12_1": 0.5, "frog_in_pan": 0.25, "high_52w": 0.25}]
SWEEP_SETS = 8


class SettingsError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------
# parsing
_BLANK = object()


def _parse(v, kind: str, where: str, default=_BLANK):
    opt = "?" in kind
    base = kind.replace("?", "")
    if v is None or (isinstance(v, str) and not v.strip()):
        if base == "list":
            return []
        if opt:
            return None
        if default is not _BLANK:
            return default
        raise SettingsError(f"{where}: this cell is required")
    if isinstance(v, str):
        v = v.strip()
    try:
        if base.startswith("choice"):
            opts = base.split(":", 1)[1].split("|")
            m = {o.lower(): o for o in opts}
            if str(v).lower() not in m:
                raise SettingsError(f"{where}: '{v}' is not one of {', '.join(opts)}")
            out = m[str(v).lower()]
            return None if out == "off" else out
        if base == "bool":
            if isinstance(v, bool):
                return v
            s = str(v).upper()
            if s in ("YES", "Y", "TRUE", "1"):
                return True
            if s in ("NO", "N", "FALSE", "0"):
                return False
            raise SettingsError(f"{where}: expected YES or NO, got '{v}'")
        if base == "int":
            f = float(str(v).replace(",", ""))
            if not f.is_integer():
                raise SettingsError(f"{where}: expected a whole number, got {v}")
            return int(f)
        if base == "num":
            return float(str(v).replace(",", "").replace("₹", ""))
        if base == "pct":
            if isinstance(v, str):
                f = float(v.rstrip("%")) / 100 if v.endswith("%") else float(v)
            else:
                f = float(v)
            if abs(f) > 1.5:
                raise SettingsError(f"{where}: {v} looks like a whole-number percent; type it as {v}% (e.g. 8%)")
            return f
        if base == "date":
            if isinstance(v, (datetime, date)):
                return v.strftime("%Y-%m-%d")
            return datetime.strptime(str(v)[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        if base == "month":
            if isinstance(v, (datetime, date)):
                return v.strftime("%Y-%m")
            return datetime.strptime(str(v)[:7], "%Y-%m").strftime("%Y-%m")
        if base == "list":
            out = []
            for x in str(v).split(","):
                x = x.strip()
                if x:
                    out.append(int(x) if re.fullmatch(r"-?\d+", x) else float(x) if re.fullmatch(r"-?\d*\.\d+", x) else x)
            return out
        return str(v)
    except SettingsError:
        raise
    except (ValueError, TypeError):
        hint = {"date": " (use YYYY-MM-DD)", "month": " (use YYYY-MM)"}.get(base, "")
        raise SettingsError(f"{where}: cannot read '{v}' as {base}{hint}") from None


def _kv_rows(ws) -> dict:
    return {r[0].value: r for r in ws.iter_rows() if isinstance(r[0].value, str) and r[0].value != SECTION}


def upgrade_workbook(path: Path = WORKBOOK) -> list[str]:
    """Add settings rows that newer code knows about but the workbook lacks, in place (values = current
    configs or defaults), and refresh drop-down lists. Everything already in the workbook is kept."""
    from copy import copy
    from openpyxl import load_workbook
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = load_workbook(path)
    added = []
    names = {}
    for r in wb["Strategies"].iter_rows(min_row=4, max_row=3 + SLOTS):
        if isinstance(r[0].value, (int, float)) and r[1].value:
            names[int(r[0].value)] = str(r[1].value).strip()

    def cfg_for(name):
        for f in (CONFIGS / f"{name}.json", INACTIVE / f"{name}.json"):
            if f.exists():
                return StrategyConfig.from_json(f)
        return StrategyConfig()

    cfgs = {k: cfg_for(n) for k, n in names.items()}
    g = _state()

    def choices(kind):
        return kind.replace("?", "").split(":", 1)[1].split("|")

    def add_rows(ws, spec, ncol, values):
        rows = _kv_rows(ws)
        style_row = max(r[0].row for r in rows.values()) if rows else 4
        sec_row = next((r[0].row for r in ws.iter_rows() if r[0].value is None and r[1].value
                        and ws.cell(r[0].row, 2).font.bold), None)
        pending, r = None, ws.max_row + 1
        for item in spec:
            if item[0] == SECTION:
                pending = item[1]
                continue
            key, label, kind = item[0], item[1], item[2]
            if key in rows:
                if kind.startswith("choice"):  # refresh the drop-down for an existing row
                    for dv in ws.data_validations.dataValidation:
                        if any(rows[key][2].coordinate in rg for rg in dv.sqref.ranges):
                            dv.formula1 = '"' + ",".join(choices(kind)) + '"'
                pending = None
                continue
            if pending:
                for j in range(1, ncol + 1):
                    c = ws.cell(r, j)
                    if sec_row:
                        src = ws.cell(sec_row, j)
                        c.fill, c.font = copy(src.fill), copy(src.font)
                ws.cell(r, 2, pending)
                r, pending = r + 1, None
            ws.cell(r, 1, key)
            ws.cell(r, 2, label)
            for j in range(1, ncol + 1):
                a, b = ws.cell(style_row, j), ws.cell(r, j)
                b.font, b.fill, b.border, b.alignment = copy(a.font), copy(a.fill), copy(a.border), copy(a.alignment)
                b.number_format = a.number_format if j <= 2 or j == ncol else \
                    {"pct": "0.0%", "int": "0", "num": "#,##0.###"}.get(kind.replace("?", ""), "General")
            ws.cell(r, ncol, item[-1])
            for col, v in values(key, kind):
                ws.cell(r, col, v)
            if kind.startswith("choice"):
                dv = DataValidation(type="list", formula1='"' + ",".join(choices(kind)) + '"', allow_blank=True)
                ws.add_data_validation(dv)
                dv.add(f"{ws.cell(r, 3).column_letter}{r}:{ws.cell(r, ncol - 1).column_letter}{r}")
            added.append(f"{ws.title}!{key}")
            r += 1

    def show(v, kind):
        if kind.startswith("choice"):
            return v if v is not None else "off"
        return ("YES" if v else "NO") if kind.replace("?", "") == "bool" else v

    for sheet, spec in STRATEGY_SHEETS.items():
        add_rows(wb[sheet], spec, SLOT_COL + SLOTS,
                 lambda key, kind: [(SLOT_COL + k - 1, show(getattr(c, key), kind)) for k, c in cfgs.items()])
    for sheet, (skey, spec) in GLOBAL_SHEETS.items():
        add_rows(wb[sheet], spec, 4, lambda key, kind, skey=skey, spec=spec: [
            (3, show(g.get(skey, {}).get(key, next(i[3] for i in spec if i[0] == key)), kind))])
    if added:
        wb.save(path)
    return added


def read_workbook(path: Path = WORKBOOK) -> dict:
    from openpyxl import load_workbook
    from .signals import SIGNALS

    wb = load_workbook(path, data_only=True)
    need = ["Strategies", "Signals", *STRATEGY_SHEETS, *GLOBAL_SHEETS, "Decision Checks", "Market Phases",
            "Sweep Signal Sets"]
    missing = [s for s in need if s not in wb.sheetnames]
    if missing:
        raise SettingsError(f"missing sheet(s): {', '.join(missing)} (do not rename sheets)")

    # ---- strategies -----------------------------------------------------------------------------
    slots = {}
    for r in wb["Strategies"].iter_rows(min_row=4, max_row=3 + SLOTS):
        k, name = r[0].value, r[1].value
        if not isinstance(k, (int, float)) or name is None or not str(name).strip():
            continue
        name, where = str(name).strip(), f"Strategies!{r[1].coordinate}"
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
            raise SettingsError(f"{where}: strategy name '{name}' may only use letters, digits, _ and -")
        if name in [s["name"] for s in slots.values()]:
            raise SettingsError(f"{where}: duplicate strategy name '{name}'")
        slots[int(k)] = {"name": name,
                         "active": _parse(r[2].value, "bool", f"Strategies!{r[2].coordinate}", False),
                         "rolling": _parse(r[3].value, "bool", f"Strategies!{r[3].coordinate}", False),
                         "notes": r[4].value or ""}
    if not slots:
        raise SettingsError("Strategies: no strategy names filled in")

    defaults = StrategyConfig()
    raw = {k: {"name": s["name"]} for k, s in slots.items()}
    for sheet, spec in STRATEGY_SHEETS.items():
        rows = _kv_rows(wb[sheet])
        for item in spec:
            if item[0] == SECTION:
                continue
            key, label, kind, _ = item
            if key not in rows:
                raise SettingsError(f"{sheet}: row '{key}' is missing (do not edit column A). "
                                    f"Run 'python -m momentum_lab settings --upgrade' to add new settings rows.")
            for k in slots:
                cell = rows[key][SLOT_COL - 1 + k - 1]
                v = _parse(cell.value, kind, f"{sheet}!{cell.coordinate} ({slots[k]['name']}: {label})",
                           getattr(defaults, key))
                raw[k][key] = v
    sig_rows = _kv_rows(wb["Signals"])
    for k in slots:
        w = {}
        for sig in SIGNALS:
            if sig in sig_rows:
                cell = sig_rows[sig][SLOT_COL - 1 + k - 1]
                x = _parse(cell.value, "num?", f"Signals!{cell.coordinate} ({slots[k]['name']}: {sig})")
                if x:
                    w[sig] = x
        if not w:
            raise SettingsError(f"Signals: strategy '{slots[k]['name']}' has no signal weights (give at least one a weight)")
        raw[k]["signal_weights"] = w

    # ---- global sheets --------------------------------------------------------------------------
    g = {}
    for sheet, (skey, spec) in GLOBAL_SHEETS.items():
        rows = _kv_rows(wb[sheet])
        g[skey] = {}
        for item in spec:
            if item[0] == SECTION:
                continue
            key, label, kind, dflt, _ = item
            cell = rows[key][2] if key in rows else None
            if cell is None:
                raise SettingsError(f"{sheet}: row '{key}' is missing (do not edit column A)")
            g[skey][key] = _parse(cell.value, kind, f"{sheet}!{cell.coordinate} ({label})", dflt)
    costs = Costs(**g["costs"])

    rows = _kv_rows(wb["Decision Checks"])
    g["checks"] = []
    for cid, label, thr in CHECK_DEFS:
        if cid not in rows:
            raise SettingsError(f"Decision Checks: row '{cid}' is missing")
        r = rows[cid]
        g["checks"].append({"id": cid,
                            "threshold": _parse(r[2].value, "pct", f"Decision Checks!{r[2].coordinate}", thr),
                            "enabled": _parse(r[3].value, "bool", f"Decision Checks!{r[3].coordinate}", True)})
    g["phases"] = []
    for r in wb["Market Phases"].iter_rows(min_row=4, max_row=3 + MAX_PHASES):
        if r[0].value is None or not str(r[0].value).strip():
            continue
        a = _parse(r[1].value, "month", f"Market Phases!{r[1].coordinate}")
        z = _parse(r[2].value, "month", f"Market Phases!{r[2].coordinate}")
        if z < a:
            raise SettingsError(f"Market Phases!{r[2].coordinate}: 'to' month is before 'from' month")
        g["phases"].append([str(r[0].value).strip(), a, z])
    sets_rows = _kv_rows(wb["Sweep Signal Sets"])
    g["sweep_sets"] = []
    for k in range(SWEEP_SETS):
        w = {}
        for sig in SIGNALS:
            if sig in sets_rows:
                cell = sets_rows[sig][2 + k]
                x = _parse(cell.value, "num?", f"Sweep Signal Sets!{cell.coordinate}")
                if x:
                    w[sig] = x
        if w:
            g["sweep_sets"].append(w)

    # ---- build + cross-check configs ------------------------------------------------------------
    strategies = []
    for k, s in sorted(slots.items()):
        d = raw[k]
        where = f"strategy '{s['name']}'"
        try:
            cfg = StrategyConfig.from_dict({**d, "costs": asdict(costs)})
        except (TypeError, ValueError) as e:
            raise SettingsError(f"{where}: {e}") from None
        if cfg.top_n < 1:
            raise SettingsError(f"Portfolio: {where} needs at least 1 stock")
        if cfg.buffer_rank is not None and cfg.buffer_rank < cfg.top_n:
            raise SettingsError(f"Portfolio: {where} hold buffer ({cfg.buffer_rank}) must be >= number of stocks ({cfg.top_n})")
        if cfg.end and cfg.end <= cfg.start:
            raise SettingsError(f"Universe & Filters: {where} end date is before the start date")
        if not universe_path(cfg.universe_csv).exists():
            raise SettingsError(f"Universe & Filters: {where} stock list '{cfg.universe_csv}' not found in data/ or the project folder")
        strategies.append({**s, "config": cfg})
    names = {s["name"] for s in strategies}
    if g["research"]["base_strategy"] not in names:
        raise SettingsError(f"Research: base strategy '{g['research']['base_strategy']}' is not on the Strategies sheet")
    if g["rolling"]["first"] > g["rolling"]["last"]:
        raise SettingsError("Rolling Backtest: first window month is after the last month")
    g["rolling_strategies"] = [s["name"] for s in strategies if s["rolling"] and s["active"]]
    if not g["rolling_strategies"]:
        raise SettingsError("Strategies: set Active and Include in rolling backtest to YES for at least one strategy")
    return {"strategies": strategies, "globals": g}


# ------------------------------------------------------------------------------------------------
# sync
def _defaults() -> dict:
    g = {skey: {it[0]: it[3] for it in spec if it[0] != SECTION} for skey, spec in GLOBAL_SHEETS.values()}
    g["checks"] = [{"id": c, "threshold": t, "enabled": True} for c, _, t in CHECK_DEFS]
    g["phases"] = [list(p) for p in DEFAULT_PHASES]
    g["sweep_sets"] = DEFAULT_SWEEP_SETS
    g["rolling_strategies"] = ["baseline_12_1", "nse_style", "quant_composite", "quant_composite_ls"]
    g["managed"] = []
    return g


def _state() -> dict:
    g = _defaults()
    if STATE.exists():
        try:
            saved = json.loads(STATE.read_text(encoding="utf-8"))
            for k, v in saved.items():
                g[k] = {**g[k], **v} if isinstance(g.get(k), dict) and isinstance(v, dict) else v
        except (OSError, ValueError):
            pass
    return g


def _write_if_changed(path: Path, text: str) -> None:
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def sync(force: bool = False, quiet: bool = False) -> bool:
    """Apply the workbook if it was saved after the last sync. Returns True if settings were (re)applied."""
    if not WORKBOOK.exists():
        return False
    if not force and STATE.exists() and STATE.stat().st_mtime >= WORKBOOK.stat().st_mtime:
        return False
    try:
        s = read_workbook()
    except SettingsError as e:
        print(f"\n!! SETTINGS NOT APPLIED -- fix '{WORKBOOK.name}': {e}\n!! Using the previous settings.\n",
              file=sys.stderr)
        return False
    except Exception as e:  # locked mid-save, corrupt file, ...
        print(f"\n!! Could not read '{WORKBOOK.name}' ({type(e).__name__}: {e}). Using the previous settings.\n",
              file=sys.stderr)
        return False
    CONFIGS.mkdir(exist_ok=True)
    prev = set(_state().get("managed", []))
    names = []
    for st in s["strategies"]:
        name, text = st["name"], json.dumps(st["config"].to_dict(), indent=2) + "\n"
        names.append(name)
        live, parked = CONFIGS / f"{name}.json", INACTIVE / f"{name}.json"
        if st["active"]:
            _write_if_changed(live, text)
            parked.unlink(missing_ok=True)
        else:
            INACTIVE.mkdir(exist_ok=True)
            _write_if_changed(parked, text)
            live.unlink(missing_ok=True)
    for name in prev - set(names):  # removed from the workbook: park its config, keep its journal
        live = CONFIGS / f"{name}.json"
        if live.exists():
            INACTIVE.mkdir(exist_ok=True)
            shutil.move(str(live), str(INACTIVE / f"{name}.json"))
    g = s["globals"]
    g["managed"] = names
    g["synced_at"] = datetime.now().isoformat(timespec="seconds")
    STATE.write_text(json.dumps(g, indent=2), encoding="utf-8")
    if not quiet:
        act = [st["name"] for st in s["strategies"] if st["active"]]
        print(f"[settings] Applied '{WORKBOOK.name}': active strategies = {', '.join(act) or 'none'}", file=sys.stderr)
    return True


def globals_() -> dict:
    """Global settings (costs, tax, rolling, live, research, checks, phases, sweep_sets, rolling_strategies)."""
    sync()
    return _state()


# ------------------------------------------------------------------------------------------------
# building the workbook
def _current_strategies() -> list[tuple[StrategyConfig, bool]]:
    out = [(StrategyConfig.from_json(p), True) for p in sorted(CONFIGS.glob("*.json")) if not p.name.startswith(".")]
    out += [(StrategyConfig.from_json(p), False) for p in sorted(INACTIVE.glob("*.json"))] if INACTIVE.exists() else []
    order = {"baseline_12_1": 0, "nse_style": 1, "quant_composite": 2, "quant_composite_ls": 3}
    return sorted(out, key=lambda x: (order.get(x[0].name, 9), x[0].name))[:SLOTS]


def build_workbook(path: Path = WORKBOOK) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation

    from .signals import SIGNALS

    g = _state()
    strategies = _current_strategies()
    rolling_set = set(g.get("rolling_strategies", []))

    INPUT = PatternFill("solid", fgColor="FFF2CC")
    HEAD = PatternFill("solid", fgColor="1F4E78")
    SEC = PatternFill("solid", fgColor="DDEBF7")
    thin = Side(style="thin", color="BFBFBF")
    BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
    WHITE_B = Font(bold=True, color="FFFFFF")
    KEYF = Font(color="808080", size=8)
    WRAP = Alignment(wrap_text=True, vertical="center")

    wb = Workbook()
    wb.remove(wb.active)

    def sheet(title, subtitle, headers, widths):
        ws = wb.create_sheet(title)
        ws["A1"] = title
        ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
        ws["A2"] = subtitle
        ws["A2"].font = Font(italic=True, color="595959")
        for j, h in enumerate(headers, 1):
            c = ws.cell(3, j, h)
            c.font, c.fill, c.alignment, c.border = WHITE_B, HEAD, WRAP, BOX
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[ws.cell(3, j).column_letter].width = w
        ws.row_dimensions[3].height = 32
        ws.freeze_panes = "C4"
        return ws

    def fmt_for(kind):
        b = kind.replace("?", "")
        return {"pct": "0.0%", "int": "0", "num": "#,##0.###", "bool": "@", "date": "@", "month": "@"}.get(b, "General")

    def validate(ws, kind, rng):
        b = kind.replace("?", "")
        dv = None
        if b == "bool":
            dv = DataValidation(type="list", formula1='"YES,NO"', allow_blank=True)
        elif b.startswith("choice"):
            dv = DataValidation(type="list", formula1='"' + ",".join(b.split(":", 1)[1].split("|")) + '"',
                                allow_blank=True)
        elif b == "int":
            dv = DataValidation(type="whole", operator="greaterThanOrEqual", formula1="0", allow_blank=True)
        elif b in ("num", "pct"):
            dv = DataValidation(type="decimal", operator="between", formula1="-1000000000", formula2="1000000000",
                                allow_blank=True)
        if dv:
            dv.error, dv.errorTitle, dv.showErrorMessage = "Not a valid value for this setting", "Invalid", True
            ws.add_data_validation(dv)
            dv.add(rng)

    def show(v, kind):
        if v is None:
            return None
        b = kind.replace("?", "")
        if b == "bool":
            return "YES" if v else "NO"
        if b.startswith("choice"):
            return v
        if b == "list":
            return ", ".join(str(x) for x in v) if isinstance(v, list) else v
        return v

    def section(ws, row, text, ncol):
        ws.cell(row, 2, text).font = Font(bold=True, color="1F4E78")
        for j in range(1, ncol + 1):
            ws.cell(row, j).fill = SEC


    # ---- Start here ---------------------------------------------------------------------------
    ws = wb.create_sheet("Start here")
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 28, 120
    lines = [
        ("MOMENTUM LAB SETTINGS", ""),
        ("", "Every setting of the Momentum Lab lives in this workbook. Edit the YELLOW cells, save the file, and "
             "the next time you run anything (the app, rolling_backtest.py, risk_compare.py ...) the "
             "changes are picked up automatically."),
        ("", ""),
        ("HOW IT WORKS", ""),
        ("Strategies", "Up to 8 strategies. Each one is a column (C to J) on the Signals, Universe & Filters, "
                       "Portfolio and Risk & Shorts sheets; column C is slot 1, D is slot 2, and so on. Name them on "
                       "the 'Strategies' sheet. To add one, type a name in an empty slot and fill its column on "
                       "the other sheets (blank cells use the default value)."),
        ("Active", "YES = shows up in the app and the scripts (written to configs/<name>.json). NO = parked in "
                   "configs/inactive/; its journal history is kept."),
        ("Blank cells", "In strategy columns a blank cell means 'off' where the description says so (e.g. vol "
                        "target, sector cap); otherwise it means the built-in default."),
        ("Percentages", "Type percentages with a % sign (8%) -- the cells are already formatted as %."),
        ("Column A", "Grey internal keys. Do not edit or move them; rows can be read in any order."),
        ("Errors", "If a cell is invalid, nothing is applied, the previous settings stay in force, and the run "
                   "prints which cell to fix. Check any time with:  python -m momentum_lab settings"),
        ("Renaming", "Renaming a strategy starts a new journal folder (journal/<new name>/). The old journal stays."),
        ("Rebuild", "python -m momentum_lab settings --rebuild  recreates this workbook from configs/ "
                    "(the old copy is backed up first)."),
        ("", ""),
        ("SHEETS", ""),
        ("Strategies", "Names, active switch, which ones the rolling backtest compares."),
        ("Signals", "What each strategy ranks stocks on: weights of the 8 momentum signals (z-scored, then summed)."),
        ("Universe & Filters", "Stock list, benchmark, backtest period and the filters a stock must pass."),
        ("Portfolio", "Capital, rebalance frequency, number of stocks, hold buffer, weighting, caps."),
        ("Risk & Shorts", "200DMA regime filter, volatility target, idle-cash interest, short book."),
        ("Costs", "Your broker's charges plus statutory costs. Applies to every strategy."),
        ("Tax", "Capital-gains tax used by the rolling backtest's after-tax results."),
        ("Rolling Backtest", "Capital, window length, roll step, period and cost-stress multiple."),
        ("Decision Checks", "Pass/fail thresholds shown on the rolling backtest's Verdict sheet."),
        ("Market Phases", "Date ranges for the rolling backtest's regime breakdown."),
        ("Live & Data", "Month-end no-trade band and price-history start date."),
        ("Research", "research_signals.py: base strategy, in/out-of-sample split and sweep grid."),
        ("Sweep Signal Sets", "Signal-weight combinations the parameter sweep tries."),
    ]
    for i, (a, b) in enumerate(lines, 1):
        ws.cell(i, 1, a).font = Font(bold=True, size=14 if i == 1 else 11, color="1F4E78" if a.isupper() else "000000")
        ws.cell(i, 2, b).alignment = Alignment(wrap_text=True, vertical="top")

    # ---- Strategies ---------------------------------------------------------------------------
    ws = sheet("Strategies", "Name each strategy (letters, digits, _ and -). Slot n = column n on the strategy sheets.",
               ["Slot", "Strategy name", "Active", "Include in rolling backtest", "Notes"], [8, 28, 12, 16, 70])
    notes = {"baseline_12_1": "Plain 12-1 momentum, top 25, monthly. The simple benchmark strategy.",
             "nse_style": "NSE Momentum-index style score, top 30, quarterly, sector cap 6.",
             "quant_composite": "Composite of 4 signals, inverse-vol weights, 200DMA regime filter, vol target.",
             "quant_composite_ls": "quant_composite plus a 10-stock short book in stock futures."}
    for k in range(1, SLOTS + 1):
        r = 3 + k
        cfg, active = strategies[k - 1] if k <= len(strategies) else (None, False)
        ws.cell(r, 1, k).font = Font(bold=True)
        vals = [cfg.name if cfg else None, ("YES" if active else "NO") if cfg else None,
                ("YES" if cfg and cfg.name in rolling_set else "NO") if cfg else None,
                notes.get(cfg.name, "") if cfg else None]
        for j, v in enumerate(vals, 2):
            c = ws.cell(r, j, v)
            c.fill, c.border = INPUT, BOX
    validate(ws, "bool", f"C4:D{3 + SLOTS}")

    def strat_header(ws):
        for k in range(1, SLOTS + 1):
            c = ws.cell(3, SLOT_COL + k - 1, f'=IF(Strategies!B{3 + k}="","(slot {k})",Strategies!B{3 + k})')
            c.font, c.fill, c.alignment, c.border = WHITE_B, HEAD, WRAP, BOX

    # ---- Signals ------------------------------------------------------------------------------
    ws = sheet("Signals", "Weight of each signal in the strategy's score (blank = not used). Weights need not sum to 1.",
               ["key", "Signal"] + [""] * SLOTS + ["What it measures"], [16, 26] + [16] * SLOTS + [70])
    strat_header(ws)
    r = 4
    for sig in SIGNALS:
        ws.cell(r, 1, sig).font = KEYF
        ws.cell(r, 2, sig).font = Font(bold=True)
        for k in range(1, SLOTS + 1):
            cfg = strategies[k - 1][0] if k <= len(strategies) else None
            c = ws.cell(r, SLOT_COL + k - 1, cfg.signal_weights.get(sig) if cfg else None)
            c.fill, c.border, c.number_format = INPUT, BOX, "0.00"
        ws.cell(r, SLOT_COL + SLOTS, SIGNAL_INFO.get(sig, "")).alignment = WRAP
        r += 1
    ws.cell(r, 2, "Total weight (info)").font = Font(italic=True)
    for k in range(1, SLOTS + 1):
        L = ws_letter(SLOT_COL + k - 1)
        ws.cell(r, SLOT_COL + k - 1, f"=SUM({L}4:{L}{r - 1})").number_format = "0.00"
    validate(ws, "num", f"{ws_letter(SLOT_COL)}4:{ws_letter(SLOT_COL + SLOTS - 1)}{r - 1}")

    # ---- per-strategy sheets ------------------------------------------------------------------
    for title, spec in STRATEGY_SHEETS.items():
        ws = sheet(title, "One column per strategy. Yellow = editable.",
                   ["key", "Setting"] + [""] * SLOTS + ["Description / allowed values"],
                   [18, 44] + [16] * SLOTS + [80])
        strat_header(ws)
        r = 4
        for item in spec:
            if item[0] == SECTION:
                section(ws, r, item[1], SLOT_COL + SLOTS)
                r += 1
                continue
            key, label, kind, help_ = item
            ws.cell(r, 1, key).font = KEYF
            ws.cell(r, 2, label)
            for k in range(1, SLOTS + 1):
                cfg = strategies[k - 1][0] if k <= len(strategies) else None
                v = show(getattr(cfg, key), kind) if cfg else None
                if kind.startswith("choice?") and cfg and v is None:
                    v = "off"
                c = ws.cell(r, SLOT_COL + k - 1, v)
                c.fill, c.border, c.number_format = INPUT, BOX, fmt_for(kind)
                c.alignment = Alignment(horizontal="center")
            ws.cell(r, SLOT_COL + SLOTS, help_).alignment = WRAP
            validate(ws, kind, f"{ws_letter(SLOT_COL)}{r}:{ws_letter(SLOT_COL + SLOTS - 1)}{r}")
            r += 1

    # ---- global key/value sheets --------------------------------------------------------------
    subtitles = {"Costs": "Per-order costs for every strategy. Set brokerage to 0 for a zero-brokerage delivery broker.",
                 "Tax": "Used by rolling_backtest.py for after-tax results.",
                 "Rolling Backtest": "Defaults for rolling_backtest.py (command-line options still override them).",
                 "Live & Data": "Month-end journal and data download.",
                 "Research": "Defaults for research_signals.py."}
    for title, (skey, spec) in GLOBAL_SHEETS.items():
        ws = sheet(title, subtitles[title], ["key", "Setting", "Value", "Description"], [20, 48, 18, 90])
        ws.freeze_panes = "A4"
        r = 4
        for item in spec:
            if item[0] == SECTION:
                section(ws, r, item[1], 4)
                r += 1
                continue
            key, label, kind, dflt, help_ = item
            ws.cell(r, 1, key).font = KEYF
            ws.cell(r, 2, label)
            v = g.get(skey, {}).get(key, dflt)
            c = ws.cell(r, 3, show(v, kind))
            c.fill, c.border, c.number_format = INPUT, BOX, fmt_for(kind)
            c.alignment = Alignment(horizontal="center")
            ws.cell(r, 4, help_).alignment = WRAP
            validate(ws, kind, f"C{r}")
            r += 1

    # ---- Decision checks ----------------------------------------------------------------------
    ws = sheet("Decision Checks", "Pass/fail tests on the rolling backtest's Verdict sheet. Set them to your own risk tolerance.",
               ["key", "Check ({} = threshold)", "Threshold", "Enabled", "Notes"], [16, 80, 14, 12, 50])
    ws.freeze_panes = "A4"
    saved = {c["id"]: c for c in g.get("checks", [])}
    for i, (cid, label, thr) in enumerate(CHECK_DEFS):
        r = 4 + i
        ws.cell(r, 1, cid).font = KEYF
        ws.cell(r, 2, label).alignment = WRAP
        c = ws.cell(r, 3, saved.get(cid, {}).get("threshold", thr))
        c.fill, c.border, c.number_format = INPUT, BOX, "0.0%"
        c = ws.cell(r, 4, "YES" if saved.get(cid, {}).get("enabled", True) else "NO")
        c.fill, c.border = INPUT, BOX
    ws.cell(4 + 4, 5, "Haircut subtracted from every window's alpha before testing it")
    ws.cell(4 + 5, 5, "0% = must be no worse than Nifty's drawdown")
    validate(ws, "pct", f"C4:C{3 + len(CHECK_DEFS)}")
    validate(ws, "bool", f"D4:D{3 + len(CHECK_DEFS)}")

    # ---- Market phases ------------------------------------------------------------------------
    ws = sheet("Market Phases", "Regime breakdown periods (inclusive months, YYYY-MM). Up to 20 rows; blank rows are ignored.",
               ["Phase", "From (YYYY-MM)", "To (YYYY-MM)"], [34, 16, 16])
    ws.freeze_panes = "A4"
    phases = g.get("phases", DEFAULT_PHASES)
    for i in range(MAX_PHASES):
        p = phases[i] if i < len(phases) else (None, None, None)
        for j, v in enumerate(p, 1):
            c = ws.cell(4 + i, j, v)
            c.fill, c.border, c.number_format = INPUT, BOX, "@"

    # ---- Sweep signal sets --------------------------------------------------------------------
    ws = sheet("Sweep Signal Sets", "Each column is one signal combination the parameter sweep tries. Blank column = unused.",
               ["key", "Signal"] + [f"Set {k}" for k in range(1, SWEEP_SETS + 1)], [16, 26] + [12] * SWEEP_SETS)
    sets = g.get("sweep_sets", DEFAULT_SWEEP_SETS)
    for i, sig in enumerate(SIGNALS):
        r = 4 + i
        ws.cell(r, 1, sig).font = KEYF
        ws.cell(r, 2, sig).font = Font(bold=True)
        for k in range(SWEEP_SETS):
            c = ws.cell(r, 3 + k, sets[k].get(sig) if k < len(sets) else None)
            c.fill, c.border, c.number_format = INPUT, BOX, "0.00"

    for ws in wb.worksheets:
        ws.sheet_view.zoomScale = 100
        for row in ws.iter_rows(min_row=4):
            for c in row:
                if c.column == 2 and ws.title not in ("Start here",):
                    c.alignment = WRAP
    if path.exists():
        backup = path.with_name(f"{path.stem} (backup {datetime.now():%Y-%m-%d %H%M}).xlsx")
        shutil.copy2(path, backup)
        print(f"Backed up the old workbook to {backup.name}")
    wb.save(path)
    return path


def ws_letter(col: int) -> str:
    from openpyxl.utils import get_column_letter
    return get_column_letter(col)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="recreate the workbook from configs/ (backs up the old one)")
    ap.add_argument("--upgrade", action="store_true", help="add rows for new settings to the workbook, keep the rest")
    a = ap.parse_args()
    if a.upgrade and WORKBOOK.exists():
        added = upgrade_workbook()
        print(f"Added {len(added)} new settings row(s): {', '.join(added) or 'none needed'}")
    if a.rebuild or not WORKBOOK.exists():
        print(f"Wrote {build_workbook()}")
    try:
        s = read_workbook()
    except SettingsError as e:
        raise SystemExit(f"Workbook has an error, nothing applied: {e}")
    sync(force=True)
    for st in s["strategies"]:
        c = st["config"]
        print(f"  {'ACTIVE  ' if st['active'] else 'inactive'} {c.name:22s} top {c.top_n}, {c.rebalance}, "
              f"{c.weighting}, signals {c.signal_weights}")
    g = s["globals"]
    print(f"  costs: Rs {g['costs']['brokerage_per_order']:.0f}/order, DP Rs {g['costs']['dp_charge_per_sell']:.0f}/sell | "
          f"rolling: Rs {g['rolling']['capital']:,.0f}, {g['rolling']['first']}..{g['rolling']['last']}, "
          f"{g['rolling']['years']}y every {g['rolling']['step']}m on {', '.join(g['rolling_strategies'])}")


if __name__ == "__main__":
    main()
