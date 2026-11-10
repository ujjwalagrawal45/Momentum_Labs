# Momentum Lab

A momentum strategy builder, backtester and month-end trading assistant for NSE stocks (Nifty 500).

It ranks every stock in the universe on one or more momentum signals and keeps the top N that pass your
filters. It then sizes them with your weighting and risk rules and tells you exactly what to buy and sell.
It also journals every run so you build a real track record, and it can stress-test a strategy over many
2-year windows after costs and tax before you put money in.

**All settings live in one Excel file, `Momentum Lab Settings.xlsx`.** You never need to edit code or JSON.

---

## Contents

1. [One-time setup](#1-one-time-setup)
2. [The settings workbook](#2-the-settings-workbook-momentum-lab-settingsxlsx)
3. [The app: monthly picks, next month, month-end journal, results](#3-the-app-monthly-picks-next-month-month-end-journal-results)
4. [Analyses and command-line tools](#4-analyses-and-command-line-tools)
5. [Deciding whether to trade a strategy for real](#5-deciding-whether-to-trade-a-strategy-for-real)
6. [Recipes: common changes](#6-recipes-common-changes)
7. [Folder layout and output files](#7-folder-layout-and-output-files)
8. [Troubleshooting](#8-troubleshooting)
9. [Limitations: read before trusting any number](#9-limitations-read-before-trusting-any-number)
10. [Background: how the strategy works](#10-background-how-the-strategy-works)

---

## 1. One-time setup

### 1.1 Install Python and packages
You need Python 3.11 or later (this machine has 3.13). Open a terminal in the project folder and run:

```
python -m venv .venv
.venv\Scripts\activate          (macOS / Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
```
`requirements.txt` pins the exact versions the lab was verified with; `pip install -e .` registers the
`momentum_lab` package and a `momentum-lab` command (so `momentum-lab rolling` = `python -m momentum_lab rolling`).
The virtual environment is optional but keeps these packages apart from anything else on the machine.

Every tool is one command: `python -m momentum_lab <command>`. Run `python -m momentum_lab` to list them,
and `python -m momentum_lab <command> --help` for one command's options.

### 1.2 Download price history
```
python -m momentum_lab fetch
```
This downloads daily prices for every stock in `data/ind_nifty500list.csv`, plus the Nifty 500 index, from Upstox
into `data/raw/`. It starts from the **Price history start date** on the workbook's *Live & Data* sheet
(2014-01-01). The first download takes a while. After that only new days are fetched.

Upstox historical candles are already adjusted for splits and bonuses and need no login. If Upstox ever starts
requiring one, copy `.env.example` to `.env`, fill in `UPSTOX_ACCESS_TOKEN` and it is picked up automatically.

### 1.3 Check the settings
```
python -m momentum_lab settings
```
This validates the workbook, writes the strategy files and prints a one-line summary of each strategy.
If anything is wrong it names the sheet and cell to fix.

You're ready. Double-click **`Start Momentum Lab.bat`** to open the app.

---

## 2. The settings workbook (`Momentum Lab Settings.xlsx`)

### 2.1 Golden rules
* **Edit only the yellow cells.** Save the file. The next thing you run (the app, or any script) picks up the
  changes automatically. You don't need to restart the app: it re-reads the workbook every time you open
  or refresh the page.
* **Don't touch column A** (grey internal keys) and don't move or rename sheets.
* **Type percentages with a % sign** (`8%`). The cells are already formatted as percentages.
* **Blank cell** = "off" where the description says so (vol target, sector cap, max vol...), otherwise the
  built-in default.
* **If a cell is invalid, nothing is applied.** The previous settings stay in force and the run prints which cell
  to fix. Run `python -m momentum_lab settings` any time to check.
* **Never edit `configs/*.json` by hand.** They are generated from the workbook and get overwritten.
* If you break the workbook badly, `python -m momentum_lab settings --rebuild` recreates it from the current
  `configs/`. The old copy is backed up first as `Momentum Lab Settings (backup <date>).xlsx`.

### 2.2 How strategies are laid out
There are up to **8 strategy slots**. Slot 1 is column C, slot 2 is column D and so on, up to slot 8 in column J.
The same column on the *Signals*, *Universe & Filters*, *Portfolio* and *Risk & Shorts* sheets belongs to the
same strategy. The column headers show the strategy name automatically.

### 2.3 Sheet-by-sheet reference

#### Strategies
| Column | Meaning |
|---|---|
| Strategy name | Letters, digits, `_` and `-` only. Becomes `configs/<name>.json` and `journal/<name>/`. |
| Active | `YES` = appears in the app and scripts. `NO` = parked in `configs/inactive/`; its journal is kept. |
| Include in rolling backtest | `YES` = compared when you run `python -m momentum_lab rolling` with no arguments. |
| Notes | Free text for yourself. |

> Renaming a strategy starts a **new** journal folder. The old one stays under the old name.

#### Signals
The weight of each momentum signal in the strategy's score. Each signal is z-scored across stocks, multiplied by
its weight, and summed. Blank = not used. The weights don't need to add up to 1 (the *Total weight* row is
for information only).

| Signal | What it measures |
|---|---|
| `mom_12_1` | Return from 12 months ago to 1 month ago (classic momentum; skipping the last month avoids short-term reversal) |
| `mom_6_1` | Return from 6 months ago to 1 month ago |
| `mom_3` | Last 3 months' return (no skip, so prone to reversal) |
| `risk_adj_12_1` | 12-1 return ÷ 1-year volatility: steady trends rank higher than wild ones |
| `nse_momentum` | NSE Momentum-index style: 6m and 12m risk-adjusted returns, 50/50 |
| `high_52w` | Closeness to the 52-week high (1 = at the high) |
| `frog_in_pan` | Smoothness: momentum built from many small up-days beats a few big jumps |
| `residual_momentum` | Momentum after removing the market's move (beta); lowers crash risk |

#### Universe & Filters
| Setting | Meaning |
|---|---|
| Stock list | CSV in `data/` (or the project folder) with Symbol, Company Name, Industry (default `ind_nifty500list.csv`) |
| Benchmark | Index to compare against and to use for the regime filter (`NIFTY500`) |
| Backtest start / end | `YYYY-MM-DD`. End blank = latest data. Signals need ~1 year of data before the start. |
| Min. trading history | Days of history a stock needs (252 ≈ 1 year) |
| Min. share price | Skips penny stocks (₹) |
| Min. median daily traded value | Liquidity filter in ₹ crore (5 = ₹5 crore a day) |
| Traded-value lookback | Days used for the liquidity median (63 ≈ 3 months) |
| Max. annual volatility | Blank = no cap; e.g. `80%` |
| Require positive 12-1 momentum | `YES` = never buy a stock whose own trend is down, even if it ranks well |
| Require price above N-day average | Blank = off; e.g. `200` |

#### Portfolio
| Setting | Meaning |
|---|---|
| Starting capital | Default capital for the app's first run and the Next month view |
| Rebalance frequency | `M` monthly, `Q` quarterly, `W` weekly (always the last trading day of the period) |
| Order fill price | `next_open`: signal on the close, trade at the next day's open |
| Number of stocks to hold | Top N. Fewer stocks = bigger positions, which helps small accounts |
| Keep a holding while rank is within | The **hold buffer**. Example: N = 25, buffer 50 → a holding is kept until it falls below rank 50. Blank = sell as soon as it leaves the top N. **This is the biggest lever on turnover and costs.** |
| Position weighting | `equal`; `inverse_vol` = less money in volatile stocks; `score` = more money in higher scores |
| Max. weight per stock | e.g. `8%` |
| Max. stocks per sector | Blank = no cap |

#### Risk & Shorts
| Setting | Meaning |
|---|---|
| Market regime filter | `bench_200dma` = cut exposure when the benchmark is below its 200-day average; `off` = always fully invested |
| Exposure when below 200DMA | e.g. `50%` = hold half in cash |
| Volatility target | Blank = off. e.g. `20%`: scale exposure down when the strategy's recent volatility is higher than this |
| Vol-target lookback | Days (63 ≈ 3 months) |
| Interest on idle cash | e.g. `6.5%` if idle cash sits in a liquid fund |
| Risk-free rate | Used only for the Sharpe ratio |
| Number of shorts | `0` = long-only. Shorts use stock futures and need several lakhs per lot |
| Short book size | % of capital |
| Keep a short while bottom rank is within | Hold buffer for shorts. Blank = same as number of shorts. Ignored when shorts = 0 |
| Short only F&O stocks | Keep `YES`: in India an overnight short has to be a stock future |
| **Exposure** (the smallest of these, the regime filter and the vol target wins; the rest earns cash interest) | |
| Max. share of capital in stocks | Static cap, e.g. drawdown you accept ÷ strategy's worst drawdown: 20% ÷ 40% = `50%` |
| Drawdown control: max. drawdown | Blank = off. Grossman-Zhou: exposure = multiplier × cushion above (1 − D) × peak |
| Drawdown control: multiplier / min. exposure | e.g. `3` and `25%` (the floor stops it getting stuck in cash) |
| Skip exposure changes smaller than | e.g. `10%`: fewer orders when exposure only wobbles |
| Also check exposure between rebalances | `off`, `M` or `W`. With `M`, exposure is re-checked every month-end and the book is rebalanced early if it must move by the band (min. 5 points). A drawdown limit needs this with quarterly rebalancing. **Then run the app at every month-end**: on a check-only month it carries the book forward unchanged unless exposure must move |
| How to cut exposure | `trim` every holding, or `drop` the weakest names (fewer orders and DP charges) |
| **Forecasts and risk model** (only used by the risk-model weightings) | |
| Expected-return model | `calibrated` = how stocks with this score did vs Nifty in the past; `grinold` = IC × volatility × z-score |
| Shrink forecasts to | `50%` = use half the estimate. The forecast report shows how over-confident raw forecasts are |
| Forecast history / covariance lookback | Blank = all history; `252` days of returns for correlations |
| Risk aversion | Black-Litterman: higher = smaller tilts from equal weight (`10` = moderate) |
| Black-Litterman confidence, Kelly fraction, equity premium | `50%`, `50%` (of full Kelly), `6%` a year |

#### Costs (all strategies)
Your broker's charges plus statutory costs. Set **brokerage per order** to `0` for a zero-brokerage delivery
broker, or to ₹20 for a typical discount broker. **Get this right**: at small capital, fixed per-order charges
(brokerage and the ₹20 DP charge per stock sold) are the biggest drag on returns. Slippage and the market-impact
coefficient are estimates of execution quality. Leave the statutory rows (STT, stamp duty, NSE, SEBI) alone unless
the law changes.

#### Tax (all strategies)
STCG (≤ 1 year), LTCG (> 1 year), the yearly LTCG exemption, the rate on futures profits (business income:
use your slab rate) and cess. Only `python -m momentum_lab rolling` uses these, for its after-tax results.

#### Rolling Backtest
Defaults for `python -m momentum_lab rolling`: capital per window, first and last month, window length in years, months
between window starts (`1` monthly roll, `3` quarterly, `12` yearly), the cost-stress multiple, and an optional
second capital so you can see how account size changes the results (blank = skip).

#### Decision Checks
The pass/fail tests on the rolling backtest's **Verdict** sheet. Set the thresholds to your own risk tolerance
and switch each check on or off with Enabled.

| Check | Default |
|---|---|
| Beats Nifty 500 after tax & costs in at least X of the 2-year windows | 70% |
| Median after-tax alpha per year at least | 3% |
| Worst-window after-tax alpha better than | −5% |
| Beats Nifty in at least X of windows under cost stress | 60% |
| Median alpha still positive after a survivorship haircut of | 4% per year |
| Median max drawdown no worse than Nifty's by more than | 0% |
| At least X of target capital actually invested (whole-share rounding) | 80% |

#### Market Phases
Named date ranges (`YYYY-MM`) for the rolling backtest's regime breakdown. Up to 20 rows; add or rename freely.

#### Live & Data
* **Skip resizes smaller than** (no-trade band, default 25%): a position you already hold is left alone unless
  it is more than 25% away from its target size. This avoids paying fees for tiny resizes.
* **Price history start date**: where `python -m momentum_lab fetch` and the app download from.

#### Research and Sweep Signal Sets
Defaults for `python -m momentum_lab signals`: which strategy to research, the in-sample/out-of-sample split date, and
the **sweep grid**. On the *Research* sheet, type comma-separated values (e.g. `20, 30`) for each parameter
you want to try; blank = not swept. Each column of *Sweep Signal Sets* is one signal-weight combination to try.

> Every combination is one "trial". The more you try, the more likely the best one is luck. The sweep reports
> a *deflated Sharpe* to account for this. Keep the grid small.

---

## 3. The app: orders, history, research

### 3.1 Start it
Double-click **`Start Momentum Lab.bat`** (or run `python -m momentum_lab app`). Your browser opens
**http://127.0.0.1:5000**. Keep the black window open while you use the app; close it to stop.

Pick the strategy once, in the top-right box. It applies to every tab. There are three tabs:

* **Orders** (opens first): what to trade.
  1. **Preview orders**: next month's buys and sells from the latest close, sized for your capital.
     Nothing is saved. It runs by itself the first time you open a strategy. A blue box tells you whether
     the strategy actually rebalances at this month-end (quarterly strategies only rebalance in March,
     June, September and December).
  2. **Save month-end run**: the real month-end run described below. It saves the orders to the journal so
     the track record builds up.

  Below that, **Showing** switches between the preview and any saved run. Each one shows capital, invested,
  cash left, orders and cost, then **Orders to place**, **Holdings after these orders** and, for a saved run,
  **How last month's book did**. Every table has a **CSV** button. Rarely used switches (start from cash,
  paper shorts, backfill) are under **More options**.
* **History**: **Performance** (live journal vs Nifty, model backtest CAGR / worst fall / Sharpe / this year;
  charts and the live track record are under *Charts and live track record*), then **Month by month**: one
  row per month with picks return, return after costs, Nifty 500 and how much it beat Nifty. Click a month to
  list its stocks as **ADDED / KEPT / REMOVED** with weight, rank and return (removed stocks: return while
  held and since sold). The drop-down switches between *Picks as traded* and *Fresh ranking every month*.
  Data comes from the model backtest (survivorship bias applies; see section 9). The first load takes about
  20 seconds.
* **Research**: every saved analysis (rolling backtest, risk comparison, forecasts, track record, holdings
  review, signal research) with headline numbers, charts and every table. **Run an analysis** starts one in
  the background; it appears in *Saved results* when it finishes.

### 3.2 When to run
Run on the **last trading day of each rebalance period, after 4 PM IST** (once the closing prices are final).

* Rebalance `M` → the last trading day of every month.
* Rebalance `Q` → the last trading day of **March, June, September and December**.
  All four strategies are currently set to `Q`. The app warns you if the date isn't a month-end, but it will
  still rebalance on any date you give it. Running it in a non-rebalance month makes the journal trade more
  often than the strategy was designed and backtested for.

### 3.3 Month-end run, step by step (Orders tab)
1. **Strategy**: pick it in the top-right box. Hover over it to see its top N, rebalance frequency and weighting.
2. **Signal date**: the date whose closing prices are used (normally today).
3. **Capital (₹)**: leave blank to compound (last run's capital plus its P&L), or type an amount for the first
   run, or after you add or withdraw money.
4. **Update prices from Upstox first**: keep ticked. Before 4 PM it fetches only up to yesterday, because today's
   candle isn't final.
5. **Ignore futures lot sizes (paper shorts)**: only for long/short strategies. Tick it to track shorts on
   paper when your capital is too small for one futures lot.
6. **Backfill monthly runs from…**: optional. It replays the strategy's rebalances from a past date up to the run
   date so the journal has a history from day one. It **rewrites** journal runs inside that window.
7. Click **Save month-end run**. Options 5 and 6 are under **More options**. A progress bar shows the steps (about 1 to 3 minutes with a data update).

### 3.4 Reading the result
* **Numbers at the top**: capital, invested, cash left, number of orders and estimated cost, market regime
  and, for a saved run, last month's return vs Nifty 500.
* **Orders to place**: what to do tomorrow morning. Each row has the action (`BUY` / `SELL`; `SHORT` / `COVER`
  for futures), the stock, the quantity change, the shares you hold afterwards, the reference price (the
  signal-date close), the value and the estimated cost. The CSV has every column, including type and lot size.
* **Holdings after these orders**: the book you should hold, with weight, shares, value after whole-share
  rounding and 12-1 momentum. If many picks cost more than their target for one share, a yellow note lists them.
* **How last month's book did** (saved runs only): each position's return and P&L since the last run, entering
  at the next open and marked at the signal-date close.
* The live track record and the model backtest chart are in **History → Performance**.

### 3.5 Placing the orders
Place the orders at the **next trading day's open**: sells first, then buys. The app assumes you filled the
previous run's orders exactly as recommended. If you didn't (partial fills, skipped trades), the next run's
quantities will be off by that much, so check your real holdings before trading.

### 3.6 Fixing a mistake
The **Journal** panel lists past runs. Click one to view it. Only the **latest** run can be deleted, for example
if you ran with the wrong date or capital. Delete it and run again.

---

## 4. Analyses and command-line tools

The analyses can all be started from the app (**Results → Run an analysis**), so you don't need a terminal.
They can also be run from a terminal in the project folder. Every script reads the workbook first, so your
latest settings always apply; command-line options override the workbook for that run only. Results are
saved in `output/results/` and shown in the app's **Research** tab; there are no Excel reports any more.

| Goal | Command |
|---|---|
| Check and apply the settings | `python -m momentum_lab settings` |
| Recreate the workbook from configs | `python -m momentum_lab settings --rebuild` |
| Add rows for new settings, keep everything else | `python -m momentum_lab settings --upgrade` |
| Download missing price history | `python -m momentum_lab fetch` |
| Append new days only (fast) | `python -m momentum_lab fetch --update` |
| Re-download everything | `python -m momentum_lab fetch --refresh` |
| Test whether each signal ranks stocks | `python -m momentum_lab signals` |
| Parameter sweep with in/out-of-sample split | `python -m momentum_lab signals --sweep` |
| Rolling 2-year walk-forward test | `python -m momentum_lab rolling` |
| Are the forecasts calibrated? Model weights for today | `python -m momentum_lab forecast` |
| Which weighting / exposure settings keep falls within 20%? | `python -m momentum_lab risk --dd 0.20` |
| Month-by-month track record | `python -m momentum_lab track configs/quant_composite.json --from 2025-01-01 --capital 1000000` |
| Track record from your live journal | `python -m momentum_lab track configs/quant_composite.json --journal` |

What to buy and sell is in the app's **Next month** tab, and the full model backtest (equity curve, drawdown,
CAGR, Sharpe) for each strategy is in **History → Performance**.

### 4.1 `python -m momentum_lab signals`: does a signal work at all?
Run this **before** building portfolios. For each signal it builds monthly quintile portfolios and reports
returns, Q5 − Q1 spread and rank-IC. A useful signal has Q5 > Q1, quintiles rising steadily, and a positive IC.
If a signal fails this test, no portfolio tweak will save it. Shown as *Signal research · quintile scoreboard*.

With `--sweep` it tries every combination on the *Research* / *Sweep Signal Sets* sheets. It picks the best
in-sample combination, shows its out-of-sample Sharpe and the deflated-Sharpe probability (want > 0.95).
Shown as *Signal research · parameter sweep*. Use `--split 2023-01-01` to change the split date for one run.

### 4.2 `python -m momentum_lab rolling`: the deployment test
See [section 5](#5-deciding-whether-to-trade-a-strategy-for-real). Options: `--capital 50000`,
`--mode compound|fresh`, `--from 2019-01`, `--to 2026-09`, `--years 2`, `--step 1`, and specific config files.

### 4.3 `python -m momentum_lab forecast`: probability-backed forecasts
Walk-forward check of the expected-return model behind the risk-model weightings. For each score decile it
compares the **predicted** chance of beating Nifty over the next rebalance period with what **actually**
happened, and reports the rank IC. The *Today* table lists the latest picks with P(beat Nifty), expected alpha
and the weight each model would give: equal, inverse_vol, erc, erc_tilt, hrp, min_variance, mean_variance,
black_litterman and kelly. *Today - risk* shows the picks' average correlation, the effective number of
independent bets and what share of capital Kelly would invest.

### 4.4 `python -m momentum_lab risk`: settings vs your drawdown budget
Stage 1 screens every strategy × weighting (own, equal, erc_tilt, black_litterman, kelly) × exposure rule (as
is, static allocation, drawdown control at multipliers 2/3/5, drawdown control with `drop`) in one continuous
backtest. Stage 2 runs the best few per strategy through the rolling simulator at your capital, with whole
shares, costs and tax: one compounding pot and fresh capital in every overlapping window. It reports the
pot's final value, its worst fall from peak, and the share of windows that stayed within the budget.
The static rule is sized with hindsight (D ÷ the strategy's worst drawdown), so treat it as an upper bound.

### 4.5 `python -m momentum_lab track`
Replays the month-end process (the same logic as the app) in a scratch folder, so your real journal isn't
touched. It shows what was BUY / HOLD / SELL at each rebalance, the money in each position, and how the
picks did vs Nifty 500 (alpha, hit rate, Sharpe, exit timing). `--journal` uses your real `journal/<name>/`
runs instead.

---

## 5. Deciding whether to trade a strategy for real

A single backtest from 2019 to today is one path and can flatter a strategy. Before committing money, run:

```
python -m momentum_lab rolling --capital 50000        # use YOUR real capital
```

It starts the strategy from cash in **every 2-year window** (e.g. Jan 2019 to Dec 2020, Apr 2019 to Mar 2021, …).
Each window is traded the way you really would: whole shares, futures lots, the no-trade band, the full cost
stack and daily mark-to-market. Each window runs in three variants: **pre-tax**, **after-tax** (capital-gains
tax paid every 31 March and on exit) and **stress** (2× slippage and market impact, after tax).

It appears in the app under **Research** as *Rolling backtest · Rs <capital> · <mode>*.

| Table | What to look at |
|---|---|
| Read me | Method and assumptions for this run |
| **Verdict** | Each decision check: PASS / FAIL per strategy. **Start here.** |
| Windows | Every window × strategy: CAGR, alpha, Sharpe, drawdown, costs, tax, % invested |
| Alpha by window | After-tax alpha per window, side by side; shows whether the edge is fading |
| Growth | Month-end value of each strategy vs Nifty 500 over the continuous period |
| Actions / Positions / Rebalances | Every BUY, SELL, HOLD, ADD, TRIM and SKIP decision, every round trip, every rebalance |
| Regimes | Performance in each Market Phase and when Nifty is below its 200DMA |
| Survivorship | How much alpha survives each survivorship haircut |

**Always test at the capital you will actually trade.** At small sizes, fixed per-order charges and whole-share
rounding (which leaves cash idle) can turn a winning strategy into a losing one. Set **Also run at this
capital** on the *Rolling Backtest* sheet to see a larger account side by side.

---

## 6. Recipes: common changes

**Add a new strategy**
1. *Strategies* sheet: type a name in an empty slot, set Active = `YES`.
2. Fill that slot's column on *Signals* (at least one weight), *Universe & Filters*, *Portfolio* and
   *Risk & Shorts*. Blanks take defaults.
3. Save, run `python -m momentum_lab settings`, then open the app: **History → Performance** shows its
   backtest, and **Results → Run an analysis → Rolling backtest** tests it properly.

**Copy an existing strategy to try a variation**: copy its column on each of the four strategy sheets into an
empty slot's column, give it a new name on *Strategies*, then change the one setting you're testing. Compare
both with `python -m momentum_lab rolling configs/old.json configs/new.json` (or include both in the rolling
backtest on the *Strategies* sheet and run it from the app).

**Retire a strategy**: set Active = `NO`. Its config moves to `configs/inactive/` and its journal is kept.

**Switch to monthly rebalancing**: *Portfolio* → Rebalance frequency = `M`. Then run the app at every month-end.

**Reduce trading costs**: raise the hold buffer (e.g. 50 for 25 stocks), reduce the number of stocks, rebalance
quarterly, or raise the no-trade band on *Live & Data*.

**Small account (under ₹1–2 lakh)**: hold fewer stocks (10 to 15), use `equal` weighting so positions aren't
tiny, keep shorts at 0, enter your broker's real brokerage on *Costs*, and check the "% invested" column in
the rolling backtest.

**Go long/short**: *Risk & Shorts* → Number of shorts > 0. Needs F&O margin (several lakhs per lot). In the app,
tick "paper shorts" if your capital is below one lot.

---

## 7. Folder layout and output files

```
Momentum Lab Settings.xlsx   <- ALL settings (edit this)
Start Momentum Lab.bat       <- double-click to open the app
README.md                    <- this guide
pyproject.toml               <- package definition (pip install -e .)
requirements.txt             <- exact package versions
.env.example                 <- template for .env (copy, then fill in)
.env                         <- API keys (private; git-ignored; never share)

momentum_lab/                <- all the code
  __main__.py                  the command list: python -m momentum_lab <command>
  paths.py                     every file location in one place
  app.py                       the web app (command: app)
  fetch.py                     download prices from Upstox (command: fetch)
  settings.py                  read / apply / rebuild the workbook (command: settings)
  config.py signals.py backtest.py risk.py expected_returns.py metrics.py
  picks.py journal.py research.py results.py data.py     <- the engine
  analyses/                    one module per analysis command
    rolling_backtest.py          rolling           rolling 2-year deployment test
    risk_compare.py              risk              weightings x exposure rules vs a drawdown budget
    forecast_report.py           forecast          forecast calibration and today's model weights
    track_record.py              track             month-by-month track record
    holding_review.py            review            leaders vs laggards in the latest book
    circuit_review.py            circuit           lower-circuit review of the latest book
    distribution_report.py       dist              rolling 1-month return distribution
    research_signals.py          signals           signal scoreboard and parameter sweep
tests/                       <- smoke tests (pytest); run on a fresh copy without any data

data/ind_nifty500list.csv    <- the stock universe (shared)
data/raw/                    <- daily prices, one CSV per stock; _NIFTY500.csv = benchmark (downloaded)
data/fno_symbols.csv         <- F&O stocks and lot sizes, for shorts (downloaded)
configs/                     <- GENERATED from the workbook: do not edit
  <strategy>.json              one per active strategy
  inactive/                    strategies set to Active = NO
  .lab_settings.json           costs, tax, rolling, checks, phases, live, research
journal/<strategy>/          <- your live record from the app (private)
  runs.json                    every run: date, capital, regime, exposure, review
  orders_<date>.csv            orders to place after that run
  book_<date>.csv              target portfolio after that run
  review_<date>.csv            how the previous book did
  model_equity.csv / model_stats.json   model backtest shown in the app
output/results/              <- saved analysis results (one JSON each), shown in the app's Research tab
```

To keep data, journal, configs and results outside the code folder (say on another drive), set the
environment variable `MOMENTUM_LAB_HOME` to that folder; the settings workbook is then read from there too.

**Back up** `Momentum Lab Settings.xlsx` and the `journal/` folder. They are the only things you can't
regenerate. Everything in `configs/` and `output/` can be rebuilt.

### 7.1 Sharing the lab and setting it up elsewhere
The project is a git repository. `.gitignore` already keeps out everything private or re-downloadable:
`.env`, `journal/`, `output/`, `configs/`, `data/raw/`, `Portfolio Allocator.xlsx` and `backup/`. So what
you share is the code, the settings workbook and the stock list, and nothing else.

* **Share**: push to a private GitHub repository (or zip the folder from `git archive -o lab.zip HEAD`,
  which leaves out the ignored files).
* **Someone else (or a new machine)**: clone it, do section 1 (install, `python -m momentum_lab fetch`),
  then `pytest` to confirm the install works. Strategy configs are regenerated from the workbook
  on the first run.
* **Move your own history too**: copy `journal/` (and `.env` if you use it) across by hand; it is never
  committed.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| "Settings error … sheet X, cell Y" | Fix that cell and save. Until then the previous settings are still used. |
| My workbook changes don't show up | Make sure you **saved** the file. Run `python -m momentum_lab settings` to see what it read. Refresh the app page. |
| A strategy is missing from the app | Check Active = `YES` on *Strategies* and that the name has only letters, digits, `_` and `-`. |
| Some symbols failed to download | Usually a renamed or delisted stock or a network blip. Re-run `python -m momentum_lab fetch --update`. A few failures are harmless. |
| Data date is older than today | Before 4 PM IST today's candle is skipped on purpose. Run again after 4 PM. |
| "A run is already in progress" | Wait for the current run to finish (watch the progress bar). |
| "Capital too small for one futures lot per short" | Expected for small accounts with shorts. Tick "paper shorts" or set Number of shorts = 0. |
| Port 5000 already in use | Another copy of the app is running. Close the other black window. |
| `ModuleNotFoundError` | Re-run the `pip install` line from section 1.1. |
| Broke the workbook completely | `python -m momentum_lab settings --rebuild` (backs up the old copy first). |

---

## 9. Limitations: read before trusting any number

* **Survivorship bias.** The universe is *today's* Nifty 500. Stocks that crashed out of the index or were delisted
  since 2014 are missing, and some of today's members got in *because* they went up. Both effects inflate
  backtest returns, and momentum is especially exposed. Treat absolute CAGR as optimistic. Comparisons
  *between* strategies are more reliable. The rolling backtest's survivorship haircut is a rough adjustment.
* **Costs and capital size.** Results depend heavily on your brokerage, the DP charge and how much idle cash
  whole-share rounding leaves. Always test at your real capital.
* **Tax** is modelled only in `python -m momentum_lab rolling`. The other backtests are pre-tax. A quarterly or monthly
  momentum strategy realises mostly short-term gains.
* **Fills.** Backtests assume you trade at the next open plus slippage. Real fills, especially in less liquid
  names, can be worse.
* **Recent decay.** A strategy that worked from 2019 to 2023 may not work now. Look at the *Alpha by window*
  sheet for a fading edge.
* This is a research tool, not investment advice.

---

## 10. Background: how the strategy works

### 10.1 Signals (`momentum_lab/signals.py`)
| Idea | Why | Signal |
|---|---|---|
| 12-1 momentum (Jegadeesh & Titman, 1993) | Skipping the last month avoids the short-term reversal effect | `mom_12_1` |
| Risk-adjusted momentum (MSCI and NSE momentum indices) | Raw momentum fills up with the most volatile stocks; dividing by volatility picks steadier trends | `risk_adj_12_1`, `nse_momentum` |
| Residual momentum (Blitz, Huij & Martens) | Removes market beta so you don't hold only high-beta names after a rally; roughly halves crash risk | `residual_momentum` |
| Frog-in-the-pan (Da, Gurun & Warachka) | Momentum from many small up-days lasts longer than momentum from one news jump | `frog_in_pan` |
| 52-week-high proximity (George & Hwang) | Anchoring: stocks near their highs keep drifting up | `high_52w` |
| Multi-signal composite | Signals are imperfectly correlated, so a blend has a higher Sharpe than any single one | Signals sheet weights |

### 10.2 Universe hygiene (point-in-time)
Minimum history, minimum price and minimum median traded value. Momentum profits in illiquid names can't be
captured. The absolute-momentum filter never buys a top-ranked stock whose own trend is negative. Optional
volatility-cap and above-moving-average filters.

### 10.3 Portfolio construction
Hold the top N (20 to 50) rather than a decile, which is easier to trade. The **rank buffer** keeps holdings
until they fall well out of the top N, which is the main turnover control. The **sector cap** limits
single-theme blow-ups, since momentum clusters in hot sectors. Weighting is equal, inverse-volatility or
score-weighted, with a per-stock cap, or one of the risk-model weightings (`momentum_lab/risk.py`):
* **Covariance**: 1 year of daily returns, shrunk towards a single-index market model (Ledoit-Wolf style).
* **Forecasts** (`momentum_lab/expected_returns.py`): walk-forward isotonic calibration of score → P(beat
  Nifty) and → excess return, centred to strip the survivorship drift in today's index members, then shrunk.
* **erc / erc_tilt**: equal risk contribution (Maillard, Roncalli & Teiletche 2010), or risk budgets
  ∝ P(beat). **hrp**: hierarchical risk parity (López de Prado 2016). **min_variance**, **mean_variance**.
* **black_litterman**: equal-weight equilibrium plus the forecasts as views; tilts smoothly away from 1/N.
* **kelly**: fractional multi-asset Kelly `f × Σ⁻¹μ`. If the weights add up to less than 100% the rest stays
  in cash; with our backtest edge it usually wants more than 100% (leverage), which is capped.
Equal weight is hard to beat out of sample (DeMiguel, Garlappi & Uppal 2009), so a model has to beat it in the
rolling backtest, not just in one continuous run.

### 10.4 Risk management: the momentum crash problem
Momentum's weakness is sharp crashes at market reversals (2009, March to April 2020). The **regime filter** cuts
exposure when Nifty 500 is below its 200-day average. **Volatility targeting** (Barroso & Santa-Clara, 2015)
scales exposure down when the strategy's realised volatility spikes. **Drawdown control** (Grossman & Zhou
1993) shrinks exposure as the strategy approaches a chosen maximum fall from its peak, with a floor so it does
not get stuck in cash. A static **max allocation** keeps part of the capital out of the strategy altogether.
The overlays combine by taking the smallest, and the rolling backtest's *Rebalances* sheet shows which one bound.

### 10.5 Realistic execution
Signal on the close of day t, trade at the open of day t+1, never on the same bar. The full Indian delivery cost
stack: STT, stamp duty, NSE and SEBI charges, GST, the DP charge per sell, brokerage, slippage, plus
square-root market impact (`impact_coef × daily vol × √(order / daily volume)`), so bigger orders pay more.

### 10.6 Research discipline
Quantile test first. Choose parameters in-sample only and check them out-of-sample. Use the deflated Sharpe
ratio (Bailey & López de Prado) to account for how many configurations were tried. Prefer robust settings,
where neighbouring parameter values give similar results, over the single best one.
