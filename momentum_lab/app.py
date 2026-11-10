"""Momentum Lab: the one local front end for everything.

    python -m momentum_lab app          (or double-click "Start Momentum Lab.bat")  ->  http://127.0.0.1:5000

  Orders   : 1) Preview next month's trades from the latest close (nothing saved), or 2) on the last trading
             day after 4 PM IST, save the month-end run (refreshes Upstox data, reviews last month, journals
             this month's orders under journal/<strategy>/). The "Showing" box switches between the preview
             and any saved run.
  History  : performance (live journal + model backtest) and the month-by-month picks; click a month to see
             which stocks were added, kept or removed and how each did.
  Research : every saved analysis with its tables and charts, and a form to re-run each one.
"""
from __future__ import annotations

import base64
import glob
import io
import json
import os
import subprocess
import sys
import threading
import traceback
import webbrowser
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request

from momentum_lab import StrategyConfig, download_all, load_env, load_panel, load_universe, picks, results, settings
from momentum_lab.backtest import decision_dates
from momentum_lab.paths import CONFIGS, ENV_FILE, RAW_DIR, ROOT
from momentum_lab.journal import (Context, delete_run, history, load_runs, model_stats, month_end, preview,
                                  strategy_dir)

IST = timezone(timedelta(hours=5, minutes=30))
app = Flask(__name__)
app.json.sort_keys = False  # keep table columns in the order the journal writes them
load_env(ENV_FILE)

JOB = {"running": False, "pct": 0, "step": "", "log": [], "error": None, "result": None}
AJOB = {"running": False, "kind": None, "title": "", "log": [], "error": None, "result": None, "started": None}
_ctx_cache: dict = {}
_ctx_lock = threading.Lock()  # the picks page and a run job may both ask for the same context
_picks_cache: dict = {}
_previews: dict = {}  # strategy -> last "next month" preview (in memory; recomputed after a restart)
_proc: dict = {}

# analysis kind -> script and the options the Results page offers (validated in analysis_argv)
ANALYSES = {
    "rolling": {"title": "Rolling backtest", "eta": "10-40 min",
                "help": "Every strategy marked 'Include in rolling backtest' through 2-year windows, after tax."},
    "risk": {"title": "Risk comparison", "eta": "20-60 min",
             "help": "Weightings x exposure rules against a largest-fall budget, at your capital."},
    "forecast": {"title": "Forecasts & risk models", "eta": "2-5 min",
                 "help": "Is the score a calibrated forecast? And today's picks under every weighting model."},
    "track": {"title": "Track record", "eta": "1-3 min",
              "help": "Month-by-month BUY / HOLD / SELL replay of one strategy vs Nifty 500."},
    "review": {"title": "Holdings review", "eta": "2-4 min",
               "help": "Latest journal book: leaders vs laggards since the book date, and whether cutting early laggards ever paid in the backtest."},
    "circuit": {"title": "Lower-circuit review", "eta": "3-6 min",
                "help": "Latest journal book: which stocks keep hitting their lower price band, and whether avoiding or selling lower-circuit stocks ever paid in the backtest."},
    "dist": {"title": "Return distribution", "eta": "1-2 min",
             "help": "The strategy's rolling 1-month returns from the backtest against a normal (bell-curve) fit: average, spread, tails, chance of a losing month."},
    "signals": {"title": "Signal scoreboard", "eta": "2-5 min",
                "help": "Does each momentum signal rank stocks? Quintile returns per signal."},
    "sweep": {"title": "Parameter sweep", "eta": "5-30 min",
              "help": "The Research sheet's grid, in-sample vs out-of-sample."},
}


def strategies() -> dict[str, Path]:
    settings.sync()  # pick up edits to the settings workbook without restarting
    return {Path(p).stem: Path(p) for p in sorted(glob.glob(str(CONFIGS / "*.json")))}


def default_strategy(strats: dict) -> str:
    return "quant_composite" if "quant_composite" in strats else next(iter(strats))


def data_date() -> str | None:
    f = RAW_DIR / "_NIFTY500.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, usecols=["date"])["date"].max()[:10]


def log(msg: str, pct: float | None = None) -> None:
    JOB["log"].append(f"{datetime.now(IST):%H:%M:%S}  {msg}")
    JOB["log"] = JOB["log"][-200:]
    JOB["step"] = msg
    if pct is not None:
        JOB["pct"] = round(pct)


def get_context(cfg: StrategyConfig) -> Context:
    key = (cfg.name, data_date(), json.dumps(cfg.__dict__, default=str, sort_keys=True))
    with _ctx_lock:
        if key not in _ctx_cache:
            _ctx_cache.clear()
            log("Loading price panel...")
            p = load_panel(load_universe(cfg.universe_csv), cfg.benchmark)
            log("Computing signals, filters and model backtest (~20s)...")
            _ctx_cache[key] = Context(p, cfg)
        return _ctx_cache[key]


def refresh_prices(cfg: StrategyConfig, until: date) -> None:
    now = datetime.now(IST)
    end = until
    if end >= now.date() and now.hour < 16:
        end = now.date() - timedelta(days=1)
        log("Before 4 PM IST: today's candle is incomplete, fetching up to yesterday.")
    log("Updating prices from Upstox...", 2)
    start = date.fromisoformat(settings.globals_()["live"]["data_start"])
    failed = download_all(load_universe(cfg.universe_csv), start, end=end, update=True,
                          progress=lambda i, n, m: (JOB.__setitem__("pct", 2 + 58 * i / n),
                                                    JOB.__setitem__("step", f"Upstox {i}/{n}: {m}")))
    log(f"Data updated to {data_date()}" + (f" ({len(failed)} symbols failed)" if failed else ""), 60)


def preview_view(r: dict) -> dict:
    book = r["book"].copy()
    cheap = (book["side"] == "LONG") & (book["qty"] == 0)
    book.loc[cheap, "note"] = "1 share costs more than its target"
    out = {k: v for k, v in r.items() if k not in ("book", "orders")}
    out.update(book=records(book), orders=records(r["orders"]), n_target_long=int((book["side"] == "LONG").sum()),
               computed=f"{datetime.now(IST):%d %b %H:%M} IST")
    return out


def job(params: dict) -> None:
    try:
        cfg = StrategyConfig.from_json(strategies()[params["strategy"]])
        mode = params.get("mode", "month_end")
        run_date = date.fromisoformat(params["date"]) if params.get("date") else datetime.now(IST).date()
        if params.get("refresh"):
            refresh_prices(cfg, run_date)
        ctx = get_context(cfg)
        JOB["pct"] = 75
        cap = float(params["capital"]) if params.get("capital") else None
        if mode == "preview":
            log("Building next month's book...", 85)
            _previews[cfg.name] = {**preview_view(preview(ctx, cap, fresh=bool(params.get("fresh")),
                                                          paper_shorts=bool(params.get("paper_shorts")))),
                                   "fresh": bool(params.get("fresh")), "paper_shorts": bool(params.get("paper_shorts"))}
            JOB["result"] = {"mode": "preview", "strategy": cfg.name}
            log("Done.", 100)
            return
        dates = [pd.Timestamp(run_date)]
        if params.get("backfill_from"):
            idx = ctx.p.close.index
            sim = idx[(idx >= pd.Timestamp(params["backfill_from"])) & (idx <= pd.Timestamp(run_date))]
            dates = sorted(set(decision_dates(sim, cfg)) | {sim[-1]})
            for r in load_runs(cfg.name):  # backfill rewrites this window
                if r["date"] >= str(dates[0].date()):
                    delete_run(cfg.name, r["date"])
        for k, d in enumerate(dates):
            log(f"Month-end run for {d.date()}...", 75 + 20 * k / len(dates))
            r = month_end(ctx, str(d.date()), capital=cap if k == 0 else None,
                          paper_shorts=bool(params.get("paper_shorts")))
            JOB["result"] = {"mode": "month_end", "strategy": cfg.name, "date": r["date"]}
        (strategy_dir(cfg.name) / "model_stats.json").write_text(json.dumps(model_stats(ctx), default=float))
        ctx.model.nav.to_frame("strategy").join(ctx.model.benchmark.rename("benchmark")).to_csv(
            strategy_dir(cfg.name) / "model_equity.csv")
        log("Done.", 100)
    except Exception as e:
        JOB["error"] = str(e)
        log("ERROR: " + str(e))
        traceback.print_exc()
    finally:
        JOB["running"] = False


def start_job(params: dict):
    if JOB["running"]:
        return jsonify({"error": "A run is already in progress"}), 409
    JOB.update(running=True, pct=0, step="Starting...", log=[], error=None, result=None)
    threading.Thread(target=job, args=(params,), daemon=True).start()
    return jsonify({"ok": True})


# ------------------------------------------------------------------------------------- analyses
def analysis_argv(kind: str, o: dict) -> list[str]:
    """Build the command line from the form; every value is checked, nothing is passed through a shell."""
    if kind not in ANALYSES:
        raise ValueError(f"Unknown analysis '{kind}'")
    argv = [sys.executable, "-u", "-m", "momentum_lab", "signals" if kind == "sweep" else kind]
    if o.get("capital"):
        cap = float(o["capital"])
        if not 1_000 <= cap <= 1e10:
            raise ValueError("Capital must be between Rs 1,000 and Rs 1,000 crore")
        if kind in ("rolling", "risk", "track"):
            argv += ["--capital", f"{cap:.0f}"]
    if kind == "rolling" and o.get("mode"):
        if o["mode"] not in ("compound", "fresh"):
            raise ValueError("Mode must be compound or fresh")
        argv += ["--mode", o["mode"]]
    if kind == "risk":
        dd = float(o.get("dd") or 20)
        if not 1 <= dd <= 90:
            raise ValueError("The fall budget must be between 1% and 90%")
        argv += ["--dd", f"{dd / 100:.4f}"]
    if kind in ("track", "review", "circuit", "dist"):
        strats = strategies()
        if o.get("strategy") not in strats:
            raise ValueError("Pick a strategy")
        argv.insert(5, str(strats[o["strategy"]]))
        if kind in ("review", "circuit", "dist"):
            pass
        elif o.get("journal"):
            argv.append("--journal")
        elif o.get("from"):
            argv += ["--from", date.fromisoformat(o["from"]).isoformat()]
    if kind == "sweep":
        argv.append("--sweep")
    return argv


def run_analysis(kind: str, argv: list[str]) -> None:
    before = {d["key"]: d["saved"] for d in results.available()}
    try:
        pkg_parent = str(Path(__file__).resolve().parent.parent)  # so `-m momentum_lab` resolves even uninstalled
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
               "PYTHONPATH": os.pathsep.join(filter(None, [pkg_parent, os.environ.get("PYTHONPATH")]))}
        p = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=env,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _proc["p"] = p
        for line in p.stdout:
            AJOB["log"].append(line.rstrip())
            AJOB["log"] = AJOB["log"][-400:]
        rc = p.wait()
        if rc != 0:
            AJOB["error"] = "Stopped." if AJOB.get("stopped") else f"Exited with code {rc}: see the log."
        new = [d for d in results.available() if before.get(d["key"]) != d["saved"]]
        AJOB["result"] = new[0]["key"] if new else None
    except Exception as e:
        AJOB["error"] = str(e)
    finally:
        AJOB["running"] = False
        _proc.pop("p", None)


def chart_png(doc: dict, i: int) -> str:
    c = doc["charts"][i]
    sheet = next(s for s in doc["sheets"] if s["name"] == c["sheet"])
    df = pd.DataFrame(sheet["rows"], columns=sheet["columns"])
    x = df[c["x"]].astype(str).str.replace(r"T00:00:00(\.000)?$", "", regex=True)
    ys = [y for y in c["y"] if y in df]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2"]
    pos = np.arange(len(df))
    if c.get("kind") == "bar":
        width = 0.8 / max(len(ys), 1)
        for k, y in enumerate(ys):
            ax.bar(pos + (k - (len(ys) - 1) / 2) * width, df[y].astype(float), width, label=y,
                   color="#9ca3af" if y == "Nifty 500" else colors[k % len(colors)])
        for k, y in enumerate(v for v in c.get("overlay", []) if v in df):  # e.g. a fitted curve over a histogram
            ax.plot(pos, df[y].astype(float), label=y, color="#dc2626", lw=2)
        ax.axhline(0, color="k", lw=0.6)
    else:
        for k, y in enumerate(ys):
            nifty = y == "Nifty 500" or "nifty" in y.lower()
            ax.plot(pos, df[y].astype(float), label=y, color="#6b7280" if nifty else colors[k % len(colors)],
                    ls="--" if nifty else "-", marker="o" if len(df) <= 24 else None, lw=1.6)
    step = max(1, len(df) // 14)
    ax.set_xticks(pos[::step], x.iloc[::step], rotation=30 if len(df) > 8 else 0, ha="right" if len(df) > 8 else "center",
                  fontsize=8)
    if c.get("log"):
        ax.set_yscale("log")
    if c.get("pct"):
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    elif not c.get("log"):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    else:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(c.get("title", ""), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return png(fig)


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def charts(name: str) -> dict:
    out = {}
    h = history(name)
    if len(h):
        fig, ax = plt.subplots(figsize=(9, 3.4))
        x = np.arange(len(h))
        ax.bar(x - 0.2, h["return"], 0.4, label="Strategy", color="#2563eb")
        ax.bar(x + 0.2, h["benchmark_return"], 0.4, label="Nifty 500", color="#9ca3af")
        ax2 = ax.twinx()
        ax2.plot(x, h["cum_strategy"], color="#1e3a8a", marker="o", label="Cumulative strategy")
        ax2.plot(x, h["cum_benchmark"], color="#4b5563", marker="o", ls="--", label="Cumulative Nifty 500")
        ax.set_xticks(x, [t[:7] for t in h["to"]])
        for a in (ax, ax2):
            a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("Live journal: monthly and cumulative return")
        l1, n1 = ax.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, n1 + n2, fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
        out["track"] = png(fig)
    f = strategy_dir(name) / "model_equity.csv"
    if f.exists():
        e = pd.read_csv(f, index_col=0, parse_dates=True)
        fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
        ax[0].plot(e["strategy"], color="#2563eb", label="Strategy (model backtest)")
        ax[0].plot(e["benchmark"], color="#9ca3af", label="Nifty 500")
        ax[0].set_yscale("log")
        ax[0].legend(fontsize=8)
        ax[0].set_title("Model backtest since " + str(e.index[0].date()) + " (growth of Rs 10 lakh, log scale)")
        ax[1].fill_between(e.index, e["strategy"] / e["strategy"].cummax() - 1, color="#dc2626", alpha=0.45)
        ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax[1].set_title("Drawdown", fontsize=9)
        for a in ax:
            a.grid(alpha=0.25)
        out["model"] = png(fig)
    return out


def clean(v):
    """NaN / inf -> None so the JSON stays valid."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def records(df: pd.DataFrame) -> list:
    return json.loads(df.to_json(orient="records", date_format="iso")) if len(df) else []


# ------------------------------------------------------------------------------------- API
@app.get("/api/state")
def state():
    strats = strategies()
    name = request.args.get("strategy") or default_strategy(strats)
    cfg = StrategyConfig.from_json(strats[name])
    runs = load_runs(cfg.name)
    nxt = runs[-1]["capital"] if runs else None
    return jsonify({
        "strategies": list(strats), "strategy": name, "data_date": data_date(),
        "today": str(datetime.now(IST).date()), "now_ist_hour": datetime.now(IST).hour,
        "runs": [{"date": r["date"], "capital": r["capital"], "n_orders": r["n_orders"],
                  "ret": (r.get("review") or {}).get("return"),
                  "bench": (r.get("review") or {}).get("benchmark_return")} for r in runs][::-1],
        "last_capital": nxt, "initial_capital": cfg.initial_capital,
        "config": {"short_n": cfg.short_n, "top_n": cfg.top_n, "rebalance": cfg.rebalance,
                   "signals": cfg.signal_weights, "regime": cfg.regime_filter, "name": cfg.name,
                   "weighting": cfg.weighting},
    })


@app.get("/api/picks")
def picks_get():
    strats = strategies()
    name = request.args.get("strategy") or default_strategy(strats)
    mode = request.args.get("mode", "traded")
    if name not in strats or mode not in ("traded", "monthly"):
        abort(400)
    ctx = get_context(StrategyConfig.from_json(strats[name]))
    key = (id(ctx), mode, str(datetime.now(IST).date()))
    if key not in _picks_cache:
        if len(_picks_cache) > 8:
            _picks_cache.clear()
        _picks_cache[key] = clean(picks.history(ctx, mode, pd.Timestamp(datetime.now(IST).date())))
    return jsonify(_picks_cache[key])


@app.post("/api/run")
def run():
    return start_job({**request.get_json(), "mode": "month_end"})


@app.post("/api/preview")
def preview_run():
    return start_job({**request.get_json(), "mode": "preview"})


@app.get("/api/preview/<strategy>")
def preview_get(strategy):
    return jsonify(_previews.get(strategy))


@app.get("/api/job")
def job_status():
    return jsonify(JOB)


@app.get("/api/run/<strategy>/<ds>")
def run_detail(strategy, ds):
    cfg = StrategyConfig.from_json(strategies()[strategy])
    d = strategy_dir(cfg.name)
    run_ = next((r for r in load_runs(cfg.name) if r["date"] == ds), None)
    if not run_:
        abort(404)
    rd = lambda f: pd.read_csv(d / f) if (d / f).exists() else pd.DataFrame()
    ms = d / "model_stats.json"
    return jsonify({"run": run_, "orders": records(rd(f"orders_{ds}.csv")), "review": records(rd(f"review_{ds}.csv")),
                    "book": records(rd(f"book_{ds}.csv")), "history": records(history(cfg.name)),
                    "model": json.loads(ms.read_text()) if ms.exists() else None, "charts": charts(cfg.name)})


@app.get("/api/performance")
def performance():
    strats = strategies()
    name = request.args.get("strategy") or default_strategy(strats)
    if name not in strats:
        abort(400)
    ms = strategy_dir(name) / "model_stats.json"
    return jsonify(clean({"history": records(history(name)), "model": json.loads(ms.read_text()) if ms.exists() else None,
                          "charts": charts(name)}))


@app.post("/api/delete")
def delete():
    p = request.get_json()
    cfg = StrategyConfig.from_json(strategies()[p["strategy"]])
    runs = load_runs(cfg.name)
    if not runs or runs[-1]["date"] != p["date"]:
        return jsonify({"error": "Only the latest run can be deleted"}), 400
    delete_run(cfg.name, p["date"])
    return jsonify({"ok": True})


@app.get("/api/results")
def results_list():
    return jsonify({"results": results.available(), "analyses": ANALYSES, "strategies": list(strategies()),
                    "rolling_capital": settings.globals_()["rolling"]["capital"]})


@app.get("/api/results/<key>")
def result_get(key):
    doc = results.load(key)
    if doc is None:
        abort(404)
    return jsonify(doc)


@app.get("/api/results/<key>/chart/<int:i>")
def result_chart(key, i):
    doc = results.load(key)
    if doc is None or i >= len(doc["charts"]):
        abort(404)
    return jsonify({"img": chart_png(doc, i)})


@app.post("/api/results/<key>/delete")
def result_delete(key):
    results.delete(key)
    return jsonify({"ok": True})


@app.post("/api/analysis/run")
def analysis_run():
    if AJOB["running"]:
        return jsonify({"error": f"{AJOB['title']} is still running"}), 409
    p = request.get_json()
    try:
        argv = analysis_argv(p.get("kind"), p.get("options") or {})
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    AJOB.update(running=True, kind=p["kind"], title=ANALYSES[p["kind"]]["title"], error=None, result=None,
                stopped=False, started=f"{datetime.now(IST):%H:%M}",
                log=[f"$ python {' '.join(Path(a).name if k == 4 else a for k, a in enumerate(argv[1:]))}"])
    threading.Thread(target=run_analysis, args=(p["kind"], argv), daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/analysis/stop")
def analysis_stop():
    if _proc.get("p"):
        AJOB["stopped"] = True
        _proc["p"].terminate()
    return jsonify({"ok": True})


@app.get("/api/analysis/job")
def analysis_job():
    return jsonify(AJOB)


@app.get("/")
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Momentum Lab</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#111827;--mute:#6b7280;--line:#e5e7eb;--accent:#2563eb;--pos:#047857;--neg:#b91c1c;--warn:#92400e;--warnbg:#fef3c7;--info:#1e3a8a;--infobg:#eff6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{background:#0f172a;color:#fff;padding:0 24px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}
header h1{font-size:18px;margin:0;padding:14px 0}
nav{display:flex;gap:2px}nav button{background:none;color:#cbd5e1;border-radius:0;padding:16px 14px;border-bottom:3px solid transparent;font-weight:500;font-size:15px}
nav button.on{color:#fff;border-bottom-color:#60a5fa}nav button:hover{color:#fff}
.hctl{margin-left:auto;display:flex;gap:12px;align-items:center;padding:8px 0}.hctl select{width:auto;min-width:200px}.hctl span{color:#94a3b8;font-size:13px}
main{max-width:1200px;margin:0 auto;padding:18px 24px}
@media(max-width:800px){main{padding:12px 16px}header{padding:0 16px}.hctl{margin-left:0}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
h2{font-size:16px;margin:0 0 10px}h3{font-size:14px;margin:16px 0 8px}
label{display:block;font-size:12px;color:var(--mute);margin:0 0 4px}
input,select{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:7px;font:inherit;background:#fff;color:var(--ink)}
.chk{display:flex;gap:8px;align-items:center;font-size:13px;margin:6px 0}.chk input{width:auto}
button{font:inherit;border:0;border-radius:7px;padding:9px 16px;cursor:pointer}
.primary{background:var(--accent);color:#fff;font-weight:600}.secondary{background:#e0e7ff;color:#1e3a8a;font-weight:600}
button:disabled{opacity:.5;cursor:wait}.small{background:#f1f5f9;color:var(--ink);padding:5px 10px;font-size:12px}.link{background:none;color:var(--neg);padding:4px 6px;font-size:12px}
.note{font-size:12px;color:var(--mute)}.warn{background:var(--warnbg);color:var(--warn);padding:8px 12px;border-radius:7px;font-size:13px;margin:8px 0}
.info{background:var(--infobg);color:var(--info);padding:10px 14px;border-radius:8px;font-size:14px;margin-bottom:14px}
.steps{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:800px){.steps{grid-template-columns:1fr}}
.step{border:1px solid var(--line);border-radius:8px;padding:12px 14px;display:flex;flex-direction:column;gap:8px}.step b{font-size:14px}
.step .row{margin-top:auto}
.row{display:flex;gap:10px;align-items:flex-end;justify-content:space-between;flex-wrap:wrap}
.common{display:flex;gap:18px;align-items:flex-end;flex-wrap:wrap;margin-bottom:14px}.common>div{min-width:220px}
details{margin-top:6px}summary{cursor:pointer;color:var(--accent);font-size:13px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:12px}.bar div{height:100%;background:var(--accent);width:0;transition:width .3s}
.log{font:11px/1.5 ui-monospace,Consolas,monospace;color:var(--mute);max-height:160px;overflow:auto;white-space:pre-wrap}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:14px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.kpi span{font-size:12px;color:var(--mute)}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:12px;flex-wrap:wrap}
.tabs button{background:none;border-radius:7px 7px 0 0;color:var(--mute);padding:8px 12px}.tabs button.on{color:var(--accent);border-bottom:2px solid var(--accent);font-weight:600}
.tw{overflow:auto;max-height:65vh}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--mute);font-size:12px;border-bottom:1px solid var(--line);padding:6px 8px;white-space:nowrap;position:sticky;top:0;background:#fff}
td{padding:6px 8px;border-bottom:1px solid #f1f5f9;white-space:nowrap}.n{text-align:right}td.wrap{white-space:normal;min-width:320px}
td small{color:var(--mute);display:block;font-size:11px;max-width:240px;overflow:hidden;text-overflow:ellipsis}
tr.hd td{font-weight:700;background:#f8fafc}tr.click{cursor:pointer}tr.click:hover td{background:#f8fafc}tr.cur td{background:#eff6ff}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600}
.BUY,.LONG,.YES,.WIN,.PASS,.ADDED{background:#dcfce7;color:#166534}.SELL,.NO,.LOSS,.FAIL,.REMOVED{background:#fee2e2;color:#991b1b}.SHORT{background:#fde68a;color:#92400e}.COVER{background:#e0e7ff;color:#3730a3}
.HOLD,.KEPT{background:#dbeafe;color:#1e40af}.ADD{background:#d1fae5;color:#065f46}.TRIM{background:#ffedd5;color:#9a3412}.SKIP{background:#f3f4f6;color:#4b5563}
img{max-width:100%;border:1px solid var(--line);border-radius:8px;margin-bottom:10px}
.empty{color:var(--mute);padding:28px;text-align:center}
.split{display:grid;grid-template-columns:260px minmax(0,1fr);gap:16px}@media(max-width:800px){.split{grid-template-columns:1fr}}
.rlist div{display:flex;justify-content:space-between;gap:8px;padding:7px 8px;border-radius:6px;cursor:pointer;font-size:13px}
.rlist div:hover,.rlist div.sel{background:#eff6ff}.rlist small{color:var(--mute);white-space:nowrap}
.tools{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap}.tools input{width:240px}
</style></head><body>
<header><h1>Momentum Lab</h1>
<nav><button data-v="orders" class="on">Orders</button><button data-v="history">History</button><button data-v="research">Research</button></nav>
<div class="hctl"><select id="strategy" title="Strategy"></select><span id="hdr"></span></div></header>
<main>

<!-- ============================ ORDERS ============================ -->
<div id="v-orders">
 <div class="card">
  <div class="common">
   <div><label>Capital (Rs)</label><input type="number" id="o-capital" step="1000"><div class="note" id="o-capnote"></div></div>
   <div><label class="chk"><input type="checkbox" id="o-refresh" checked><span>Update prices from Upstox first</span></label></div>
  </div>
  <div class="steps">
   <div class="step"><b>1 · Preview next month</b><span class="note">What to buy and sell based on the latest close. Nothing is saved.</span>
    <div class="row"><span></span><button class="secondary" id="o-preview">Preview orders</button></div></div>
   <div class="step"><b>2 · Save the month-end run</b><span class="note">On the last trading day, after 4 PM IST. Reviews last month and saves this month's orders to the journal.</span>
    <div class="row"><div style="flex:1"><label>Signal date</label><input type="date" id="o-date"></div><button class="primary" id="o-save">Save month-end run</button></div>
    <div id="o-datenote"></div></div>
  </div>
  <details><summary>More options</summary>
   <label class="chk"><input type="checkbox" id="o-fresh"><span>Preview: I hold nothing yet, so build the whole book from cash</span></label>
   <label class="chk"><input type="checkbox" id="o-paper"><span>Ignore futures lot sizes (paper shorts)</span></label>
   <label class="chk"><input type="checkbox" id="o-bf"><span>Save: backfill monthly runs starting from</span><input type="date" id="o-bffrom" style="width:170px"></label>
  </details>
  <div id="o-prog" hidden><div class="bar"><div id="o-pb"></div></div><div class="note" id="o-step"></div><details><summary>Log</summary><div class="log" id="o-log"></div></details></div>
 </div>
 <div class="row" style="margin-bottom:10px;align-items:center"><div style="min-width:280px"><label>Showing</label><select id="o-show"></select></div><span id="o-del"></span></div>
 <div id="o-main"><div class="card empty">Loading…</div></div>
</div>

<!-- ============================ HISTORY ============================ -->
<div id="v-history" hidden>
 <div id="h-perf"></div>
 <div class="card">
  <div class="row" style="align-items:center;margin-bottom:10px"><h2 style="margin:0">Month by month</h2>
   <div style="min-width:260px"><select id="h-mode"><option value="traded">Picks as traded</option><option value="monthly">Fresh ranking every month</option></select></div></div>
  <div class="note" id="h-explain" style="margin-bottom:8px"></div>
  <div id="h-months"><div class="empty">Loading the month-by-month picks (about 20 seconds the first time)…</div></div>
 </div>
 <div id="h-detail"></div>
</div>

<!-- ============================ RESEARCH ============================ -->
<div id="v-research" hidden><div class="split">
 <aside>
  <div class="card"><h2>Saved results</h2><div class="rlist" id="r-list"></div></div>
  <div class="card"><h2>Run an analysis</h2>
   <label>Analysis</label><select id="a-kind"></select><div class="note" id="a-help" style="margin:6px 0"></div>
   <div id="a-opts"></div><button class="primary" id="a-go" style="width:100%;margin-top:10px">Run</button>
   <div id="a-status" class="note" style="margin-top:8px"></div><details><summary>Log</summary><div class="log" id="a-log"></div></details>
  </div>
 </aside>
 <section id="r-main"><div class="card empty">Select a saved result, or run an analysis.</div></section>
</div></div>
</main>

<script>
const $=s=>document.querySelector(s),$$=s=>document.querySelectorAll(s);
const esc=v=>String(v??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct=v=>v==null?"–":(v>0?"+":"")+(v*100).toFixed(2)+"%";
const rs=v=>v==null?"–":"Rs "+Math.round(v).toLocaleString("en-IN");
const cls=v=>v>0?"pos":v<0?"neg":"";
const kpi=(t,b,c="")=>`<div class="kpi"><span>${t}</span><b class="${c}">${b}</b></div>`;
const post=(u,b)=>fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b||{})});
const getj=async u=>(await fetch(u)).json();
const MN=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const mlab=m=>{const[y,mo]=m.split("-");return MN[+mo-1]+" "+y};
const dlab=d=>d?new Date(d.slice(0,10)+"T00:00").toLocaleDateString("en-IN",{day:"numeric",month:"short",year:"numeric"}):"";
const REB={M:"monthly",Q:"quarterly",W:"weekly"};
function csv(name,rows,cols){cols=cols||Object.keys(rows[0]||{});const q=v=>v==null?"":/[",\n]/.test(String(v))?'"'+String(v).replace(/"/g,'""')+'"':v;
 const b=new Blob(["﻿"+[cols.join(","),...rows.map(r=>cols.map((c,i)=>q(Array.isArray(r)?r[i]:r[c])).join(","))].join("\n")],{type:"text/csv"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=name+".csv";a.click()}

/* One table helper with explicit column formats: [key, label, format]. */
const FMT={
 pct:v=>[pct(v),"n "+cls(v)], w:v=>[(Math.abs(v)*100).toFixed(1)+"%","n"], rs:v=>[Math.round(v).toLocaleString("en-IN"),"n"],
 srs:v=>[(v>0?"+":"")+Math.round(v).toLocaleString("en-IN"),"n "+cls(v)], px:v=>[v.toLocaleString("en-IN",{minimumFractionDigits:2,maximumFractionDigits:2}),"n"],
 int:v=>[Math.round(v).toLocaleString("en-IN"),"n"], sint:v=>[(v>0?"+":"")+Math.round(v),"n "+cls(v)],
 tag:v=>[`<span class="tag ${esc(v)}">${esc(v)}</span>`,""], month:v=>[mlab(v),""], date:v=>[esc(String(v).slice(0,10)),""], txt:v=>[esc(v),""]};
const stockCell=r=>`<td><b>${esc(r.symbol)}</b><small title="${esc(r.name)}">${esc(r.name)}</small></td>`;
function tbl(rows,cols,o={}){if(!rows||!rows.length)return`<div class="empty">${o.empty||"Nothing here."}</div>`;
 const num=t=>!["tag","month","date","txt","stock"].includes(t);
 return`<div class="tw"><table><thead><tr>${cols.map(([,l,t])=>`<th class="${num(t)?"n":""}">${l}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>
  `<tr class="${o.click?"click":""} ${o.cur&&o.cur(r)?"cur":""}" ${o.key?`data-k="${esc(r[o.key])}"`:""}>${cols.map(([k,,t])=>{
   if(t==="stock")return stockCell(r);const v=r[k];if(v==null||v==="")return`<td class="${num(t)?"n":""}">${t==="txt"?"":"–"}</td>`;
   const[h,c]=FMT[t](v);return`<td class="${c}">${h}</td>`}).join("")}</tr>`).join("")}</tbody></table></div>`}

/* ---------- global state ---------- */
let ST={},view="orders",strategy=null;
$$("nav button").forEach(b=>b.onclick=()=>go(b.dataset.v));
function go(v){if(!["orders","history","research"].includes(v))v="orders";view=v;history.replaceState(null,"","#"+v);
 $$("nav button").forEach(b=>b.classList.toggle("on",b.dataset.v===v));["orders","history","research"].forEach(x=>$("#v-"+x).hidden=x!==v);
 if(v==="orders")oLoad();if(v==="history")hLoad();if(v==="research")rLoad()}
async function loadState(name){ST=await getj("/api/state"+(name?"?strategy="+encodeURIComponent(name):""));strategy=ST.strategy;
 $("#strategy").innerHTML=ST.strategies.map(s=>`<option ${s===strategy?"selected":""}>${esc(s)}</option>`).join("");
 const c=ST.config;$("#strategy").title=`Top ${c.top_n} long${c.short_n?` / ${c.short_n} short`:""} · rebalances ${REB[c.rebalance]||c.rebalance} · weighting ${c.weighting}`;
 $("#hdr").textContent=`Prices to ${dlab(ST.data_date)||"—"}`}
$("#strategy").onchange=async e=>{await loadState(e.target.value);O.shown=null;H.data=null;H.perf=null;go(view)};

/* ================= ORDERS ================= */
let O={shown:null,auto:new Set()};
function oLoad(){const last=ST.last_capital||ST.initial_capital;
 $("#o-capital").value="";$("#o-capital").placeholder=Math.round(last).toLocaleString("en-IN");
 $("#o-capnote").textContent=ST.runs.length?"Blank = last run's capital (+ its P&L when saving).":"Blank = the strategy's starting capital.";
 $("#o-fresh").checked=!ST.runs.length;if(!$("#o-date").value)$("#o-date").value=ST.data_date||ST.today;dateNote();
 $("#o-show").innerHTML=`<option value="preview">Preview of next month (not saved)</option>`+ST.runs.map(r=>`<option value="${r.date}">Saved run ${r.date}${r.ret!=null?" · last month "+pct(r.ret):""}</option>`).join("");
 $("#o-show").value=O.shown||"preview";oShow()}
$("#o-show").onchange=e=>{O.shown=e.target.value;oShow()};
async function oShow(){const w=$("#o-show").value;$("#o-del").innerHTML="";
 if(w==="preview"){const p=await getj("/api/preview/"+encodeURIComponent(strategy));
  if(p)return oRender(p,null);
  if(!O.auto.has(strategy)){O.auto.add(strategy);return oRun("preview",true)}
  $("#o-main").innerHTML='<div class="card empty">Click <b>Preview orders</b> to see next month\'s trades.</div>';return}
 const r=await getj(`/api/run/${encodeURIComponent(strategy)}/${w}`);oRender(null,r);
 if(ST.runs.length&&ST.runs[0].date===w){$("#o-del").innerHTML='<button class="link" id="o-delbtn">Delete this run</button>';
  $("#o-delbtn").onclick=async()=>{if(!confirm("Delete saved run "+w+"?"))return;await post("/api/delete",{strategy,date:w});O.shown=null;await loadState(strategy);oLoad()}}}
function oRender(p,r){const R=r&&r.run,book=p?p.book:r.book,orders=p?p.orders:r.orders,cap=p?p.capital:R.capital;
 const invested=p?p.invested_rs:(R.invested_rs??book.filter(x=>x.qty>0).reduce((a,x)=>a+(x.actual_rs||0),0));
 const held=book.filter(x=>x.qty),skipped=book.filter(x=>x.side==="LONG"&&!x.qty),shorts=book.some(x=>x.side==="SHORT");
 let h="";
 if(p){const sc=p.schedule,reb=REB[sc.rebalance]||sc.rebalance;
  h+=`<div class="info">${sc.rebalances_at_month_end?`<b>Rebalance due at the ${dlab(sc.month_end)} month-end.</b> These orders are from the ${dlab(p.asof)} close; prices can move before then, so save the month-end run after that close.`
   :`<b>No rebalance at the ${dlab(sc.month_end)} month-end.</b> This strategy rebalances ${reb}; the next one is ${dlab(sc.next_rebalance)}. If you already hold the book, keep it. The orders below are what a rebalance today would do.`}</div>`}
 else{const v=R.review||{};h+=`<div class="info">Saved run of the <b>${dlab(R.date)}</b> close. Place these orders at the next open.</div>`}
 h+=`<div class="kpis">${kpi("Capital",rs(cap))}${kpi("Invested",rs(invested)+` <span class="note">${(invested/cap*100).toFixed(0)}%</span>`)}${kpi("Cash left",rs(cap-invested))}${kpi("Orders · est. cost",`${orders.length} · ${rs(p?p.est_costs:R.est_costs)}`)}${kpi("Market",esc(p?p.regime:R.regime))}`;
 if(R&&R.review)h+=kpi("Last month vs Nifty 500",`<span class="${cls(R.review.return)}">${pct(R.review.return)}</span> <span class="note">vs ${pct(R.review.benchmark_return)}</span>`);
 h+=`</div>`;
 if(skipped.length>held.length/2)h+=`<div class="warn">${skipped.length} picks are skipped because one share costs more than their target amount at this capital: ${skipped.map(x=>esc(x.symbol)).join(", ")}. That money stays in cash.</div>`;
 if(R&&R.n_short===0&&ST.config.short_n&&!R.paper_shorts)h+='<div class="warn">Capital too small for one futures lot per short, so shorts were skipped. Tick "paper shorts" under More options to track them anyway.</div>';
 const oc=[["action","Action","tag"],["symbol","Stock","stock"],["qty_change","Qty","sint"],["new_qty","Hold after","int"],["ref_price","Price","px"],["value_rs","Value (Rs)","srs"],["est_cost_rs","Est. cost","rs"]];
 h+=`<div class="card"><div class="row" style="align-items:center"><h2>Orders to place</h2><button class="small" id="o-csv1">⬇ CSV</button></div>
  ${p&&p.held_from?`<div class="note" style="margin-bottom:6px">Changes vs your saved book of ${p.held_from}.</div>`:""}${tbl(orders,oc,{empty:"No trades needed."})}</div>`;
 const bc=[["symbol","Stock","stock"],["sector","Sector","txt"],...(shorts?[["side","Side","tag"]]:[]),["weight","Weight","w"],["qty","Shares","int"],["price","Price","px"],["actual_rs","Value (Rs)","rs"],["ret_12_1","12-1 momentum","pct"]];
 h+=`<div class="card"><div class="row" style="align-items:center"><h2>Holdings after these orders (${held.length})</h2><button class="small" id="o-csv2">⬇ CSV</button></div>${tbl(held,bc)}</div>`;
 if(R&&R.review){const v=R.review;h+=`<div class="card"><h2>How last month's book did</h2><div class="note" style="margin-bottom:8px">Book of ${v.from}, marked at the ${v.to} close · long ${rs(v.pnl_long)}${v.pnl_short?" · short "+rs(v.pnl_short):""} · costs −${rs(v.costs)} · cash interest ${rs(v.cash_interest)} · net ${rs(v.net_pnl)} · winners ${pct(v.hit_rate)}</div>
  ${tbl(r.review,[["symbol","Stock","stock"],["qty","Shares","int"],["entry_price","Bought at","px"],["mark_price","Now","px"],["return","Return","pct"],["pnl_rs","P&L (Rs)","srs"]])}</div>`}
 $("#o-main").innerHTML=h;const tag=p?`${strategy}_${p.asof}_preview`:`${strategy}_${R.date}`;
 $("#o-csv1").onclick=()=>csv(tag+"_orders",orders);$("#o-csv2").onclick=()=>csv(tag+"_holdings",held)}
function dateNote(){const d=$("#o-date").value,n=[];if(!d)return;const dt=new Date(d+"T00:00");let lw=new Date(dt.getFullYear(),dt.getMonth()+1,0);
 while(lw.getDay()===0||lw.getDay()===6)lw.setDate(lw.getDate()-1);
 if(dt.toDateString()!==lw.toDateString())n.push("Not the last weekday of the month. The strategy is designed for month-end runs.");
 if(d===ST.today&&ST.now_ist_hour<16)n.push("Before 4 PM IST, so today's close isn't final yet.");
 if(ST.data_date&&d>ST.data_date&&!$("#o-refresh").checked)n.push(`Prices only go to ${ST.data_date}. Tick "Update prices".`);
 $("#o-datenote").innerHTML=n.map(x=>`<div class="warn">${x}</div>`).join("")}
$("#o-date").onchange=dateNote;$("#o-refresh").onchange=dateNote;
async function oRun(mode,auto){const body={strategy,capital:$("#o-capital").value||null,refresh:!auto&&$("#o-refresh").checked,paper_shorts:$("#o-paper").checked};
 if(mode==="preview")body.fresh=$("#o-fresh").checked;
 else{body.date=$("#o-date").value;body.backfill_from=$("#o-bf").checked?$("#o-bffrom").value:null;
  if(body.backfill_from&&!confirm("Backfill replaces saved runs from "+body.backfill_from+" onward. Continue?"))return}
 const r=await post(mode==="preview"?"/api/preview":"/api/run",body);
 if(!r.ok){$("#o-prog").hidden=false;$("#o-step").innerHTML=`<div class="warn">${esc((await r.json()).error)}</div>`;return}
 $("#o-main").innerHTML=`<div class="card empty">${mode==="preview"?"Working out next month's orders":"Running the month-end"} (about 30 seconds the first time)…</div>`;poll()}
$("#o-preview").onclick=()=>{O.shown="preview";$("#o-show").value="preview";oRun("preview")};
$("#o-save").onclick=()=>oRun("save");
async function poll(){const j=await getj("/api/job");$("#o-prog").hidden=false;
 $("#o-pb").style.width=j.pct+"%";$("#o-step").textContent=j.step;$("#o-log").textContent=j.log.join("\n");
 $("#o-preview").disabled=$("#o-save").disabled=j.running;if(j.running){setTimeout(poll,800);return}
 if(j.error){$("#o-step").innerHTML=`<div class="warn">${esc(j.error)}</div>`;$("#o-main").innerHTML=`<div class="card"><div class="warn">${esc(j.error)}</div></div>`;return}
 if(!j.result||j.result.strategy!==strategy)return;
 await loadState(strategy);O.shown=j.result.mode==="preview"?"preview":j.result.date;if(view==="orders")oLoad()}

/* ================= HISTORY ================= */
let H={data:null,perf:null,sel:0,req:0,all:false};
async function hLoad(){if(!H.perf){H.perf=await getj("/api/performance?strategy="+encodeURIComponent(strategy));hPerf()}
 if(H.data)return hMonths();const q=++H.req,mode=$("#h-mode").value;
 $("#h-months").innerHTML='<div class="empty">Loading the month-by-month picks (about 20 seconds the first time)…</div>';$("#h-detail").innerHTML="";
 const r=await fetch(`/api/picks?strategy=${encodeURIComponent(strategy)}&mode=${mode}`);if(q!==H.req)return;
 if(!r.ok){$("#h-months").innerHTML='<div class="warn">Could not build the pick history. The error is in the Momentum Lab terminal window.</div>';return}
 H.data=await r.json();H.sel=0;hMonths()}
$("#h-mode").onchange=()=>{H.data=null;hLoad()};
function hPerf(){const m=H.perf.model,hist=H.perf.history,ch=H.perf.charts;let h='<div class="card"><h2>Performance</h2>';
 const last=hist.length?hist[hist.length-1]:null;
 h+=`<div class="kpis">${last?kpi(`Live journal since ${hist[0].from}`,`<span class="${cls(last.cum_strategy)}">${pct(last.cum_strategy)}</span> <span class="note">Nifty ${pct(last.cum_benchmark)}</span>`):""}`;
 if(m)h+=kpi("Backtest CAGR",`${pct(m.cagr)} <span class="note">Nifty ${pct(m.bench_cagr)}</span>`)+kpi("Backtest worst fall",pct(m.max_dd),"neg")+kpi("Sharpe",m.sharpe.toFixed(2))+kpi("This year (model)",`<span class="${cls(m.ytd)}">${pct(m.ytd)}</span> <span class="note">Nifty ${pct(m.bench_ytd)}</span>`);
 h+=`</div>`;if(!m&&!last)h+='<div class="note">No saved runs yet. Performance appears after the first month-end run.</div>';
 if(ch.track||ch.model||hist.length)h+=`<details><summary>Charts and live track record</summary><div style="margin-top:10px">${ch.track?`<img src="${ch.track}">`:""}${ch.model?`<img src="${ch.model}">`:""}
  ${tbl(hist,[["from","From","date"],["to","To","date"],["capital_start","Capital","rs"],["net_pnl","Net P&L","srs"],["return","Return","pct"],["benchmark_return","Nifty 500","pct"],["excess","Beat by","pct"],["hit_rate","Winners","w"]],{empty:"The track record starts after the second run."})}
  ${m?`<div class="note">Backtest ${m.start} → ${m.end}, after costs. It uses today's Nifty 500 members (survivorship bias), so treat it as optimistic.</div>`:""}</div></details>`;
 $("#h-perf").innerHTML=h+'</div>'}
function hMonths(){const D=H.data,ms=D.months,reb=REB[D.rebalance]||D.rebalance,traded=$("#h-mode").value==="traded";
 $("#h-explain").innerHTML=traded?`Each month's picks as the strategy actually traded them (it rebalances ${reb}). Decided at a month-end close, bought at the next open. Click a month to see its stocks.`
  :`What the model would pick at every month-end if it rebalanced monthly. The strategy itself trades ${reb}. Returns are before costs.`;
 if(!ms.length){$("#h-months").innerHTML='<div class="empty">No picks yet: the model needs about a year of prices before its start date.</div>';return}
 const rows=ms.map((m,i)=>({...m,i,beat:m.ret_picks!=null&&m.ret_bench!=null?m.ret_picks-m.ret_bench:null,
  state:m.status==="upcoming"?"next month, to buy":m.status==="in_progress"?"this month, running":m.rebalanced?"rebalance":"",chg:m.rebalanced?`+${m.n_new} / −${m.n_exited}`:"no change"}));
 const shown=H.all?rows:rows.slice(0,Math.max(12,H.sel+1));
 $("#h-months").innerHTML=tbl(shown,[["month","Month","month"],["state","","txt"],["n","Stocks","int"],["chg","Added / removed","txt"],["ret_picks","Picks return","pct"],...(traded?[["ret_model","After costs","pct"]]:[]),["ret_bench","Nifty 500","pct"],["beat","Beat Nifty by","pct"]],{click:1,key:"i",cur:r=>r.i===H.sel})+(shown.length<rows.length?`<div class="row" style="margin-top:8px"><span class="note">Last ${shown.length} of ${rows.length} months.</span><button class="small" id="h-all">Show all months</button></div>`:"");
 const ab=$("#h-all");if(ab)ab.onclick=()=>{H.all=true;hMonths()};
 $$("#h-months [data-k]").forEach(t=>t.onclick=()=>{H.sel=+t.dataset.k;hMonths()});hDetail()}
function hDetail(){const ms=H.data.months,m=ms[H.sel],pv=ms[H.sel+1],up=m.status==="upcoming";
 const rows=[...m.holdings.map(x=>({...x,st:x.status==="NEW"?"ADDED":"KEPT",r:x.ret})),...m.exited.map(x=>({...x,st:"REMOVED",r:x.ret_while_held}))];
 let h=`<div class="card"><div class="row" style="align-items:center"><div><h2 style="margin:0">${mlab(m.month)} · ${m.n} stocks</h2><div class="note">Decided at the ${dlab(m.signal_date)} close${m.end_date?` · measured to ${dlab(m.end_date)}`:""}${pv?` · compared with ${mlab(pv.month)}`:""}</div></div><button class="small" id="h-csv">⬇ CSV</button></div>`;
 if(up)h+=`<div class="info" style="margin-top:10px">${m.rebalanced?`These are next month's stocks. For share quantities at your capital, use <b>Orders → Preview</b>.`:`No rebalance at this month-end: keep holding last month's stocks.`}</div>`;
 h+=`<div class="tools" style="margin-top:10px"><input id="h-filt" placeholder="Find a stock or sector"></div><div id="h-table"></div>
  <div class="note" style="margin-top:6px">Return = the stock's return this month${up?" (not started yet)":""}; for removed stocks, its total return while held. "Since sold" runs to the latest close.</div></div>`;
 $("#h-detail").innerHTML=h;
 const draw=()=>{const q=$("#h-filt").value.toLowerCase();$("#h-table").innerHTML=tbl(rows.filter(x=>!q||(x.symbol+" "+x.name+" "+x.sector).toLowerCase().includes(q)),
  [["st","Status","tag"],["symbol","Stock","stock"],["sector","Sector","txt"],["weight","Weight","w"],["rank","Rank","int"],["r","Return","pct"],["held_since","Held since","month"],["ret_after_exit","Since sold","pct"]])};
 $("#h-filt").oninput=draw;draw();
 $("#h-csv").onclick=()=>csv(`${strategy}_picks_${m.month}`,rows,["symbol","name","sector","side","st","weight","rank","score","mom_12_1","entry_price","ret","held_since","ret_since_added","ret_while_held","exit_price","ret_after_exit"])}

/* ================= RESEARCH ================= */
let RL={},rsel=null;
const isDate=v=>typeof v==="string"&&/^\d{4}-\d\d-\d\d[T ]00:00:00(\.000)?$/.test(v);
function kindOf(col,label,fmts){if(fmts[col])return fmts[col];const L=label||"";
 if(label!=null){if(/%\s*$/.test(L))return"pct";if(/₹|\(Rs\)/.test(L))return"rs";if(/\(x\)/.test(L))return"x"}
 if(/%\s*$/.test(col))return"pct";if(/₹|(^|_)rs$|^nav$|capital/.test(col))return"rs";
 if(/P_beat|(^|_)(return|alpha|excess|exposure|weight|hit_rate|accuracy)(_|$)/i.test(col))return"pct";return null}
function rcell(col,v,label,fmts){if(v==null||v==="")return["",""];
 if(typeof v==="number"){const t=kindOf(col,label,fmts),sg=t==="pct"&&/alpha|return|excess|pnl|CAGR|drawdown|fall/i.test(label||col)?cls(v):"";
  if(t==="pct")return[(v*100).toFixed(Math.abs(v)<0.1?2:1)+"%","n "+sg];
  if(t==="rs")return[Math.round(v).toLocaleString("en-IN"),"n"+(/pnl|alpha/i.test(label||col)?" "+cls(v):"")];
  if(t==="x")return[v.toFixed(2)+"x","n"];
  return[Math.abs(v)>=1000?Math.round(v).toLocaleString("en-IN"):Number.isInteger(v)?String(v):v.toFixed(3),"n"]}
 if(isDate(v))return[v.slice(0,10),""];
 const s=String(v),m=/^(BUY|SELL|HOLD|ADD|TRIM|SKIP|SHORT|COVER|LONG|YES|NO|WIN|LOSS|PASS|FAIL)\b/.exec(s);
 if(m&&s.length<40)return[`<span class="tag ${m[1]}">${esc(s)}</span>`,""];
 return[esc(s),s.length>60?"wrap":""]}
function rtable(sh,fmts,filt,limit){const cols=sh.columns,metric=cols[0]==="Metric"||cols[0]==="Item";
 let rows=sh.rows;if(filt){const q=filt.toLowerCase();rows=rows.filter(r=>r.some(v=>v!=null&&String(v).toLowerCase().includes(q)))}
 const shown=rows.slice(0,limit);
 const body=shown.map(r=>{const lab=metric?r[0]:null,hd=metric&&lab&&String(lab)===String(lab).toUpperCase()&&/[A-Z]/.test(lab)&&r.slice(1).every(x=>x==null);
  return`<tr class="${hd?"hd":""}">${r.map((v,i)=>{const[t,c]=rcell(cols[i],v,metric&&i>0?String(lab||""):null,fmts);return`<td class="${c}">${t}</td>`}).join("")}</tr>`}).join("");
 return{html:`<div class="tw"><table><thead><tr>${cols.map(c=>`<th>${esc(c).replaceAll("_"," ")}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`,n:rows.length,shown:shown.length}}
async function rLoad(){RL=await getj("/api/results");
 $("#r-list").innerHTML=RL.results.length?RL.results.map(r=>`<div data-k="${esc(r.key)}" class="${rsel===r.key?"sel":""}"><span>${esc(r.title)}</span><small>${r.saved.slice(5,16).replace("T"," ")}</small></div>`).join(""):'<div class="note">Nothing saved yet. Run an analysis below.</div>';
 if(!$("#a-kind").options.length){$("#a-kind").innerHTML=Object.entries(RL.analyses).map(([k,a])=>`<option value="${k}">${a.title}</option>`).join("");aOpts()}
 if(!rsel&&RL.results.length)rShow(RL.results[0].key)}
function aOpts(){const kd=$("#a-kind").value,a=RL.analyses[kd],cap=Math.round(RL.rolling_capital);
 $("#a-help").textContent=`${a.help} Takes ${a.eta}.`;
 const so=RL.strategies.map(s=>`<option ${s===strategy?"selected":""}>${esc(s)}</option>`).join("");
 const o={rolling:`<label>Starting capital (Rs)</label><input id="o-x-capital" type="number" value="${cap}"><label style="margin-top:8px">Capital between windows</label><select id="o-x-mode"><option value="compound">compound: one pot rolled through</option><option value="fresh">fresh: new capital every window</option></select>`,
  risk:`<label>Largest fall from peak you accept (%)</label><input id="o-x-dd" type="number" value="20" min="1" max="90"><label style="margin-top:8px">Capital (Rs)</label><input id="o-x-capital" type="number" value="${cap}">`,
  track:`<label>Strategy</label><select id="o-x-strategy">${so}</select><label style="margin-top:8px">Replay from</label><input id="o-x-from" type="date" value="2025-01-01"><label style="margin-top:8px">Capital (Rs)</label><input id="o-x-capital" type="number" placeholder="strategy default"><label class="chk"><input type="checkbox" id="o-x-journal"><span>Use my live journal instead of a replay</span></label>`,
  review:`<label>Strategy</label><select id="o-x-strategy">${so}</select><div class="note">Reviews that strategy's latest saved book.</div>`,
  circuit:`<label>Strategy</label><select id="o-x-strategy">${so}</select><div class="note">Reviews that strategy's latest saved book.</div>`,
  dist:`<label>Strategy</label><select id="o-x-strategy">${so}</select><div class="note">Uses that strategy's model backtest.</div>`}[kd]||"";
 $("#a-opts").innerHTML=o+`<div class="note" style="margin-top:6px">Other settings come from the settings workbook.</div>`}
$("#a-kind").onchange=aOpts;
$("#a-go").onclick=async()=>{const v=id=>{const e=$("#o-x-"+id);return e?(e.type==="checkbox"?e.checked:e.value):undefined};
 const body={kind:$("#a-kind").value,options:{capital:v("capital"),mode:v("mode"),dd:v("dd"),strategy:v("strategy"),from:v("from"),journal:v("journal")}};
 const r=await post("/api/analysis/run",body);if(!r.ok){$("#a-status").innerHTML=`<div class="warn">${esc((await r.json()).error)}</div>`;return}apoll()};
async function apoll(){const j=await getj("/api/analysis/job");$("#a-log").textContent=j.log.slice(-30).join("\n");$("#a-log").scrollTop=1e9;
 if(j.running){$("#a-go").disabled=true;$("#a-status").innerHTML=`<b>${esc(j.title)}</b> running since ${j.started}… <button class="link" id="a-stop">stop</button>`;$("#a-stop").onclick=()=>post("/api/analysis/stop");setTimeout(apoll,1500);return}
 $("#a-go").disabled=false;if(!j.kind)return;
 $("#a-status").innerHTML=j.error?`<div class="warn">${esc(j.title)}: ${esc(j.error)}</div>`:`<b>${esc(j.title)}</b> finished.`;
 if(j.result&&view==="research"){rsel=null;await rLoad();rShow(j.result)}}
$("#r-list").onclick=e=>{const d=e.target.closest("[data-k]");if(d)rShow(d.dataset.k)};
async function rShow(key){rsel=key;$$("#r-list div").forEach(e=>e.classList.toggle("sel",e.dataset.k===key));
 $("#r-main").innerHTML='<div class="card empty">Loading…</div>';const d=await getj("/api/results/"+encodeURIComponent(key));
 const fv=x=>x.fmt==="pct"?pct(x.value):x.fmt==="rs"?rs(x.value):x.fmt==="num"?Number(x.value).toFixed(3):esc(x.value);
 const params=Object.entries(d.params||{}).filter(([,v])=>v!=null&&v!=="").map(([k,v])=>`${esc(k)} ${esc(v)}`).join(" · ");
 let h=`<div class="row" style="margin-bottom:12px;align-items:center"><div><h2 style="margin:0">${esc(d.title)}</h2><span class="note">saved ${d.saved.replace("T"," ")}${params?" · "+params:""}</span></div><button class="link" id="r-del">Delete</button></div>`;
 if(d.notes)h+=`<div class="info">${esc(d.notes)}</div>`;
 if(d.headline.length)h+=`<div class="kpis">${d.headline.map(x=>kpi(esc(x.label),fv(x),x.fmt==="pct"?cls(x.value):"")).join("")}</div>`;
 if(d.charts.length)h+=`<div class="card">${d.charts.map((c,i)=>`<div id="r-ch${i}" class="note">Drawing chart: ${esc(c.title)}…</div>`).join("")}</div>`;
 h+=`<div class="card"><div class="tabs">${d.sheets.map((s,i)=>`<button data-i="${i}" class="${i?"":"on"}">${esc(s.name)} <span class="note">${s.rows.length}</span></button>`).join("")}</div>
  <div class="tools"><input id="r-filt" placeholder="Filter rows"><button class="small" id="r-csv">⬇ CSV</button><span class="note" id="r-count"></span></div><div id="r-table"></div></div>`;
 $("#r-main").innerHTML=h;let si=0,limit=500;
 const draw=()=>{const sh=d.sheets[si];if(!sh){$("#r-table").innerHTML='<div class="empty">No tables.</div>';return}const t=rtable(sh,d.formats||{},$("#r-filt").value,limit);
  $("#r-table").innerHTML=t.html+(t.n>t.shown?`<div class="row" style="margin-top:8px"><span class="note">Showing ${t.shown} of ${t.n} rows.</span><button class="small" id="r-more">Show all</button></div>`:"");
  $("#r-count").textContent=`${t.n} rows`;const mb=$("#r-more");if(mb)mb.onclick=()=>{limit=1e9;draw()}};
 $$("#r-main .tabs button").forEach(b=>b.onclick=()=>{$$("#r-main .tabs button").forEach(x=>x.classList.toggle("on",x===b));si=+b.dataset.i;limit=500;$("#r-filt").value="";draw()});
 $("#r-filt").oninput=draw;$("#r-csv").onclick=()=>{const sh=d.sheets[si];csv(`${d.key}_${sh.name}`,sh.rows,sh.columns)};
 $("#r-del").onclick=async()=>{if(!confirm("Delete this saved result? You can re-run it later."))return;await post(`/api/results/${encodeURIComponent(key)}/delete`);rsel=null;$("#r-main").innerHTML='<div class="card empty">Deleted.</div>';rLoad()};
 draw();d.charts.forEach(async(c,i)=>{const r=await getj(`/api/results/${encodeURIComponent(key)}/chart/${i}`);const el=$("#r-ch"+i);if(el)el.outerHTML=`<img src="${r.img}" alt="${esc(c.title)}">`})}

/* ---------- start ---------- */
(async()=>{await loadState();const j=await getj("/api/job");if(j.running){O.auto.add(strategy);poll()}
 const h=location.hash.slice(1);go({picks:"history",next:"orders",journal:"orders",results:"research"}[h]||h);
 const a=await getj("/api/analysis/job");if(a.running){if(view!=="research")RL=await getj("/api/results");apoll()}})();
</script></body></html>"""

def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Momentum Lab web app")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from other devices (no login!)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    a = ap.parse_args(argv)
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}")).start()
    app.run(host=a.host, port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
