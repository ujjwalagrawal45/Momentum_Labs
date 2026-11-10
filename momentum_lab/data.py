"""Data layer: Upstox historical candles -> local cache -> aligned price panels.

Upstox's historical-candle endpoint serves split/bonus-adjusted daily OHLCV and
does not need an access token. If UPSTOX_ACCESS_TOKEN is set (env var or .env),
it is sent anyway (useful if Upstox ever gates the endpoint).
"""
from __future__ import annotations

import os
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from .paths import ENV_FILE, RAW_DIR, universe_path

API = "https://api.upstox.com/v3/historical-candle"
BENCHMARKS = {"NIFTY500": "NSE_INDEX|Nifty 500", "NIFTY50": "NSE_INDEX|Nifty 50"}


def load_env(path: str | Path = ENV_FILE) -> None:
    """Minimal .env loader (KEY=VALUE lines) so python-dotenv isn't required."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_universe(csv_path: str | Path) -> pd.DataFrame:
    u = pd.read_csv(universe_path(csv_path))
    u = u.rename(columns={"Company Name": "name", "Industry": "sector", "Symbol": "symbol",
                          "Series": "series", "ISIN Code": "isin", "Market Cap": "mcap"})
    u["instrument_key"] = "NSE_EQ|" + u["isin"]
    return u


class UpstoxClient:
    def __init__(self, token: str | None = None, max_retries: int = 8, min_interval: float = 0.35):
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        token = token or os.environ.get("UPSTOX_ACCESS_TOKEN")
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.max_retries = max_retries
        self.min_interval = min_interval  # global spacing between requests (shared across threads)
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _throttle(self) -> None:
        with self._lock:
            wait = self._next_slot - time.monotonic()
            self._next_slot = max(time.monotonic(), self._next_slot) + self.min_interval
        if wait > 0:
            time.sleep(wait)

    def _get(self, url: str) -> list:
        for attempt in range(self.max_retries):
            self._throttle()
            r = self.session.get(url, timeout=30)
            if r.status_code == 429:  # rate limited: back off up to ~1 min
                time.sleep(min(60, 2 ** attempt))
                continue
            data = r.json()
            if data.get("status") != "success":
                raise RuntimeError(f"{url}: {data.get('errors')}")
            return data["data"]["candles"]
        raise RuntimeError(f"{url}: rate limited after {self.max_retries} retries")

    def daily_candles(self, instrument_key: str, start: date, end: date) -> pd.DataFrame:
        """Fetch daily candles, chunked into <10y windows (Upstox's per-call limit)."""
        key = urllib.parse.quote(instrument_key, safe="")
        rows, chunk_end = [], end
        while chunk_end >= start:
            chunk_start = max(start, chunk_end - timedelta(days=365 * 9))
            rows += self._get(f"{API}/{key}/days/1/{chunk_end}/{chunk_start}")
            chunk_end = chunk_start - timedelta(days=1)
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "oi"])
        df["date"] = pd.to_datetime(df["ts"].str[:10])
        df = df.drop(columns=["ts", "oi"]).drop_duplicates("date").set_index("date").sort_index()
        return df


FNO_FILE = RAW_DIR.parent / "fno_symbols.csv"


def fetch_fno_list() -> pd.DataFrame:
    """Stocks with NSE stock futures (the only way to hold a short overnight in India),
    with lot sizes, from Upstox's public instrument master. Saved to data/fno_symbols.csv."""
    import gzip
    import io
    import json
    r = requests.get("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", timeout=60)
    r.raise_for_status()
    inst = json.load(gzip.open(io.BytesIO(r.content)))
    fut = pd.DataFrame([x for x in inst if x.get("segment") == "NSE_FO" and x.get("instrument_type") == "FUT"
                        and x.get("underlying_type") == "EQUITY"])
    fut = fut.sort_values("expiry").groupby("underlying_symbol").first()  # near-month contract
    out = fut[["lot_size", "trading_symbol", "instrument_key"]].rename_axis("symbol")
    out.to_csv(FNO_FILE)
    return out


def load_fno_list() -> pd.DataFrame:
    return pd.read_csv(FNO_FILE, index_col="symbol") if FNO_FILE.exists() else fetch_fno_list()


@dataclass
class Panel:
    """Wide (date x symbol) price panels plus static metadata."""
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.Series
    meta: pd.DataFrame  # indexed by symbol: name, sector, ...

    @property
    def traded_value(self) -> pd.DataFrame:
        return self.close * self.volume


def download_all(universe: pd.DataFrame, start: date, end: date | None = None,
                 workers: int = 3, refresh: bool = False, update: bool = False,
                 progress=None) -> list:
    """Download every symbol + benchmarks into data/raw/<SYMBOL>.csv.
    refresh: re-download full history.  update: append only the days missing from each cached
    file (re-downloads a symbol in full if a split/bonus changed its adjusted history).
    progress(done, total, msg) is called as work completes. Returns [(symbol, error)]."""
    end = end or date.today()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = UpstoxClient()
    say = progress or (lambda i, n, m: (i % 50 == 0 or i == n) and print(f"  {i}/{n} {m}"))
    jobs = {s: k for s, k in zip(universe["symbol"], universe["instrument_key"])}
    jobs.update({f"_{b}": k for b, k in BENCHMARKS.items()})
    if refresh or update:
        todo = jobs
    else:
        todo = {s: k for s, k in jobs.items() if not (RAW_DIR / f"{s}.csv").exists()}
    print(f"{'Updating' if update else 'Downloading'} {len(todo)} instruments...")

    def fetch(sym, key):
        path = RAW_DIR / f"{sym}.csv"
        if update and not refresh and path.exists():
            old = pd.read_csv(path, index_col="date", parse_dates=True)
            last = old.index.max().date()
            if not len(pd.bdate_range(last + timedelta(days=1), end)):  # no weekday since
                return "up to date"
            new = client.daily_candles(key, last - timedelta(days=10), end)
            overlap = old.index.intersection(new.index)
            if len(overlap) and (abs(new.loc[overlap, "close"] / old.loc[overlap, "close"] - 1) > 0.005).any():
                full = client.daily_candles(key, start, end)  # corporate action: history re-adjusted
                full.to_csv(path)
                return "re-adjusted"
            pd.concat([old[old.index < new.index.min()] if len(new) else old, new]).to_csv(path)
            return f"+{len(new.index.difference(old.index))} days"
        client.daily_candles(key, start, end).to_csv(path)
        return "downloaded"

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fetch, s, k): s for s, k in todo.items()}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                msg = f"{futs[f]}: {f.result()}"
            except Exception as e:  # keep going; report at the end
                failed.append((futs[f], str(e)[:120]))
                msg = f"{futs[f]}: FAILED"
            say(i, len(todo), msg)
    if failed:
        print(f"{len(failed)} failed:")
        for s, e in failed:
            print(f"  {s}: {e}")
    return failed


def load_panel(universe: pd.DataFrame, benchmark: str = "NIFTY500") -> Panel:
    frames = {}
    for sym in universe["symbol"]:
        p = RAW_DIR / f"{sym}.csv"
        if p.exists():
            df = pd.read_csv(p, index_col="date", parse_dates=True)
            if len(df):
                frames[sym] = df
    if not frames:
        raise FileNotFoundError("No cached data. Run: python -m momentum_lab fetch")
    bench = pd.read_csv(RAW_DIR / f"_{benchmark}.csv", index_col="date", parse_dates=True)["close"]
    # Trading calendar = benchmark days, minus partial sessions (e.g. Diwali Muhurat trading,
    # special weekend sessions) where only a handful of stocks print -- a rebalance landing
    # on one of those would see an almost empty universe.
    close = pd.DataFrame({s: d["close"] for s, d in frames.items()}).reindex(bench.index)
    n = close.notna().sum(axis=1)
    cal = n.index[n >= 0.8 * n.rolling(21, center=True, min_periods=1).median()]
    bench = bench.reindex(cal)
    fields = {f: pd.DataFrame({s: d[f] for s, d in frames.items()}).reindex(cal)
              for f in ["open", "high", "low", "close", "volume"]}
    for f in ["open", "high", "low", "close"]:  # zero prints are bad ticks -> missing
        fields[f] = fields[f].where(fields[f] > 0)
    meta = universe.set_index("symbol").loc[list(frames)]
    return Panel(**fields, benchmark=bench, meta=meta)
