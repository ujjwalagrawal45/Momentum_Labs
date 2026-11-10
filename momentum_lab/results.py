"""Saved analysis results, shown in the app (python -m momentum_lab app -> Results) instead of Excel workbooks.

Each analysis script calls save(key, title, sheets, ...) once at the end; the app lists everything in
output/results/ and renders the tables and charts. One JSON file per result, overwritten on a re-run.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .paths import OUTPUT as OUT, RESULTS  # noqa: F401


def _table(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:  # dicts / sets / dates -> plain text, so the JSON is always flat
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v if v is None or isinstance(v, (str, int, float, bool)) else str(v))
    s = json.loads(df.to_json(orient="split", index=False, date_format="iso", default_handler=str))
    return {"columns": s["columns"], "rows": s["data"]}


def save(key: str, title: str, sheets: dict[str, pd.DataFrame], kind: str, params: dict | None = None,
         notes: str = "", charts: list[dict] | None = None, headline: list[dict] | None = None,
         formats: dict[str, str] | None = None) -> Path:
    """charts: [{"sheet", "x", "y": [cols], "title", "kind": "line"|"bar", "log": bool, "pct": bool}]
    headline: [{"label", "value", "fmt": "pct"|"rs"|"num"|"text"}] -- the KPI tiles above the tables.
    formats: {column: "pct"|"rs"|"num"} for columns the app can't guess (it already treats names ending
    in '%' as percentages and names with '(₹)' or ending in '_rs' as rupees)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    doc = {"key": key, "title": title, "kind": kind, "saved": datetime.now().isoformat(timespec="seconds"),
           "params": params or {}, "notes": notes, "charts": charts or [], "headline": headline or [],
           "formats": formats or {},
           "sheets": [{"name": n, **_table(df)} for n, df in sheets.items() if df is not None]}
    path = RESULTS / f"{key}.json"
    path.write_text(json.dumps(doc, default=str), encoding="utf-8")
    return path


def available() -> list[dict]:
    out = []
    for f in sorted(RESULTS.glob("*.json")) if RESULTS.exists() else []:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"key": d["key"], "title": d["title"], "kind": d["kind"], "saved": d["saved"],
                    "params": d["params"]})
    return sorted(out, key=lambda d: d["saved"], reverse=True)


def load(key: str) -> dict | None:
    f = RESULTS / f"{Path(key).name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def delete(key: str) -> None:
    (RESULTS / f"{Path(key).name}.json").unlink(missing_ok=True)
