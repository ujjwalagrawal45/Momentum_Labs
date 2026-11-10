"""Strategy configuration. Every knob of the strategy lives here so a strategy is fully
described by one JSON file (see configs/*.json) -- this makes runs reproducible and lets
the research tools sweep parameters."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Costs:
    """Indian equity delivery costs (verify against your broker's current tariff)."""
    brokerage_per_order: float = 20.0     # Rs flat per executed order
    brokerage_bps: float = 0.0            # alternative %-based brokerage
    stt_bps: float = 10.0                 # 0.1% on buy and sell (delivery)
    stamp_duty_buy_bps: float = 1.5       # 0.015% on buys
    exchange_bps: float = 0.297           # NSE transaction charge 0.00297%
    sebi_bps: float = 0.01                # Rs 10 / crore
    gst_rate: float = 0.18                # on brokerage + exchange + SEBI fees
    dp_charge_per_sell: float = 20.0      # depository charge per scrip sold
    slippage_bps: float = 10.0            # half-spread + timing slippage
    impact_coef: float = 0.5              # square-root impact: coef * sigma_daily * sqrt(Q / ADV)


@dataclass
class StrategyConfig:
    name: str = "momentum"
    universe_csv: str = "ind_nifty500list.csv"
    benchmark: str = "NIFTY500"
    start: str = "2016-01-01"
    end: str | None = None
    initial_capital: float = 1_000_000.0

    # --- Signal ---------------------------------------------------------------
    # weights on z-scored signals from momentum_lab.signals.SIGNALS
    signal_weights: dict = field(default_factory=lambda: {"mom_12_1": 1.0})

    # --- Universe filters (all evaluated point-in-time) -----------------------
    min_history_days: int = 252           # avoids freshly-listed names with no momentum history
    min_price: float = 20.0
    min_adv_cr: float = 5.0               # median daily traded value over adv_window, Rs crore
    adv_window: int = 63
    max_vol: float | None = None          # drop names with annualised vol above this (e.g. 0.8)
    abs_momentum: bool = True             # require positive 12-1 return (absolute/dual momentum)
    above_ma: int | None = None           # require close > N-day MA (e.g. 200)

    # --- Portfolio construction ----------------------------------------------
    rebalance: str = "M"                  # "W", "M", "Q" (last trading day of period)
    execution: str = "next_open"          # "next_open" or "next_close"
    top_n: int = 25
    buffer_rank: int | None = 50          # keep a holding while its rank <= this (cuts turnover)
    weighting: str = "equal"              # "equal", "inverse_vol", "score", or a risk-model weighting
                                          # (momentum_lab.risk.MODEL_WEIGHTINGS: erc, hrp, black_litterman, kelly, ...)
    max_weight: float = 0.10
    max_per_sector: int | None = 6        # max names per industry

    # --- Short leg (optional) ---------------------------------------------------
    # Shorts the lowest-ranked stocks with negative 12-1 momentum. In India an overnight
    # short needs stock futures, so the short universe is restricted to F&O names.
    short_n: int = 0                      # 0 = long-only
    short_exposure: float = 0.5           # short book size as a fraction of NAV
    short_buffer_rank: int | None = None  # keep a short while its bottom-rank <= this
    short_fno_only: bool = True

    # --- Risk overlays ----------------------------------------------------------
    regime_filter: str | None = "bench_200dma"  # None or "bench_200dma"
    regime_exposure: float = 0.5          # gross exposure when benchmark is below its 200DMA
    vol_target: float | None = None       # e.g. 0.18 annualised; scales exposure down only
    vol_lookback: int = 63

    # --- Forecasts & risk model (used by the risk-model weightings) ------------
    alpha_model: str = "calibrated"       # "calibrated" (isotonic score -> excess return) or "grinold" (IC x sigma x z)
    alpha_shrink: float = 0.5             # multiply forecasts by this (they are noisy)
    forecast_lookback_years: int | None = None  # None = all history (expanding window)
    cov_lookback: int = 252               # days of returns for the covariance matrix
    risk_aversion: float = 10.0           # mean-variance / Black-Litterman lambda (higher = closer to equal weight in BL)
    view_confidence: float = 0.5          # Black-Litterman: 0 = ignore forecasts, 1 = trust them fully
    kelly_fraction: float = 0.5           # fraction of full Kelly
    equity_premium: float = 0.06          # assumed yearly excess return of equities over cash (Kelly only)

    # --- Exposure: how much of the capital is at risk (smallest overlay wins) ---
    max_allocation: float = 1.0           # static cap, e.g. drawdown budget / stressed strategy drawdown
    dd_limit: float | None = None         # Grossman-Zhou drawdown control: max drawdown D, None = off
    dd_multiplier: float = 2.0            # exposure = multiplier x cushion above the (1-D) x peak floor
    dd_floor: float = 0.25                # never go below this exposure (avoids getting stuck in cash)
    exposure_band: float = 0.0            # skip exposure changes smaller than this (e.g. 0.10)
    derisk: str = "trim"                  # "trim" every holding, or "drop" the weakest names (fewer orders)
    risk_check: str | None = None         # None = exposure only re-checked at rebalances; "M" / "W" = also
                                          # monthly / weekly, rebalancing early if exposure must move >= band

    # --- Costs / cash -------------------------------------------------------------
    costs: Costs = field(default_factory=Costs)
    cash_rate: float = 0.0                # annual yield on idle cash (e.g. 0.065 for a liquid fund)
    rf_rate: float = 0.065                # risk-free rate for Sharpe/Sortino

    @classmethod
    def from_json(cls, path: str | Path) -> "StrategyConfig":
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "StrategyConfig":
        raw = dict(raw)
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"Unknown config keys: {bad}")
        if "costs" in raw:
            raw["costs"] = Costs(**raw["costs"])
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def replace(self, **kw) -> "StrategyConfig":
        d = self.to_dict()
        d.update(kw)
        return StrategyConfig.from_dict(d)
