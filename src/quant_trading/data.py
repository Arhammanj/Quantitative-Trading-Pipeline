from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketDataRequest:
    symbol: str = "AAPL"
    start: str = "2018-01-01"
    end: str | None = None
    cache_dir: Path | None = None


def generate_synthetic_market_data(
    periods: int = 756,
    start: str = "2018-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    daily_returns = rng.normal(loc=0.0004, scale=0.018, size=periods)
    close = 100.0 * np.exp(np.cumsum(daily_returns))
    open_ = close * (1 + rng.normal(0.0, 0.002, size=periods))
    high = np.maximum(open_, close) * (1 + rng.random(size=periods) * 0.01)
    low = np.minimum(open_, close) * (1 - rng.random(size=periods) * 0.01)
    volume = rng.integers(1_000_000, 5_000_000, size=periods)

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=dates,
    )
    frame.index.name = "date"
    return frame


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = ["_".join(str(part) for part in column if part) for column in normalized.columns]
    normalized.columns = [str(column).lower().replace(" ", "_") for column in normalized.columns]
    return normalized


def load_market_data(request: MarketDataRequest) -> pd.DataFrame:
    cache_path: Path | None = None
    if request.cache_dir is not None:
        request.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{request.symbol}_{request.start}_{request.end or 'latest'}.csv"
        cache_path = request.cache_dir / cache_name
        if cache_path.exists():
            cached = pd.read_csv(cache_path, index_col="date", parse_dates=True)
            return _normalize_columns(cached).sort_index()

    raw = yf.download(
        request.symbol,
        start=request.start,
        end=request.end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        raise ValueError(f"No market data returned for symbol {request.symbol!r}.")

    frame = _normalize_columns(raw)
    required_columns = ["open", "high", "low", "close", "volume"]
    available_columns = [column for column in required_columns if column in frame.columns]
    if len(available_columns) < 5:
        raise ValueError("Downloaded data is missing required OHLCV columns.")

    frame = frame[available_columns].copy()
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]

    frame.index.name = "date"
    frame = frame.sort_index()
    if cache_path is not None:
        frame.to_csv(cache_path)
    return frame
