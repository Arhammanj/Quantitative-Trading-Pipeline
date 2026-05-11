from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "log_return_1d",
    "momentum_5",
    "momentum_10",
    "ma_ratio_5",
    "ma_ratio_20",
    "ema_ratio_12",
    "ema_ratio_26",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility_10",
    "volatility_20",
    "rsi_14",
    "volume_ratio_20",
    "range_ratio",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.rolling(window=window, min_periods=window).mean()
    average_loss = losses.rolling(window=window, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def create_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_index()
    for column in ["open", "high", "low", "close", "volume"]:
        if column not in data.columns:
            raise ValueError(f"Missing required column: {column}")

    close = data["close"]
    volume = data["volume"]

    data["return_1d"] = close.pct_change()
    data["log_return_1d"] = np.log(close / close.shift(1))
    data["momentum_5"] = close.pct_change(5)
    data["momentum_10"] = close.pct_change(10)

    sma_5 = close.rolling(window=5, min_periods=5).mean()
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()

    data["ma_ratio_5"] = close / sma_5 - 1
    data["ma_ratio_20"] = close / sma_20 - 1
    data["ema_ratio_12"] = close / ema_12 - 1
    data["ema_ratio_26"] = close / ema_26 - 1

    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    data["volatility_10"] = data["return_1d"].rolling(window=10, min_periods=10).std()
    data["volatility_20"] = data["return_1d"].rolling(window=20, min_periods=20).std()
    data["rsi_14"] = _rsi(close, window=14)
    data["volume_ratio_20"] = volume / volume.rolling(window=20, min_periods=20).mean() - 1
    data["range_ratio"] = (data["high"] - data["low"]) / close

    data["next_return"] = close.pct_change().shift(-1)
    data["target"] = (data["next_return"] > 0).astype(int)

    feature_frame = data.dropna().copy()
    return feature_frame
