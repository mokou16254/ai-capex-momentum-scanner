from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculate RSI using Wilder-style exponential smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Calculate average true range."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(length).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, RSI, ATR, volume, distance, and high/low columns."""
    data = df.copy()
    data["EMA10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
    data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["RSI14"] = calculate_rsi(data["Close"], 14)
    data["ATR14"] = calculate_atr(data, 14)
    data["VolumeMA20"] = data["Volume"].rolling(20).mean()
    data["RelativeVolume"] = data["Volume"] / data["VolumeMA20"]
    data["DistanceEMA10Pct"] = (data["Close"] / data["EMA10"] - 1) * 100
    data["DistanceEMA21Pct"] = (data["Close"] / data["EMA21"] - 1) * 100
    data["DistanceEMA50Pct"] = (data["Close"] / data["EMA50"] - 1) * 100
    data["High20"] = data["High"].rolling(20).max()
    data["High50"] = data["High"].rolling(50).max()
    data["Low20"] = data["Low"].rolling(20).min()
    data["Low50"] = data["Low"].rolling(50).min()
    data["PctChange"] = data["Close"].pct_change() * 100
    data["RangePosition"] = (data["Close"] - data["Low"]) / (data["High"] - data["Low"]).replace(0, np.nan)
    return data
