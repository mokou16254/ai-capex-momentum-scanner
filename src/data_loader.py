from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def download_price_data(ticker: str, months: int = 8) -> pd.DataFrame:
    """Download daily OHLCV data for a ticker with yfinance."""
    end = datetime.today()
    start = end - timedelta(days=int(months * 31))
    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for {ticker}: {sorted(missing)}")
    return data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
