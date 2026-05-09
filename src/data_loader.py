from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def download_price_data(ticker: str, months: int = 8) -> pd.DataFrame:
    """Download daily OHLCV data for a ticker with yfinance.

    yfinance's `end` date is exclusive. Use tomorrow as the end date so that
    the latest completed daily candle is included when Yahoo has published it.
    This matters when running the scanner after market close.
    """
    today = datetime.today()
    end = today + timedelta(days=1)
    start = today - timedelta(days=int(months * 31))
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
