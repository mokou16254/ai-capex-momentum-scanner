from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a dictionary.

    PyYAML's default loader treats tickers like ON as booleans in YAML 1.1.
    BaseLoader keeps scalar values as strings, which is safer for ticker lists.
    Numeric config values are converted to float/int later by the scanner.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.BaseLoader) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def ensure_output_dir(path: str | Path = "output") -> Path:
    """Create the output directory if it does not exist."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def flatten_watchlist(watchlist: dict[str, list[str]]) -> dict[str, dict[str, object]]:
    """Deduplicate tickers while preserving category labels."""
    ticker_map: dict[str, dict[str, object]] = {}
    for category, tickers in watchlist.items():
        if not isinstance(tickers, list):
            continue
        for raw_ticker in tickers:
            ticker = str(raw_ticker).strip().upper()
            if not ticker:
                continue
            if ticker not in ticker_map:
                ticker_map[ticker] = {
                    "ticker": ticker,
                    "primary_category": category,
                    "all_categories": [],
                    "in_core_watchlist": False,
                }
            ticker_map[ticker]["all_categories"].append(category)
            if category == "core_watchlist":
                ticker_map[ticker]["in_core_watchlist"] = True
    return ticker_map
