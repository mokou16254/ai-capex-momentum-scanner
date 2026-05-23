from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
import yaml

from src.utils import ensure_output_dir, load_yaml


ETF_SUFFIXES = {"ETF", "ETN", "TRUST", "FUND", "INDEX"}
LEVERAGED_HINTS = {"2X", "3X", "ULTRA", "BEAR", "BULL", "INVERSE", "DAILY"}


def _normalize_ticker(raw: object) -> str | None:
    ticker = str(raw).strip().upper()
    if not ticker or ticker in {"NAN", "--", "-"}:
        return None
    # yfinance/ETF holdings may include cash, futures, or non-US suffixes.
    if ticker.startswith(("CASH", "USD", "US DOLLAR")):
        return None
    if " " in ticker or "/" in ticker:
        return None
    return ticker.replace(".", "-")


def _looks_like_fund(ticker: str, name: str, config: dict[str, Any]) -> bool:
    if not bool(config.get("exclude_etfs_and_leveraged_products", True)):
        return False
    name_upper = name.upper()
    if any(word in name_upper for word in ETF_SUFFIXES):
        return True
    if any(word in name_upper for word in LEVERAGED_HINTS):
        return True
    # Broad fund tickers can leak into holdings lists.
    if ticker in {"SPY", "QQQ", "DIA", "IWM", "VOO", "VTI", "IVV"}:
        return True
    return False


def _extract_holdings_table(ticker: str) -> pd.DataFrame:
    fund = yf.Ticker(ticker)
    funds_data = getattr(fund, "funds_data", None)
    if funds_data is not None:
        holdings = getattr(funds_data, "top_holdings", None)
        if isinstance(holdings, pd.DataFrame) and not holdings.empty:
            return holdings.reset_index()
    raise ValueError(f"No holdings table available for {ticker}")


def _row_to_holding(row: pd.Series) -> tuple[str | None, str, float]:
    ticker = None
    name = ""
    weight = 0.0

    for key in ["symbol", "Symbol", "ticker", "Ticker", "index"]:
        if key in row and pd.notna(row[key]):
            maybe = _normalize_ticker(row[key])
            if maybe:
                ticker = maybe
                break

    for key in ["holdingName", "Holding Name", "name", "Name", "Security Name"]:
        if key in row and pd.notna(row[key]):
            name = str(row[key]).strip()
            break

    for key in ["holdingPercent", "% Assets", "weight", "Weight", "percent", "%"]:
        if key in row and pd.notna(row[key]):
            try:
                weight = float(row[key])
                if weight <= 1:
                    weight *= 100
                break
            except (TypeError, ValueError):
                continue
    return ticker, name, weight


def fetch_theme_holdings(theme_etfs: dict[str, list[str]], settings: dict[str, Any]) -> tuple[dict[str, set[str]], list[str]]:
    min_weight = float(settings.get("min_etf_weight_percent", 0.0))
    max_holdings = int(settings.get("max_holdings_per_etf", 80))
    expanded: dict[str, set[str]] = {theme: set() for theme in theme_etfs}
    failures: list[str] = []

    for theme, etfs in theme_etfs.items():
        for etf in etfs:
            try:
                holdings = _extract_holdings_table(etf)
            except Exception as exc:
                failures.append(f"{etf}: {exc}")
                continue
            added = 0
            for _, row in holdings.iterrows():
                ticker, name, weight = _row_to_holding(row)
                if ticker is None:
                    continue
                if weight < min_weight:
                    continue
                if _looks_like_fund(ticker, name, settings):
                    continue
                expanded[theme].add(ticker)
                added += 1
                if added >= max_holdings:
                    break
    return expanded, failures


def merge_manual_additions(expanded: dict[str, set[str]], manual_additions: dict[str, list[str]]) -> None:
    for theme, tickers in manual_additions.items():
        expanded.setdefault(theme, set())
        for ticker in tickers:
            normalized = _normalize_ticker(ticker)
            if normalized:
                expanded[theme].add(normalized)


def apply_exclusions(expanded: dict[str, set[str]], exclusions: list[str]) -> None:
    excluded = {_normalize_ticker(ticker) for ticker in exclusions}
    excluded.discard(None)
    for tickers in expanded.values():
        tickers.difference_update(excluded)


def write_generated_watchlist(expanded: dict[str, set[str]], output_path: str | Path) -> None:
    serializable = {
        f"generated_{theme}": sorted(tickers)
        for theme, tickers in expanded.items()
        if tickers
    }
    with open(output_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(serializable, file, sort_keys=False)


def main() -> None:
    sources = load_yaml("universe_sources.yaml")
    settings = sources.get("settings", {}) or {}
    theme_etfs = sources.get("theme_etfs", {}) or {}
    manual_additions = sources.get("manual_additions", {}) or {}
    manual_exclusions = sources.get("manual_exclusions", []) or []
    output_path = settings.get("output_path", "generated_watchlist.yaml")

    expanded, failures = fetch_theme_holdings(theme_etfs, settings)
    merge_manual_additions(expanded, manual_additions)
    apply_exclusions(expanded, manual_exclusions)
    write_generated_watchlist(expanded, output_path)

    output_dir = ensure_output_dir("output")
    flat_rows = []
    for theme, tickers in expanded.items():
        for ticker in sorted(tickers):
            flat_rows.append({"theme": theme, "ticker": ticker})
    pd.DataFrame(flat_rows).to_csv(output_dir / "generated_universe.csv", index=False)

    total = len({row["ticker"] for row in flat_rows})
    print(f"Generated {output_path} with {total} unique tickers across {len(expanded)} themes.")
    print(f"Wrote {output_dir / 'generated_universe.csv'}")
    if failures:
        failure_path = output_dir / "universe_failures.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"Completed with {len(failures)} ETF holding failures. See {failure_path}")


if __name__ == "__main__":
    main()
