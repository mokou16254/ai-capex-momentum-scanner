from __future__ import annotations

import traceback

from src.data_loader import download_price_data
from src.indicators import add_indicators
from src.report_generator import write_csv, write_markdown_report
from src.scanner import scan_ticker
from src.utils import ensure_output_dir, flatten_watchlist, load_yaml


def main() -> None:
    config = load_yaml("config.yaml")
    watchlist = load_yaml("watchlist.yaml")
    ticker_map = flatten_watchlist(watchlist)
    output_dir = ensure_output_dir("output")

    results: list[dict] = []
    failures: list[str] = []
    min_days = int(config.get("minimum_history_days", 80))

    for ticker, meta in sorted(ticker_map.items()):
        try:
            raw_data = download_price_data(ticker)
            if len(raw_data) < min_days:
                failures.append(f"{ticker}: not enough history ({len(raw_data)} days)")
                continue
            data = add_indicators(raw_data).dropna(subset=["EMA10", "EMA21", "EMA50", "RSI14"])
            if data.empty:
                failures.append(f"{ticker}: indicators could not be calculated")
                continue
            result = scan_ticker(ticker, meta, data, config)
            results.append(result.row)
            print(f"Scanned {ticker}: {result.row['setup_classification']} score={result.row['score']}")
        except Exception as exc:  # Keep scanning the rest of the list.
            failures.append(f"{ticker}: {exc}")
            print(f"Failed {ticker}: {exc}")
            traceback.print_exc(limit=1)

    write_csv(results, output_dir / "scan_results.csv")
    write_markdown_report(results, output_dir / "daily_report.md")

    if failures:
        failure_path = output_dir / "failures.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"Completed with {len(failures)} failures. See {failure_path}")
    print(f"Wrote {output_dir / 'scan_results.csv'}")
    print(f"Wrote {output_dir / 'daily_report.md'}")


if __name__ == "__main__":
    main()
