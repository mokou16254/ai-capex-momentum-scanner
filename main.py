from __future__ import annotations

from pathlib import Path

from src.data_loader import download_intraday_price_data, download_price_data
from src.indicators import add_indicators
from src.report_generator import write_candidate_list, write_csv, write_markdown_report
from src.scanner import scan_ticker
from src.utils import ensure_output_dir, flatten_watchlist, load_yaml


PULLBACK_LABELS = {"Pullback Trigger", "Pullback Watch"}

STATUS_EMOJI = {
    "Pullback Trigger": "🟣",
    "Pullback Watch": "🟢",
    "Extended / Hold, Do Not Chase": "🟠",
    "Breakdown Risk": "🔴",
    "Neutral": "⚪",
}

CATEGORY_ORDER = [
    "Pullback Trigger",
    "Pullback Watch",
    "Extended / Hold, Do Not Chase",
    "Breakdown Risk",
    "Neutral",
]


def print_header(title: str) -> None:
    line = "=" * 100
    print(f"\n{line}\n{title}\n{line}")


def load_benchmarks() -> dict[str, object]:
    benchmarks = {}
    for ticker in ["QQQ", "SMH"]:
        try:
            benchmarks[ticker] = download_price_data(ticker)
            print(f"Loaded benchmark {ticker}")
        except Exception as exc:
            print(f"Could not load benchmark {ticker}: {exc}")
    return benchmarks


def load_watchlists() -> dict[str, list[str]]:
    watchlist = load_yaml("watchlist.yaml")
    generated_path = Path("generated_watchlist.yaml")
    if generated_path.exists():
        generated = load_yaml(generated_path)
        for category, tickers in generated.items():
            if isinstance(tickers, list):
                watchlist.setdefault(category, [])
                watchlist[category].extend(tickers)
        print(f"Loaded generated universe from {generated_path}")
    return watchlist


def main() -> None:
    config = load_yaml("config.yaml")
    watchlist = load_watchlists()
    ticker_map = flatten_watchlist(watchlist)
    output_dir = ensure_output_dir("output")

    results: list[dict] = []
    failures: list[str] = []
    min_days = int(config.get("minimum_history_days", 80))
    use_intraday = bool(config.get("use_intraday_trigger", True))
    intraday_period = str(config.get("intraday_period", "60d"))
    intraday_interval = str(config.get("intraday_interval", "4h"))

    print_header(f"AI Capex Pullback Scanner | {len(ticker_map)} tickers")
    benchmark_data = load_benchmarks()

    for index, (ticker, meta) in enumerate(sorted(ticker_map.items()), start=1):
        try:
            raw_data = download_price_data(ticker)
            if len(raw_data) < min_days:
                failures.append(f"{ticker}: not enough history ({len(raw_data)} days)")
                print(f"[{index:>3}/{len(ticker_map)}] ⚠️  {ticker:<6} skipped: not enough history")
                continue
            data = add_indicators(raw_data).dropna(subset=["EMA10", "EMA21", "EMA50", "RSI14"])
            if data.empty:
                failures.append(f"{ticker}: indicators could not be calculated")
                print(f"[{index:>3}/{len(ticker_map)}] ⚠️  {ticker:<6} skipped: indicators unavailable")
                continue

            intraday_data = None
            if use_intraday:
                try:
                    intraday_data = download_intraday_price_data(ticker, period=intraday_period, interval=intraday_interval)
                except Exception as exc:
                    failures.append(f"{ticker}: intraday data unavailable ({exc})")

            result = scan_ticker(ticker, meta, data, config, benchmark_data, intraday_data)
            results.append(result.row)
            label = result.row["setup_classification"]
            emoji = STATUS_EMOJI.get(label, "•")
            print(
                f"[{index:>3}/{len(ticker_map)}] {emoji} {ticker:<6} "
                f"{label:<30} "
                f"tech={result.row['technical_score']:>3} prio={result.row['priority_score']:>3} "
                f"raw={result.row.get('raw_priority_score', 0):>6} "
                f"RS20Q={str(result.row['rs_20d_vs_qqq']):>6} "
                f"ATR%={result.row['atr_percent']:>5} "
                f"Risk%={str(result.row.get('pullback_risk_percent')):>5} "
                f"{intraday_interval}={str(result.row.get('intraday_trigger')):<5} "
                f"RVOL={result.row['relative_volume']:>4}"
            )
        except Exception as exc:  # Keep scanning the rest of the list.
            failures.append(f"{ticker}: {exc}")
            print(f"[{index:>3}/{len(ticker_map)}] ❌ {ticker:<6} failed: {exc}")

    write_csv(results, output_dir / "pullback_scan_results.csv")
    write_markdown_report(results, output_dir / "pullback_daily_report.md")
    write_candidate_list(results, output_dir / "pullback_candidates.txt")

    print_header("Pullback Scan Summary")
    if results:
        counts: dict[str, int] = {}
        for row in results:
            label = row["setup_classification"]
            counts[label] = counts.get(label, 0) + 1
        for label in CATEGORY_ORDER:
            print(f"{STATUS_EMOJI.get(label, '•')} {label:<30} {counts.get(label, 0):>3}")

        pullback_rows = [row for row in results if row["setup_classification"] in PULLBACK_LABELS]
        print("\nTop pullback candidates:")
        if not pullback_rows:
            print("  None.")
        else:
            top_rows = sorted(pullback_rows, key=lambda row: row.get("raw_priority_score", row.get("priority_score", 0)), reverse=True)[:15]
            for row in top_rows:
                print(
                    f"  {row['ticker']:<6} prio={row['priority_score']:>3} raw={row.get('raw_priority_score', 0):>6} "
                    f"risk={str(row.get('pullback_risk_percent')):>5}% stop={str(row.get('suggested_stop')):>8} "
                    f"{row['setup_classification']:<18} {row['conclusion']}"
                )
    if failures:
        failure_path = output_dir / "failures.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"\n⚠️  Completed with {len(failures)} failures. See {failure_path}")
    print(f"\n✅ Wrote {output_dir / 'pullback_scan_results.csv'}")
    print(f"✅ Wrote {output_dir / 'pullback_daily_report.md'}")
    print(f"✅ Wrote {output_dir / 'pullback_candidates.txt'}")


if __name__ == "__main__":
    main()
