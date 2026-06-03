from __future__ import annotations

from src.data_loader import download_price_data
from src.indicators import add_indicators
from src.report_generator import write_csv, write_markdown_report
from src.scanner import scan_ticker
from src.theme_rotation import analyze_theme_rotation, write_theme_rotation_csv, write_theme_rotation_report
from src.utils import ensure_output_dir, flatten_watchlist, load_yaml


STATUS_EMOJI = {
    "Pullback Setup": "🟢",
    "Breakout Watch": "🚀",
    "Extended / Hold, Do Not Chase": "🟠",
    "Breakdown Risk": "🔴",
    "Neutral": "⚪",
}

CATEGORY_ORDER = [
    "Pullback Setup",
    "Breakout Watch",
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


def main() -> None:
    config = load_yaml("config.yaml")
    watchlist = load_yaml("watchlist.yaml")
    ticker_map = flatten_watchlist(watchlist)
    output_dir = ensure_output_dir("output")

    results: list[dict] = []
    failures: list[str] = []
    min_days = int(config.get("minimum_history_days", 80))

    print_header(f"AI Capex Momentum Scanner | {len(ticker_map)} tickers")
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
            result = scan_ticker(ticker, meta, data, config, benchmark_data)
            results.append(result.row)
            label = result.row["setup_classification"]
            emoji = STATUS_EMOJI.get(label, "•")
            print(
                f"[{index:>3}/{len(ticker_map)}] {emoji} {ticker:<6} "
                f"{label:<30} "
                f"tech={result.row['technical_score']:>3} prio={result.row['priority_score']:>3} "
                f"raw={result.row.get('raw_priority_score', 0):>6} "
                f"RS5Q={str(result.row.get('rs_5d_vs_qqq')):>6} "
                f"RS20Q={str(result.row['rs_20d_vs_qqq']):>6} "
                f"ATR%={result.row['atr_percent']:>5} "
                f"RVOL={result.row['relative_volume']:>4}"
            )
        except Exception as exc:  # Keep scanning the rest of the list.
            failures.append(f"{ticker}: {exc}")
            print(f"[{index:>3}/{len(ticker_map)}] ❌ {ticker:<6} failed: {exc}")

    theme_rows = analyze_theme_rotation(results, config)

    write_csv(results, output_dir / "scan_results.csv")
    write_markdown_report(results, output_dir / "daily_report.md")
    write_theme_rotation_csv(theme_rows, output_dir / "theme_rotation.csv")
    write_theme_rotation_report(theme_rows, output_dir / "theme_rotation_report.md")

    print_header("Scan Summary")
    if results:
        counts: dict[str, int] = {}
        for row in results:
            label = row["setup_classification"]
            counts[label] = counts.get(label, 0) + 1
        for label in CATEGORY_ORDER:
            print(f"{STATUS_EMOJI.get(label, '•')} {label:<30} {counts.get(label, 0):>3}")
        print("\nTop priority names:")
        top_rows = sorted(results, key=lambda row: row.get("raw_priority_score", row.get("priority_score", 0)), reverse=True)[:10]
        for row in top_rows:
            print(
                f"  {row['ticker']:<6} prio={row['priority_score']:>3} raw={row.get('raw_priority_score', 0):>6} "
                f"tech={row['technical_score']:>3} {row['setup_classification']:<30} {row['conclusion']}"
            )
    if theme_rows:
        print("\nTop active themes:")
        for row in theme_rows[: int(config.get("theme_top_n", 8))]:
            print(
                f"  #{row['theme_rank']:<2} {row['theme']:<32} "
                f"score={row['rotation_score']:>6} {row['theme_state']} | leaders: {row['top_leaders']}"
            )
    if failures:
        failure_path = output_dir / "failures.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"\n⚠️  Completed with {len(failures)} failures. See {failure_path}")
    print(f"\n✅ Wrote {output_dir / 'scan_results.csv'}")
    print(f"✅ Wrote {output_dir / 'daily_report.md'}")
    print(f"✅ Wrote {output_dir / 'theme_rotation.csv'}")
    print(f"✅ Wrote {output_dir / 'theme_rotation_report.md'}")


if __name__ == "__main__":
    main()
