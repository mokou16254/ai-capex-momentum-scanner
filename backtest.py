from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import download_price_data
from src.indicators import add_indicators
from src.scanner import scan_ticker
from src.utils import ensure_output_dir, flatten_watchlist, load_yaml


def truncate_to_date(df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Return rows up to and including as_of_date."""
    return df[df.index <= as_of_date].copy()


def future_return(df: pd.DataFrame, signal_pos: int, horizon: int) -> float | None:
    """Return close-to-close future return over horizon trading days."""
    future_pos = signal_pos + horizon
    if signal_pos < 0 or future_pos >= len(df):
        return None
    start = float(df["Close"].iloc[signal_pos])
    end = float(df["Close"].iloc[future_pos])
    if start <= 0:
        return None
    return (end / start - 1) * 100


def evaluate_one_snapshot(days_ago: int, horizon: int) -> pd.DataFrame:
    """Evaluate scanner scores from `days_ago` trading days ago against later performance."""
    config = load_yaml("config.yaml")
    watchlist = load_yaml("watchlist.yaml")
    ticker_map = flatten_watchlist(watchlist)
    min_days = int(config.get("minimum_history_days", 80))

    benchmark_raw: dict[str, pd.DataFrame] = {}
    benchmark_indicated: dict[str, pd.DataFrame] = {}
    for benchmark in ["QQQ", "SMH"]:
        raw = download_price_data(benchmark, months=14)
        benchmark_raw[benchmark] = raw
        benchmark_indicated[benchmark] = add_indicators(raw).dropna(subset=["EMA10", "EMA21", "EMA50", "RSI14"])

    rows: list[dict] = []
    failures: list[str] = []

    for ticker, meta in sorted(ticker_map.items()):
        try:
            raw = download_price_data(ticker, months=14)
            data = add_indicators(raw).dropna(subset=["EMA10", "EMA21", "EMA50", "RSI14"])
            signal_pos = len(data) - 1 - days_ago
            if signal_pos < min_days or signal_pos < 0:
                failures.append(f"{ticker}: not enough data for days_ago={days_ago}")
                continue
            if signal_pos + horizon >= len(data):
                failures.append(f"{ticker}: not enough future bars for horizon={horizon}")
                continue

            as_of_date = data.index[signal_pos]
            signal_data = data.iloc[: signal_pos + 1].copy()
            signal_benchmarks = {
                name: truncate_to_date(bench_df, as_of_date)
                for name, bench_df in benchmark_indicated.items()
            }
            result = scan_ticker(ticker, meta, signal_data, config, signal_benchmarks).row

            ticker_future = future_return(data, signal_pos, horizon)
            qqq_data = truncate_to_date(benchmark_raw["QQQ"], data.index[signal_pos + horizon])
            smh_data = truncate_to_date(benchmark_raw["SMH"], data.index[signal_pos + horizon])
            qqq_future = future_return(qqq_data, len(truncate_to_date(benchmark_raw["QQQ"], as_of_date)) - 1, horizon)
            smh_future = future_return(smh_data, len(truncate_to_date(benchmark_raw["SMH"], as_of_date)) - 1, horizon)

            result.update(
                {
                    "as_of_date": as_of_date.strftime("%Y-%m-%d"),
                    "horizon_days": horizon,
                    "future_return_pct": round(ticker_future, 2) if ticker_future is not None else None,
                    "future_return_vs_qqq_pct": round(ticker_future - qqq_future, 2) if ticker_future is not None and qqq_future is not None else None,
                    "future_return_vs_smh_pct": round(ticker_future - smh_future, 2) if ticker_future is not None and smh_future is not None else None,
                    "positive_forward_return": ticker_future is not None and ticker_future > 0,
                    "beat_qqq": ticker_future is not None and qqq_future is not None and ticker_future > qqq_future,
                    "beat_smh": ticker_future is not None and smh_future is not None and ticker_future > smh_future,
                }
            )
            rows.append(result)
            print(
                f"{ticker:<6} {result['setup_classification']:<24} "
                f"prio={result['priority_score']:>3} fwd={result['future_return_pct']:>6}%"
            )
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
            print(f"Failed {ticker}: {exc}")

    output_dir = ensure_output_dir("output")
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["setup_classification", "priority_score"], ascending=[True, False], inplace=True)
    df.to_csv(output_dir / "backtest_results.csv", index=False)
    write_summary(df, failures, output_dir / "backtest_summary.md", days_ago, horizon)
    return df


def write_summary(df: pd.DataFrame, failures: list[str], path: Path, days_ago: int, horizon: int) -> None:
    lines = [
        "# Backtest Snapshot Summary",
        "",
        f"Signal date: {days_ago} trading days ago",
        f"Forward horizon: {horizon} trading days",
        "",
        "This is a validation snapshot, not a full trading system backtest.",
        "It checks whether scanner scores from the signal date were followed by positive/relative performance.",
        "",
    ]

    if df.empty:
        lines.append("No rows generated.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    summary = (
        df.groupby("setup_classification")
        .agg(
            count=("ticker", "count"),
            avg_priority_score=("priority_score", "mean"),
            avg_future_return_pct=("future_return_pct", "mean"),
            win_rate=("positive_forward_return", "mean"),
            beat_qqq_rate=("beat_qqq", "mean"),
            beat_smh_rate=("beat_smh", "mean"),
        )
        .reset_index()
    )

    lines.append("## Summary by setup")
    lines.append("")
    lines.append(summary.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    lines.append("## Top priority names from signal date")
    lines.append("")
    top_cols = [
        "ticker",
        "setup_classification",
        "priority_score",
        "technical_score",
        "future_return_pct",
        "future_return_vs_qqq_pct",
        "future_return_vs_smh_pct",
        "conclusion",
    ]
    lines.append(df.sort_values("priority_score", ascending=False)[top_cols].head(15).to_markdown(index=False))
    lines.append("")

    lines.append("## Best forward performers")
    lines.append("")
    lines.append(df.sort_values("future_return_pct", ascending=False)[top_cols].head(15).to_markdown(index=False))
    lines.append("")

    lines.append("## Worst forward performers")
    lines.append("")
    lines.append(df.sort_values("future_return_pct", ascending=True)[top_cols].head(15).to_markdown(index=False))
    lines.append("")

    if failures:
        lines.append("## Failures")
        lines.append("")
        for failure in failures:
            lines.append(f"- {failure}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate scanner scores from a past snapshot.")
    parser.add_argument("--days-ago", type=int, default=5, help="Signal date in trading days ago. Default: 5")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon in trading days. Default: 5")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_one_snapshot(days_ago=args.days_ago, horizon=args.horizon)
