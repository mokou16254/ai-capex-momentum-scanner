from __future__ import annotations

from pathlib import Path

import pandas as pd


CATEGORY_ORDER = [
    "Pullback Setup",
    "Breakout Watch",
    "Extended / Do Not Chase",
    "Breakdown Risk",
    "Neutral",
]

CSV_COLUMNS = [
    "date",
    "ticker",
    "setup_classification",
    "score",
    "conclusion",
    "trend_state",
    "momentum_state",
    "volume_state",
    "position_state",
    "in_core_watchlist",
    "primary_category",
    "close",
    "percent_change",
    "rsi14",
    "relative_volume",
    "distance_to_ema10_percent",
    "distance_to_ema21_percent",
    "above_ema10",
    "above_ema21",
    "above_ema50",
    "near_50d_high",
    "notes",
    "all_categories",
    "atr14",
    "volume",
    "ema10",
    "ema21",
    "ema50",
]


def write_csv(results: list[dict], path: str | Path) -> None:
    df = pd.DataFrame(results)
    if df.empty:
        df.to_csv(path, index=False)
        return
    df.sort_values(["setup_classification", "score"], ascending=[True, False], inplace=True)
    ordered_columns = [column for column in CSV_COLUMNS if column in df.columns]
    extra_columns = [column for column in df.columns if column not in ordered_columns]
    df = df[ordered_columns + extra_columns]
    df.to_csv(path, index=False)


def _format_ticker_section(row: pd.Series) -> str:
    core_tag = "core watchlist" if bool(row["in_core_watchlist"]) else "outside core"
    lines = [
        f"### {row['ticker']} - {row['setup_classification']}",
        f"**Conclusion:** {row['conclusion']}",
        f"Score: {row['score']} ({core_tag})",
        f"Close: ${row['close']} | Change: {row['percent_change']}% | RSI: {row['rsi14']} | RVOL: {row['relative_volume']}",
        f"Trend: {row['trend_state']}",
        f"Momentum: {row['momentum_state']} | Volume: {row['volume_state']} | Position: {row['position_state']}",
        f"Category: {row['primary_category']} | All categories: {row['all_categories']}",
        "",
        "Key notes:",
    ]
    for note in str(row["notes"]).split(" | "):
        if note.strip():
            lines.append(f"- {note.strip()}")
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(results: list[dict], path: str | Path) -> None:
    df = pd.DataFrame(results)
    if df.empty:
        Path(path).write_text("# AI Capex Momentum Scanner - Daily Report\n\nNo results generated.\n", encoding="utf-8")
        return

    report_date = df["date"].max()
    lines = [
        "# AI Capex Momentum Scanner - Daily Report",
        "",
        f"Date: {report_date}",
        "",
        "This report is a screening aid only. It does not place trades and is not financial advice.",
        "",
    ]

    interesting = df[(~df["in_core_watchlist"]) & (df["score"] >= 70) & (df["setup_classification"].isin(["Pullback Setup", "Breakout Watch"]))]
    if not interesting.empty:
        lines.append("## New Momentum Candidates Outside Core Watchlist")
        lines.append("")
        for _, row in interesting.sort_values("score", ascending=False).head(15).iterrows():
            lines.append(_format_ticker_section(row))

    for category in CATEGORY_ORDER:
        subset = df[df["setup_classification"] == category].sort_values("score", ascending=False)
        lines.append(f"## {category}")
        lines.append("")
        if subset.empty:
            lines.append("None.\n")
            continue
        limit = 20 if category != "Neutral" else 30
        for _, row in subset.head(limit).iterrows():
            lines.append(_format_ticker_section(row))

    Path(path).write_text("\n".join(lines), encoding="utf-8")
