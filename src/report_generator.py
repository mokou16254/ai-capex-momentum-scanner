from __future__ import annotations

from pathlib import Path

import pandas as pd


CATEGORY_ORDER = [
    "Pullback Setup",
    "Breakout Watch",
    "Extended / Hold, Do Not Chase",
    "Breakdown Risk",
    "Neutral",
]

CSV_COLUMNS = [
    "date",
    "ticker",
    "setup_classification",
    "rs_percentile_20d",
    "rs_percentile_60d",
    "priority_score",
    "raw_priority_score",
    "priority_rank",
    "priority_percentile",
    "category_rank",
    "category_percentile",
    "technical_score",
    "category_weight",
    "conclusion",
    "trend_state",
    "momentum_state",
    "volume_state",
    "position_state",
    "volatility_state",
    "liquidity_state",
    "in_core_watchlist",
    "primary_category",
    "close",
    "percent_change",
    "rsi14",
    "relative_volume",
    "atr_percent",
    "avg_dollar_volume_m",
    "return_20d",
    "return_60d",
    "rs_20d_vs_qqq",
    "rs_20d_vs_smh",
    "rs_60d_vs_qqq",
    "rs_60d_vs_smh",
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


def add_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add global/within-category priority ranks and RS percentile ranks."""
    data = df.copy()
    if data.empty:
        return data

    if "raw_priority_score" in data.columns:
        data["priority_rank"] = data["raw_priority_score"].rank(method="min", ascending=False).astype(int)
        data["priority_percentile"] = (data["raw_priority_score"].rank(pct=True) * 100).round(1)
        data["category_rank"] = data.groupby("primary_category")["raw_priority_score"].rank(method="min", ascending=False).astype(int)
        data["category_percentile"] = (data.groupby("primary_category")["raw_priority_score"].rank(pct=True) * 100).round(1)

    if "return_20d" in data.columns:
        data["rs_percentile_20d"] = (data["return_20d"].rank(pct=True) * 100).round(1)
    if "return_60d" in data.columns:
        data["rs_percentile_60d"] = (data["return_60d"].rank(pct=True) * 100).round(1)
    return data


def _sort_key(df: pd.DataFrame) -> list[str]:
    if "rs_percentile_20d" in df.columns and "raw_priority_score" in df.columns:
        return ["setup_classification", "rs_percentile_20d", "raw_priority_score"]
    if "raw_priority_score" in df.columns:
        return ["setup_classification", "raw_priority_score"]
    if "priority_score" in df.columns:
        return ["setup_classification", "priority_score"]
    if "score" in df.columns:
        return ["setup_classification", "score"]
    return ["setup_classification"]


def _score_column(df: pd.DataFrame) -> str:
    if "raw_priority_score" in df.columns:
        return "raw_priority_score"
    if "priority_score" in df.columns:
        return "priority_score"
    if "score" in df.columns:
        return "score"
    return "setup_classification"


def write_csv(results: list[dict], path: str | Path) -> None:
    df = pd.DataFrame(results)
    if df.empty:
        df.to_csv(path, index=False)
        return
    df = add_rank_columns(df)
    sort_columns = _sort_key(df)
    ascending = [True] + [False] * (len(sort_columns) - 1)
    df.sort_values(sort_columns, ascending=ascending, inplace=True)
    ordered_columns = [column for column in CSV_COLUMNS if column in df.columns]
    extra_columns = [column for column in df.columns if column not in ordered_columns]
    df = df[ordered_columns + extra_columns]
    df.to_csv(path, index=False)


def _format_ticker_section(row: pd.Series) -> str:
    core_tag = "core watchlist" if bool(row["in_core_watchlist"]) else "outside core"
    priority = row.get("priority_score", row.get("score", "N/A"))
    raw_priority = row.get("raw_priority_score", "N/A")
    technical = row.get("technical_score", row.get("score", "N/A"))
    rank = row.get("priority_rank", "N/A")
    category_rank = row.get("category_rank", "N/A")
    lines = [
        f"### {row['ticker']} - {row['setup_classification']}",
        f"**Conclusion:** {row['conclusion']}",
        f"RS20P: {row.get('rs_percentile_20d', 'N/A')} | RS60P: {row.get('rs_percentile_60d', 'N/A')}",
        f"Priority: {priority} raw={raw_priority} rank={rank} | Category rank: {category_rank} ({core_tag})",
        f"Technical score: {technical}",
        f"Close: ${row['close']} | Change: {row['percent_change']}% | RSI: {row['rsi14']} | RVOL: {row['relative_volume']}",
        f"20d return: {row.get('return_20d', 'N/A')}% | 60d return: {row.get('return_60d', 'N/A')}%",
        f"RS 20d vs QQQ: {row.get('rs_20d_vs_qqq', 'N/A')} | RS 20d vs SMH: {row.get('rs_20d_vs_smh', 'N/A')} | ATR%: {row.get('atr_percent', 'N/A')}",
        f"Trend: {row['trend_state']}",
        f"Momentum: {row['momentum_state']} | Volume: {row['volume_state']} | Position: {row['position_state']}",
        f"Volatility: {row.get('volatility_state', 'N/A')} | Liquidity: {row.get('liquidity_state', 'N/A')}",
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

    df = add_rank_columns(df)
    score_column = _score_column(df)
    report_date = df["date"].max()
    lines = [
        "# AI Capex Momentum Scanner - Daily Report",
        "",
        f"Date: {report_date}",
        "",
        "This report is a screening aid only. It does not place trades and is not financial advice.",
        "",
    ]

    rs_leaders = df[(df.get("rs_percentile_20d", 0) >= 90) | (df.get("rs_percentile_60d", 0) >= 90)]
    if not rs_leaders.empty:
        lines.append("## RS > 90 Leaders")
        lines.append("")
        for _, row in rs_leaders.sort_values(["rs_percentile_20d", score_column], ascending=[False, False]).head(20).iterrows():
            lines.append(_format_ticker_section(row))

    interesting = df[(~df["in_core_watchlist"]) & (df[score_column] >= 70) & (df["setup_classification"].isin(["Pullback Setup", "Breakout Watch"]))]
    if not interesting.empty:
        lines.append("## New Momentum Candidates Outside Core Watchlist")
        lines.append("")
        for _, row in interesting.sort_values(score_column, ascending=False).head(15).iterrows():
            lines.append(_format_ticker_section(row))

    for category in CATEGORY_ORDER:
        subset = df[df["setup_classification"] == category].sort_values(score_column, ascending=False)
        lines.append(f"## {category}")
        lines.append("")
        if subset.empty:
            lines.append("None.\n")
            continue
        limit = 20 if category != "Neutral" else 30
        for _, row in subset.head(limit).iterrows():
            lines.append(_format_ticker_section(row))

    Path(path).write_text("\n".join(lines), encoding="utf-8")
