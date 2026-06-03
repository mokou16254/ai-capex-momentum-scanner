from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


THEME_COLUMNS = [
    "date",
    "theme",
    "rotation_score",
    "theme_rank",
    "theme_state",
    "member_count",
    "median_return_5d",
    "median_return_20d",
    "median_rs_5d_vs_qqq",
    "median_rs_20d_vs_qqq",
    "median_relative_volume",
    "median_priority_score",
    "percent_positive_5d",
    "percent_above_ema10",
    "percent_above_ema21",
    "percent_above_ema50",
    "percent_near_50d_high",
    "pullback_count",
    "breakout_count",
    "extended_count",
    "breakdown_count",
    "top_leaders",
    "buyable_names",
    "extended_names",
]


SETUP_BUYABLE = {"Pullback Setup", "Breakout Watch"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _split_categories(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return round(float(numeric.median()), 2)


def _pct_true(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return round(sum(_as_bool(value) for value in series) / len(series) * 100, 1)


def _pct_positive(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return round(float((numeric > 0).mean() * 100), 1)


def _theme_state(row: dict[str, Any]) -> str:
    if row["breakdown_count"] >= max(2, row["member_count"] // 3):
        return "Weak / breakdown risk"
    if row["extended_count"] >= max(2, row["member_count"] // 2):
        return "Strong but chase risk"
    if row["median_rs_5d_vs_qqq"] > 0 and row["median_relative_volume"] >= 1.15 and row["percent_positive_5d"] >= 55:
        return "Early rotation / active inflow"
    if row["percent_above_ema21"] >= 65 and row["median_rs_20d_vs_qqq"] > 0:
        return "Leadership trend"
    return "Mixed / watch only"


def _leader_score(row: pd.Series) -> float:
    score = _as_float(row.get("raw_priority_score", row.get("priority_score", 0)))
    score += max(_as_float(row.get("rs_5d_vs_qqq", 0)), 0) * 0.8
    score += max(_as_float(row.get("rs_20d_vs_qqq", 0)), 0) * 0.4
    score += max(_as_float(row.get("relative_volume", 0)) - 1, 0) * 6
    if _as_bool(row.get("near_50d_high", False)):
        score += 5
    if row.get("setup_classification") in SETUP_BUYABLE:
        score += 6
    if row.get("setup_classification") == "Extended / Hold, Do Not Chase":
        score -= 4
    if row.get("setup_classification") == "Breakdown Risk":
        score -= 12
    return round(score, 2)


def _names(df: pd.DataFrame, limit: int = 5) -> str:
    if df.empty:
        return ""
    data = df.copy()
    data["leader_score"] = data.apply(_leader_score, axis=1)
    return ", ".join(data.sort_values("leader_score", ascending=False)["ticker"].head(limit).astype(str))


def _explode_theme_rows(results: list[dict], excluded_categories: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        categories = _split_categories(result.get("all_categories")) or [str(result.get("primary_category", "")).strip()]
        for category in categories:
            if not category or category in excluded_categories:
                continue
            row = dict(result)
            row["theme"] = category
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_theme_rotation(results: list[dict], config: dict[str, Any]) -> list[dict]:
    """Aggregate ticker scan rows into AI-capex theme rotation scores."""
    excluded = set(config.get("theme_rotation_exclude_categories", ["core_watchlist"]) or [])
    min_members = int(config.get("theme_min_members", 3))
    data = _explode_theme_rows(results, excluded)
    if data.empty:
        return []

    theme_rows: list[dict[str, Any]] = []
    for theme, subset in data.groupby("theme"):
        if len(subset) < min_members:
            continue

        setup_counts = subset["setup_classification"].value_counts().to_dict()
        row = {
            "date": subset["date"].max(),
            "theme": theme,
            "member_count": int(len(subset)),
            "median_return_5d": _median(subset.get("return_5d", pd.Series(dtype=float))),
            "median_return_20d": _median(subset.get("return_20d", pd.Series(dtype=float))),
            "median_rs_5d_vs_qqq": _median(subset.get("rs_5d_vs_qqq", pd.Series(dtype=float))),
            "median_rs_20d_vs_qqq": _median(subset.get("rs_20d_vs_qqq", pd.Series(dtype=float))),
            "median_relative_volume": _median(subset.get("relative_volume", pd.Series(dtype=float))),
            "median_priority_score": _median(subset.get("raw_priority_score", subset.get("priority_score", pd.Series(dtype=float)))),
            "percent_positive_5d": _pct_positive(subset.get("return_5d", pd.Series(dtype=float))),
            "percent_above_ema10": _pct_true(subset.get("above_ema10", pd.Series(dtype=bool))),
            "percent_above_ema21": _pct_true(subset.get("above_ema21", pd.Series(dtype=bool))),
            "percent_above_ema50": _pct_true(subset.get("above_ema50", pd.Series(dtype=bool))),
            "percent_near_50d_high": _pct_true(subset.get("near_50d_high", pd.Series(dtype=bool))),
            "pullback_count": int(setup_counts.get("Pullback Setup", 0)),
            "breakout_count": int(setup_counts.get("Breakout Watch", 0)),
            "extended_count": int(setup_counts.get("Extended / Hold, Do Not Chase", 0)),
            "breakdown_count": int(setup_counts.get("Breakdown Risk", 0)),
        }
        row["rotation_score"] = round(
            row["median_rs_5d_vs_qqq"] * 1.4
            + row["median_rs_20d_vs_qqq"] * 0.8
            + row["median_return_5d"] * 0.7
            + (row["median_relative_volume"] - 1.0) * 10
            + row["percent_positive_5d"] * 0.20
            + row["percent_above_ema10"] * 0.25
            + row["percent_above_ema21"] * 0.20
            + row["percent_near_50d_high"] * 0.15
            + row["breakout_count"] * 3
            + row["pullback_count"] * 2
            - row["extended_count"] * 1.5
            - row["breakdown_count"] * 4,
            2,
        )
        row["theme_state"] = _theme_state(row)
        row["top_leaders"] = _names(subset, limit=5)
        row["buyable_names"] = _names(subset[subset["setup_classification"].isin(SETUP_BUYABLE)], limit=5)
        row["extended_names"] = _names(subset[subset["setup_classification"] == "Extended / Hold, Do Not Chase"], limit=5)
        theme_rows.append(row)

    ranked = sorted(theme_rows, key=lambda item: item["rotation_score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["theme_rank"] = rank
    return ranked


def write_theme_rotation_csv(theme_rows: list[dict], path: str | Path) -> None:
    df = pd.DataFrame(theme_rows)
    if df.empty:
        df.to_csv(path, index=False)
        return
    columns = [column for column in THEME_COLUMNS if column in df.columns]
    extra_columns = [column for column in df.columns if column not in columns]
    df[columns + extra_columns].to_csv(path, index=False)


def write_theme_rotation_report(theme_rows: list[dict], path: str | Path) -> None:
    if not theme_rows:
        Path(path).write_text("# Theme Rotation Report\n\nNo theme rotation results generated.\n", encoding="utf-8")
        return

    report_date = max(row.get("date", "") for row in theme_rows)
    lines = [
        "# Theme Rotation Report",
        "",
        f"Date: {report_date}",
        "",
        "This report ranks watchlist themes by recent relative strength, volume activity, breadth, and buyable setups.",
        "It is a screening aid only, not financial advice.",
        "",
        "## Top Active Themes",
        "",
    ]

    for row in theme_rows:
        lines.extend(
            [
                f"### {row['theme_rank']}. {row['theme']} - {row['theme_state']}",
                f"Rotation score: {row['rotation_score']} | Members: {row['member_count']}",
                f"5d return median: {row['median_return_5d']}% | 5d RS vs QQQ median: {row['median_rs_5d_vs_qqq']} | RVOL median: {row['median_relative_volume']}",
                f"20d return median: {row['median_return_20d']}% | 20d RS vs QQQ median: {row['median_rs_20d_vs_qqq']}",
                f"Breadth: {row['percent_positive_5d']}% positive 5d | {row['percent_above_ema10']}% above EMA10 | {row['percent_above_ema21']}% above EMA21",
                f"Setups: {row['pullback_count']} pullback | {row['breakout_count']} breakout | {row['extended_count']} extended | {row['breakdown_count']} breakdown",
                f"Leaders: {row['top_leaders'] or 'None'}",
                f"Buyable / chart-check names: {row['buyable_names'] or 'None'}",
                f"Extended / do-not-chase names: {row['extended_names'] or 'None'}",
                "",
            ]
        )

    Path(path).write_text("\n".join(lines), encoding="utf-8")
