from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ScanResult:
    row: dict[str, Any]


def _is_down_day(latest: pd.Series) -> bool:
    return float(latest["Close"]) < float(latest["Open"])


def _near_high(latest: pd.Series, high_col: str, threshold_pct: float) -> bool:
    high = float(latest[high_col])
    if high <= 0:
        return False
    return (high - float(latest["Close"])) / high * 100 <= threshold_pct


def classify_setup(latest: pd.Series, config: dict[str, Any]) -> tuple[str, list[str], int]:
    """Classify a ticker and return label, notes, and score."""
    close = float(latest["Close"])
    ema10 = float(latest["EMA10"])
    ema21 = float(latest["EMA21"])
    ema50 = float(latest["EMA50"])
    rsi = float(latest["RSI14"])
    rel_vol = float(latest.get("RelativeVolume", 0) or 0)
    dist10 = float(latest["DistanceEMA10Pct"])
    dist21 = float(latest["DistanceEMA21Pct"])
    pct_change = float(latest.get("PctChange", 0) or 0)
    range_position = float(latest.get("RangePosition", 0.5) or 0.5)

    notes: list[str] = []
    score = 0

    if close > ema10:
        score += 12
        notes.append("Close is above EMA10")
    if close > ema21:
        score += 14
        notes.append("Close is above EMA21")
    if close > ema50:
        score += 14
        notes.append("Close is above EMA50")

    if rsi > 60:
        score += 16
        notes.append("RSI is above 60, showing strong momentum")
    elif 45 <= rsi <= 65:
        score += 12
        notes.append("RSI is in a constructive pullback/reset zone")
    elif rsi < 50:
        notes.append("RSI is below 50, showing weaker momentum")

    if rel_vol >= float(config["relative_volume_breakout_min"]):
        score += 12
        notes.append("Relative volume is elevated")
    elif rel_vol < 1:
        score += 5
        notes.append("Volume is calm relative to the 20-day average")

    if abs(dist10) <= float(config["pullback_distance_to_ema_percent"]) or abs(dist21) <= float(config["pullback_distance_to_ema_percent"]):
        score += 10
        notes.append("Price is near a key short/intermediate EMA")

    if _near_high(latest, "High50", float(config["breakout_distance_to_50d_high_percent"])):
        score += 12
        notes.append("Close is near the 50-day high")

    if dist10 > float(config["extended_distance_to_ema10_percent"]):
        score -= 10
        notes.append("Price is extended above EMA10")
    if dist21 > float(config["extended_distance_to_ema21_percent"]):
        score -= 8
        notes.append("Price is extended above EMA21")

    breakdown = (
        close < ema21
        and rsi < 50
        and (_is_down_day(latest) and rel_vol >= float(config["relative_volume_breakdown_min"]) or close <= float(latest["Low20"]))
    )
    extended = (
        rsi >= float(config["rsi_extended"])
        or dist10 > float(config["extended_distance_to_ema10_percent"])
        or dist21 > float(config["extended_distance_to_ema21_percent"])
    )
    breakout = (
        close > ema10 > ema21 > ema50
        and _near_high(latest, "High50", float(config["breakout_distance_to_50d_high_percent"]))
        and rsi >= float(config["rsi_breakout_min"])
        and rel_vol >= float(config["relative_volume_breakout_min"])
        and not extended
    )
    pullback = (
        close > ema21
        and close > ema50
        and float(config["rsi_pullback_min"]) <= rsi <= float(config["rsi_pullback_max"])
        and (abs(dist10) <= float(config["pullback_distance_to_ema_percent"]) or abs(dist21) <= float(config["pullback_distance_to_ema_percent"]))
        and rel_vol < float(config["relative_volume_breakdown_min"])
        and not breakdown
    )

    if breakdown:
        label = "Breakdown Risk"
        score = min(score, 45)
        notes.append("Possible breakdown: below EMA21 with weak RSI and risk signal")
    elif extended:
        label = "Extended / Do Not Chase"
        notes.append("Trend may be strong, but new-entry risk/reward is poor")
    elif breakout:
        label = "Breakout Watch"
        notes.append("Meets breakout-watch criteria")
    elif pullback:
        label = "Pullback Setup"
        notes.append("Meets constructive pullback criteria")
    else:
        label = "Neutral"

    if pct_change > 0 and range_position >= 0.7:
        score += 5
        notes.append("Latest candle closed in the upper part of the daily range")
    elif pct_change > 0 and range_position < 0.4 and rel_vol > 1.5:
        score -= 8
        notes.append("Warning: high-volume up day closed weakly off the highs")

    return label, notes, max(0, min(100, int(round(score))))


def scan_ticker(ticker: str, category_meta: dict[str, Any], data: pd.DataFrame, config: dict[str, Any]) -> ScanResult:
    latest = data.iloc[-1]
    label, notes, score = classify_setup(latest, config)
    row = {
        "date": data.index[-1].strftime("%Y-%m-%d"),
        "ticker": ticker,
        "primary_category": category_meta["primary_category"],
        "all_categories": ", ".join(category_meta["all_categories"]),
        "in_core_watchlist": bool(category_meta["in_core_watchlist"]),
        "close": round(float(latest["Close"]), 2),
        "percent_change": round(float(latest.get("PctChange", 0) or 0), 2),
        "volume": int(latest["Volume"]),
        "relative_volume": round(float(latest.get("RelativeVolume", 0) or 0), 2),
        "ema10": round(float(latest["EMA10"]), 2),
        "ema21": round(float(latest["EMA21"]), 2),
        "ema50": round(float(latest["EMA50"]), 2),
        "rsi14": round(float(latest["RSI14"]), 1),
        "atr14": round(float(latest.get("ATR14", 0) or 0), 2),
        "distance_to_ema10_percent": round(float(latest["DistanceEMA10Pct"]), 2),
        "distance_to_ema21_percent": round(float(latest["DistanceEMA21Pct"]), 2),
        "distance_to_ema50_percent": round(float(latest["DistanceEMA50Pct"]), 2),
        "setup_classification": label,
        "score": score,
        "notes": " | ".join(notes),
    }
    return ScanResult(row=row)
