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


def _safe_pct_return(data: pd.DataFrame, days: int) -> float | None:
    if len(data) <= days:
        return None
    start = float(data["Close"].iloc[-days - 1])
    end = float(data["Close"].iloc[-1])
    if start <= 0:
        return None
    return (end / start - 1) * 100


def _ema_state(close: float, ema10: float, ema21: float, ema50: float) -> str:
    if close > ema10 > ema21 > ema50:
        return "Strong uptrend: above EMA10/21/50"
    if close > ema21 > ema50:
        return "Uptrend: above EMA21/50"
    if close > ema50:
        return "Mixed: above EMA50, below short EMAs"
    return "Weak: below EMA50"


def _momentum_state(rsi: float) -> str:
    if rsi >= 75:
        return "Very hot / extended"
    if rsi >= 60:
        return "Strong momentum"
    if rsi >= 50:
        return "Constructive reset"
    if rsi >= 40:
        return "Weakening momentum"
    return "Weak momentum"


def _volume_state(rel_vol: float) -> str:
    if rel_vol >= 2.0:
        return "Very high volume"
    if rel_vol >= 1.3:
        return "Elevated volume"
    if rel_vol >= 0.8:
        return "Normal volume"
    return "Quiet volume"


def _position_state(dist10: float, dist21: float, near_50d_high: bool) -> str:
    if dist10 > 10 or dist21 > 15:
        return "Extended from EMAs"
    if near_50d_high:
        return "Near 50-day high"
    if abs(dist10) <= 5 or abs(dist21) <= 5:
        return "Near EMA support"
    return "Middle of range"


def _volatility_state(atr_percent: float, config: dict[str, Any]) -> str:
    high = float(config.get("high_atr_percent", 6.0))
    very_high = float(config.get("very_high_atr_percent", 10.0))
    if atr_percent >= very_high:
        return "Very high volatility"
    if atr_percent >= high:
        return "High volatility"
    if atr_percent >= 3.0:
        return "Medium volatility"
    return "Low volatility"


def _liquidity_state(avg_dollar_volume_m: float, config: dict[str, Any]) -> str:
    minimum = float(config.get("min_avg_dollar_volume_millions", 30.0))
    if avg_dollar_volume_m >= 500:
        return "Very liquid"
    if avg_dollar_volume_m >= 100:
        return "Liquid"
    if avg_dollar_volume_m >= minimum:
        return "Tradable liquidity"
    return "Thin liquidity"


def _category_weight(category_meta: dict[str, Any], config: dict[str, Any]) -> float:
    weights = config.get("category_weights", {}) or {}
    categories = category_meta.get("all_categories", []) or []
    category_weights = [float(weights.get(category, 1.0)) for category in categories]
    weight = max(category_weights) if category_weights else 1.0
    if bool(category_meta.get("in_core_watchlist", False)):
        weight = max(weight, float(weights.get("core_watchlist", 1.2)))
    return weight


def _priority_score_raw(technical_score: int, category_weight: float, rs_20d_vs_qqq: float | None, rs_20d_vs_smh: float | None, atr_percent: float, avg_dollar_volume_m: float, config: dict[str, Any]) -> float:
    score = technical_score * category_weight
    if rs_20d_vs_qqq is not None and rs_20d_vs_qqq > 0:
        score += min(rs_20d_vs_qqq, 15) * 0.7
    if rs_20d_vs_smh is not None and rs_20d_vs_smh > 0:
        score += min(rs_20d_vs_smh, 15) * 0.5
    if avg_dollar_volume_m < float(config.get("min_avg_dollar_volume_millions", 30.0)):
        score -= 15
    if atr_percent >= float(config.get("very_high_atr_percent", 10.0)):
        score -= 8
    return round(score, 2)


def _priority_score(raw_score: float) -> int:
    return max(0, min(100, int(round(raw_score))))


def _conclusion(label: str, rsi: float, dist10: float, dist21: float, rel_vol: float, near_50d_high: bool, priority_score: int | None = None) -> str:
    if label == "Pullback Setup":
        return "Watch for bounce / constructive pullback"
    if label == "Breakout Watch":
        return "Watch for breakout confirmation"
    if label == "Extended / Hold, Do Not Chase":
        return "Strong trend; hold/watch, but avoid fresh chasing"
    if label == "Breakdown Risk":
        return "Caution: possible structure damage"
    if priority_score is not None and priority_score >= 80:
        return "High-priority strength; manual chart check"
    if near_50d_high and rsi >= 60 and dist10 <= 10:
        return "Strong trend; needs manual chart check"
    if rel_vol >= 1.3 and rsi >= 60:
        return "Active momentum; check chart manually"
    if abs(dist10) <= 5 or abs(dist21) <= 5:
        return "Near support; monitor reaction"
    return "No clean setup"


def classify_setup(latest: pd.Series, config: dict[str, Any]) -> tuple[str, list[str], int]:
    """Classify a ticker and return label, notes, and technical score."""
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
        notes.append("Above EMA10")
    if close > ema21:
        score += 14
        notes.append("Above EMA21")
    if close > ema50:
        score += 14
        notes.append("Above EMA50")

    if rsi > 60:
        score += 16
        notes.append("RSI shows strong momentum")
    elif 45 <= rsi <= 65:
        score += 12
        notes.append("RSI is in reset zone")
    elif rsi < 50:
        notes.append("RSI below 50")

    if rel_vol >= float(config["relative_volume_breakout_min"]):
        score += 12
        notes.append("Volume is elevated")
    elif rel_vol < 1:
        score += 5
        notes.append("Volume is calm")

    if abs(dist10) <= float(config["pullback_distance_to_ema_percent"]) or abs(dist21) <= float(config["pullback_distance_to_ema_percent"]):
        score += 10
        notes.append("Near EMA support")

    if _near_high(latest, "High50", float(config["breakout_distance_to_50d_high_percent"])):
        score += 12
        notes.append("Near 50-day high")

    if dist10 > float(config["extended_distance_to_ema10_percent"]):
        score -= 10
        notes.append("Extended above EMA10")
    if dist21 > float(config["extended_distance_to_ema21_percent"]):
        score -= 8
        notes.append("Extended above EMA21")

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
        notes.append("Below EMA21 with weak RSI / breakdown risk")
    elif extended:
        label = "Extended / Hold, Do Not Chase"
        notes.append("Trend may continue, but entry is stretched")
    elif breakout:
        label = "Breakout Watch"
        notes.append("Meets breakout-watch criteria")
    elif pullback:
        label = "Pullback Setup"
        notes.append("Constructive pullback criteria")
    else:
        label = "Neutral"

    if pct_change > 0 and range_position >= 0.7:
        score += 5
        notes.append("Strong close in daily range")
    elif pct_change > 0 and range_position < 0.4 and rel_vol > 1.5:
        score -= 8
        notes.append("High-volume up day closed weakly")

    return label, notes, max(0, min(100, int(round(score))))


def scan_ticker(
    ticker: str,
    category_meta: dict[str, Any],
    data: pd.DataFrame,
    config: dict[str, Any],
    benchmark_data: dict[str, pd.DataFrame] | None = None,
) -> ScanResult:
    latest = data.iloc[-1]
    label, notes, technical_score = classify_setup(latest, config)

    close = float(latest["Close"])
    ema10 = float(latest["EMA10"])
    ema21 = float(latest["EMA21"])
    ema50 = float(latest["EMA50"])
    rsi = float(latest["RSI14"])
    rel_vol = float(latest.get("RelativeVolume", 0) or 0)
    dist10 = float(latest["DistanceEMA10Pct"])
    dist21 = float(latest["DistanceEMA21Pct"])
    atr = float(latest.get("ATR14", 0) or 0)
    atr_percent = (atr / close * 100) if close > 0 else 0
    avg_dollar_volume_m = float((data["Close"] * data["Volume"]).rolling(20).mean().iloc[-1] / 1_000_000)
    near_50d_high = _near_high(latest, "High50", float(config["breakout_distance_to_50d_high_percent"]))

    short_days = int(config.get("rs_lookback_days_short", 20))
    long_days = int(config.get("rs_lookback_days_long", 60))
    ticker_ret_20 = _safe_pct_return(data, short_days)
    ticker_ret_60 = _safe_pct_return(data, long_days)
    qqq_ret_20 = _safe_pct_return(benchmark_data.get("QQQ"), short_days) if benchmark_data and benchmark_data.get("QQQ") is not None else None
    smh_ret_20 = _safe_pct_return(benchmark_data.get("SMH"), short_days) if benchmark_data and benchmark_data.get("SMH") is not None else None
    qqq_ret_60 = _safe_pct_return(benchmark_data.get("QQQ"), long_days) if benchmark_data and benchmark_data.get("QQQ") is not None else None
    smh_ret_60 = _safe_pct_return(benchmark_data.get("SMH"), long_days) if benchmark_data and benchmark_data.get("SMH") is not None else None
    rs_20d_vs_qqq = ticker_ret_20 - qqq_ret_20 if ticker_ret_20 is not None and qqq_ret_20 is not None else None
    rs_20d_vs_smh = ticker_ret_20 - smh_ret_20 if ticker_ret_20 is not None and smh_ret_20 is not None else None
    rs_60d_vs_qqq = ticker_ret_60 - qqq_ret_60 if ticker_ret_60 is not None and qqq_ret_60 is not None else None
    rs_60d_vs_smh = ticker_ret_60 - smh_ret_60 if ticker_ret_60 is not None and smh_ret_60 is not None else None

    category_weight = _category_weight(category_meta, config)
    raw_priority_score = _priority_score_raw(technical_score, category_weight, rs_20d_vs_qqq, rs_20d_vs_smh, atr_percent, avg_dollar_volume_m, config)
    priority_score = _priority_score(raw_priority_score)

    row = {
        "date": data.index[-1].strftime("%Y-%m-%d"),
        "ticker": ticker,
        "setup_classification": label,
        "technical_score": technical_score,
        "priority_score": priority_score,
        "raw_priority_score": raw_priority_score,
        "category_weight": round(category_weight, 2),
        "conclusion": _conclusion(label, rsi, dist10, dist21, rel_vol, near_50d_high, priority_score),
        "trend_state": _ema_state(close, ema10, ema21, ema50),
        "momentum_state": _momentum_state(rsi),
        "volume_state": _volume_state(rel_vol),
        "position_state": _position_state(dist10, dist21, near_50d_high),
        "volatility_state": _volatility_state(atr_percent, config),
        "liquidity_state": _liquidity_state(avg_dollar_volume_m, config),
        "in_core_watchlist": bool(category_meta["in_core_watchlist"]),
        "primary_category": category_meta["primary_category"],
        "all_categories": ", ".join(category_meta["all_categories"]),
        "close": round(close, 2),
        "percent_change": round(float(latest.get("PctChange", 0) or 0), 2),
        "rsi14": round(rsi, 1),
        "relative_volume": round(rel_vol, 2),
        "atr_percent": round(atr_percent, 2),
        "avg_dollar_volume_m": round(avg_dollar_volume_m, 1),
        "return_20d": round(ticker_ret_20, 2) if ticker_ret_20 is not None else None,
        "return_60d": round(ticker_ret_60, 2) if ticker_ret_60 is not None else None,
        "rs_20d_vs_qqq": round(rs_20d_vs_qqq, 2) if rs_20d_vs_qqq is not None else None,
        "rs_20d_vs_smh": round(rs_20d_vs_smh, 2) if rs_20d_vs_smh is not None else None,
        "rs_60d_vs_qqq": round(rs_60d_vs_qqq, 2) if rs_60d_vs_qqq is not None else None,
        "rs_60d_vs_smh": round(rs_60d_vs_smh, 2) if rs_60d_vs_smh is not None else None,
        "distance_to_ema10_percent": round(dist10, 2),
        "distance_to_ema21_percent": round(dist21, 2),
        "distance_to_ema50_percent": round(float(latest["DistanceEMA50Pct"]), 2),
        "above_ema10": close > ema10,
        "above_ema21": close > ema21,
        "above_ema50": close > ema50,
        "near_50d_high": near_50d_high,
        "notes": " | ".join(notes),
        # Raw fields kept at the end for debugging/backtesting.
        "volume": int(latest["Volume"]),
        "atr14": round(atr, 2),
        "ema10": round(ema10, 2),
        "ema21": round(ema21, 2),
        "ema50": round(ema50, 2),
    }
    return ScanResult(row=row)
