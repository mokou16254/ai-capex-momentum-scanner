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
    if label == "Pullback Trigger":
        return "Daily pullback candidate with 2h trigger; confirm on live chart"
    if label == "Pullback Watch":
        return "Daily pullback candidate; watch for intraday trendline break"
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


def _pullback_days_from_high(data: pd.DataFrame, lookback: int) -> int | None:
    recent = data.tail(lookback)
    if recent.empty:
        return None
    high_pos = int(recent["High"].to_numpy().argmax())
    return len(recent) - high_pos - 1


def _recent_pullback_low(data: pd.DataFrame, days_from_high: int | None, fallback_days: int = 8) -> float | None:
    if data.empty:
        return None
    lookback = fallback_days if days_from_high is None else max(2, min(days_from_high + 1, fallback_days))
    return float(data["Low"].tail(lookback).min())


def _intraday_trigger(intraday_data: pd.DataFrame | None, config: dict[str, Any]) -> tuple[bool, str]:
    if intraday_data is None or intraday_data.empty:
        return False, "No 2h data loaded"
    min_bars = int(config.get("intraday_min_bars", 20))
    if len(intraday_data) < min_bars:
        return False, f"Not enough 2h bars ({len(intraday_data)})"

    data = intraday_data.copy()
    data["EMA10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    lookback = int(config.get("intraday_trigger_lookback_bars", 5))
    prior = data.iloc[-lookback - 1 : -1]
    prior_high = float(prior["High"].max()) if not prior.empty else float(previous["High"])

    close = float(latest["Close"])
    range_position = float((latest["Close"] - latest["Low"]) / (latest["High"] - latest["Low"])) if float(latest["High"]) != float(latest["Low"]) else 0.5
    breakout = close > prior_high
    regained_ema10 = close > float(latest["EMA10"]) and float(previous["Close"]) <= float(previous["EMA10"])
    above_ema21 = close > float(latest["EMA21"])
    green = close > float(latest["Open"])
    strong_close = range_position >= float(config.get("intraday_strong_close_min", 0.60))

    reasons: list[str] = []
    if breakout:
        reasons.append(f"2h close broke prior {lookback}-bar high")
    if regained_ema10:
        reasons.append("2h close regained EMA10")
    if above_ema21:
        reasons.append("2h close above EMA21")
    if green and strong_close:
        reasons.append("2h green strong close")

    trigger = (breakout or regained_ema10) and above_ema21 and green and strong_close
    return trigger, "; ".join(reasons) if reasons else "No 2h trigger"


def classify_setup(data: pd.DataFrame, config: dict[str, Any], intraday_data: pd.DataFrame | None = None) -> tuple[str, list[str], int, dict[str, Any]]:
    """Classify a ticker and return label, notes, technical score, and extra fields."""
    latest = data.iloc[-1]
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

    near_ema_support = abs(dist10) <= float(config["pullback_distance_to_ema_percent"]) or abs(dist21) <= float(config["pullback_distance_to_ema_percent"])
    if near_ema_support:
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

    days_lookback = int(config.get("pullback_high_lookback_days", 20))
    days_from_high = _pullback_days_from_high(data, days_lookback)
    min_days = int(config.get("pullback_min_days_from_high", 2))
    max_days = int(config.get("pullback_max_days_from_high", 12))
    duration_ok = days_from_high is not None and min_days <= days_from_high <= max_days

    recent_low = _recent_pullback_low(data, days_from_high, int(config.get("pullback_stop_lookback_days", 8)))
    stop_buffer = float(config.get("pullback_stop_buffer_percent", 0.5)) / 100
    suggested_stop = recent_low * (1 - stop_buffer) if recent_low is not None else None
    risk_percent = (close - suggested_stop) / close * 100 if suggested_stop is not None and close > 0 else None
    max_risk = float(config.get("pullback_max_risk_percent", 10.0))
    risk_ok = risk_percent is not None and 0 < risk_percent <= max_risk

    pullback_min_20d_return = float(config.get("pullback_min_20d_return_percent", 8.0))
    pullback_min_60d_return = float(config.get("pullback_min_60d_return_percent", 15.0))
    pullback_max_rvol = float(config.get("pullback_max_relative_volume", 1.25))
    support_buffer = float(config.get("pullback_support_buffer_percent", 3.0)) / 100

    ret20 = _safe_pct_return(data, 20)
    ret60 = _safe_pct_return(data, 60)
    trend_qualified = close > ema50 and ema10 > ema21 > ema50 and (
        (ret20 is not None and ret20 >= pullback_min_20d_return)
        or (ret60 is not None and ret60 >= pullback_min_60d_return)
        or _near_high(latest, "High50", 8.0)
    )
    support_test = float(latest["Low"]) <= ema10 * (1 + support_buffer) or float(latest["Low"]) <= ema21 * (1 + support_buffer) or near_ema_support
    controlled_pullback = close > ema21 and close > ema50 and float(config["rsi_pullback_min"]) <= rsi <= float(config["rsi_pullback_max"]) and rel_vol <= pullback_max_rvol
    intraday_trigger, intraday_note = _intraday_trigger(intraday_data, config)

    if trend_qualified:
        score += 10
        notes.append("Prior trend qualifies for pullback trade")
    if support_test:
        score += 8
        notes.append("Tested EMA10/EMA21 support zone")
    if controlled_pullback:
        score += 8
        notes.append("Controlled pullback: RSI reset and no heavy distribution volume")
    if duration_ok:
        score += 6
        notes.append(f"Pullback duration is controlled ({days_from_high} days from high)")
    elif days_from_high is not None:
        notes.append(f"Pullback duration outside ideal range ({days_from_high} days from high)")
    if risk_ok:
        score += 8
        notes.append(f"Stop distance is controlled ({risk_percent:.1f}%)")
    elif risk_percent is not None:
        notes.append(f"Stop distance is wide or invalid ({risk_percent:.1f}%)")
    if intraday_trigger:
        score += 10
        notes.append("2h trigger present")

    breakdown = close < ema21 and rsi < 50 and (_is_down_day(latest) and rel_vol >= float(config["relative_volume_breakdown_min"]) or close <= float(latest["Low20"]))
    extended = rsi >= float(config["rsi_extended"]) or dist10 > float(config["extended_distance_to_ema10_percent"]) or dist21 > float(config["extended_distance_to_ema21_percent"])
    breakout = close > ema10 > ema21 > ema50 and _near_high(latest, "High50", float(config["breakout_distance_to_50d_high_percent"])) and rsi >= float(config["rsi_breakout_min"]) and rel_vol >= float(config["relative_volume_breakout_min"]) and not extended
    pullback_watch = trend_qualified and support_test and controlled_pullback and duration_ok and risk_ok and not breakdown and not extended

    if breakdown:
        label = "Breakdown Risk"
        score = min(score, 45)
        notes.append("Below EMA21 with weak RSI / breakdown risk")
    elif extended:
        label = "Extended / Hold, Do Not Chase"
        notes.append("Trend may continue, but entry is stretched")
    elif pullback_watch and intraday_trigger:
        label = "Pullback Trigger"
        notes.append("Daily pullback plus 2h trigger")
    elif pullback_watch:
        label = "Pullback Watch"
        notes.append("Daily pullback watch candidate")
    elif breakout:
        label = "Breakout Watch"
        notes.append("Meets breakout-watch criteria")
    else:
        label = "Neutral"

    if pct_change > 0 and range_position >= 0.7:
        score += 5
        notes.append("Strong close in daily range")
    elif pct_change > 0 and range_position < 0.4 and rel_vol > 1.5:
        score -= 8
        notes.append("High-volume up day closed weakly")

    extras = {
        "pullback_days_from_high": days_from_high,
        "suggested_stop": round(suggested_stop, 2) if suggested_stop is not None else None,
        "pullback_risk_percent": round(risk_percent, 2) if risk_percent is not None else None,
        "intraday_trigger": intraday_trigger,
        "intraday_trigger_note": intraday_note,
    }
    return label, notes, max(0, min(100, int(round(score)))), extras


def scan_ticker(
    ticker: str,
    category_meta: dict[str, Any],
    data: pd.DataFrame,
    config: dict[str, Any],
    benchmark_data: dict[str, pd.DataFrame] | None = None,
    intraday_data: pd.DataFrame | None = None,
) -> ScanResult:
    latest = data.iloc[-1]
    label, notes, technical_score, extras = classify_setup(data, config, intraday_data)

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
        **extras,
        "above_ema10": close > ema10,
        "above_ema21": close > ema21,
        "above_ema50": close > ema50,
        "near_50d_high": near_50d_high,
        "notes": " | ".join(notes),
        "volume": int(latest["Volume"]),
        "atr14": round(atr, 2),
        "ema10": round(ema10, 2),
        "ema21": round(ema21, 2),
        "ema50": round(ema50, 2),
    }
    return ScanResult(row=row)
