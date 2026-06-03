from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class VisualScanResult:
    ticker: str
    screenshot_path: str
    image_width: int
    image_height: int
    volume_bar_count: int
    volume_purple_last_10: int
    volume_purple_last_20: int
    volume_cyan_last_10: int
    volume_cyan_last_20: int
    volume_red_last_10: int
    volume_red_last_20: int
    volume_green_last_10: int
    volume_green_last_20: int
    latest_purple_age: int | None
    momentum_bar_count: int
    momentum_blue_last_10: int
    momentum_blue_last_20: int
    momentum_red_last_10: int
    momentum_red_last_20: int
    latest_momentum_color: str
    visual_signal_score: int
    visual_signal: str

    def to_dict(self) -> dict:
        row = asdict(self)
        row["latest_purple_age"] = "" if row["latest_purple_age"] is None else row["latest_purple_age"]
        return row


@dataclass
class DetectedBar:
    x_center: float
    width: int
    color: str
    pixel_count: int


def parse_box(value: str) -> tuple[float, float, float, float]:
    """Parse a crop box string: left,top,right,bottom."""
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Crop box must have four comma-separated numbers: left,top,right,bottom")
    left, top, right, bottom = parts
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("Crop box values must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1")
    return left, top, right, bottom


def _pixel_box(image: Image.Image, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    width, height = image.size
    left, top, right, bottom = box
    return int(width * left), int(height * top), int(width * right), int(height * bottom)


def _crop_by_ratio(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Crop an image using fractional coordinates: left, top, right, bottom."""
    return image.crop(_pixel_box(image, box))


def save_debug_crops(
    image_path: str | Path,
    output_dir: str | Path,
    volume_box: tuple[float, float, float, float] = (0.025, 0.620, 0.705, 0.770),
    momentum_box: tuple[float, float, float, float] = (0.025, 0.770, 0.705, 0.900),
) -> None:
    """Save crop calibration images for one screenshot.

    Creates:
    - <ticker>_overlay.png: original screenshot with crop rectangles
    - <ticker>_volume_crop.png
    - <ticker>_momentum_crop.png
    """
    path = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(path).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    volume_px = _pixel_box(image, volume_box)
    momentum_px = _pixel_box(image, momentum_box)
    draw.rectangle(volume_px, outline=(255, 0, 255), width=4)
    draw.text((volume_px[0] + 6, max(0, volume_px[1] - 22)), "volume crop", fill=(255, 0, 255))
    draw.rectangle(momentum_px, outline=(0, 200, 255), width=4)
    draw.text((momentum_px[0] + 6, max(0, momentum_px[1] - 22)), "momentum crop", fill=(0, 200, 255))

    stem = path.stem
    overlay.save(out_dir / f"{stem}_overlay.png")
    _crop_by_ratio(image, volume_box).save(out_dir / f"{stem}_volume_crop.png")
    _crop_by_ratio(image, momentum_box).save(out_dir / f"{stem}_momentum_crop.png")


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    """Return a rough color mask for TradingView dark-theme screenshots."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    if color == "purple":
        return (r >= 115) & (b >= 115) & (g <= 140) & ((r + b) / 2 - g >= 25)
    if color == "cyan":
        return (g >= 120) & (b >= 140) & (r <= 110) & (b - r >= 50)
    if color == "red":
        return (r >= 140) & (g <= 100) & (b <= 120) & (r - g >= 45)
    if color == "green":
        return (g >= 90) & (r <= 120) & (b <= 120) & (g - r >= 25)
    if color == "blue":
        return (g >= 115) & (b >= 145) & (r <= 110) & (b - r >= 45)
    raise ValueError(f"Unknown color: {color}")


def _group_columns(mask: np.ndarray, min_col_pixels: int, min_width: int = 2, merge_gap: int = 2) -> list[tuple[int, int, int]]:
    """Group active x-columns into bars.

    Returns tuples of (start_x, end_x, pixel_count). end_x is exclusive.
    """
    col_counts = mask.sum(axis=0)
    active = np.where(col_counts >= min_col_pixels)[0]
    if active.size == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = prev = int(active[0])
    for x in active[1:]:
        x = int(x)
        if x - prev <= merge_gap + 1:
            prev = x
        else:
            groups.append((start, prev + 1))
            start = prev = x
    groups.append((start, prev + 1))

    output: list[tuple[int, int, int]] = []
    for start, end in groups:
        width = end - start
        pixels = int(mask[:, start:end].sum())
        if width >= min_width and pixels >= min_col_pixels * min_width:
            output.append((start, end, pixels))
    return output


def _dedupe_bars(bars: Iterable[DetectedBar], max_distance: int = 4) -> list[DetectedBar]:
    """Remove duplicate detections that hit the same visual bar with nearby colors."""
    sorted_bars = sorted(bars, key=lambda bar: (bar.x_center, -bar.pixel_count))
    deduped: list[DetectedBar] = []
    for bar in sorted_bars:
        if deduped and abs(bar.x_center - deduped[-1].x_center) <= max_distance:
            if bar.pixel_count > deduped[-1].pixel_count:
                deduped[-1] = bar
        else:
            deduped.append(bar)
    return deduped


def _detect_colored_bars(crop: Image.Image, colors: list[str], min_col_pixel_ratio: float = 0.06) -> list[DetectedBar]:
    rgb = _rgb_array(crop)
    height = rgb.shape[0]
    min_col_pixels = max(3, int(height * min_col_pixel_ratio))

    bars: list[DetectedBar] = []
    for color in colors:
        mask = _color_mask(rgb, color)
        for start, end, pixels in _group_columns(mask, min_col_pixels=min_col_pixels):
            bars.append(
                DetectedBar(
                    x_center=(start + end - 1) / 2,
                    width=end - start,
                    color=color,
                    pixel_count=pixels,
                )
            )
    return _dedupe_bars(bars)


def _estimate_bar_spacing(bars: list[DetectedBar]) -> float:
    """Estimate x-spacing between chart bars from detected colored bars."""
    if len(bars) < 2:
        return 8.0
    xs = np.array(sorted({round(bar.x_center, 1) for bar in bars}), dtype=float)
    if len(xs) < 2:
        return 8.0
    diffs = np.diff(xs)
    diffs = diffs[(diffs >= 2) & (diffs <= 40)]
    if diffs.size == 0:
        return 8.0
    return float(np.median(diffs))


def _recent_by_x_window(bars: list[DetectedBar], periods: int, spacing: float | None = None) -> list[DetectedBar]:
    """Return bars inside the recent x-axis window.

    This approximates the most recent N chart bars by x-position instead of
    selecting the rightmost N detected colored bars. That matters because not
    every chart bar is detected as a saturated Kova color.
    """
    if not bars:
        return []
    spacing = spacing or _estimate_bar_spacing(bars)
    latest_x = max(bar.x_center for bar in bars)
    left_edge = latest_x - periods * spacing
    return [bar for bar in bars if bar.x_center >= left_edge]


def _count_color(bars: list[DetectedBar], color: str) -> int:
    return sum(1 for bar in bars if bar.color == color)


def _latest_color(bars: list[DetectedBar]) -> str:
    if not bars:
        return "unknown"
    return max(bars, key=lambda bar: bar.x_center).color


def _latest_age(bars: list[DetectedBar], color: str, spacing: float | None = None) -> int | None:
    color_bars = [bar for bar in bars if bar.color == color]
    if not bars or not color_bars:
        return None
    spacing = spacing or _estimate_bar_spacing(bars)
    latest_x = max(bar.x_center for bar in bars)
    latest_color_x = max(bar.x_center for bar in color_bars)
    return max(0, int(round((latest_x - latest_color_x) / max(spacing, 1))))


def _visual_signal_score(
    purple_10: int,
    purple_20: int,
    cyan_10: int,
    blue_momentum_10: int,
    red_momentum_10: int,
    latest_purple_age: int | None,
) -> int:
    score = 0
    score += min(purple_10, 5) * 10
    score += min(max(purple_20 - purple_10, 0), 5) * 4
    score += min(cyan_10, 4) * 6
    score += min(blue_momentum_10, 10) * 4
    score -= min(red_momentum_10, 10) * 3
    if latest_purple_age is not None:
        if latest_purple_age <= 2:
            score += 15
        elif latest_purple_age <= 5:
            score += 8
    return max(0, min(100, int(score)))


def _visual_signal_label(score: int, purple_10: int, blue_momentum_10: int, red_momentum_10: int, latest_purple_age: int | None) -> str:
    if red_momentum_10 >= 7 and blue_momentum_10 <= 2:
        return "Negative momentum / avoid"
    if purple_10 >= 3 and blue_momentum_10 >= 5:
        return "Purple cluster + strong momentum"
    if purple_10 >= 2 and latest_purple_age is not None and latest_purple_age <= 5:
        return "Recent purple accumulation cluster"
    if blue_momentum_10 >= 7:
        return "Strong momentum"
    if score >= 55:
        return "Interesting visual signal"
    return "No strong visual signal"


def analyze_kova_screenshot(
    image_path: str | Path,
    ticker: str | None = None,
    volume_box: tuple[float, float, float, float] = (0.025, 0.620, 0.705, 0.770),
    momentum_box: tuple[float, float, float, float] = (0.025, 0.770, 0.705, 0.900),
) -> VisualScanResult:
    """Analyze one TradingView screenshot using fixed-layout crop ratios."""
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    inferred_ticker = ticker or path.stem.upper()

    volume_crop = _crop_by_ratio(image, volume_box)
    momentum_crop = _crop_by_ratio(image, momentum_box)

    volume_bars = _detect_colored_bars(volume_crop, ["purple", "cyan", "red", "green"], min_col_pixel_ratio=0.05)
    momentum_bars = _detect_colored_bars(momentum_crop, ["blue", "red"], min_col_pixel_ratio=0.04)

    volume_spacing = _estimate_bar_spacing(volume_bars)
    momentum_spacing = _estimate_bar_spacing(momentum_bars)
    volume_last_10 = _recent_by_x_window(volume_bars, 10, volume_spacing)
    volume_last_20 = _recent_by_x_window(volume_bars, 20, volume_spacing)
    momentum_last_10 = _recent_by_x_window(momentum_bars, 10, momentum_spacing)
    momentum_last_20 = _recent_by_x_window(momentum_bars, 20, momentum_spacing)

    purple_10 = _count_color(volume_last_10, "purple")
    purple_20 = _count_color(volume_last_20, "purple")
    cyan_10 = _count_color(volume_last_10, "cyan")
    cyan_20 = _count_color(volume_last_20, "cyan")
    red_10 = _count_color(volume_last_10, "red")
    red_20 = _count_color(volume_last_20, "red")
    green_10 = _count_color(volume_last_10, "green")
    green_20 = _count_color(volume_last_20, "green")
    latest_purple_age = _latest_age(volume_bars, "purple", volume_spacing)

    blue_momentum_10 = _count_color(momentum_last_10, "blue")
    blue_momentum_20 = _count_color(momentum_last_20, "blue")
    red_momentum_10 = _count_color(momentum_last_10, "red")
    red_momentum_20 = _count_color(momentum_last_20, "red")

    score = _visual_signal_score(
        purple_10=purple_10,
        purple_20=purple_20,
        cyan_10=cyan_10,
        blue_momentum_10=blue_momentum_10,
        red_momentum_10=red_momentum_10,
        latest_purple_age=latest_purple_age,
    )

    return VisualScanResult(
        ticker=inferred_ticker,
        screenshot_path=str(path),
        image_width=image.size[0],
        image_height=image.size[1],
        volume_bar_count=len(volume_bars),
        volume_purple_last_10=purple_10,
        volume_purple_last_20=purple_20,
        volume_cyan_last_10=cyan_10,
        volume_cyan_last_20=cyan_20,
        volume_red_last_10=red_10,
        volume_red_last_20=red_20,
        volume_green_last_10=green_10,
        volume_green_last_20=green_20,
        latest_purple_age=latest_purple_age,
        momentum_bar_count=len(momentum_bars),
        momentum_blue_last_10=blue_momentum_10,
        momentum_blue_last_20=blue_momentum_20,
        momentum_red_last_10=red_momentum_10,
        momentum_red_last_20=red_momentum_20,
        latest_momentum_color=_latest_color(momentum_bars),
        visual_signal_score=score,
        visual_signal=_visual_signal_label(score, purple_10, blue_momentum_10, red_momentum_10, latest_purple_age),
    )
