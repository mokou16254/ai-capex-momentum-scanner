from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_VOLUME_BOX = (0.025, 0.620, 0.705, 0.770)
DEFAULT_MOMENTUM_BOX = (0.025, 0.770, 0.705, 0.900)

# Calibrated from the fixed TradingView layout/crop.
# The slots are intentionally fixed rather than inferred from detected colors.
DEFAULT_LATEST_X_RATIO = 0.820
DEFAULT_BAR_SPACING_RATIO = 0.0104
DEFAULT_SLOT_RADIUS = 3
RECENT_SLOT_COUNT = 60


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


@dataclass
class SlotSample:
    index: int
    x_center: float
    color: str
    pixel_count: int


def parse_box(value: str) -> tuple[float, float, float, float]:
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
    return image.crop(_pixel_box(image, box))


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    if color == "purple":
        return (r >= 115) & (b >= 115) & (g <= 145) & ((r + b) / 2 - g >= 22)
    if color == "cyan":
        return (g >= 120) & (b >= 140) & (r <= 115) & (b - r >= 45)
    if color == "red":
        return (r >= 140) & (g <= 105) & (b <= 125) & (r - g >= 40)
    if color == "green":
        return (g >= 90) & (r <= 130) & (b <= 125) & (g - r >= 20)
    if color == "blue":
        return (g >= 135) & (b >= 155) & (r <= 105) & (b - r >= 50)
    raise ValueError(f"Unknown color: {color}")


def _group_columns(mask: np.ndarray, min_col_pixels: int, min_width: int = 2, merge_gap: int = 2) -> list[tuple[int, int, int]]:
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
            bars.append(DetectedBar(x_center=(start + end - 1) / 2, width=end - start, color=color, pixel_count=pixels))
    return _dedupe_bars(bars)


def _fixed_slot_geometry(crop: Image.Image) -> tuple[float, float]:
    """Return fixed latest_x and bar spacing for the calibrated TradingView layout."""
    latest_x = crop.width * DEFAULT_LATEST_X_RATIO
    spacing = crop.width * DEFAULT_BAR_SPACING_RATIO
    return latest_x, spacing


def _sample_slot_color(rgb: np.ndarray, x_center: float, colors: list[str], radius: int = DEFAULT_SLOT_RADIUS) -> tuple[str, int]:
    height, width = rgb.shape[:2]
    left = max(0, int(round(x_center - radius)))
    right = min(width, int(round(x_center + radius + 1)))
    if left >= right:
        return "none", 0

    slot = rgb[:, left:right, :]
    best_color = "none"
    best_count = 0
    for color in colors:
        count = int(_color_mask(slot, color).sum())
        if count > best_count:
            best_color = color
            best_count = count

    min_pixels = max(4, int(height * (right - left) * 0.015))
    if best_count < min_pixels:
        return "none", best_count
    return best_color, best_count


def _sample_slots(
    crop: Image.Image,
    colors: list[str],
    latest_x: float,
    spacing: float,
    count: int = RECENT_SLOT_COUNT,
    radius: int = DEFAULT_SLOT_RADIUS,
) -> list[SlotSample]:
    rgb = _rgb_array(crop)
    samples: list[SlotSample] = []
    for index in range(count):
        x = latest_x - index * spacing
        if x < 0:
            break
        color, pixels = _sample_slot_color(rgb, x, colors, radius=radius)
        samples.append(SlotSample(index=index, x_center=x, color=color, pixel_count=pixels))
    return samples


def _count_slot_color(slots: list[SlotSample], color: str, periods: int) -> int:
    return sum(1 for slot in slots[:periods] if slot.color == color)


def _latest_slot_age(slots: list[SlotSample], color: str) -> int | None:
    for slot in slots:
        if slot.color == color:
            return slot.index
    return None


def _latest_slot_color(slots: list[SlotSample]) -> str:
    for slot in slots:
        if slot.color != "none":
            return slot.color
    return "unknown"


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


def _draw_slot_debug(
    crop: Image.Image,
    color_bars: list[DetectedBar],
    slots: list[SlotSample],
    output_path: str | Path,
    panel_label: str,
    spacing: float,
    latest_x: float,
) -> None:
    debug = crop.convert("RGB").copy()
    draw = ImageDraw.Draw(debug, "RGBA")
    width, height = debug.size

    if not slots:
        draw.text((8, 8), f"{panel_label}: no slots", fill=(255, 255, 255, 255))
        debug.save(output_path)
        return

    left_10 = latest_x - 9 * spacing - DEFAULT_SLOT_RADIUS
    left_20 = latest_x - 19 * spacing - DEFAULT_SLOT_RADIUS
    right_edge = latest_x + DEFAULT_SLOT_RADIUS

    draw.rectangle((max(0, left_20), 0, min(width, right_edge), height), outline=(255, 165, 0, 230), fill=(255, 165, 0, 25), width=2)
    draw.rectangle((max(0, left_10), 0, min(width, right_edge), height), outline=(255, 255, 0, 255), fill=(255, 255, 0, 45), width=3)

    color_map = {
        "purple": (255, 0, 255, 255),
        "cyan": (0, 255, 255, 255),
        "red": (255, 70, 70, 255),
        "green": (80, 255, 80, 255),
        "blue": (80, 180, 255, 255),
        "none": (170, 170, 170, 120),
    }

    # Thin lines for raw colored detections, useful for color threshold debugging.
    for bar in color_bars:
        raw_color = color_map.get(bar.color, (220, 220, 220, 255))
        draw.line((bar.x_center, 0, bar.x_center, height), fill=raw_color, width=1)

    # Slot centers are the source of truth for last10/last20.
    for slot in slots[:20]:
        slot_color = color_map.get(slot.color, (170, 170, 170, 120))
        width_px = 3 if slot.index < 10 else 2
        draw.line((slot.x_center, 0, slot.x_center, height), fill=slot_color, width=width_px)
        draw.text((slot.x_center + 2, 2), str(slot.index), fill=(235, 235, 235, 200))
        draw.ellipse((slot.x_center - 3, height - 9, slot.x_center + 3, height - 3), fill=slot_color)

    draw.line((latest_x, 0, latest_x, height), fill=(255, 255, 255, 255), width=3)
    summary = (
        f"{panel_label} | fixed spacing={spacing:.1f} | fixed latest_x={latest_x:.1f} | "
        f"p10={_count_slot_color(slots, 'purple', 10)} p20={_count_slot_color(slots, 'purple', 20)} | "
        f"blue10={_count_slot_color(slots, 'blue', 10)} red10={_count_slot_color(slots, 'red', 10)}"
    )
    draw.rectangle((0, height - 24, min(width, 1050), height), fill=(0, 0, 0, 180))
    draw.text((8, height - 20), summary, fill=(255, 255, 255, 255))
    debug.save(output_path)


def _panel_state(crop: Image.Image, colors: list[str]) -> tuple[list[DetectedBar], list[SlotSample], float, float]:
    color_bars = _detect_colored_bars(crop, colors, min_col_pixel_ratio=0.05)
    latest_x, spacing = _fixed_slot_geometry(crop)
    slots = _sample_slots(crop, colors, latest_x, spacing)
    return color_bars, slots, spacing, latest_x


def save_debug_crops(
    image_path: str | Path,
    output_dir: str | Path,
    volume_box: tuple[float, float, float, float] = DEFAULT_VOLUME_BOX,
    momentum_box: tuple[float, float, float, float] = DEFAULT_MOMENTUM_BOX,
    include_windows: bool = True,
) -> None:
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
    volume_crop = _crop_by_ratio(image, volume_box)
    momentum_crop = _crop_by_ratio(image, momentum_box)
    overlay.save(out_dir / f"{stem}_overlay.png")
    volume_crop.save(out_dir / f"{stem}_volume_crop.png")
    momentum_crop.save(out_dir / f"{stem}_momentum_crop.png")

    if include_windows:
        volume_color_bars, volume_slots, volume_spacing, volume_latest_x = _panel_state(volume_crop, ["purple", "cyan", "red", "green"])
        momentum_color_bars, momentum_slots, momentum_spacing, momentum_latest_x = _panel_state(momentum_crop, ["blue", "red"])
        _draw_slot_debug(volume_crop, volume_color_bars, volume_slots, out_dir / f"{stem}_volume_debug.png", "volume", volume_spacing, volume_latest_x)
        _draw_slot_debug(momentum_crop, momentum_color_bars, momentum_slots, out_dir / f"{stem}_momentum_debug.png", "momentum", momentum_spacing, momentum_latest_x)


def analyze_kova_screenshot(
    image_path: str | Path,
    ticker: str | None = None,
    volume_box: tuple[float, float, float, float] = DEFAULT_VOLUME_BOX,
    momentum_box: tuple[float, float, float, float] = DEFAULT_MOMENTUM_BOX,
) -> VisualScanResult:
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    inferred_ticker = ticker or path.stem.upper()

    volume_crop = _crop_by_ratio(image, volume_box)
    momentum_crop = _crop_by_ratio(image, momentum_box)

    volume_color_bars, volume_slots, _, _ = _panel_state(volume_crop, ["purple", "cyan", "red", "green"])
    momentum_color_bars, momentum_slots, _, _ = _panel_state(momentum_crop, ["blue", "red"])

    purple_10 = _count_slot_color(volume_slots, "purple", 10)
    purple_20 = _count_slot_color(volume_slots, "purple", 20)
    cyan_10 = _count_slot_color(volume_slots, "cyan", 10)
    cyan_20 = _count_slot_color(volume_slots, "cyan", 20)
    red_10 = _count_slot_color(volume_slots, "red", 10)
    red_20 = _count_slot_color(volume_slots, "red", 20)
    green_10 = _count_slot_color(volume_slots, "green", 10)
    green_20 = _count_slot_color(volume_slots, "green", 20)
    latest_purple_age = _latest_slot_age(volume_slots, "purple")

    blue_momentum_10 = _count_slot_color(momentum_slots, "blue", 10)
    blue_momentum_20 = _count_slot_color(momentum_slots, "blue", 20)
    red_momentum_10 = _count_slot_color(momentum_slots, "red", 10)
    red_momentum_20 = _count_slot_color(momentum_slots, "red", 20)

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
        volume_bar_count=len(volume_color_bars),
        volume_purple_last_10=purple_10,
        volume_purple_last_20=purple_20,
        volume_cyan_last_10=cyan_10,
        volume_cyan_last_20=cyan_20,
        volume_red_last_10=red_10,
        volume_red_last_20=red_20,
        volume_green_last_10=green_10,
        volume_green_last_20=green_20,
        latest_purple_age=latest_purple_age,
        momentum_bar_count=len(momentum_color_bars),
        momentum_blue_last_10=blue_momentum_10,
        momentum_blue_last_20=blue_momentum_20,
        momentum_red_last_10=red_momentum_10,
        momentum_red_last_20=red_momentum_20,
        latest_momentum_color=_latest_slot_color(momentum_slots),
        visual_signal_score=score,
        visual_signal=_visual_signal_label(score, purple_10, blue_momentum_10, red_momentum_10, latest_purple_age),
    )
