from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

import src.kova_visual as kova_visual


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def find_images(screenshot_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in screenshot_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_report(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("# Kova Visual Scan Report\n\nNo screenshots analyzed.\n", encoding="utf-8")
        return

    lines = [
        "# Kova Visual Scan Report",
        "",
        "This report is based on TradingView screenshot color detection. It is a screening aid only.",
        "",
        "## Top Visual Signals",
        "",
    ]

    ranked = df.sort_values("visual_signal_score", ascending=False)
    for _, row in ranked.head(50).iterrows():
        lines.extend(
            [
                f"### {row['ticker']} - {row['visual_signal']}",
                f"Score: {row['visual_signal_score']} | Screenshot: `{row['screenshot_path']}`",
                f"Volume purple bars: last 10 = {row['volume_purple_last_10']}, last 20 = {row['volume_purple_last_20']} | latest purple age = {row['latest_purple_age']}",
                f"Volume cyan bars: last 10 = {row['volume_cyan_last_10']}, last 20 = {row['volume_cyan_last_20']}",
                f"Momentum blue bars: last 10 = {row['momentum_blue_last_10']}, last 20 = {row['momentum_blue_last_20']} | latest momentum color = {row['latest_momentum_color']}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TradingView Kova indicator screenshots.")
    parser.add_argument("--screenshots", default="screenshots", help="Directory containing screenshots named like TICKER.png")
    parser.add_argument("--output", default="output/kova_visual_scan.csv", help="CSV output path")
    parser.add_argument("--report", default="output/kova_visual_report.md", help="Markdown report output path")
    parser.add_argument("--debug-crops", action="store_true", help="Save overlay/crop images to help calibrate panel coordinates")
    parser.add_argument("--debug-dir", default="output/debug_crops", help="Directory for debug crop images")
    parser.add_argument("--volume-box", default="0.025,0.620,0.705,0.770", help="Volume crop box as left,top,right,bottom ratios")
    parser.add_argument("--momentum-box", default="0.025,0.770,0.705,0.900", help="Momentum crop box as left,top,right,bottom ratios")
    parser.add_argument("--latest-x-ratio", type=float, default=None, help="Override fixed latest slot x ratio, e.g. 0.8185")
    parser.add_argument("--bar-spacing-ratio", type=float, default=None, help="Override fixed bar spacing ratio, e.g. 0.0104")
    parser.add_argument("--slot-radius", type=int, default=None, help="Override slot sampling radius in pixels, e.g. 3")
    args = parser.parse_args()

    if args.latest_x_ratio is not None:
        kova_visual.DEFAULT_LATEST_X_RATIO = args.latest_x_ratio
    if args.bar_spacing_ratio is not None:
        kova_visual.DEFAULT_BAR_SPACING_RATIO = args.bar_spacing_ratio
    if args.slot_radius is not None:
        kova_visual.DEFAULT_SLOT_RADIUS = args.slot_radius

    volume_box = kova_visual.parse_box(args.volume_box)
    momentum_box = kova_visual.parse_box(args.momentum_box)
    screenshot_dir = Path(args.screenshots)
    output_path = Path(args.output)
    report_path = Path(args.report)
    debug_dir = Path(args.debug_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "Kova calibration: "
        f"latest_x_ratio={kova_visual.DEFAULT_LATEST_X_RATIO}, "
        f"bar_spacing_ratio={kova_visual.DEFAULT_BAR_SPACING_RATIO}, "
        f"slot_radius={kova_visual.DEFAULT_SLOT_RADIUS}"
    )

    if not screenshot_dir.exists():
        raise FileNotFoundError(f"Screenshot directory does not exist: {screenshot_dir}")

    rows: list[dict] = []
    failures: list[str] = []
    images = find_images(screenshot_dir)

    for index, image_path in enumerate(images, start=1):
        try:
            if args.debug_crops:
                kova_visual.save_debug_crops(image_path, debug_dir, volume_box=volume_box, momentum_box=momentum_box)
            result = kova_visual.analyze_kova_screenshot(image_path, volume_box=volume_box, momentum_box=momentum_box)
            rows.append(result.to_dict())
            print(
                f"[{index:>3}/{len(images)}] {result.ticker:<8} "
                f"score={result.visual_signal_score:>3} "
                f"purple10={result.volume_purple_last_10} "
                f"blueMom10={result.momentum_blue_last_10} "
                f"{result.visual_signal}"
            )
        except Exception as exc:
            failures.append(f"{image_path}: {exc}")
            print(f"[{index:>3}/{len(images)}] ❌ {image_path.name}: {exc}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("visual_signal_score", ascending=False, inplace=True)
    df.to_csv(output_path, index=False)
    write_report(df, report_path)

    if failures:
        failure_path = output_path.parent / "kova_visual_failures.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"\n⚠️ Completed with {len(failures)} failures. See {failure_path}")

    if args.debug_crops:
        print(f"✅ Wrote debug crop images to {debug_dir}")
    print(f"\n✅ Wrote {output_path}")
    print(f"✅ Wrote {report_path}")


if __name__ == "__main__":
    main()
