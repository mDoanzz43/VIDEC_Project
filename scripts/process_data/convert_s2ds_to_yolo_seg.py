#!/usr/bin/env python3
import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CLASS_COLORS = {
    0: ("crack", [(255, 255, 255)]),
    1: ("spalling", [(255, 0, 0)]),
    2: ("corrosion", [(255, 255, 0)]),
    3: ("hard_negative", [(0, 255, 255), (0, 255, 0), (0, 0, 255)]),
}


def log(level, message):
    print(f"[{level}] {message}")


def require_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for YOLO segmentation conversion. Install opencv-python.") from exc
    return cv2


def find_pairs(split_dir):
    pairs = []
    for image_path in sorted(split_dir.glob("*.png")):
        if image_path.name.endswith("_lab.png"):
            continue
        mask_path = image_path.with_name(f"{image_path.stem}_lab.png")
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def color_mask(mask_rgb, colors):
    binary = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for color in colors:
        binary |= np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=2).astype(np.uint8)
    return binary


def contour_to_yolo(contour, width, height):
    points = contour.reshape(-1, 2)
    if len(points) < 3:
        return None
    values = []
    for x, y in points:
        values.append(max(0.0, min(1.0, float(x) / width)))
        values.append(max(0.0, min(1.0, float(y) / height)))
    return values


def convert_component(cv2, component_mask, class_id, width, height, epsilon_ratio):
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        coords = contour_to_yolo(approx, width, height)
        if coords and len(coords) >= 6:
            lines.append(" ".join([str(class_id)] + [f"{value:.6f}" for value in coords]))
    return lines


def draw_overlay(image_path, label_lines, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    colors = {0: "white", 1: "red", 2: "yellow", 3: "cyan"}
    for line in label_lines:
        parts = line.split()
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
        pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
        if len(pts) >= 3:
            draw.line(pts + [pts[0]], fill=colors.get(class_id, "lime"), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_dataset_yaml():
    yaml_path = Path("configs/yolo/s2ds_seg4.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "path: data/yolo_s2ds_seg4\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: crack\n"
        "  1: spalling\n"
        "  2: corrosion\n"
        "  3: hard_negative\n",
        encoding="utf-8",
    )
    log("INFO", f"Wrote {yaml_path}")


def convert(args):
    cv2 = require_cv2()
    output_root = Path(args.output_root)
    summary_rows = []
    class_counts = Counter()
    overlay_count = 0

    for split in ("train", "val", "test"):
        image_out = output_root / "images" / split
        label_out = output_root / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        pairs = find_pairs(Path(args.input_root) / split)
        log("INFO", f"{split}: converting {len(pairs)} pairs")

        for image_path, mask_path in pairs:
            out_image_path = image_out / image_path.name
            out_label_path = label_out / f"{image_path.stem}.txt"
            shutil.copy2(image_path, out_image_path)

            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            height, width = mask_rgb.shape[:2]
            label_lines = []
            present = defaultdict(int)

            for class_id, (class_name, colors) in CLASS_COLORS.items():
                if class_id == 3 and not args.include_hard_negative:
                    continue
                binary = color_mask(mask_rgb, colors)
                if class_id == 0 and args.crack_dilate > 0:
                    kernel = np.ones((args.crack_dilate, args.crack_dilate), dtype=np.uint8)
                    binary = cv2.dilate(binary, kernel, iterations=1)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                for component_id in range(1, num_labels):
                    area = int(stats[component_id, cv2.CC_STAT_AREA])
                    if area < args.min_area:
                        continue
                    component = (labels == component_id).astype(np.uint8) * 255
                    lines = convert_component(cv2, component, class_id, width, height, args.epsilon_ratio)
                    if lines:
                        label_lines.extend(lines)
                        present[class_name] += len(lines)
                        class_counts[class_name] += len(lines)

            out_label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            if overlay_count < args.debug_overlays:
                draw_overlay(out_image_path, label_lines, output_root / "overlays_debug" / f"{split}_{image_path.stem}.png")
                overlay_count += 1
            summary_rows.append(
                {
                    "split": split,
                    "image_path": str(out_image_path),
                    "mask_path": str(mask_path),
                    "label_path": str(out_label_path),
                    "num_objects": len(label_lines),
                    "present_classes": ";".join(sorted(present.keys())),
                }
            )

    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "s2ds_to_yolo_seg4_conversion_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["split"])
        writer.writeheader()
        writer.writerows(summary_rows)
    dist_path = results_dir / "yolo_seg4_class_distribution.csv"
    with dist_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "object_count"])
        writer.writeheader()
        for class_name, count in sorted(class_counts.items()):
            writer.writerow({"class_name": class_name, "object_count": count})
    write_dataset_yaml()
    log("INFO", f"Saved conversion summary: {summary_path}")
    log("INFO", f"Saved class distribution: {dist_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert S2DS color masks to YOLO segmentation format.")
    parser.add_argument("--input-root", default="data/augmented/s2ds_underwater")
    parser.add_argument("--output-root", default="data/yolo_s2ds_seg4")
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--epsilon-ratio", type=float, default=0.002)
    parser.add_argument("--crack-dilate", type=int, default=1)
    parser.add_argument("--include-hard-negative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-overlays", type=int, default=50)
    return parser.parse_args()


def main():
    convert(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
