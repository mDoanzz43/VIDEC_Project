#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mask_array(path):
    return np.array(Image.open(path).convert("L")) > 0


def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def mask_metrics(gt_mask, pred_mask):
    if gt_mask.shape != pred_mask.shape:
        pred_mask = np.array(
            Image.fromarray(pred_mask.astype(np.uint8) * 255).resize(
                (gt_mask.shape[1], gt_mask.shape[0]),
                Image.NEAREST,
            )
        ) > 0

    tp = int(np.logical_and(gt_mask, pred_mask).sum())
    fp = int(np.logical_and(~gt_mask, pred_mask).sum())
    fn = int(np.logical_and(gt_mask, ~pred_mask).sum())
    union = tp + fp + fn
    mask_iou = tp / union if union else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return mask_iou, precision, recall, f1, tp, union


def evaluate_sample(gt_annotation_path, pred_root):
    gt = read_json(gt_annotation_path)
    image_id = gt["image_id"]
    pred_path = pred_root / "annotations" / f"{image_id}.json"
    if not pred_path.exists():
        print(f"[WARN] Missing prediction annotation, skipping: {pred_path}")
        return None

    pred = read_json(pred_path)
    gt_defect = gt["defects"][0]
    pred_defect = pred["predictions"][0]
    gt_mask_path = Path(gt_defect["mask_path"])
    pred_mask_path = Path(pred_defect["mask_path"])

    if not gt_mask_path.exists():
        print(f"[WARN] Missing ground-truth mask, skipping: {gt_mask_path}")
        return None
    if not pred_mask_path.exists():
        print(f"[WARN] Missing prediction mask, skipping: {pred_mask_path}")
        return None

    gt_mask = mask_array(gt_mask_path)
    pred_mask = mask_array(pred_mask_path)
    mask_iou, precision, recall, f1, intersection_area, union_area = mask_metrics(gt_mask, pred_mask)
    gt_bbox = gt_defect["bbox"]
    pred_bbox = pred_defect["bbox"]
    bbox_iou_value = bbox_iou(gt_bbox, pred_bbox)

    result_row = {
        "image_id": image_id,
        "class": gt_defect["class"],
        "mask_iou": round(mask_iou, 6),
        "bbox_iou": round(bbox_iou_value, 6),
        "pixel_precision": round(precision, 6),
        "pixel_recall": round(recall, 6),
        "pixel_f1": round(f1, 6),
    }
    debug_row = {
        "image_id": image_id,
        "class": gt_defect["class"],
        "gt_mask_area": int(gt_mask.sum()),
        "pred_mask_area": int(pred_mask.sum()),
        "intersection_area": int(intersection_area),
        "union_area": int(union_area),
        "mask_iou": round(mask_iou, 6),
        "gt_bbox": json.dumps(gt_bbox),
        "pred_bbox": json.dumps(pred_bbox),
        "bbox_iou": round(bbox_iou_value, 6),
    }
    return result_row, debug_row


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "class",
                "mask_iou",
                "bbox_iou",
                "pixel_precision",
                "pixel_recall",
                "pixel_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_debug_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "class",
                "gt_mask_area",
                "pred_mask_area",
                "intersection_area",
                "union_area",
                "mask_iou",
                "gt_bbox",
                "pred_bbox",
                "bbox_iou",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def class_iou(rows):
    result = {}
    for row in rows:
        result.setdefault(row["class"], []).append(float(row["mask_iou"]))
    return {name: sum(values) / len(values) for name, values in sorted(result.items())}


def draw_iou_by_class(rows, path):
    values = class_iou(rows)
    if not values:
        return False

    width, height = 820, 480
    left, bottom = 110, 80
    chart_w, chart_h = width - left - 50, height - 130
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    draw.text((left, 28), "Sanity Baseline Mask IoU by Class", fill=(0, 0, 0), font=title_font)
    axis_y = height - bottom
    draw.line((left, 70, left, axis_y), fill=(0, 0, 0), width=2)
    draw.line((left, axis_y, width - 35, axis_y), fill=(0, 0, 0), width=2)
    draw.text((28, 70), "1.0", fill=(0, 0, 0), font=label_font)
    draw.text((34, axis_y - 12), "0", fill=(0, 0, 0), font=label_font)

    bar_w = max(40, chart_w // max(len(values) * 2, 1))
    for index, (class_name, value) in enumerate(values.items()):
        bar_h = int(value * chart_h)
        x0 = left + 55 + index * bar_w * 2
        y0 = axis_y - bar_h
        draw.rectangle((x0, y0, x0 + bar_w, axis_y), fill=(70, 150, 110))
        draw.text((x0 - 5, max(72, y0 - 22)), f"{value:.2f}", fill=(0, 0, 0), font=label_font)
        draw.text((x0 - 12, axis_y + 12), class_name, fill=(0, 0, 0), font=label_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate minimal ViDEC predictions.")
    parser.add_argument("--data", default="data/generated", help="Generated ground-truth data root.")
    parser.add_argument("--pred", default="data/predictions", help="Prediction data root.")
    parser.add_argument("--output", default="reports/results", help="Results output root.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data)
    pred_root = Path(args.pred)
    output_root = Path(args.output)

    annotation_dir = data_root / "annotations"
    gt_paths = sorted(annotation_dir.glob("*.json"))
    if not gt_paths:
        raise RuntimeError(f"No generated annotations found in {annotation_dir}")

    rows = []
    debug_rows = []
    for gt_path in gt_paths:
        evaluated = evaluate_sample(gt_path, pred_root)
        if evaluated is not None:
            row, debug_row = evaluated
            rows.append(row)
            debug_rows.append(debug_row)

    csv_path = output_root / "perception_results.csv"
    write_csv(rows, csv_path)
    print(f"[INFO] Saved {csv_path}")

    debug_csv_path = output_root / "evaluation_debug.csv"
    write_debug_csv(debug_rows, debug_csv_path)
    print(f"[INFO] Saved {debug_csv_path}")

    figure_path = Path("reports/figures/iou_by_class.png")
    if draw_iou_by_class(rows, figure_path):
        print(f"[INFO] Saved {figure_path}")
    else:
        print("[WARN] No class IoU data available; skipped iou_by_class.png")

    print(f"[INFO] Evaluated {len(rows)} prediction samples")


if __name__ == "__main__":
    raise SystemExit(main())
