#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def log(level, message):
    print(f"[{level}] {message}")


def metric_value(obj, path):
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    try:
        return float(current)
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO11s segmentation model on S2DS.")
    parser.add_argument("--model", default="runs/videc_yolo/yolo11s_seg_s2ds4_underwater_v1/weights/best.pt")
    parser.add_argument("--data", default="configs/yolo/s2ds_seg4.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as exc:
        log("ERROR", f"Could not import ultralytics: {type(exc).__name__}: {exc}")
        log("INFO", "Install with: pip install ultralytics")
        return 1

    model = YOLO(args.model)
    results = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device, task="segment")
    metrics = {
        "box_map": metric_value(results, "box.map"),
        "box_map50": metric_value(results, "box.map50"),
        "mask_map": metric_value(results, "seg.map"),
        "mask_map50": metric_value(results, "seg.map50"),
    }

    output_dir = Path("reports/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "yolo11s_seg_s2ds4_test_metrics.json"
    csv_path = output_dir / "yolo11s_seg_s2ds4_test_metrics.csv"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})

    log("INFO", f"Metrics: {metrics}")
    log("INFO", f"Saved {json_path}")
    log("INFO", f"Saved {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
