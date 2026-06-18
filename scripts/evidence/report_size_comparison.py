#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


SIZE_KEYS = ["full_frame", "roi_crop", "mask_bbox", "metadata_compact", "evidence_packet_json"]
RATIO_KEYS = ["roi_crop_vs_full", "mask_bbox_vs_full", "metadata_compact_vs_full", "evidence_packet_vs_full"]


def log(level, message):
    print(f"[{level}] {message}")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def draw_bar_chart(summary, output_path):
    means = [summary["sizes"][key]["mean"] or 0 for key in SIZE_KEYS]
    labels = ["Full frame", "ROI crop", "Mask+BBox", "Metadata", "Evidence JSON"]
    width, height = 1000, 620
    margin = 90
    chart_h = 380
    max_value = max(means) if means else 1
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 24), "ViDEC Size Comparison: full frame vs compact inspection evidence", fill=(0, 0, 0))
    draw.line([margin, height - margin, width - margin, height - margin], fill=(0, 0, 0), width=2)
    bar_w = 120
    gap = 60
    x = margin + 25
    for label, value in zip(labels, means):
        bar_h = int((value / max_value) * chart_h) if max_value else 0
        y1 = height - margin - bar_h
        draw.rectangle([x, y1, x + bar_w, height - margin], fill=(80, 140, 220), outline=(20, 20, 20))
        draw.text((x, y1 - 20), f"{value / 1024:.1f} KB", fill=(0, 0, 0))
        draw.text((x, height - margin + 12), label, fill=(0, 0, 0))
        x += bar_w + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    log("INFO", f"Saved size comparison figure: {output_path}")


def report(args):
    packets = [load_json(path) for path in sorted(Path(args.packets).glob("*_evidence.json"))]
    sizes = defaultdict(list)
    ratios = defaultdict(list)
    class_counts = Counter()
    class_conf = defaultdict(list)
    status_counts = Counter()

    for packet in packets:
        status_counts[packet.get("verification_status", "unknown")] += 1
        for key in SIZE_KEYS:
            sizes[key].append(packet.get("size_bytes", {}).get(key))
        for key in RATIO_KEYS:
            ratios[key].append(packet.get("compression_vs_full", {}).get(key))
        for detection in packet.get("detections", []):
            class_name = detection.get("class_name", "unknown")
            class_counts[class_name] += 1
            if detection.get("confidence") is not None:
                class_conf[class_name].append(float(detection["confidence"]))

    summary = {
        "num_packets": len(packets),
        "sizes": {key: stats(values) for key, values in sizes.items()},
        "compression_vs_full": {key: stats(values) for key, values in ratios.items()},
        "detection_count_per_class": dict(class_counts),
        "average_confidence_per_class": {
            key: statistics.mean(values) for key, values in class_conf.items() if values
        },
        "verification_status_counts": dict(status_counts),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric_group", "metric", "mean", "median", "min", "max"])
        writer.writeheader()
        for key, values in summary["sizes"].items():
            writer.writerow({"metric_group": "size_bytes", "metric": key, **values})
        for key, values in summary["compression_vs_full"].items():
            writer.writerow({"metric_group": "compression_vs_full", "metric": key, **values})

    draw_bar_chart(summary, Path(args.output_figure))
    log("INFO", f"Saved size comparison CSV: {output_csv}")
    log("INFO", f"Saved size comparison JSON: {output_json}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Report ViDEC inspection evidence size comparison.")
    parser.add_argument("--packets", default="data/inspection_evidence_packets/packets")
    parser.add_argument("--output-csv", default="reports/results/inspection_size_comparison.csv")
    parser.add_argument("--output-json", default="reports/results/inspection_size_comparison.json")
    parser.add_argument("--output-figure", default="reports/figures/inspection_size_comparison.png")
    return parser.parse_args()


def main():
    return report(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
