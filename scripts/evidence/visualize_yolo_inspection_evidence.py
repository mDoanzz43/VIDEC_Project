#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def log(level, message):
    print(f"[{level}] {message}")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def short_pose(runtime_metadata):
    pose = runtime_metadata.get("PoseSensor")
    if not pose or len(pose) < 3:
        return "pose: n/a"
    try:
        x = pose[0][3]
        y = pose[1][3]
        z = pose[2][3]
        return f"pose xyz=({x:.1f},{y:.1f},{z:.1f})"
    except Exception:
        return "pose: n/a"


def depth_text(runtime_metadata):
    depth = runtime_metadata.get("DepthSensor")
    if isinstance(depth, list) and depth:
        try:
            return f"depth={float(depth[0]):.2f}"
        except Exception:
            pass
    return "depth: n/a"


def make_grid(evidence_paths, output_path, max_items):
    tiles = []
    counts = Counter()
    confidences = []
    for evidence_path in evidence_paths[:max_items]:
        evidence = load_json(evidence_path)
        overlay_path = evidence.get("overlay_path")
        if not overlay_path or not Path(overlay_path).exists():
            continue
        image = Image.open(overlay_path).convert("RGB")
        image.thumbnail((360, 260))
        canvas = Image.new("RGB", (380, 340), "white")
        canvas.paste(image, ((380 - image.width) // 2, 8))
        draw = ImageDraw.Draw(canvas)
        preds = evidence.get("yolo_predictions", [])
        for pred in preds:
            counts[pred.get("class_name", "unknown")] += 1
            if pred.get("confidence") is not None:
                confidences.append(float(pred["confidence"]))
        label = ", ".join(f"{p.get('class_name')} {p.get('confidence', 0):.2f}" for p in preds[:3]) or "no detections"
        runtime = evidence.get("runtime_metadata", {})
        draw.text((8, 276), label[:58], fill=(0, 0, 0))
        draw.text((8, 296), depth_text(runtime), fill=(0, 0, 0))
        draw.text((8, 316), short_pose(runtime), fill=(0, 0, 0))
        tiles.append(canvas)

    if not tiles:
        log("WARN", "No evidence overlays available for visualization.")
        return counts, confidences

    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * 380, rows * 340), "white")
    for idx, tile in enumerate(tiles):
        grid.paste(tile, ((idx % cols) * 380, (idx // cols) * 340))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    log("INFO", f"Saved evidence grid: {output_path}")
    return counts, confidences


def visualize(args):
    evidence_paths = sorted(Path(args.evidence).glob("*_evidence.json"))
    output_path = Path("reports/figures/yolo_runtime_inspection_evidence_grid.png")
    counts, confidences = make_grid(evidence_paths, output_path, args.max_items)

    counts_path = Path("reports/results/yolo_runtime_detection_counts.csv")
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    with counts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "detection_count"])
        writer.writeheader()
        for class_name, count in sorted(counts.items()):
            writer.writerow({"class_name": class_name, "detection_count": count})
    log("INFO", f"Saved detection counts: {counts_path}")
    if confidences:
        log("INFO", f"Confidence range: min={min(confidences):.3f}, max={max(confidences):.3f}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO runtime inspection evidence.")
    parser.add_argument("--evidence", default="data/inspection_evidence")
    parser.add_argument("--max-items", type=int, default=12)
    return parser.parse_args()


def main():
    return visualize(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
