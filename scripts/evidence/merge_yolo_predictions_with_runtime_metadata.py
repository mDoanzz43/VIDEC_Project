#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def log(level, message):
    print(f"[{level}] {message}")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def frame_id_from_stem(stem):
    return stem.replace("_overlay", "")


def merge(args):
    pred_root = Path(args.predictions)
    pred_dir = pred_root / "predictions"
    overlay_dir = pred_root / "overlays"
    metadata_dir = Path(args.metadata)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = Counter()
    for pred_path in sorted(pred_dir.glob("*.json")):
        stem = frame_id_from_stem(pred_path.stem)
        metadata_path = metadata_dir / f"{stem}.json"
        if not metadata_path.exists():
            log("WARN", f"Missing metadata for {pred_path.name}: {metadata_path}")
            continue

        pred_payload = load_json(pred_path)
        runtime = load_json(metadata_path)
        overlay_path = pred_payload.get("overlay_path") or str(overlay_dir / f"{stem}_overlay.png")
        predictions = pred_payload.get("predictions", [])
        for prediction in predictions:
            counts[prediction.get("class_name", "unknown")] += 1

        evidence = {
            "rgb_path": runtime.get("rgb_path") or pred_payload.get("image_path"),
            "overlay_path": overlay_path,
            "yolo_predictions": predictions,
            "runtime_metadata": {
                "t": runtime.get("t"),
                "PoseSensor": runtime.get("PoseSensor"),
                "VelocitySensor": runtime.get("VelocitySensor"),
                "IMUSensor": runtime.get("IMUSensor"),
                "DVLSensor": runtime.get("DVLSensor"),
                "DepthSensor": runtime.get("DepthSensor"),
                "action": runtime.get("action"),
                "pressed_keys": runtime.get("pressed_keys"),
            },
            "evidence_source": "holoocean_runtime_yolo11s_seg",
        }
        evidence_path = output_dir / f"{stem}_evidence.json"
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        rows.append(
            {
                "frame_id": stem,
                "rgb_path": evidence["rgb_path"],
                "overlay_path": overlay_path,
                "metadata_path": str(metadata_path),
                "evidence_path": str(evidence_path),
                "num_predictions": len(predictions),
                "classes": ";".join(sorted({p.get("class_name", "unknown") for p in predictions})),
            }
        )
        log("INFO", f"Saved evidence: {evidence_path}")

    summary_path = Path("reports/results/yolo_runtime_inspection_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["frame_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log("INFO", f"Saved summary: {summary_path}")
    log("INFO", f"Detections by class: {dict(counts)}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Merge YOLO runtime predictions with HoloOcean teleop metadata.")
    parser.add_argument("--predictions", default="data/yolo_runtime_predictions")
    parser.add_argument("--metadata", default="data/runtime_teleop/metadata")
    parser.add_argument("--output", default="data/inspection_evidence")
    return parser.parse_args()


def main():
    return merge(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
