#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from PIL import Image


def log(level, message):
    print(f"[{level}] {message}")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def size_bytes(path):
    if path is None:
        return 0
    path = Path(path)
    return path.stat().st_size if path.exists() else 0


def kb(value):
    return round(value / 1024.0, 3)


def ratio_vs_full(size, full_size):
    if not full_size:
        return None
    return round(size / full_size, 6)


def pose_xyz(metadata):
    pose = metadata.get("PoseSensor")
    if not pose or len(pose) < 3:
        return None
    try:
        return [pose[0][3], pose[1][3], pose[2][3]]
    except Exception:
        return None


def depth_value(metadata):
    depth = metadata.get("DepthSensor")
    if isinstance(depth, list) and depth:
        return depth[0]
    return None


def choose_top_detection(predictions, defect_classes):
    if not predictions:
        return None, "no_detection"
    defect_predictions = [p for p in predictions if p.get("class_name") in defect_classes]
    candidates = defect_predictions or predictions
    return max(candidates, key=lambda item: item.get("confidence", 0.0)), None


def crop_roi(frame_path, detection, output_path, padding):
    image = Image.open(frame_path).convert("RGB")
    w, h = image.size
    if detection is None:
        crop_w = min(w, max(1, w // 3))
        crop_h = min(h, max(1, h // 3))
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        box = (x1, y1, x1 + crop_w, y1 + crop_h)
    else:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        box = (
            max(0, int(x1) - padding),
            max(0, int(y1) - padding),
            min(w, int(x2) + padding),
            min(h, int(y2) + padding),
        )
    roi = image.crop(box)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    roi.save(output_path)
    return output_path


def write_json_stable(path, payload):
    previous_size = -1
    for _ in range(4):
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        current_size = path.stat().st_size
        if current_size == previous_size:
            return current_size
        previous_size = current_size
        payload.setdefault("size_bytes", {})["evidence_packet_json"] = current_size
        payload.setdefault("size_kb", {})["evidence_packet_json"] = kb(current_size)
    return path.stat().st_size


def build_packet(args, pred_path):
    prediction_payload = load_json(pred_path)
    image_stem = prediction_payload.get("image_stem") or pred_path.stem
    frame_path = Path(args.frames) / f"{image_stem}.png"
    metadata_path = Path(args.metadata) / f"{image_stem}.json"
    overlay_path = Path(args.overlays) / f"{image_stem}_overlay.png"
    mask_bbox_path = Path(args.mask_bbox) / f"{image_stem}_mask_bbox.png"

    runtime = load_json(metadata_path) if metadata_path.exists() else {}
    detections = prediction_payload.get("predictions", [])
    top_detection, reason = choose_top_detection(detections, set(args.defect_classes))
    verification_status = (
        "confirmed"
        if top_detection and float(top_detection.get("confidence", 0.0)) >= args.min_conf
        else "uncertain"
    )

    roi_path = Path(args.output) / "roi" / f"{image_stem}_roi.png"
    if frame_path.exists():
        crop_roi(frame_path, top_detection, roi_path, args.roi_padding)
    else:
        roi_path = None
        reason = "missing_full_frame"

    compact = {
        "image_id": image_stem,
        "frame_id": image_stem,
        "top_class": top_detection.get("class_name") if top_detection else None,
        "top_confidence": top_detection.get("confidence") if top_detection else None,
        "bbox": top_detection.get("bbox_xyxy") if top_detection else None,
        "area_px": top_detection.get("mask_area_px") if top_detection else None,
        "depth": depth_value(runtime),
        "pose_xyz": pose_xyz(runtime),
        "recommended_action": "transmit_roi" if verification_status == "confirmed" else "request_review",
        "verification_status": verification_status,
    }
    metadata_compact_path = Path(args.output) / "metadata_compact" / f"{image_stem}_metadata_compact.json"
    metadata_compact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_compact_path.write_text(json.dumps(compact, indent=2), encoding="utf-8")

    full_size = size_bytes(frame_path)
    roi_size = size_bytes(roi_path)
    mask_bbox_size = size_bytes(mask_bbox_path)
    prediction_json_size = size_bytes(pred_path)
    compact_size = size_bytes(metadata_compact_path)

    packet = {
        "frame_id": image_stem,
        "image_stem": image_stem,
        "full_frame_path": str(frame_path),
        "overlay_path": str(overlay_path),
        "mask_bbox_path": str(mask_bbox_path),
        "roi_path": str(roi_path) if roi_path else None,
        "yolo_prediction_path": str(pred_path),
        "metadata_path": str(metadata_path),
        "metadata_compact_path": str(metadata_compact_path),
        "source": "holoocean_runtime_yolo11s_seg",
        "model_path": args.model_path,
        "detections": [
            {
                "class_name": p.get("class_name"),
                "confidence": p.get("confidence"),
                "bbox_xyxy": p.get("bbox_xyxy"),
                "bbox_xywh": p.get("bbox_xywh"),
                "mask_area_px": p.get("mask_area_px"),
                "mask_path": p.get("mask_path"),
            }
            for p in detections
        ],
        "runtime_metadata": {
            "t": runtime.get("t"),
            "PoseSensor": runtime.get("PoseSensor"),
            "VelocitySensor": runtime.get("VelocitySensor"),
            "IMUSensor": runtime.get("IMUSensor"),
            "DVLSensor": runtime.get("DVLSensor"),
            "DepthSensor": runtime.get("DepthSensor"),
            "action": runtime.get("action"),
            "pressed_keys": runtime.get("pressed_keys"),
            "active_command": runtime.get("active_command"),
        },
        "compact_metadata": compact,
        "size_bytes": {
            "full_frame": full_size,
            "roi_crop": roi_size,
            "mask_bbox": mask_bbox_size,
            "prediction_json": prediction_json_size,
            "metadata_compact": compact_size,
            "evidence_packet_json": 0,
        },
        "size_kb": {
            "full_frame": kb(full_size),
            "roi_crop": kb(roi_size),
            "mask_bbox": kb(mask_bbox_size),
            "prediction_json": kb(prediction_json_size),
            "metadata_compact": kb(compact_size),
            "evidence_packet_json": 0,
        },
        "compression_vs_full": {
            "roi_crop_vs_full": ratio_vs_full(roi_size, full_size),
            "mask_bbox_vs_full": ratio_vs_full(mask_bbox_size, full_size),
            "metadata_compact_vs_full": ratio_vs_full(compact_size, full_size),
            "evidence_packet_vs_full": None,
        },
        "verification_status": verification_status,
        "reason": reason,
    }

    packet_path = Path(args.output) / "packets" / f"{image_stem}_evidence.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_size = write_json_stable(packet_path, packet)
    packet["size_bytes"]["evidence_packet_json"] = packet_size
    packet["size_kb"]["evidence_packet_json"] = kb(packet_size)
    packet["compression_vs_full"]["evidence_packet_vs_full"] = ratio_vs_full(packet_size, full_size)
    write_json_stable(packet_path, packet)
    return packet_path, packet


def build(args):
    output_root = Path(args.output)
    for subdir in ("packets", "roi", "metadata_compact", "size_comparison"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    rows = []
    for pred_path in sorted(Path(args.predictions).glob("*.json")):
        packet_path, packet = build_packet(args, pred_path)
        size_comparison_path = output_root / "size_comparison" / f"{packet['image_stem']}_size_comparison.json"
        size_payload = {
            "frame_id": packet["frame_id"],
            "size_bytes": packet["size_bytes"],
            "size_kb": packet["size_kb"],
            "compression_vs_full": packet["compression_vs_full"],
            "verification_status": packet["verification_status"],
        }
        size_comparison_path.write_text(json.dumps(size_payload, indent=2), encoding="utf-8")
        rows.append(
            {
                "frame_id": packet["frame_id"],
                "packet_path": str(packet_path),
                "verification_status": packet["verification_status"],
                "num_detections": len(packet["detections"]),
                "full_frame_bytes": packet["size_bytes"]["full_frame"],
                "roi_crop_bytes": packet["size_bytes"]["roi_crop"],
                "mask_bbox_bytes": packet["size_bytes"]["mask_bbox"],
                "metadata_compact_bytes": packet["size_bytes"]["metadata_compact"],
                "evidence_packet_json_bytes": packet["size_bytes"]["evidence_packet_json"],
                "roi_crop_vs_full": packet["compression_vs_full"]["roi_crop_vs_full"],
                "mask_bbox_vs_full": packet["compression_vs_full"]["mask_bbox_vs_full"],
                "metadata_compact_vs_full": packet["compression_vs_full"]["metadata_compact_vs_full"],
            }
        )
        log("INFO", f"Saved evidence packet: {packet_path}")

    summary_path = Path("reports/results/inspection_evidence_packet_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["frame_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log("INFO", f"Saved summary: {summary_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Build ViDEC inspection evidence packets from YOLO predictions and runtime metadata.")
    parser.add_argument("--predictions", default="data/yolo_runtime_predictions/predictions")
    parser.add_argument("--overlays", default="data/yolo_runtime_predictions/overlays")
    parser.add_argument("--mask-bbox", default="data/yolo_runtime_predictions/mask_bbox")
    parser.add_argument("--frames", default="data/runtime_teleop/frames")
    parser.add_argument("--metadata", default="data/runtime_teleop/metadata")
    parser.add_argument("--output", default="data/inspection_evidence_packets")
    parser.add_argument("--roi-padding", type=int, default=20)
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--defect-classes", nargs="+", default=["crack", "spalling", "corrosion"])
    parser.add_argument("--model-path", default="")
    return parser.parse_args()


def main():
    return build(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
