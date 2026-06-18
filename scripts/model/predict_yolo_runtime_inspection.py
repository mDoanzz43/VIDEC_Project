#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


NAMES = {0: "crack", 1: "spalling", 2: "corrosion", 3: "hard_negative"}
COLORS = {
    "crack": (255, 255, 255),
    "spalling": (255, 70, 70),
    "corrosion": (255, 220, 0),
    "hard_negative": (0, 220, 255),
}


def log(level, message):
    print(f"[{level}] {message}")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


def image_sources(source):
    path = Path(source)
    if path.is_dir():
        files = []
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            files.extend(sorted(path.glob(pattern)))
        return files
    return [path]


def mask_to_image_size(mask, size):
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def mask_area(mask):
    if mask is None:
        return None
    return int(mask.astype(bool).sum())


def mask_polygon(result, index):
    if result.masks is None or result.masks.xyn is None:
        return None
    try:
        return result.masks.xyn[index].tolist()
    except Exception:
        return None


def result_mask(result, index, image_size):
    if result.masks is None or result.masks.data is None:
        return None
    try:
        mask = result.masks.data[index].cpu().numpy() > 0.5
    except Exception:
        return None
    return mask_to_image_size(mask, image_size)


def include_for_visualization(prediction, args):
    class_name = prediction["class_name"]
    if args.show_classes and class_name not in args.show_classes:
        return False
    if args.hide_classes and class_name in args.hide_classes:
        return False
    if args.roi:
        x1, y1, x2, y2 = prediction["bbox_xyxy"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        rx1, ry1, rx2, ry2 = args.roi
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
            return False
    return True


def save_instance_mask(mask, path):
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def draw_predictions(image_path, predictions, output_path, mask_bbox_only=False):
    image = Image.open(image_path).convert("RGB")
    if mask_bbox_only:
        canvas = Image.new("RGB", image.size, "black")
    else:
        canvas = image.copy()

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(canvas)

    for prediction in predictions:
        color = COLORS.get(prediction["class_name"], (0, 255, 0))
        mask_path = prediction.get("mask_path")
        if mask_path and Path(mask_path).exists():
            mask = Image.open(mask_path).convert("L")
            color_layer = Image.new("RGBA", image.size, (*color, 95 if not mask_bbox_only else 170))
            overlay.paste(color_layer, (0, 0), mask)

        x1, y1, x2, y2 = prediction["bbox_xyxy"]
        label = f"{prediction['class_name']} {prediction['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, max(0, y1 - 18), x1 + min(190, len(label) * 8), y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - 16)), label, fill=(0, 0, 0))

    if not mask_bbox_only:
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for prediction in predictions:
            color = COLORS.get(prediction["class_name"], (0, 255, 0))
            x1, y1, x2, y2 = prediction["bbox_xyxy"]
            label = f"{prediction['class_name']} {prediction['confidence']:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.rectangle([x1, max(0, y1 - 18), x1 + min(190, len(label) * 8), y1], fill=color)
            draw.text((x1 + 3, max(0, y1 - 16)), label, fill=(0, 0, 0))
    else:
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for prediction in predictions:
            color = COLORS.get(prediction["class_name"], (0, 255, 0))
            x1, y1, x2, y2 = prediction["bbox_xyxy"]
            label = f"{prediction['class_name']} {prediction['confidence']:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 3, max(0, y1 - 16)), label, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def extract_predictions(result, image_path, image_size, mask_dir, args):
    predictions = []
    boxes = result.boxes
    if boxes is None:
        return predictions

    for idx in range(len(boxes)):
        class_id = int(boxes.cls[idx].item())
        class_name = NAMES.get(class_id, str(class_id))
        mask = result_mask(result, idx, image_size)
        mask_path = None
        if parse_bool(args.save_masks) and mask is not None:
            mask_path = mask_dir / f"{image_path.stem}_mask_{idx:03d}_{class_name}.png"
            save_instance_mask(mask, mask_path)

        prediction = {
            "image_path": str(image_path),
            "class_id": class_id,
            "class_name": class_name,
            "confidence": float(boxes.conf[idx].item()),
            "bbox_xyxy": [float(v) for v in boxes.xyxy[idx].cpu().numpy().tolist()],
            "bbox_xywh": [float(v) for v in boxes.xywh[idx].cpu().numpy().tolist()],
            "mask_polygon_normalized": mask_polygon(result, idx),
            "mask_area_px": mask_area(mask),
            "mask_path": str(mask_path) if mask_path else None,
        }
        predictions.append(prediction)

    predictions.sort(key=lambda item: item["confidence"], reverse=True)
    if args.top_k is not None:
        predictions = predictions[: args.top_k]
    return predictions


def predict(args):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        log("ERROR", f"Could not import ultralytics: {type(exc).__name__}: {exc}")
        log("INFO", "Install with: pip install ultralytics")
        return 1

    output = Path(args.output)
    overlay_dir = output / "overlays"
    json_dir = output / "predictions"
    mask_dir = output / "masks"
    mask_bbox_dir = output / "mask_bbox"
    for directory in (overlay_dir, json_dir, mask_dir, mask_bbox_dir):
        directory.mkdir(parents=True, exist_ok=True)

    counters = {"images": 0, "json": 0, "overlays": 0, "mask_bbox": 0, "masks": 0}
    model = YOLO(args.model)
    for image_path in image_sources(args.source):
        image = Image.open(image_path).convert("RGB")
        results = model.predict(str(image_path), conf=args.conf, imgsz=args.imgsz, device=args.device, task="segment", verbose=False)
        result = results[0]
        predictions_all = extract_predictions(result, image_path, image.size, mask_dir, args)
        counters["masks"] += sum(1 for prediction in predictions_all if prediction.get("mask_path"))
        predictions_vis = [p for p in predictions_all if include_for_visualization(p, args)]
        predictions_json = predictions_vis if args.filter_json else predictions_all

        overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
        mask_bbox_path = mask_bbox_dir / f"{image_path.stem}_mask_bbox.png"
        if parse_bool(args.save_overlays):
            draw_predictions(image_path, predictions_vis, overlay_path, mask_bbox_only=False)
            counters["overlays"] += 1
            log("INFO", f"Saved overlay: {overlay_path}")
        if parse_bool(args.save_mask_bbox):
            draw_predictions(image_path, predictions_vis, mask_bbox_path, mask_bbox_only=True)
            counters["mask_bbox"] += 1
            log("INFO", f"Saved mask+bbox: {mask_bbox_path}")

        payload = {
            "image_path": str(image_path),
            "image_stem": image_path.stem,
            "overlay_path": str(overlay_path) if parse_bool(args.save_overlays) else None,
            "mask_bbox_path": str(mask_bbox_path) if parse_bool(args.save_mask_bbox) else None,
            "predictions": predictions_json,
        }
        json_path = json_dir / f"{image_path.stem}.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        counters["images"] += 1
        counters["json"] += 1
        log("INFO", f"Saved prediction: {json_path}")

    log("INFO", f"Images processed: {counters['images']}")
    log("INFO", f"JSON saved: {counters['json']}")
    log("INFO", f"Overlays saved: {counters['overlays']}")
    log("INFO", f"Mask+bbox saved: {counters['mask_bbox']}")
    log("INFO", f"Masks saved: {counters['masks']}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO segmentation on HoloOcean runtime inspection frames.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="data/runtime_teleop/frames")
    parser.add_argument("--output", default="data/yolo_runtime_predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-overlays", default="true")
    parser.add_argument("--save-masks", default="true")
    parser.add_argument("--save-mask-bbox", default="true")
    parser.add_argument("--hide-classes", nargs="*", default=[])
    parser.add_argument("--show-classes", nargs="*", default=[])
    parser.add_argument("--filter-json", action="store_true")
    parser.add_argument("--roi", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main():
    return predict(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
