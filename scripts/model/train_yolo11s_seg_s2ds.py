#!/usr/bin/env python3
import argparse
from pathlib import Path


def log(level, message):
    print(f"[{level}] {message}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11s segmentation on S2DS four-class dataset.")
    parser.add_argument("--data", default="configs/yolo/s2ds_seg4.yaml")
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/videc_yolo")
    parser.add_argument("--name", default="yolo11s_seg_s2ds4_underwater_v1")
    parser.add_argument("--degrees", type=float, default=10)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.4)
    parser.add_argument("--shear", type=float, default=2)
    parser.add_argument("--perspective", type=float, default=0.0005)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--hsv-h", dest="hsv_h", type=float, default=0.03)
    parser.add_argument("--hsv-s", dest="hsv_s", type=float, default=0.5)
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, default=0.4)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--mixup", type=float, default=0.05)
    parser.add_argument("--copy-paste", dest="copy_paste", type=float, default=0.1)
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
    model.train(
        task="segment",
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        fliplr=args.fliplr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
    )

    weights_dir = Path(args.project) / args.name / "weights"
    log("INFO", f"Expected best.pt: {weights_dir / 'best.pt'}")
    log("INFO", f"Expected last.pt: {weights_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
