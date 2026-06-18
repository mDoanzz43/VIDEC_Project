#!/usr/bin/env python3
import argparse
import csv
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def log(level, message):
    print(f"[{level}] {message}")


def require_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for geometric augmentation. Install opencv-python.") from exc
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


def apply_geometric(image, mask, rng):
    cv2 = require_cv2()
    img = np.array(image.convert("RGB"))
    msk = np.array(mask.convert("RGB"))
    h, w = img.shape[:2]

    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.9, 1.1)
    tx = rng.uniform(-0.06, 0.06) * w
    ty = rng.uniform(-0.06, 0.06) * h
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[:, 2] += [tx, ty]

    img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    msk = cv2.warpAffine(msk, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    if rng.random() < 0.45:
        jitter = 0.04
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = src + np.float32(
            [
                [rng.uniform(-jitter, jitter) * w, rng.uniform(-jitter, jitter) * h],
                [rng.uniform(-jitter, jitter) * w, rng.uniform(-jitter, jitter) * h],
                [rng.uniform(-jitter, jitter) * w, rng.uniform(-jitter, jitter) * h],
                [rng.uniform(-jitter, jitter) * w, rng.uniform(-jitter, jitter) * h],
            ]
        )
        persp = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, persp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        msk = cv2.warpPerspective(msk, persp, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return img, msk


def apply_underwater(image_array, rng, np_rng):
    cv2 = require_cv2()
    img = image_array.astype(np.float32)
    factors = np.array([rng.uniform(0.65, 0.9), rng.uniform(0.9, 1.15), rng.uniform(1.05, 1.35)], dtype=np.float32)
    img *= factors
    contrast = rng.uniform(0.75, 1.15)
    brightness = rng.uniform(-18, 12)
    img = (img - 127.5) * contrast + 127.5 + brightness

    haze_color = np.array([rng.uniform(20, 45), rng.uniform(80, 125), rng.uniform(95, 150)], dtype=np.float32)
    haze_alpha = rng.uniform(0.05, 0.22)
    img = img * (1.0 - haze_alpha) + haze_color * haze_alpha

    if rng.random() < 0.55:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.25:
        kernel = np.zeros((5, 5), dtype=np.float32)
        kernel[2, :] = 1.0 / 5.0
        img = cv2.filter2D(img, -1, kernel)
    if rng.random() < 0.65:
        sigma = rng.uniform(3, 9)
        img += np_rng.normal(0, sigma, img.shape)

    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        buffer = BytesIO()
        Image.fromarray(img).save(buffer, format="JPEG", quality=rng.randint(55, 85))
        buffer.seek(0)
        img = np.array(Image.open(buffer).convert("RGB"))
    return img


def save_preview(samples, output_path):
    if not samples:
        return
    thumbs = []
    for image_path, mask_path in samples:
        image = Image.open(image_path).convert("RGB").resize((180, 180))
        mask = Image.open(mask_path).convert("RGB").resize((180, 180), Image.Resampling.NEAREST)
        overlay = Image.blend(image, mask, 0.35)
        thumbs.append(overlay)
    cols = 6
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * 180, rows * 180), "white")
    for idx, thumb in enumerate(thumbs):
        grid.paste(thumb, ((idx % cols) * 180, (idx // cols) * 180))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    log("INFO", f"Saved preview grid: {output_path}")


def augment(args):
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    rows = []
    preview_samples = []
    output_root = Path(args.output_root)

    for split in args.splits:
        split_dir = Path(args.input_root) / split
        pairs = find_pairs(split_dir)
        log("INFO", f"{split}: found {len(pairs)} image/mask pairs")
        out_split = output_root / split
        out_split.mkdir(parents=True, exist_ok=True)

        if args.limit is not None:
            pairs = pairs[: args.limit]
            log("INFO", f"{split}: limited to {len(pairs)} pair(s)")

        for image_path, mask_path in pairs:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")
            outputs = []
            if args.include_original:
                outputs.append(("orig", np.array(image), np.array(mask)))
            for aug_idx in range(args.num_aug_per_image):
                geo_img, geo_mask = apply_geometric(image, mask, rng)
                aug_img = apply_underwater(geo_img, rng, np_rng)
                outputs.append((f"aug{aug_idx + 1:02d}", aug_img, geo_mask))

            for suffix, out_img, out_mask in outputs:
                stem = f"{image_path.stem}_{suffix}"
                out_image_path = out_split / f"{stem}.png"
                out_mask_path = out_split / f"{stem}_lab.png"
                Image.fromarray(out_img).save(out_image_path)
                Image.fromarray(out_mask.astype(np.uint8)).save(out_mask_path)
                if len(preview_samples) < args.preview_count:
                    preview_samples.append((out_image_path, out_mask_path))
                rows.append(
                    {
                        "split": split,
                        "source_image": str(image_path),
                        "source_mask": str(mask_path),
                        "output_image": str(out_image_path),
                        "output_mask": str(out_mask_path),
                        "augmentation": suffix,
                    }
                )

    summary_path = Path("reports/results/s2ds_underwater_augmentation_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["split"])
        writer.writeheader()
        writer.writerows(rows)
    log("INFO", f"Saved summary: {summary_path}")
    save_preview(preview_samples, Path("reports/figures/s2ds_underwater_augmentation_preview.png"))
    log("INFO", f"Saved {len(rows)} augmented/copy samples")


def parse_args():
    parser = argparse.ArgumentParser(description="Create underwater-style offline augmentations for S2DS.")
    parser.add_argument("--input-root", default="data/raw/s2ds")
    parser.add_argument("--output-root", default="data/augmented/s2ds_underwater")
    parser.add_argument("--num-aug-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of image/mask pairs per split for smoke tests.",
    )
    return parser.parse_args()


def main():
    augment(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
