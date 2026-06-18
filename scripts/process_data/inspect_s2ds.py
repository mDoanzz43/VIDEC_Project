#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def count_files_by_suffix(root):
    counts = {}
    for path in root.rglob("*"):
        if path.is_file():
            suffix = path.suffix.lower() or "<none>"
            counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items()))


def dataset_structure(root):
    structure = {
        "root": str(root),
        "exists": root.exists(),
        "directories": [],
        "file_counts_by_suffix": count_files_by_suffix(root) if root.exists() else {},
    }

    if not root.exists():
        return structure

    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        files = [path for path in directory.iterdir() if path.is_file()]
        masks = [path for path in files if path.name.endswith("_lab.png")]
        images = [
            path
            for path in files
            if path.suffix.lower() in IMAGE_EXTS and not path.name.endswith("_lab.png")
        ]
        structure["directories"].append(
            {
                "path": str(directory.relative_to(root)),
                "files": len(files),
                "images": len(images),
                "lab_masks": len(masks),
            }
        )

    return structure


def find_pairs(root):
    pairs = []
    for mask_path in sorted(root.rglob("*_lab.png")):
        image_path = mask_path.with_name(mask_path.name.replace("_lab.png", ".png"))
        if image_path.exists():
            pairs.append({"image": image_path, "mask": mask_path})
    return pairs


def unique_values(array, max_items=30):
    if array.ndim == 2:
        values = np.unique(array)
        return {
            "kind": "ids",
            "count": int(values.size),
            "values": [int(value) for value in values[:max_items]],
            "truncated": bool(values.size > max_items),
        }

    colors = np.unique(array.reshape(-1, array.shape[-1]), axis=0)
    return {
        "kind": "colors",
        "count": int(colors.shape[0]),
        "values": [[int(channel) for channel in color] for color in colors[:max_items]],
        "truncated": bool(colors.shape[0] > max_items),
    }


def inspect_pair(pair):
    image = np.array(Image.open(pair["image"]))
    mask = np.array(Image.open(pair["mask"]))
    return {
        "image_path": str(pair["image"]),
        "mask_path": str(pair["mask"]),
        "image_shape": list(image.shape),
        "mask_shape": list(mask.shape),
        "image_dtype": str(image.dtype),
        "mask_dtype": str(mask.dtype),
        "mask_unique": unique_values(mask),
    }


def mask_to_overlay(mask):
    if mask.ndim == 3:
        active = np.any(mask[:, :, :3] != 0, axis=2)
    else:
        active = mask != 0

    overlay = np.zeros((*active.shape, 3), dtype=np.uint8)
    overlay[active] = [255, 0, 0]
    return overlay, active


def save_overlay(pair, output_path):
    image = np.array(Image.open(pair["image"]).convert("RGB"))
    mask = np.array(Image.open(pair["mask"]))
    overlay, active = mask_to_overlay(mask)

    if active.shape != image.shape[:2]:
        raise ValueError(
            f"Image/mask size mismatch: image={image.shape[:2]} mask={active.shape}"
        )

    blended = image.copy()
    blended[active] = (0.6 * image[active] + 0.4 * overlay[active]).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(output_path)


def print_structure(structure):
    print(f"[INFO] S2DS root: {structure['root']}")
    print(f"[INFO] Exists: {structure['exists']}")
    print(f"[INFO] File counts by suffix: {structure['file_counts_by_suffix']}")
    for item in structure["directories"]:
        print(
            "[INFO] Directory "
            f"{item['path']}: files={item['files']} "
            f"images={item['images']} lab_masks={item['lab_masks']}"
        )


def print_inspections(inspections):
    for index, item in enumerate(inspections, start=1):
        print(f"[INFO] Pair {index}")
        print(f"  image: {item['image_path']}")
        print(f"  mask: {item['mask_path']}")
        print(f"  image_shape: {item['image_shape']} dtype={item['image_dtype']}")
        print(f"  mask_shape: {item['mask_shape']} dtype={item['mask_dtype']}")
        print(f"  mask_unique: {item['mask_unique']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect S2DS image/mask pairs.")
    parser.add_argument("--input", default="data/raw/s2ds", help="S2DS dataset root.")
    parser.add_argument("--output", default="reports", help="Report output root.")
    parser.add_argument("--limit", type=int, default=5, help="Number of pairs to inspect.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)

    structure = dataset_structure(input_root)
    pairs = find_pairs(input_root) if input_root.exists() else []
    inspections = [inspect_pair(pair) for pair in pairs[: args.limit]]

    print_structure(structure)
    print(f"[INFO] Detected image/mask pairs: {len(pairs)}")
    print_inspections(inspections)

    result = {
        "dataset_structure": structure,
        "pair_count": len(pairs),
        "pair_rule": "mask path uses *_lab.png; image path uses same stem without _lab",
        "inspected_count": len(inspections),
        "inspections": inspections,
    }

    results_path = output_root / "results" / "s2ds_inspection.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {results_path}")

    if pairs:
        overlay_path = output_root / "figures" / "s2ds_overlay_sample.png"
        save_overlay(pairs[0], overlay_path)
        print(f"[INFO] Saved {overlay_path}")


if __name__ == "__main__":
    raise SystemExit(main())
