#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path

import yaml
from PIL import Image


SUBSETS = {
    "crack": "01_crack",
    "spalling": "02_spalling",
    "corrosion": "03_corrosion",
    "hard_negative": "04_hard_negative",
    "clean_background": "05_clean_background",
    "mixed_skip": "06_mixed_skip",
}

DEFAULT_MAPPING = {
    "s2ds_colors": {
        "0,0,0": "background",
        "255,255,255": "crack",
        "255,0,0": "spalling",
        "255,255,0": "corrosion",
        "0,255,255": "efflorescence",
        "0,255,0": "vegetation",
        "0,0,255": "control_point",
    },
    "s2ds_to_videc": {
        "background": {"target": "clean_background", "role": "negative"},
        "crack": {"target": "crack", "role": "defect"},
        "spalling": {"target": "spalling", "role": "defect"},
        "corrosion": {"target": "corrosion", "role": "defect"},
        "efflorescence": {
            "target": "hard_negative",
            "subtype": "stain_like",
            "role": "distractor",
        },
        "vegetation": {
            "target": "hard_negative",
            "subtype": "algae_like",
            "role": "distractor",
        },
        "control_point": {
            "target": "hard_negative",
            "subtype": "marker_artifact",
            "role": "distractor",
        },
    },
}


def ensure_mapping(path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            print(f"[INFO] Using class mapping: {path}")
            return yaml.safe_load(f)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(DEFAULT_MAPPING, f, sort_keys=False)
    print(f"[INFO] Created class mapping: {path}")
    return DEFAULT_MAPPING


def make_output_dirs(output_root):
    for subset in SUBSETS.values():
        (output_root / subset / "images").mkdir(parents=True, exist_ok=True)
        (output_root / subset / "masks").mkdir(parents=True, exist_ok=True)


def find_pairs(input_root):
    pairs = []
    for split in ["train", "val", "test"]:
        split_dir = input_root / split
        if not split_dir.exists():
            print(f"[INFO] Split not found, skipping: {split_dir}")
            continue

        for image_path in sorted(split_dir.glob("*.png")):
            if image_path.name.endswith("_lab.png"):
                continue
            mask_path = image_path.with_name(f"{image_path.stem}_lab.png")
            if mask_path.exists():
                pairs.append((split, image_path, mask_path))
    return pairs


def mask_colors(mask_path):
    image = Image.open(mask_path).convert("RGBA")
    colors = image.getcolors(maxcolors=256)
    if colors is None:
        raise ValueError(f"Too many mask colors to inspect simply: {mask_path}")

    keys = []
    for _, color in colors:
        rgb = color[:3]
        keys.append(f"{rgb[0]},{rgb[1]},{rgb[2]}")
    return sorted(keys)


def present_classes(mask_path, color_to_class):
    classes = []
    unknown = []
    for color_key in mask_colors(mask_path):
        class_name = color_to_class.get(color_key)
        if class_name is None:
            unknown.append(f"unknown:{color_key}")
        elif class_name != "background":
            classes.append(class_name)
    return sorted(set(classes + unknown))


def assign_subset(classes, mapping):
    if not classes:
        return SUBSETS["clean_background"]

    if any(class_name.startswith("unknown:") for class_name in classes):
        return SUBSETS["mixed_skip"]

    roles = {class_name: mapping[class_name]["role"] for class_name in classes}
    defect_classes = [name for name, role in roles.items() if role == "defect"]
    distractor_classes = [name for name, role in roles.items() if role == "distractor"]

    if defect_classes and distractor_classes:
        return SUBSETS["mixed_skip"]
    if distractor_classes:
        return SUBSETS["hard_negative"]
    if len(defect_classes) == 1 and defect_classes[0] in SUBSETS:
        return SUBSETS[defect_classes[0]]
    return SUBSETS["mixed_skip"]


def copy_pair(split, image_path, mask_path, output_root, subset):
    dest_stem = f"{split}_{image_path.stem}"
    image_dest = output_root / subset / "images" / f"{dest_stem}{image_path.suffix}"
    mask_dest = output_root / subset / "masks" / f"{dest_stem}_lab.png"
    shutil.copy2(image_path, image_dest)
    shutil.copy2(mask_path, mask_dest)


def write_summary(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image_path",
                "mask_path",
                "present_classes",
                "assigned_subset",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare minimal S2DS subset folders.")
    parser.add_argument("--input", default="data/raw/s2ds", help="S2DS dataset root.")
    parser.add_argument(
        "--output",
        default="data/processed/s2ds_subset",
        help="Output directory for filtered subset.",
    )
    parser.add_argument(
        "--mapping",
        default="configs/class_mapping.yaml",
        help="Class mapping YAML path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs for testing.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)

    config = ensure_mapping(Path(args.mapping))
    color_to_class = config["s2ds_colors"]
    class_mapping = config["s2ds_to_videc"]

    make_output_dirs(output_root)
    pairs = find_pairs(input_root)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    print(f"[INFO] Processing {len(pairs)} S2DS pairs")

    rows = []
    subset_counts = {subset: 0 for subset in SUBSETS.values()}
    for split, image_path, mask_path in pairs:
        classes = present_classes(mask_path, color_to_class)
        subset = assign_subset(classes, class_mapping)
        copy_pair(split, image_path, mask_path, output_root, subset)
        subset_counts[subset] += 1

        rows.append(
            {
                "split": split,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "present_classes": "|".join(classes) if classes else "background",
                "assigned_subset": subset,
            }
        )

    summary_path = Path("reports/results/s2ds_filter_summary.csv")
    write_summary(rows, summary_path)

    for subset, count in subset_counts.items():
        print(f"[INFO] {subset}: {count}")
    print(f"[INFO] Saved {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
