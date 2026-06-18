#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    0: (255, 255, 255),
    1: (255, 60, 60),
    2: (255, 220, 0),
    3: (0, 220, 255),
}


def log(level, message):
    print(f"[{level}] {message}")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


def load_names(yaml_path):
    try:
        import yaml
    except Exception:
        yaml = None

    text = Path(yaml_path).read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
        return {int(key): str(value) for key, value in data.get("names", {}).items()}

    names = {}
    in_names = False
    for line in text.splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            if not line.startswith("  "):
                break
            key, value = line.strip().split(":", 1)
            names[int(key)] = value.strip()
    return names


def image_files(image_dir):
    files = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(sorted(image_dir.glob(pattern)))
    return files


def label_path_for(image_path, dataset, split):
    return Path(dataset) / "labels" / split / f"{image_path.stem}.txt"


def parse_label_file(label_path, names, epsilon=1e-6):
    if not label_path.exists():
        return [], True, True, "missing label file"
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return [], True, False, ""

    instances = []
    invalid = False
    notes = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) < 7:
            invalid = True
            notes.append(f"line {line_number}: fewer than 3 polygon points")
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            invalid = True
            notes.append(f"line {line_number}: parse error")
            continue
        if class_id not in names:
            invalid = True
            notes.append(f"line {line_number}: unknown class_id {class_id}")
        if len(coords) % 2 != 0:
            invalid = True
            notes.append(f"line {line_number}: odd coordinate count")
            continue
        if len(coords) < 6:
            invalid = True
            notes.append(f"line {line_number}: polygon has fewer than 3 points")
            continue
        outside = [value for value in coords if value < -epsilon or value > 1.0 + epsilon]
        if outside:
            invalid = True
            notes.append(f"line {line_number}: coordinates outside [0,1]")
        instances.append({"class_id": class_id, "coords": coords})
    return instances, False, invalid, "; ".join(notes)


def polygon_pixels(coords, width, height):
    points = []
    for idx in range(0, len(coords), 2):
        x = max(0, min(width - 1, coords[idx] * width))
        y = max(0, min(height - 1, coords[idx + 1] * height))
        points.append((x, y))
    return points


def draw_overlay(image_path, instances, names, empty_label, invalid_label):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    if empty_label:
        draw.rectangle([0, 0, width, 28], fill=(0, 0, 0, 170))
        draw.text((6, 6), "empty label / background", fill=(255, 255, 255, 255))
        return image

    for instance in instances:
        class_id = instance["class_id"]
        color = COLORS.get(class_id, (0, 255, 0))
        points = polygon_pixels(instance["coords"], width, height)
        if len(points) < 3:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        draw.polygon(points, fill=(*color, 70))
        draw.line(points + [points[0]], fill=(*color, 255), width=2)
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        draw.rectangle(bbox, outline=(*color, 255), width=2)
        label = names.get(class_id, str(class_id))
        label_y = max(0, bbox[1] - 16)
        draw.rectangle([bbox[0], label_y, bbox[0] + 120, label_y + 16], fill=(*color, 210))
        draw.text((bbox[0] + 3, label_y + 2), label, fill=(0, 0, 0, 255))

    if invalid_label:
        draw.rectangle([0, height - 24, width, height], fill=(180, 0, 0, 190))
        draw.text((6, height - 20), "invalid label warning", fill=(255, 255, 255, 255))
    return image


def make_grid(images, output_path):
    if not images:
        log("WARN", "No samples selected for grid.")
        return
    thumb_w, thumb_h = 240, 240
    cols = 6
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    for idx, image in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w + (thumb_w - thumb.width) // 2
        y = (idx // cols) * thumb_h + (thumb_h - thumb.height) // 2
        grid.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    log("INFO", f"Saved grid: {output_path}")


def visualize(args):
    rng = random.Random(args.seed)
    names = load_names(args.yaml)
    image_dir = Path(args.dataset) / "images" / args.split
    images = image_files(image_dir)
    rng.shuffle(images)

    selected_overlays = []
    rows = []
    individual_dir = Path(f"reports/figures/yolo_seg_label_visual_check_{args.split}")
    include_empty = parse_bool(args.include_empty)
    save_individual = parse_bool(args.save_individual)

    for image_path in images:
        label_path = label_path_for(image_path, args.dataset, args.split)
        instances, empty_label, invalid_label, notes = parse_label_file(label_path, names)
        if empty_label and not include_empty:
            continue
        if invalid_label:
            log("WARN", f"Invalid label for {image_path}: {notes}")

        class_ids = sorted({instance["class_id"] for instance in instances})
        class_names = [names.get(class_id, str(class_id)) for class_id in class_ids]
        overlay = draw_overlay(image_path, instances, names, empty_label, invalid_label)
        selected_overlays.append(overlay)
        if save_individual:
            individual_dir.mkdir(parents=True, exist_ok=True)
            overlay.save(individual_dir / f"{image_path.stem}_overlay.png")

        rows.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "num_instances": len(instances),
                "class_ids": ";".join(str(class_id) for class_id in class_ids),
                "class_names": ";".join(class_names),
                "empty_label": empty_label,
                "invalid_label": invalid_label,
                "notes": notes,
            }
        )
        if len(selected_overlays) >= args.samples:
            break

    make_grid(selected_overlays, Path(args.output))
    summary_path = Path(f"reports/results/yolo_seg_label_visual_check_{args.split}.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "image_path",
            "label_path",
            "num_instances",
            "class_ids",
            "class_names",
            "empty_label",
            "invalid_label",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log("INFO", f"Saved CSV summary: {summary_path}")
    log("INFO", f"Selected {len(rows)} sample(s)")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO segmentation labels before training.")
    parser.add_argument("--dataset", default="data/yolo_s2ds_seg4")
    parser.add_argument("--yaml", default="configs/yolo/s2ds_seg4.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="reports/figures/yolo_seg_label_visual_check_train.png")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-empty", default="false")
    parser.add_argument("--save-individual", default="false")
    return parser.parse_args()


def main():
    visualize(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
