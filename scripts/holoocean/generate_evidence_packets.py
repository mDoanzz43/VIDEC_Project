#!/usr/bin/env python3
import argparse
import csv
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


MODES = ["full_frame", "roi_crop", "mask_bbox", "metadata_only"]


def json_size_bytes(data):
    return len(json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def resolve_path(path_text):
    path = Path(path_text)
    if path.exists():
        return path
    return Path.cwd() / path_text


def read_annotations(data_root):
    annotation_dir = data_root / "annotations"
    if not annotation_dir.exists():
        raise RuntimeError(f"Annotation folder not found: {annotation_dir}")

    annotations = []
    for path in sorted(annotation_dir.glob("*.json")):
        annotations.append((path, json.loads(path.read_text(encoding="utf-8"))))
    if not annotations:
        raise RuntimeError(f"No annotation JSON files found in {annotation_dir}")
    return annotations


def crop_roi(image_path, bbox, output_path):
    x, y, w, h = bbox
    image = Image.open(image_path).convert("RGB")
    crop = image.crop((x, y, x + w, y + h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    return output_path.stat().st_size


def metadata_payload(annotation, defect):
    return {
        "image_id": annotation["image_id"],
        "defect_id": defect["defect_id"],
        "class": defect["class"],
        "bbox": defect["bbox"],
        "area_px": defect["metrology"]["area_px"],
        "severity": defect.get("metrology", {}).get("severity"),
        "recommended_action": defect["verification"]["recommended_action"],
    }


def mask_bbox_payload(annotation, defect, mask_path):
    return {
        "image_id": annotation["image_id"],
        "defect_id": defect["defect_id"],
        "mode": "mask_bbox",
        "mask_path": str(mask_path),
        "class": defect["class"],
        "bbox": defect["bbox"],
        "area_px": defect["metrology"]["area_px"],
    }


def packet_record(image_id, defect_id, mode, size_bytes, full_size, status):
    ratio = round(full_size / size_bytes, 4) if size_bytes else 0.0
    return {
        "image_id": image_id,
        "defect_id": defect_id,
        "mode": mode,
        "size_bytes": int(size_bytes),
        "compression_ratio_vs_full_frame": ratio,
        "verification_status": status,
    }


def build_packets(annotation_path, annotation, output_root):
    image_id = annotation["image_id"]
    image_path = resolve_path(annotation["image_path"])
    mask_path = resolve_path(annotation["mask_path"])
    full_size = image_path.stat().st_size

    records = []
    packet = {
        "image_id": image_id,
        "source_annotation": str(annotation_path),
        "packets": [],
    }

    for defect in annotation["defects"]:
        defect_id = defect["defect_id"]
        bbox = defect["bbox"]

        full_frame = packet_record(image_id, defect_id, "full_frame", full_size, full_size, "confirmed")
        full_frame["payload"] = {"image_path": str(image_path)}
        records.append(full_frame)

        roi_path = output_root / "roi_crops" / f"{image_id}_{defect_id}.png"
        roi_size = crop_roi(image_path, bbox, roi_path)
        roi = packet_record(image_id, defect_id, "roi_crop", roi_size, full_size, "confirmed")
        roi["payload"] = {"roi_path": str(roi_path), "bbox": bbox}
        records.append(roi)

        mask_payload = mask_bbox_payload(annotation, defect, mask_path)
        mask_meta_size = json_size_bytes(mask_payload)
        mask_size = mask_path.stat().st_size + mask_meta_size
        mask_bbox = packet_record(
            image_id,
            defect_id,
            "mask_bbox",
            mask_size,
            full_size,
            "confirmed",
        )
        mask_bbox["payload"] = mask_payload
        records.append(mask_bbox)

        meta_payload = metadata_payload(annotation, defect)
        meta_size = json_size_bytes(meta_payload)
        metadata = packet_record(
            image_id,
            defect_id,
            "metadata_only",
            meta_size,
            full_size,
            "uncertain",
        )
        metadata["payload"] = meta_payload
        records.append(metadata)

    packet["packets"] = records
    return packet, records


def write_packet(packet, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{packet['image_id']}_packet.json"
    path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    return path


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "defect_id",
                "mode",
                "size_bytes",
                "compression_ratio_vs_full_frame",
                "verification_status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def write_visualization_summary(rows, representative_rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "defect_id",
        "mode",
        "size_bytes",
        "size_kb",
        "compression_ratio_vs_full_frame",
        "verification_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in representative_rows:
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "defect_id": row["defect_id"],
                    "mode": row["mode"],
                    "size_bytes": row["size_bytes"],
                    "size_kb": round(row["size_bytes"] / 1024, 2),
                    "compression_ratio_vs_full_frame": row["compression_ratio_vs_full_frame"],
                    "verification_status": row["verification_status"],
                }
            )


def average_sizes(rows):
    averages = {}
    for mode in MODES:
        values = [row["size_bytes"] for row in rows if row["mode"] == mode]
        averages[mode] = sum(values) / len(values) if values else 0
    return averages


def draw_packet_size_chart(rows, path):
    averages = average_sizes(rows)
    width, height = 900, 520
    margin_left, margin_bottom = 120, 90
    chart_w, chart_h = width - margin_left - 60, height - 80 - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    draw.text((margin_left, 28), "Evidence Packet Size Comparison", fill=(0, 0, 0), font=title_font)
    max_value = max(averages.values()) if averages else 1
    max_value = max(max_value, 1)
    bar_w = chart_w // (len(MODES) * 2)

    axis_x = margin_left
    axis_y = height - margin_bottom
    draw.line((axis_x, 70, axis_x, axis_y), fill=(0, 0, 0), width=2)
    draw.line((axis_x, axis_y, width - 40, axis_y), fill=(0, 0, 0), width=2)

    colors = {
        "full_frame": (70, 118, 190),
        "roi_crop": (74, 160, 110),
        "mask_bbox": (215, 155, 55),
        "metadata_only": (160, 95, 180),
    }

    for index, mode in enumerate(MODES):
        value = averages[mode]
        bar_h = int((value / max_value) * chart_h)
        x0 = margin_left + 50 + index * bar_w * 2
        y0 = axis_y - bar_h
        x1 = x0 + bar_w
        draw.rectangle((x0, y0, x1, axis_y), fill=colors[mode])
        draw.text((x0 - 10, axis_y + 12), mode, fill=(0, 0, 0), font=label_font)
        draw.text((x0 - 4, max(72, y0 - 22)), f"{int(value)} B", fill=(0, 0, 0), font=label_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def load_fonts():
    try:
        return (
            ImageFont.truetype("DejaVuSans-Bold.ttf", 28),
            ImageFont.truetype("DejaVuSans-Bold.ttf", 22),
            ImageFont.truetype("DejaVuSans.ttf", 18),
            ImageFont.truetype("DejaVuSansMono.ttf", 16),
        )
    except OSError:
        font = ImageFont.load_default()
        return font, font, font, font


def fit_image(image, max_size):
    max_w, max_h = max_size
    scale = min(max_w / image.width, max_h / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def draw_wrapped_text(draw, xy, text, font, fill=(0, 0, 0), width=54, line_spacing=6):
    x, y = xy
    for paragraph in str(text).splitlines():
        lines = textwrap.wrap(paragraph, width=width) or [""]
        for line in lines:
            draw.text((x, y), line, fill=fill, font=font)
            bbox = draw.textbbox((x, y), line, font=font)
            y += bbox[3] - bbox[1] + line_spacing
    return y


def size_text(row):
    return f"{row['size_bytes']} B ({row['size_bytes'] / 1024:.2f} KB)"


def draw_panel(canvas, box, title, lines, image=None, text_card=None):
    title_font, subtitle_font, body_font, mono_font = load_fonts()
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=(248, 250, 250), outline=(32, 45, 52), width=2)
    draw.text((x0 + 24, y0 + 18), title, fill=(10, 25, 32), font=title_font)

    text_y = y0 + 58
    for line in lines:
        draw.text((x0 + 24, text_y), line, fill=(20, 38, 44), font=body_font)
        text_y += 25

    content_y = y0 + 165
    content_w = x1 - x0 - 48
    content_h = y1 - content_y - 28
    if image is not None:
        fitted = fit_image(image, (content_w, content_h))
        px = x0 + 24 + (content_w - fitted.width) // 2
        py = content_y + (content_h - fitted.height) // 2
        draw.rectangle((px - 2, py - 2, px + fitted.width + 2, py + fitted.height + 2), outline=(70, 85, 90), width=2)
        canvas.paste(fitted, (px, py))
    elif text_card is not None:
        card = Image.new("RGB", (content_w, content_h), (255, 255, 236))
        card_draw = ImageDraw.Draw(card)
        card_draw.rectangle((0, 0, content_w - 1, content_h - 1), outline=(70, 85, 90), width=2)
        draw_wrapped_text(card_draw, (22, 22), text_card, mono_font, width=56)
        canvas.paste(card, (x0 + 24, content_y))


def mask_bbox_overlay(image_path, mask_path, bbox):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    overlay = image.copy()
    pixels = overlay.load()
    mask_pixels = mask.load()
    for y in range(overlay.height):
        for x in range(overlay.width):
            if mask_pixels[x, y] > 0:
                r, g, b = pixels[x, y]
                pixels[x, y] = (int(0.65 * r + 0.35 * 255), int(0.65 * g), int(0.65 * b))
    draw = ImageDraw.Draw(overlay)
    x, y, w, h = bbox
    draw.rectangle((x, y, x + w, y + h), outline=(255, 220, 0), width=4)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 12), f"bbox={bbox}", fill=(0, 0, 0), font=font, stroke_width=3, stroke_fill=(255, 220, 0))
    return overlay


def draw_evidence_visualization(annotation, representative_rows, path):
    rows_by_mode = {row["mode"]: row for row in representative_rows}
    defect = annotation["defects"][0]
    image_path = resolve_path(annotation["image_path"])
    mask_path = resolve_path(annotation["mask_path"])
    bbox = defect["bbox"]
    x, y, w, h = bbox

    full_image = Image.open(image_path).convert("RGB")
    roi_image = full_image.crop((x, y, x + w, y + h))
    mask_overlay = mask_bbox_overlay(image_path, mask_path, bbox)

    width, height = 1920, 1180
    canvas = Image.new("RGB", (width, height), (232, 238, 240))
    draw = ImageDraw.Draw(canvas)
    title_font, _, body_font, _ = load_fonts()
    draw.text(
        (42, 28),
        "ViDEC Evidence Packet Visualization: communication-constrained defect evidence",
        fill=(8, 26, 34),
        font=title_font,
    )
    draw.text(
        (42, 68),
        "Each panel shows the information sent for one representative generated sample. Sizes are measured from actual files or compact JSON payloads.",
        fill=(30, 48, 54),
        font=body_font,
    )

    panel_boxes = [
        (40, 120, 940, 620),
        (980, 120, 1880, 620),
        (40, 660, 940, 1140),
        (980, 660, 1880, 1140),
    ]

    full_row = rows_by_mode["full_frame"]
    roi_row = rows_by_mode["roi_crop"]
    mask_row = rows_by_mode["mask_bbox"]
    meta_row = rows_by_mode["metadata_only"]
    metadata = meta_row["payload"]

    draw_panel(
        canvas,
        panel_boxes[0],
        "Full frame",
        [
            f"Size: {size_text(full_row)}",
            f"Verification: {full_row['verification_status']}",
        ],
        image=full_image,
    )
    draw_panel(
        canvas,
        panel_boxes[1],
        "ROI crop",
        [
            f"Size: {size_text(roi_row)}",
            f"Compression vs full: {roi_row['compression_ratio_vs_full_frame']}x",
            f"Verification: {roi_row['verification_status']}",
        ],
        image=roi_image,
    )
    draw_panel(
        canvas,
        panel_boxes[2],
        "Mask + BBox",
        [
            f"Size: {size_text(mask_row)}",
            f"Compression vs full: {mask_row['compression_ratio_vs_full_frame']}x",
            f"Verification: {mask_row['verification_status']}",
        ],
        image=mask_overlay,
    )
    metadata_text = "\n".join(
        [
            f"image_id: {metadata['image_id']}",
            f"defect_id: {metadata['defect_id']}",
            f"class: {metadata['class']}",
            f"bbox: {metadata['bbox']}",
            f"area_px: {metadata['area_px']}",
            f"severity: {metadata.get('severity')}",
            f"recommended_action: {metadata['recommended_action']}",
            "",
            f"size_bytes: {meta_row['size_bytes']}",
            f"size_kb: {meta_row['size_bytes'] / 1024:.2f}",
            f"verification_status: {meta_row['verification_status']}",
        ]
    )
    draw_panel(
        canvas,
        panel_boxes[3],
        "Metadata only",
        [
            f"Size: {size_text(meta_row)}",
            f"Verification: {meta_row['verification_status']}",
        ],
        text_card=metadata_text,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ViDEC evidence packet size benchmarks.")
    parser.add_argument("--data", default="data/generated", help="Generated ViDEC data root.")
    parser.add_argument("--output", default="data/generated/evidence", help="Evidence output root.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data)
    output_root = Path(args.output)

    annotations = read_annotations(data_root)
    all_rows = []
    representative_annotation = None
    representative_rows = None
    for annotation_path, annotation in annotations:
        packet, rows = build_packets(annotation_path, annotation, output_root)
        packet_path = write_packet(packet, output_root)
        all_rows.extend(rows)
        if representative_annotation is None:
            representative_annotation = annotation
            representative_rows = rows
        print(f"[INFO] Saved {packet_path}")

    csv_path = Path("reports/results/communication_results.csv")
    write_csv(all_rows, csv_path)
    print(f"[INFO] Saved {csv_path}")

    figure_path = Path("reports/figures/packet_size_comparison.png")
    draw_packet_size_chart(all_rows, figure_path)
    print(f"[INFO] Saved {figure_path}")

    visualization_path = Path("reports/figures/evidence_packet_visualization.png")
    draw_evidence_visualization(representative_annotation, representative_rows, visualization_path)
    print(f"[INFO] Saved {visualization_path}")

    visualization_summary_path = Path("reports/results/evidence_visualization_summary.csv")
    write_visualization_summary(all_rows, representative_rows, visualization_summary_path)
    print(f"[INFO] Saved {visualization_summary_path}")
    print(f"[INFO] Generated {len(all_rows)} evidence packet records")


if __name__ == "__main__":
    raise SystemExit(main())
