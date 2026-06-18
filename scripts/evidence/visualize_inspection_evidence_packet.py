#!/usr/bin/env python3
import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def log(level, message):
    print(f"[{level}] {message}")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def make_fonts(base_size):
    return {
        "title": load_font(base_size + 8, bold=True),
        "panel_title": load_font(base_size + 4, bold=True),
        "body": load_font(base_size),
        "metadata": load_font(max(12, base_size - 1)),
        "grid": load_font(max(12, base_size - 2), bold=True),
    }


def wrap_text_pixels(text, font, max_width):
    words = str(text).split()
    lines = []
    current = ""
    scratch = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(scratch)
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def select_packet(packet_dir, sample, frame_id):
    packet_paths = sorted(Path(packet_dir).glob("*_evidence.json"))
    if not packet_paths:
        raise FileNotFoundError(f"No packet JSON files found in {packet_dir}")
    if frame_id:
        for path in packet_paths:
            if frame_id in path.stem:
                return path
        raise FileNotFoundError(f"No packet matching frame id: {frame_id}")
    if sample and sample != "first":
        for path in packet_paths:
            if sample in path.stem:
                return path
        raise FileNotFoundError(f"No packet matching sample: {sample}")
    for path in packet_paths:
        packet = load_json(path)
        if resolve_packet_path(packet, "full_frame_path") or resolve_packet_path(packet, "overlay_path"):
            return path
    return packet_paths[0]


def candidate_stems(packet):
    stems = []
    for value in (packet.get("image_stem"), packet.get("frame_id")):
        if value and value not in stems:
            stems.append(str(value))
            if str(value).isdigit():
                stems.append(f"frame_{int(value):06d}")
    return stems


def path_if_exists(path):
    if path and Path(path).exists():
        return Path(path)
    return None


def resolve_packet_path(packet, key):
    direct = path_if_exists(packet.get(key))
    if direct:
        return direct

    for stem in candidate_stems(packet):
        candidates = {
            "full_frame_path": [Path("data/runtime_teleop/frames") / f"{stem}.png"],
            "overlay_path": [Path("data/yolo_runtime_predictions/overlays") / f"{stem}_overlay.png"],
            "mask_bbox_path": [Path("data/yolo_runtime_predictions/mask_bbox") / f"{stem}_mask_bbox.png"],
            "roi_path": [Path("data/inspection_evidence_packets/roi") / f"{stem}_roi.png"],
        }.get(key, [])
        for candidate in candidates:
            found = path_if_exists(candidate)
            if found:
                return found
    return None


def load_panel_image(path, size=(620, 390), fonts=None):
    resolved = path_if_exists(path)
    if resolved:
        image = Image.open(resolved).convert("RGB")
    else:
        image = Image.new("RGB", size, (235, 235, 235))
        ImageDraw.Draw(image).text(
            (20, 20),
            "Not available",
            fill=(0, 0, 0),
            font=(fonts or make_fonts(22))["body"],
        )
    image.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def load_roi_image(packet, size=(620, 390), padding=20, fonts=None):
    roi_path = resolve_packet_path(packet, "roi_path")
    if roi_path:
        return load_panel_image(roi_path, size, fonts)

    full_path = resolve_packet_path(packet, "full_frame_path")
    bbox = packet.get("compact_metadata", {}).get("bbox")
    if not full_path or not bbox:
        return load_panel_image(None, size, fonts)

    image = Image.open(full_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = bbox
    crop_box = (
        max(0, int(x1) - padding),
        max(0, int(y1) - padding),
        min(width, int(x2) + padding),
        min(height, int(y2) + padding),
    )
    roi = image.crop(crop_box)
    roi.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(roi, ((size[0] - roi.width) // 2, (size[1] - roi.height) // 2))
    return canvas


def panel_with_title(content, title, lines, fonts, size=(700, 560)):
    panel = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(30, 30, 30), width=3)
    draw.text((18, 14), title, fill=(0, 0, 0), font=fonts["panel_title"])
    panel.paste(content, ((size[0] - content.width) // 2, 56))
    y = 56 + content.height + 14
    for line in lines:
        draw.text((18, y), line, fill=(0, 0, 0), font=fonts["body"])
        y += 26
    return panel


def metadata_card(packet, fonts, size=(620, 390)):
    card = Image.new("RGB", size, (248, 248, 248))
    draw = ImageDraw.Draw(card)
    compact = packet.get("compact_metadata", {})
    lines = [
        f"image_id: {compact.get('image_id')}",
        f"frame_id: {compact.get('frame_id')}",
        f"class: {compact.get('top_class')}",
        f"confidence: {compact.get('top_confidence')}",
        f"bbox: {compact.get('bbox')}",
        f"area_px: {compact.get('area_px')}",
        f"depth: {compact.get('depth')}",
        f"pose_xyz: {compact.get('pose_xyz')}",
        f"recommended_action: {compact.get('recommended_action')}",
        f"verification_status: {compact.get('verification_status')}",
    ]
    y = 14
    for line in lines:
        for wrapped in textwrap.wrap(str(line), width=54)[:2]:
            draw.text((16, y), wrapped, fill=(0, 0, 0), font=fonts["metadata"])
            y += 27
        y += 4
        if y > size[1] - 28:
            break
    return card


def file_size(path):
    resolved = path_if_exists(path)
    return resolved.stat().st_size if resolved else 0


def bytes_line(label, packet, key, path=None):
    actual_size = file_size(path)
    size_b = actual_size or packet.get("size_bytes", {}).get(key, 0)
    size_kb = round(size_b / 1024.0, 3)
    return f"{label}: {size_b} bytes ({size_kb} KB)"


def ratio_line(label, packet, key):
    value = packet.get("compression_vs_full", {}).get(key)
    return f"{label}: {value}" if value is not None else f"{label}: n/a"


def make_four_panel(packet_path, output_path, title, font_size):
    packet = load_json(packet_path)
    fonts = make_fonts(font_size)
    status = packet.get("verification_status", "unknown")
    full_path = resolve_packet_path(packet, "full_frame_path")
    roi_path = resolve_packet_path(packet, "roi_path")
    mask_bbox_path = resolve_packet_path(packet, "mask_bbox_path")
    overlay_path = resolve_packet_path(packet, "overlay_path")
    if mask_bbox_path is None and overlay_path is not None:
        log("WARN", "mask_bbox image missing; using overlay image as fallback for Mask + BBox panel.")
        mask_bbox_path = overlay_path
    if roi_path is None:
        log("WARN", "ROI image missing; using in-memory crop from full frame if bbox is available.")

    full = panel_with_title(
        load_panel_image(full_path, fonts=fonts),
        "Full frame RGB",
        [bytes_line("size", packet, "full_frame", full_path), f"verification: {status}"],
        fonts,
    )
    roi = panel_with_title(
        load_roi_image(packet, fonts=fonts),
        "ROI crop",
        [bytes_line("size", packet, "roi_crop", roi_path), ratio_line("vs full", packet, "roi_crop_vs_full"), f"verification: {status}"],
        fonts,
    )
    mask_bbox = panel_with_title(
        load_panel_image(mask_bbox_path, fonts=fonts),
        "Mask + BBox",
        [bytes_line("size", packet, "mask_bbox", mask_bbox_path), ratio_line("vs full", packet, "mask_bbox_vs_full"), f"verification: {status}"],
        fonts,
    )
    metadata = panel_with_title(
        metadata_card(packet, fonts),
        "Metadata only",
        [bytes_line("size", packet, "metadata_compact"), ratio_line("vs full", packet, "metadata_compact_vs_full"), f"verification: {status}"],
        fonts,
    )

    canvas_width = 1400
    title_lines = wrap_text_pixels(title, fonts["title"], canvas_width - 48)
    title_line_height = font_size + 14
    panel_top = 18 + len(title_lines) * title_line_height + 18
    canvas_height = panel_top + 1120
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_y = 14
    for line in title_lines:
        draw.text((24, title_y), line, fill=(0, 0, 0), font=fonts["title"])
        title_y += title_line_height
    canvas.paste(full, (0, panel_top))
    canvas.paste(roi, (700, panel_top))
    canvas.paste(mask_bbox, (0, panel_top + 560))
    canvas.paste(metadata, (700, panel_top + 560))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    log("INFO", f"Saved 4-panel evidence figure: {output_path}")


def make_grid(packet_dir, font_size):
    fonts = make_fonts(font_size)
    packet_paths = sorted(Path(packet_dir).glob("*_evidence.json"))[:6]
    if not packet_paths:
        return
    tiles = []
    for packet_path in packet_paths:
        packet = load_json(packet_path)
        img = load_panel_image(resolve_packet_path(packet, "overlay_path"), size=(420, 300), fonts=fonts)
        draw = ImageDraw.Draw(img)
        compact = packet.get("compact_metadata", {})
        draw.text(
            (10, 10),
            f"{packet.get('frame_id')} {compact.get('top_class')} {compact.get('top_confidence')}",
            fill=(255, 255, 0),
            font=fonts["grid"],
        )
        tiles.append(img)
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * 420, rows * 300), "white")
    for idx, tile in enumerate(tiles):
        grid.paste(tile, ((idx % cols) * 420, (idx // cols) * 300))
    output_path = Path("reports/figures/inspection_evidence_packet_grid.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    log("INFO", f"Saved packet grid: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ViDEC inspection evidence packet as a 4-panel figure.")
    parser.add_argument("--packets", default="data/inspection_evidence_packets/packets")
    parser.add_argument("--output", default="reports/figures/inspection_evidence_packet_4panel.png")
    parser.add_argument("--sample", default="first")
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--title", default="ViDEC Evidence Packet Visualization: communication-constrained defect evidence")
    parser.add_argument("--make-grid", action="store_true")
    parser.add_argument("--font-size", type=int, default=22)
    return parser.parse_args()


def main():
    args = parse_args()
    packet_path = select_packet(args.packets, args.sample, args.frame_id)
    make_four_panel(packet_path, Path(args.output), args.title, args.font_size)
    if args.make_grid:
        make_grid(args.packets, args.font_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
