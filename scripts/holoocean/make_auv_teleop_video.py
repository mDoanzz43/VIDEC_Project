#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def log(level, message):
    print(f"[{level}] {message}")


def clamp(value, low, high):
    return max(low, min(high, value))


def compute_auto_fps(metadata_dir):
    metadata_paths = sorted(Path(metadata_dir).glob("frame_*.json"))
    times = []
    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log("WARN", f"Skipping unreadable metadata {metadata_path}: {type(exc).__name__}: {exc}")
            continue
        t_value = payload.get("t")
        if isinstance(t_value, (int, float)):
            times.append(float(t_value))

    if len(times) < 2:
        log("WARN", f"Not enough metadata timestamps for auto FPS in {metadata_dir}")
        return None

    duration = times[-1] - times[0]
    if duration <= 0:
        log("WARN", f"Invalid metadata timestamp duration: {duration}")
        return None

    fps = (len(times) - 1) / duration
    fps = clamp(fps, 1.0, 30.0)
    log("INFO", f"Computed FPS from metadata: {fps:.3f}")
    return fps


def make_video(args):
    frame_dir = Path(args.frames)
    output_path = Path(args.output)
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        log("ERROR", f"No frame_*.png files found in {frame_dir}")
        return 1

    fps = float(args.fps)
    if args.auto_fps:
        metadata_dir = Path(args.metadata_dir) if args.metadata_dir else frame_dir.parent / "metadata"
        computed_fps = compute_auto_fps(metadata_dir)
        if computed_fps is not None:
            fps = computed_fps

    estimated_duration = len(frames) / fps if fps > 0 else 0
    log("INFO", f"Frame count: {len(frames)}")
    log("INFO", f"FPS used: {fps:.3f}")
    log("INFO", f"Estimated duration: {estimated_duration:.2f}s")

    try:
        import cv2
    except Exception:
        log("WARN", "OpenCV is not installed. Use ffmpeg manually:")
        log(
            "INFO",
            f"ffmpeg -framerate {fps:.3f} -i {frame_dir}/frame_%06d.png "
            f"-pix_fmt yuv420p {output_path}",
        )
        return 1

    first = cv2.imread(str(frames[0]))
    if first is None:
        log("ERROR", f"Could not read first frame: {frames[0]}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        log("ERROR", f"Could not open video writer: {output_path}")
        return 1

    written = 0
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            log("WARN", f"Skipping unreadable frame: {frame_path}")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1
    writer.release()

    log("INFO", f"Wrote {written} frame(s) to video: {output_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Encode saved ViDEC AUV teleop frames into MP4.")
    parser.add_argument("--frames", default="data/runtime_teleop/frames")
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--output", default="reports/figures/auv_keyboard_teleop.mp4")
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("--auto-fps", action="store_true")
    return parser.parse_args()


def main():
    return make_video(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
