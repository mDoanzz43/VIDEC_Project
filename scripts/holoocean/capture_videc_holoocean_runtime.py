#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def log(level, message):
    print(f"[{level}] {message}")


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def is_rgb_array(value):
    return (
        isinstance(value, np.ndarray)
        and value.ndim == 3
        and value.shape[2] in (3, 4)
        and value.shape[0] > 0
        and value.shape[1] > 0
    )


def find_rgb_camera(state):
    for key in ("LeftCamera", "RightCamera"):
        if key in state and is_rgb_array(state[key]):
            return key, state[key]
    for key, value in state.items():
        if is_rgb_array(value):
            return str(key), value
    return None, None


def find_depth(state):
    if "DepthSensor" in state:
        return "DepthSensor", np.asarray(state["DepthSensor"])
    for key, value in state.items():
        if "depth" in str(key).lower():
            return str(key), np.asarray(value)
    return None, None


def depth_stats(depth):
    array = np.asarray(depth, dtype=np.float64)
    return {
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
        "shape": list(array.shape),
    }


def convert_camera_frame(frame, color_format):
    image = np.asarray(frame)
    first3 = image[:, :, :3]
    if color_format == "rgb":
        rgb = first3
        note = "Saved first three channels as RGB; alpha channel ignored if present."
    elif color_format == "bgr":
        rgb = first3[:, :, ::-1]
        note = "Saved first three channels as BGR converted to RGB; alpha channel ignored if present."
    else:
        rgb = first3[:, :, ::-1]
        note = (
            "Auto mode assumes HoloOcean camera data is BGR/BGRA and converts first "
            "three channels to RGB; alpha channel ignored if present."
        )
    return rgb.astype(np.uint8), note


def save_rgb(frame, path, color_format):
    rgb, note = convert_camera_frame(frame, color_format)
    Image.fromarray(rgb).save(path)
    return note


def safe_close(env):
    if env is None:
        return
    if hasattr(env, "close"):
        env.close()
    else:
        env.__exit__(None, None, None)


def build_metadata(args, frame_id, tick_index, state, camera_key, camera_frame, depth_key, depth, paths):
    color_format_used = args.color_format
    if args.color_format == "auto":
        color_format_used = "auto_bgr_to_rgb"
    metadata = {
        "scenario": args.scenario,
        "frame_id": frame_id,
        "tick_index": tick_index,
        "state_keys": [str(key) for key in state.keys()],
        "source": "holoocean_runtime",
        "t": json_safe(state.get("t")),
        "PoseSensor": json_safe(state.get("PoseSensor")),
        "VelocitySensor": json_safe(state.get("VelocitySensor")),
        "IMUSensor": json_safe(state.get("IMUSensor")),
        "DVLSensor": json_safe(state.get("DVLSensor")),
        "GPSSensor": json_safe(state.get("GPSSensor")),
        "DepthSensor": json_safe(state.get("DepthSensor")),
        "DepthSensor_stats": depth_stats(depth) if depth is not None else None,
        "depth_key": depth_key,
        "depth_path": str(paths["depth"]) if paths.get("depth") else None,
        "camera_key": camera_key,
        "camera_shape": list(camera_frame.shape),
        "camera_original_shape": list(camera_frame.shape),
        "color_format_used": color_format_used,
        "channel_note": paths.get("channel_note"),
        "rgb_path": str(paths["rgb"]),
    }
    return metadata


def capture(args):
    import holoocean

    output_dir = Path(args.output)
    rgb_dir = output_dir
    rgb_dir.mkdir(parents=True, exist_ok=True)

    env = None
    saved = 0
    tick_index = -1
    missed_camera_frames = 0
    try:
        log("INFO", f"Creating HoloOcean environment: {args.scenario}")
        env = holoocean.make(args.scenario, show_viewport=False)

        for _ in range(args.warmup):
            env.tick()

        while saved < args.num_frames:
            tick_index += 1
            state = env.tick()
            if tick_index % max(1, args.save_every) != 0:
                continue

            camera_key, frame = find_rgb_camera(state)
            if frame is None:
                missed_camera_frames += 1
                continue

            depth_key, depth = find_depth(state)
            frame_id = saved
            rgb_path = output_dir / f"videc_runtime_rgb_{frame_id:06d}.png"
            metadata_path = output_dir / f"videc_runtime_metadata_{frame_id:06d}.json"
            depth_path = output_dir / f"videc_runtime_depth_{frame_id:06d}.npy" if depth is not None else None

            channel_note = save_rgb(frame, rgb_path, args.color_format)
            if depth is not None:
                np.save(depth_path, depth)

            metadata = build_metadata(
                args,
                frame_id,
                tick_index,
                state,
                camera_key,
                frame,
                depth_key,
                depth,
                {"rgb": rgb_path, "depth": depth_path, "channel_note": channel_note},
            )
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            log("INFO", f"Saved RGB: {rgb_path}")
            if depth_path is not None:
                log("INFO", f"Saved depth: {depth_path}")
            log("INFO", f"Saved metadata: {metadata_path}")
            saved += 1

        log("INFO", f"Saved {saved} runtime frame(s) to {output_dir}")
        if missed_camera_frames:
            log(
                "INFO",
                f"Skipped {missed_camera_frames} tick(s) without RGB camera frames. "
                "This is expected when camera Hz is lower than ticks_per_sec.",
            )
        return 0
    except Exception as exc:
        log("ERROR", f"Runtime capture failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if env is not None:
            try:
                safe_close(env)
                log("INFO", "Closed HoloOcean environment")
            except Exception as exc:
                log("WARN", f"env close failed: {type(exc).__name__}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture RGB/depth/metadata from the ViDEC HoloOcean runtime.")
    parser.add_argument("--scenario", default="test_ocean_map-HoveringCamera")
    parser.add_argument("--output", default="data/runtime_capture")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--color-format",
        choices=["auto", "rgb", "bgr"],
        default="auto",
        help="Camera channel interpretation. auto defaults to BGR/BGRA-to-RGB conversion.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_frames < 1:
        raise ValueError("--num-frames must be at least 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be at least 1")
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
