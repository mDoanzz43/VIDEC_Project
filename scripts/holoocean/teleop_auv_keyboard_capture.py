#!/usr/bin/env python3
import argparse
import ast
import json
import math
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_CONFIG = {
    "scenario": "test_ocean_map-HoveringCamera",
    "agent_name": "auv0",
    "output": "data/runtime_teleop",
    "frame_output": "data/runtime_teleop/frames",
    "metadata_output": "data/runtime_teleop/metadata",
    "color_format": "bgr",
    "fps": 10,
    "save_every": 3,
    "warmup_ticks": 20,
    "action_mode": "thrusters_8",
    "control_mode": "thrusters_8_empirical",
    "preview": True,
    "preview_scale": 1.0,
    "hold_ticks": 10,
    "video_fps_hint": 5,
    "action_scale": 15.0,
    "vertical_scale": 15.0,
    "forward_scale": 15.0,
    "strafe_scale": 5.0,
    "yaw_scale": 5.0,
    "yaw_step_degrees": 10.0,
    "yaw_tolerance_degrees": 3.0,
    "yaw_max_ticks": 90,
    "yaw_left_sign": -1,
    "terminal_note": "Terminal fallback only captures keys typed in the terminal. Tap keys; command-hold mode keeps action for hold_ticks.",
    "thruster_actions": {
        "up": [15, 15, 15, 15, 0, 0, 0, 0],
        "down": [-15, -15, -15, -15, 0, 0, 0, 0],
        "forward": [0, 0, 0, 0, 15, 15, 15, 15],
        "backward": [0, 0, 0, 0, -15, -15, -15, -15],
        "strafe_left": [0, 0, 0, 0, 5, -5, -5, 5],
        "strafe_right": [0, 0, 0, 0, -5, 5, 5, -5],
        "yaw_left": [0, 0, 0, 0, 5, -5, 5, -5],
        "yaw_right": [0, 0, 0, 0, -5, 5, -5, 5],
    },
    "key_mapping": {
        "w": "up",
        "s": "down",
        "q": "forward",
        "e": "backward",
        "a": "strafe_left",
        "d": "strafe_right",
        "u": "yaw_left_10",
        "i": "yaw_right_10",
        "space": "stop",
        "x": "stop",
    },
}


def log(level, message):
    print(f"[{level}] {message}")


def load_config(path):
    if path is None:
        return dict(DEFAULT_CONFIG)
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
    except Exception:
        data = parse_simple_yaml(text)
    else:
        data = yaml.safe_load(text)

    config = dict(DEFAULT_CONFIG)
    if data:
        config.update(data)
        if "key_mapping" in data:
            mapping = dict(DEFAULT_CONFIG["key_mapping"])
            mapping.update(data["key_mapping"])
            config["key_mapping"] = mapping
        if "thruster_actions" in data:
            actions = dict(DEFAULT_CONFIG["thruster_actions"])
            actions.update(data["thruster_actions"])
            config["thruster_actions"] = normalize_thruster_actions(actions)
    return config


def parse_scalar(value):
    value = value.strip()
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_simple_yaml(text):
    data = {}
    current_section = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        if raw_line.startswith("  ") and current_section:
            key, value = raw_line.strip().split(":", 1)
            data[current_section][key.strip()] = parse_scalar(value)
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            data[key] = {}
            current_section = key
        else:
            data[key] = parse_scalar(value)
            current_section = None
    return data


def normalize_thruster_actions(actions):
    normalized = {}
    for name, values in actions.items():
        if isinstance(values, str):
            values = ast.literal_eval(values)
        if len(values) != 8:
            raise ValueError(f"Thruster action '{name}' must have 8 values, got {len(values)}")
        normalized[name] = [float(value) for value in values]
    return normalized


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


def safe_close(env):
    if env is None:
        return
    if hasattr(env, "close"):
        env.close()
    else:
        env.__exit__(None, None, None)


def is_camera_array(value):
    return (
        isinstance(value, np.ndarray)
        and value.ndim == 3
        and value.shape[2] in (3, 4)
        and value.shape[0] > 0
        and value.shape[1] > 0
    )


def find_camera_frame(state):
    for key in ("LeftCamera", "RightCamera"):
        if key in state and is_camera_array(state[key]):
            return key, state[key]
    for key, value in state.items():
        if is_camera_array(value):
            return str(key), value
    return None, None


def convert_camera_frame(frame, color_format):
    image = np.asarray(frame)
    first3 = image[:, :, :3]
    if color_format == "rgb":
        rgb = first3
    else:
        rgb = first3[:, :, ::-1]
    return rgb.astype(np.uint8)


def save_frame(frame, path, color_format):
    Image.fromarray(convert_camera_frame(frame, color_format)).save(path)


class KeyboardState:
    def __init__(self):
        self.old_terminal_settings = None

    def start(self):
        if sys.stdin.isatty():
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            log("INFO", "Keyboard input: nonblocking terminal fallback with command-hold")
        else:
            log("WARN", "stdin is not a TTY; keyboard fallback may not receive keys.")

    def stop(self):
        if self.old_terminal_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)

    def poll(self):
        pressed = set()
        while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            if char == " ":
                pressed.add("space")
            elif char:
                pressed.add(char.lower())
        return pressed


def action_for_command(command, config):
    action = np.zeros(8, dtype=np.float32)
    if not command or command == "stop":
        return action
    thruster_actions = config.get("thruster_actions", DEFAULT_CONFIG["thruster_actions"])
    if command in thruster_actions:
        signs = np.sign(np.asarray(thruster_actions[command], dtype=np.float32))
        group_scale = scale_for_command(command, config)
        global_scale = float(config.get("action_scale", 15.0)) / 15.0
        action += signs * group_scale * global_scale
    return action


def scale_for_command(command, config):
    if command in ("up", "down"):
        return float(config.get("vertical_scale", 15.0))
    if command in ("forward", "backward"):
        return float(config.get("forward_scale", 15.0))
    if command in ("strafe_left", "strafe_right"):
        return float(config.get("strafe_scale", 5.0))
    if command in ("yaw_left", "yaw_right"):
        return float(config.get("yaw_scale", 5.0))
    return float(config.get("action_scale", 15.0))


def command_action_name(command):
    if command in ("yaw_left_10", "yaw_left_30"):
        return "yaw_left"
    if command in ("yaw_right_10", "yaw_right_30"):
        return "yaw_right"
    return command


def log_control_parameters(config):
    log("INFO", f"control_mode={config.get('control_mode')}")
    log("INFO", f"vertical_scale={config.get('vertical_scale')}")
    log("INFO", f"forward_scale={config.get('forward_scale')}")
    log("INFO", f"strafe_scale={config.get('strafe_scale')}")
    log("INFO", f"yaw_scale={config.get('yaw_scale')}")
    log("INFO", f"yaw_step_degrees={config.get('yaw_step_degrees')}")
    log("INFO", f"hold_ticks={config.get('hold_ticks')}")


def pose_xyz(state):
    pose = state.get("PoseSensor")
    if pose is None:
        return None
    try:
        matrix = np.asarray(pose, dtype=np.float64)
        return [float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])]
    except Exception:
        return None


def yaw_degrees_from_pose(pose):
    if pose is None:
        return None
    try:
        matrix = np.asarray(pose, dtype=np.float64)
        return math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    except Exception:
        return None


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def build_metadata(
    frame_id,
    tick_index,
    config,
    action,
    pressed_keys,
    active_command,
    hold_ticks_remaining,
    state,
    camera_key,
    camera_frame,
    rgb_path,
    yaw_mode,
    target_yaw,
    current_yaw,
    yaw_error,
):
    return {
        "frame_id": frame_id,
        "tick_index": tick_index,
        "scenario": config["scenario"],
        "agent_name": config["agent_name"],
        "action": action.tolist(),
        "pressed_keys": sorted(pressed_keys),
        "active_command": active_command,
        "hold_ticks_remaining": hold_ticks_remaining,
        "yaw_mode": yaw_mode,
        "yaw_target": target_yaw,
        "current_yaw": current_yaw,
        "yaw_error": yaw_error,
        "t": json_safe(state.get("t")),
        "state_keys": [str(key) for key in state.keys()],
        "PoseSensor": json_safe(state.get("PoseSensor")),
        "VelocitySensor": json_safe(state.get("VelocitySensor")),
        "IMUSensor": json_safe(state.get("IMUSensor")),
        "DVLSensor": json_safe(state.get("DVLSensor")),
        "DepthSensor": json_safe(state.get("DepthSensor")),
        "camera_key": camera_key,
        "camera_shape": list(camera_frame.shape),
        "rgb_path": str(rgb_path),
        "source": "holoocean_runtime_keyboard_teleop",
    }


class PreviewWindow:
    def __init__(self, enabled, scale):
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self.cv2 = None
        self.available = False
        if not self.enabled:
            return
        try:
            import cv2
        except Exception as exc:
            log("WARN", f"OpenCV unavailable; live preview disabled: {type(exc).__name__}: {exc}")
            return
        self.cv2 = cv2
        self.available = True
        self.window_title = "ViDEC AUV LeftCamera Live"
        self.cv2.namedWindow(self.window_title, self.cv2.WINDOW_NORMAL)
        log("INFO", "OpenCV live preview enabled")

    def show(self, frame, color_format):
        if not self.available:
            return False
        rgb = convert_camera_frame(frame, color_format)
        preview = rgb[:, :, ::-1]
        if self.scale != 1.0:
            h, w = preview.shape[:2]
            new_w = max(1, int(w * self.scale))
            new_h = max(1, int(h * self.scale))
            preview = self.cv2.resize(preview, (new_w, new_h))
        self.cv2.imshow(self.window_title, preview)
        key = self.cv2.waitKey(1) & 0xFF
        return key in (27, ord("q"))

    def close(self):
        if self.available:
            self.cv2.destroyWindow(self.window_title)


def run_teleop(args):
    config = load_config(args.config)
    if args.scenario:
        config["scenario"] = args.scenario
    if args.output:
        config["output"] = args.output
        config["frame_output"] = str(Path(args.output) / "frames")
        config["metadata_output"] = str(Path(args.output) / "metadata")
    if args.action_scale is not None:
        config["action_scale"] = args.action_scale
    if args.strafe_scale is not None:
        config["strafe_scale"] = args.strafe_scale
    if args.yaw_scale is not None:
        config["yaw_scale"] = args.yaw_scale
    if args.yaw_step_degrees is not None:
        config["yaw_step_degrees"] = args.yaw_step_degrees
    if args.save_every is not None:
        config["save_every"] = args.save_every
    if args.preview is not None:
        config["preview"] = args.preview
    if args.preview_scale is not None:
        config["preview_scale"] = args.preview_scale
    if args.hold_ticks is not None:
        config["hold_ticks"] = args.hold_ticks

    if args.dry_run_actions:
        mapping = config.get("key_mapping", DEFAULT_CONFIG["key_mapping"])
        log_control_parameters(config)
        for key, command in mapping.items():
            action_name = command_action_name(command)
            log("INFO", f"key='{key}' command='{command}' action={action_for_command(action_name, config).tolist()}")
        return 0

    import holoocean
    log_control_parameters(config)

    frame_dir = Path(config["frame_output"])
    metadata_dir = Path(config["metadata_output"])
    if not args.no_save:
        frame_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

    keyboard = KeyboardState()
    env = None
    frame_id = 0
    camera_seen = 0
    tick_index = 0
    active_command = None
    active_until_tick = -1
    yaw_mode = False
    yaw_action_name = None
    yaw_start_tick = -1
    target_yaw = None
    current_yaw = None
    yaw_error = None
    mapping = config.get("key_mapping", DEFAULT_CONFIG["key_mapping"])
    preview = PreviewWindow(config.get("preview", True), config.get("preview_scale", 1.0))

    try:
        keyboard.start()
        log("INFO", f"Creating HoloOcean environment: {config['scenario']}")
        env = holoocean.make(config["scenario"])

        for _ in range(int(config.get("warmup_ticks", 20))):
            env.tick()

        log("INFO", "Teleop running. Press Ctrl+C to stop.")
        while True:
            pressed = keyboard.poll()
            for key in sorted(pressed):
                command = mapping.get(key)
                if not command:
                    continue
                if command == "stop":
                    active_command = None
                    active_until_tick = tick_index
                    yaw_mode = False
                    yaw_action_name = None
                    target_yaw = None
                    yaw_error = None
                elif command in ("yaw_left_10", "yaw_right_10", "yaw_left_30", "yaw_right_30"):
                    active_command = None
                    active_until_tick = tick_index
                    yaw_mode = True
                    yaw_action_name = command_action_name(command)
                    yaw_start_tick = tick_index
                    step = float(config.get("yaw_step_degrees", 30.0))
                    sign = float(config.get("yaw_left_sign", -1.0))
                    state_for_yaw = env.tick()
                    tick_index += 1
                    current_yaw = yaw_degrees_from_pose(state_for_yaw.get("PoseSensor"))
                    if current_yaw is None:
                        log("WARN", "PoseSensor unavailable; yaw target command ignored.")
                        yaw_mode = False
                        yaw_action_name = None
                        target_yaw = None
                    else:
                        direction = sign if yaw_action_name == "yaw_left" else -sign
                        target_yaw = wrap_degrees(current_yaw + direction * step)
                        yaw_error = wrap_degrees(target_yaw - current_yaw)
                        log(
                            "INFO",
                            f"key='{key}' command='{command}' current_yaw={current_yaw:.2f} "
                            f"target_yaw={target_yaw:.2f} error={yaw_error:.2f} action_name={yaw_action_name}",
                        )
                else:
                    active_command = command
                    active_until_tick = tick_index + int(config.get("hold_ticks", 15))
                log("INFO", f"key='{key}' command='{command}' active_until_tick={active_until_tick}")

            if tick_index > active_until_tick and not yaw_mode:
                active_command = None
            hold_ticks_remaining = max(0, active_until_tick - tick_index)
            action = action_for_command(yaw_action_name if yaw_mode else active_command, config)
            try:
                env.act(config["agent_name"], action)
            except Exception as exc:
                log("ERROR", f"env.act failed for 8-vector action: {type(exc).__name__}: {exc}")
                log("ERROR", "Run: python scripts/holoocean/diagnose_auv_action_space.py --scenario test_ocean_map-HoveringCamera")
                return 1

            state = env.tick()
            current_yaw = yaw_degrees_from_pose(state.get("PoseSensor"))
            if yaw_mode and current_yaw is not None and target_yaw is not None:
                yaw_error = wrap_degrees(target_yaw - current_yaw)
                yaw_ticks = tick_index - yaw_start_tick
                if tick_index % 15 == 0:
                    log(
                        "INFO",
                        f"yaw_mode current_yaw={current_yaw:.2f} target_yaw={target_yaw:.2f} "
                        f"error={yaw_error:.2f} action_name={yaw_action_name}",
                    )
                if abs(yaw_error) <= float(config.get("yaw_tolerance_degrees", 3.0)):
                    log("INFO", f"Yaw target reached: current_yaw={current_yaw:.2f}, target_yaw={target_yaw:.2f}")
                    yaw_mode = False
                    yaw_action_name = None
                    target_yaw = None
                elif yaw_ticks >= int(config.get("yaw_max_ticks", 90)):
                    log("WARN", f"Yaw target timeout after {yaw_ticks} ticks; stopping yaw action.")
                    yaw_mode = False
                    yaw_action_name = None
                    target_yaw = None

            if tick_index % args.print_pose_every == 0:
                log(
                    "INFO",
                    f"tick={tick_index} pressed={sorted(pressed)} active_command={active_command} "
                    f"yaw_mode={yaw_mode} action={action.tolist()} pose_xyz={pose_xyz(state)} yaw={current_yaw} "
                    f"action_scale={config.get('action_scale')} strafe_scale={config.get('strafe_scale')} "
                    f"yaw_scale={config.get('yaw_scale')} "
                    f"state_keys={[str(key) for key in state.keys()]}",
                )

            camera_key, camera_frame = find_camera_frame(state)
            if camera_frame is not None:
                should_stop = preview.show(camera_frame, config.get("color_format", "bgr"))
                if should_stop:
                    log("INFO", "Preview stop key received; stopping teleop.")
                    return 0
                camera_seen += 1
                if not args.no_save and camera_seen % int(config.get("save_every", 3)) == 0:
                    frame_id += 1
                    rgb_path = frame_dir / f"frame_{frame_id:06d}.png"
                    metadata_path = metadata_dir / f"frame_{frame_id:06d}.json"
                    save_frame(camera_frame, rgb_path, config.get("color_format", "bgr"))
                    metadata = build_metadata(
                        frame_id,
                        tick_index,
                        config,
                        action,
                        pressed,
                        active_command,
                        hold_ticks_remaining,
                        state,
                        camera_key,
                        camera_frame,
                        rgb_path,
                        yaw_mode,
                        target_yaw,
                        current_yaw,
                        yaw_error,
                    )
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    log("INFO", f"Saved frame: {rgb_path}")
                    log("INFO", f"Saved metadata: {metadata_path}")

            tick_index += 1
            time.sleep(max(0.0, 1.0 / float(config.get("fps", 10))))
    except KeyboardInterrupt:
        log("INFO", "Ctrl+C received; stopping teleop.")
        return 0
    finally:
        preview.close()
        keyboard.stop()
        try:
            safe_close(env)
            log("INFO", "Closed HoloOcean environment")
        except Exception as exc:
            log("WARN", f"env close failed: {type(exc).__name__}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Keyboard teleop for ViDEC HoloOcean AUV with frame capture.")
    parser.add_argument("--config", default="configs/holoocean/auv_keyboard_teleop.yaml")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--preview", dest="preview", action="store_true", default=None)
    parser.add_argument("--no-preview", dest="preview", action="store_false")
    parser.add_argument("--preview-scale", type=float, default=None)
    parser.add_argument("--hold-ticks", type=int, default=None)
    parser.add_argument("--action-scale", type=float, default=None)
    parser.add_argument("--strafe-scale", type=float, default=None)
    parser.add_argument("--yaw-scale", type=float, default=None)
    parser.add_argument("--yaw-step-degrees", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--print-pose-every", type=int, default=30)
    parser.add_argument("--dry-run-actions", action="store_true")
    return parser.parse_args()


def main():
    return run_teleop(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
