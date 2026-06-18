#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


SOURCE_LINUX_DIR = Path("/home/manhdoan/coding/holoocean/dist_videc/Linux")
SOURCE_EXECUTABLE = SOURCE_LINUX_DIR / "Holodeck/Binaries/Linux/Holodeck"
HOLOOCEAN_WORLDS_DIR = Path.home() / ".local/share/holoocean/2.3.0/worlds"
OCEAN_PACKAGE_DIR = HOLOOCEAN_WORLDS_DIR / "Ocean"
VIDEC_PACKAGE_DIR = HOLOOCEAN_WORLDS_DIR / "ViDEC"
REPO_SCENARIO_PATH = Path("configs/holoocean/test_ocean_map-HoveringCamera.json")


CONFIG = {
    "name": "ViDEC",
    "platform": "Linux",
    "version": "2.3.0",
    "path": "Linux/Holodeck/Binaries/Linux/Holodeck",
    "worlds": [
        {
            "name": "test_ocean_map",
            "pre_start_steps": 20,
            "env_min": [-1008.0, -1008.0, -440.0],
            "env_max": [1008.0, 1008.0, 100.0],
        }
    ],
}


SCENARIO = {
    "name": "HoveringCamera",
    "world": "test_ocean_map",
    "main_agent": "auv0",
    "ticks_per_sec": 30,
    "frames_per_sec": True,
    "octree_min": 0.02,
    "octree_max": 5.0,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {"sensor_type": "PoseSensor", "socket": "IMUSocket"},
                {"sensor_type": "VelocitySensor", "socket": "IMUSocket"},
                {
                    "sensor_type": "IMUSensor",
                    "socket": "IMUSocket",
                    "Hz": 30,
                    "configuration": {
                        "AccelSigma": 0.00277,
                        "AngVelSigma": 0.00123,
                        "AccelBiasSigma": 0.00141,
                        "AngVelBiasSigma": 0.00388,
                        "ReturnBias": True,
                    },
                },
                {
                    "sensor_type": "GPSSensor",
                    "socket": "IMUSocket",
                    "Hz": 5,
                    "configuration": {"Sigma": 0.5, "Depth": 1, "DepthSigma": 0.25},
                },
                {
                    "sensor_type": "DVLSensor",
                    "socket": "DVLSocket",
                    "Hz": 20,
                    "configuration": {
                        "Elevation": 22.5,
                        "VelSigma": 0.02626,
                        "ReturnRange": True,
                        "MaxRange": 50,
                        "RangeSigma": 0.1,
                    },
                },
                {
                    "sensor_type": "DepthSensor",
                    "socket": "DepthSocket",
                    "Hz": 30,
                    "configuration": {"Sigma": 0.255},
                },
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "LeftCamera",
                    "socket": "CameraLeftSocket",
                    "Hz": 5,
                    "configuration": {"CaptureWidth": 512, "CaptureHeight": 512},
                },
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "RightCamera",
                    "socket": "CameraRightSocket",
                    "Hz": 5,
                    "configuration": {"CaptureWidth": 512, "CaptureHeight": 512},
                },
            ],
            "control_scheme": 0,
            "location": [0.0, 0.0, -5.0],
            "rotation": [0.0, 0.0, 0.0],
        }
    ],
    "window_width": 1280,
    "window_height": 720,
}


def log(level, message):
    print(f"[{level}] {message}")


def require_path(path, description):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    log("INFO", f"Found {description}: {path}")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("INFO", f"Wrote {path}")


def prepare_destination(no_backup):
    if not VIDEC_PACKAGE_DIR.exists():
        VIDEC_PACKAGE_DIR.mkdir(parents=True)
        log("INFO", f"Created package folder: {VIDEC_PACKAGE_DIR}")
        return

    if no_backup:
        raise FileExistsError(
            f"Destination already exists and --no-backup was passed: {VIDEC_PACKAGE_DIR}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = VIDEC_PACKAGE_DIR.with_name(f"ViDEC_backup_{timestamp}")
    shutil.move(str(VIDEC_PACKAGE_DIR), str(backup_path))
    log("WARN", f"Existing ViDEC package moved to backup: {backup_path}")
    VIDEC_PACKAGE_DIR.mkdir(parents=True)
    log("INFO", f"Created package folder: {VIDEC_PACKAGE_DIR}")


def register_package(args):
    require_path(SOURCE_LINUX_DIR, "source packaged Linux build")
    require_path(SOURCE_EXECUTABLE, "source Holodeck executable")
    require_path(OCEAN_PACKAGE_DIR, "reference Ocean package")
    require_path(OCEAN_PACKAGE_DIR / "materials.csv", "reference Ocean materials.csv")

    prepare_destination(args.no_backup)

    target_linux_dir = VIDEC_PACKAGE_DIR / "Linux"
    shutil.copytree(SOURCE_LINUX_DIR, target_linux_dir)
    log("INFO", f"Copied Linux build to: {target_linux_dir}")

    shutil.copy2(OCEAN_PACKAGE_DIR / "materials.csv", VIDEC_PACKAGE_DIR / "materials.csv")
    log("INFO", f"Copied materials.csv to: {VIDEC_PACKAGE_DIR / 'materials.csv'}")

    config_path = VIDEC_PACKAGE_DIR / "config.json"
    scenario_path = VIDEC_PACKAGE_DIR / "test_ocean_map-HoveringCamera.json"
    write_json(config_path, CONFIG)
    write_json(scenario_path, SCENARIO)
    write_json(REPO_SCENARIO_PATH, SCENARIO)

    log("INFO", f"Package folder: {VIDEC_PACKAGE_DIR}")
    log("INFO", f"Config path: {config_path}")
    log("INFO", f"Scenario path: {scenario_path}")
    log(
        "INFO",
        "Suggested test command: python scripts/test_videc_holoocean_make.py "
        "--scenario test_ocean_map-HoveringCamera",
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register the custom ViDEC packaged build as a separate HoloOcean world package."
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Fail if the ViDEC package already exists instead of moving it to a timestamped backup.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        return register_package(args)
    except Exception as exc:
        log("ERROR", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
