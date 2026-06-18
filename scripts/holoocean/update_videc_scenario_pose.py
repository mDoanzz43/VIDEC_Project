#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_SCENARIO_NAME = "test_ocean_map-HoveringCamera"
DEFAULT_WORLD_DIR = Path.home() / ".local/share/holoocean/2.3.0/worlds/ViDEC"
DEFAULT_REPO_COPY = Path("configs/holoocean/test_ocean_map-HoveringCamera.json")


def log(level, message):
    print(f"[{level}] {message}")


def scenario_path(world_dir, scenario_name):
    return Path(world_dir).expanduser() / f"{scenario_name}.json"


def load_scenario(path):
    if not path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_scenario(scenario):
    agents = scenario.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Scenario must contain agents[0].")
    if not isinstance(agents[0], dict):
        raise ValueError("Scenario agents[0] must be an object.")


def update_dvl_hz(scenario):
    changed = False
    for agent in scenario.get("agents", []):
        for sensor in agent.get("sensors", []):
            if sensor.get("sensor_type") == "DVLSensor" and sensor.get("Hz") != 30:
                log("WARN", f"Updating DVLSensor Hz from {sensor.get('Hz')} to 30")
                sensor["Hz"] = 30
                changed = True
    return changed


def write_scenario(path, scenario):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    log("INFO", f"Wrote scenario JSON: {path}")


def update_pose(location, rotation, scenario_name, world_dir, repo_copy):
    installed_path = scenario_path(world_dir, scenario_name)
    repo_copy = Path(repo_copy)

    if "Ocean" in installed_path.parts and "ViDEC" not in installed_path.parts:
        raise ValueError(f"Refusing to modify non-ViDEC world path: {installed_path}")

    scenario = load_scenario(installed_path)
    validate_scenario(scenario)

    agent = scenario["agents"][0]
    old_location = agent.get("location")
    old_rotation = agent.get("rotation")
    log("INFO", f"Old location: {old_location}")
    log("INFO", f"Old rotation: {old_rotation}")

    agent["location"] = [float(value) for value in location]
    agent["rotation"] = [float(value) for value in rotation]
    update_dvl_hz(scenario)

    write_scenario(installed_path, scenario)
    write_scenario(repo_copy, scenario)

    log("INFO", f"New location: {agent['location']}")
    log("INFO", f"New rotation: {agent['rotation']}")
    log("INFO", "Did not modify the Ocean package.")
    return installed_path, repo_copy


def parse_args():
    parser = argparse.ArgumentParser(description="Update the ViDEC HoloOcean scenario AUV pose.")
    parser.add_argument("--location", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--rotation", nargs=3, type=float, required=True, metavar=("R", "P", "Y"))
    parser.add_argument("--scenario-name", default=DEFAULT_SCENARIO_NAME)
    parser.add_argument("--world-dir", default=str(DEFAULT_WORLD_DIR))
    parser.add_argument("--repo-copy", default=str(DEFAULT_REPO_COPY))
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        update_pose(
            args.location,
            args.rotation,
            args.scenario_name,
            Path(args.world_dir).expanduser(),
            Path(args.repo_copy),
        )
        return 0
    except Exception as exc:
        log("ERROR", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
