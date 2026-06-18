#!/usr/bin/env python3
import argparse
from pathlib import Path


VIDEC_PACKAGE_DIR = Path.home() / ".local/share/holoocean/2.3.0/worlds/ViDEC"


def log(level, message):
    print(f"[{level}] {message}")


def print_diagnostics(scenario):
    scenario_path = VIDEC_PACKAGE_DIR / f"{scenario}.json"
    executable_path = VIDEC_PACKAGE_DIR / "Linux/Holodeck/Binaries/Linux/Holodeck"
    checks = {
        "ViDEC package folder": VIDEC_PACKAGE_DIR,
        "config.json": VIDEC_PACKAGE_DIR / "config.json",
        "scenario JSON": scenario_path,
        "Linux executable": executable_path,
    }
    for label, path in checks.items():
        log("INFO", f"{label}: {path} exists={path.exists()}")


def main():
    parser = argparse.ArgumentParser(description="Test holoocean.make() for the ViDEC custom world.")
    parser.add_argument("--scenario", default="test_ocean_map-HoveringCamera")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    env = None
    try:
        import holoocean

        log("INFO", f"holoocean module: {getattr(holoocean, '__file__', 'not_available')}")
        if hasattr(holoocean, "installed_packages"):
            try:
                log("INFO", f"installed packages: {holoocean.installed_packages()}")
            except Exception as exc:
                log("WARN", f"installed_packages() failed: {type(exc).__name__}: {exc}")

        scenario = holoocean.get_scenario(args.scenario)
        log("INFO", f"Loaded scenario '{args.scenario}': {scenario}")

        log("INFO", f"Creating environment with holoocean.make('{args.scenario}')")
        env = holoocean.make(args.scenario, show_viewport=False)
        print_interval = max(1, args.steps // 5)
        expected_keys = {
            "LeftCamera",
            "RightCamera",
            "DepthSensor",
            "PoseSensor",
            "VelocitySensor",
            "IMUSensor",
            "DVLSensor",
            "GPSSensor",
            "t",
        }

        seen_keys = set()
        for step_index in range(args.steps):
            state = env.tick()
            keys = [str(key) for key in state.keys()]
            seen_keys.update(keys)
            if step_index == 0 or step_index % print_interval == 0 or step_index == args.steps - 1:
                log("INFO", f"step={step_index} state_keys={keys}")

        detected = sorted(expected_keys.intersection(seen_keys))
        missing = sorted(expected_keys.difference(seen_keys))
        log("INFO", f"Detected expected keys: {detected}")
        log("WARN", f"Missing expected keys: {missing}")
        log("INFO", "holoocean.make() test completed")
        return 0
    except Exception as exc:
        log("ERROR", f"holoocean.make() test failed: {type(exc).__name__}: {exc}")
        print_diagnostics(args.scenario)
        return 1
    finally:
        if env is not None:
            try:
                env.close()
                log("INFO", "Closed HoloOcean environment")
            except Exception as exc:
                log("WARN", f"env.close() failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
