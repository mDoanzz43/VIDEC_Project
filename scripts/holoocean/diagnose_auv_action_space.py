#!/usr/bin/env python3
import argparse

import numpy as np


def log(level, message):
    print(f"[{level}] {message}")


def safe_close(env):
    if env is None:
        return
    if hasattr(env, "close"):
        env.close()
    else:
        env.__exit__(None, None, None)


def print_matching_methods(env):
    terms = ("act", "action", "control", "scheme")
    names = []
    for name in dir(env):
        lower = name.lower()
        if any(term in lower for term in terms):
            names.append(name)
    log("INFO", f"Env methods/attrs containing {terms}: {names}")


def try_action(env, action):
    try:
        env.act("auv0", action)
        state = None
        for _ in range(3):
            state = env.tick()
        keys = [str(key) for key in state.keys()] if state is not None else []
        return True, keys, None
    except Exception as exc:
        return False, [], f"{type(exc).__name__}: {exc}"


def diagnose(args):
    import holoocean

    env = None
    try:
        log("INFO", f"Creating HoloOcean environment: {args.scenario}")
        env = holoocean.make(args.scenario)
        print_matching_methods(env)

        if hasattr(env, "action_space"):
            try:
                log("INFO", f"env.action_space: {env.action_space}")
            except Exception as exc:
                log("WARN", f"Could not print env.action_space: {type(exc).__name__}: {exc}")

        state = env.tick()
        log("INFO", f"Initial state keys: {[str(key) for key in state.keys()]}")

        for length in (8, 6, 5):
            action = np.zeros(length, dtype=np.float32)
            success, keys, error = try_action(env, action)
            if success:
                log("INFO", f"Action zeros length {length}: success; state_keys={keys}")
            else:
                log("WARN", f"Action zeros length {length}: failed; error={error}")

        return 0
    except Exception as exc:
        log("ERROR", f"Diagnostic failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        try:
            safe_close(env)
            log("INFO", "Closed HoloOcean environment")
        except Exception as exc:
            log("WARN", f"env close failed: {type(exc).__name__}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose ViDEC HoveringAUV action vector dimensions.")
    parser.add_argument("--scenario", default="test_ocean_map-HoveringCamera")
    return parser.parse_args()


def main():
    return diagnose(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
