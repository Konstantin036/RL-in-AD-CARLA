"""
test_observation.py
-------------------
Phase 3 — Live Observation Test

Purpose:
    Spawn a vehicle, drive it with small throttle, and print the
    observation vector every 10 ticks. This confirms the observation
    functions return sensible values before we plug them into the env.

How to run:
    python scripts/test_observation.py

What to look for:
    - lateral_distance near 0 at start (spawned on lane center)
    - heading_error near 0 at start (aligned with road)
    - speed climbs from 0 as throttle is applied
    - steering shows 0 (we are not steering)
    - obs array values all in [-1, 1]
"""

import sys
import time
import math

# Add project root to path so we can import carla_env
sys.path.insert(0, __import__("os").path.join(__file__, "..", ".."))

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carla_env.observation import (
    compute_observation,
    get_lateral_distance,
    get_heading_error,
    get_speed_kmh,
    ObservationData,
)


def log(msg):
    print(f"[INFO]  {msg}")

def error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)


def run_observation_test(
    host="localhost",
    port=2000,
    map_name="Town03",
    num_ticks=300,
    delta_seconds=0.05,
):
    try:
        import carla
    except ImportError:
        error("Cannot import carla. Check PYTHONPATH.")
        return

    client = None
    world = None
    vehicle = None

    try:
        # ── Connect ────────────────────────────────────────────────────────────
        log(f"Connecting to CARLA at {host}:{port} ...")
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        log(f"Connected. Server: {client.get_server_version()}")

        # ── Load world ─────────────────────────────────────────────────────────
        world = client.load_world(map_name)
        time.sleep(2.0)
        carla_map = world.get_map()

        # ── Sync mode ──────────────────────────────────────────────────────────
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta_seconds
        world.apply_settings(settings)
        world.tick()

        # ── Spawn vehicle ──────────────────────────────────────────────────────
        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        spawn_tf = world.get_map().get_spawn_points()[0]
        vehicle = world.try_spawn_actor(bp, spawn_tf)
        if vehicle is None:
            raise RuntimeError("Spawn failed.")
        world.tick()
        log(f"Vehicle spawned at {spawn_tf.location}")

        # ── Apply throttle ────────────────────────────────────────────────────
        control = carla.VehicleControl(throttle=0.4, steer=0.0, brake=0.0)
        vehicle.apply_control(control)

        # ── Print header ───────────────────────────────────────────────────────
        log("")
        log("Observation vector over time (raw physical values + normalized):")
        log("-" * 80)
        log(f"{'Tick':>5}  {'lat(m)':>8}  {'hdg(°)':>8}  {'spd(km/h)':>10}  "
            f"{'steer':>6}  │  {'obs[0]':>7}  {'obs[1]':>7}  {'obs[2]':>7}  {'obs[3]':>7}")
        log("-" * 80)

        for tick in range(num_ticks):
            world.tick()

            # ── Compute observation ────────────────────────────────────────────
            obs_array, obs_data = compute_observation(vehicle, carla_map)

            # Print every 10 ticks
            if tick % 10 == 0:
                hdg_deg = math.degrees(obs_data.heading_error_rad)
                log(
                    f"{tick:>5}  "
                    f"{obs_data.lateral_distance_m:>+8.3f}  "
                    f"{hdg_deg:>+8.2f}  "
                    f"{obs_data.speed_kmh:>10.2f}  "
                    f"{obs_data.steering:>+6.2f}  │  "
                    f"{obs_array[0]:>+7.3f}  "
                    f"{obs_array[1]:>+7.3f}  "
                    f"{obs_array[2]:>+7.3f}  "
                    f"{obs_array[3]:>+7.3f}"
                )

        log("-" * 80)
        log("Observation test complete.")
        log("")
        log("What you should see:")
        log("  lateral(m) near 0 at start → vehicle starts at lane center")
        log("  heading(°) near 0 at start → vehicle aligned with road")
        log("  speed rises from 0         → throttle is working")
        log("  obs values all in [-1, 1]  → normalization is correct")

    except KeyboardInterrupt:
        log("Interrupted by user.")

    except Exception as e:
        error(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if vehicle:
            vehicle.destroy()
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        log("Cleanup done.")


if __name__ == "__main__":
    run_observation_test()
