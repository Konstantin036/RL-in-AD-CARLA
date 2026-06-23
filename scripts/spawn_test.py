"""
spawn_test.py
-------------
Phase 2 — Spawn a Vehicle and Tick the World

Purpose:
    Demonstrate the full actor lifecycle in synchronous mode:
        1. Connect and load world
        2. Enable synchronous mode
        3. Spawn a vehicle
        4. Run a tick loop (this is the RL step loop skeleton)
        5. Destroy all actors
        6. Restore asynchronous mode

    This script is the foundation of the RL environment.
    Every concept here reappears in carla_env/env.py.

How to run:
    1. Launch CARLA:  ./CarlaUE4.sh -quality-level=Low -fps=20
    2. Run this:      python scripts/spawn_test.py

What to observe:
    - The CARLA window should show a vehicle appearing and driving forward.
    - The terminal prints speed readings each tick.
    - On exit (Ctrl+C or natural end), the vehicle disappears cleanly.
"""

import sys
import time
import math


# ── Logging helpers ───────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[INFO]  {msg}")

def warn(msg: str) -> None:
    print(f"[WARN]  {msg}")

def error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


# ── CARLA helpers ─────────────────────────────────────────────────────────────

def get_vehicle_speed_kmh(vehicle) -> float:
    """
    Convert CARLA's velocity vector to a scalar speed in km/h.

    CARLA gives velocity as a 3D vector (vx, vy, vz) in m/s.
    We compute the magnitude, then convert to km/h.

    Why not use a CARLA built-in? There isn't one for speed directly.
    This two-liner is the standard pattern you will see everywhere.
    """
    v = vehicle.get_velocity()                          # carla.Vector3D in m/s
    speed_ms = math.sqrt(v.x**2 + v.y**2 + v.z**2)    # magnitude in m/s
    return speed_ms * 3.6                               # convert to km/h


# ── Synchronous mode context ──────────────────────────────────────────────────

def enable_sync_mode(world, delta_seconds: float = 0.05):
    """
    Switch the CARLA world to synchronous mode.

    Parameters
    ----------
    world          : carla.World object
    delta_seconds  : Fixed timestep in seconds.
                     0.05 = 20 physics steps per second (standard for RL).

    What this does internally:
    - Sets synchronous_mode = True  → server waits for tick() calls
    - Sets fixed_delta_seconds      → each tick advances time by exactly this amount
    - Calls apply_settings()        → sends the new config to the server

    IMPORTANT: After calling this, you MUST call world.tick() regularly.
    If you stop calling tick(), CARLA freezes completely.
    """
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta_seconds
    world.apply_settings(settings)
    log(f"Synchronous mode ON  (delta = {delta_seconds}s = {1/delta_seconds:.0f} FPS)")


def disable_sync_mode(world):
    """
    Restore asynchronous mode.

    Call this:
    - At the end of every script
    - In every exception handler
    - Basically: always, before you exit

    If you leave sync mode on and your script crashes, the CARLA server
    freezes and you need to restart it. This is the most common beginner
    mistake with synchronous CARLA.
    """
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    log("Synchronous mode OFF (restored async)")


# ── Vehicle spawning ──────────────────────────────────────────────────────────

def spawn_vehicle(world, vehicle_model: str = "vehicle.tesla.model3"):
    """
    Spawn a vehicle at a random spawn point on the current map.

    Parameters
    ----------
    world          : carla.World
    vehicle_model  : Blueprint ID string. We use Tesla Model 3 as the
                     standard ego vehicle throughout this project.

    Returns
    -------
    carla.Vehicle actor

    Key concepts:
    - Blueprint: the "recipe" for an actor (model, physics properties)
    - Transform: a position (Location) + orientation (Rotation) in the world
    - Spawn point: a pre-validated Transform on the map (no walls, no overlaps)
    """
    blueprint_library = world.get_blueprint_library()

    # find() returns the first blueprint matching the ID string.
    # The Tesla Model 3 is a common choice for RL — mid-size, stable physics.
    vehicle_bp = blueprint_library.find(vehicle_model)

    # Optionally set the vehicle color (purely cosmetic)
    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", "255,0,0")   # red — easy to spot

    # Get all valid spawn points for this map
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points found on this map.")

    # Pick spawn point index 0 for determinism during testing.
    # Later in the RL environment we will randomize this.
    spawn_transform = spawn_points[0]

    # try_spawn_actor() is safer than spawn_actor():
    # - spawn_actor() raises an exception if the spot is occupied
    # - try_spawn_actor() returns None instead
    # This matters during training when we reset episodes rapidly.
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)

    if vehicle is None:
        raise RuntimeError(
            f"Failed to spawn vehicle at spawn point 0. "
            f"The spot may be occupied. Try a different index."
        )

    log(f"Vehicle spawned: {vehicle.type_id}  (id={vehicle.id})")
    log(f"Spawn location: x={spawn_transform.location.x:.1f}, "
        f"y={spawn_transform.location.y:.1f}, "
        f"z={spawn_transform.location.z:.1f}")

    return vehicle


# ── Main test loop ────────────────────────────────────────────────────────────

def run_spawn_test(
    host: str = "localhost",
    port: int = 2000,
    map_name: str = "Town03",
    num_ticks: int = 200,
    delta_seconds: float = 0.05,
):
    """
    Full lifecycle test: connect → sync → spawn → tick → destroy → cleanup.

    Parameters
    ----------
    host           : CARLA server host
    port           : CARLA server port
    map_name       : Map to load
    num_ticks      : How many physics steps to run
    delta_seconds  : Timestep size (0.05s = 20 FPS)
    """

    # ── Import ─────────────────────────────────────────────────────────────────
    try:
        import carla
    except ImportError:
        error("Cannot import 'carla'. Check your PYTHONPATH and README.md Step 4.")
        return False

    client = None
    world = None
    vehicle = None

    try:
        # ── Connect ────────────────────────────────────────────────────────────
        log(f"Connecting to CARLA at {host}:{port} ...")
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        log(f"Connected. Server version: {client.get_server_version()}")

        # ── Load map ───────────────────────────────────────────────────────────
        log(f"Loading map: {map_name} ...")
        world = client.load_world(map_name)
        time.sleep(2.0)   # let the world fully initialize
        log(f"Map loaded: {world.get_map().name}")

        # ── Enable synchronous mode ────────────────────────────────────────────
        # Do this BEFORE spawning actors.
        # Some tutorials enable sync after spawning — this can cause a race
        # condition where the vehicle gets its first physics tick in async mode.
        enable_sync_mode(world, delta_seconds)

        # After enabling sync, we must tick once to let the server acknowledge
        # the new settings before we do anything else.
        world.tick()

        # ── Spawn vehicle ──────────────────────────────────────────────────────
        vehicle = spawn_vehicle(world, vehicle_model="vehicle.tesla.model3")

        # Tick once more after spawning so the vehicle registers in the physics
        # engine before we try to read its state.
        world.tick()

        # ── Apply a small constant throttle ───────────────────────────────────
        # carla.VehicleControl is the action object.
        # throttle: 0.0 (none) to 1.0 (full)
        # steer:    -1.0 (full left) to 1.0 (full right)
        # brake:    0.0 (none) to 1.0 (full)
        #
        # This is the same structure the RL agent will use to send actions.
        # For now we just send a fixed small throttle to see the vehicle move.
        control = carla.VehicleControl(throttle=0.3, steer=0.0, brake=0.0)
        vehicle.apply_control(control)

        # ── Tick loop ──────────────────────────────────────────────────────────
        # This is the skeleton of the RL environment's step() function.
        # In the real environment:
        #   - Instead of fixed control, the agent sends an action
        #   - Instead of just printing, we compute observation and reward
        #   - We check termination conditions each tick

        log(f"Starting tick loop: {num_ticks} ticks × {delta_seconds}s "
            f"= {num_ticks * delta_seconds:.1f}s simulated time")
        log("Watch the CARLA window — the red vehicle should start moving.")
        log("-" * 55)
        log(f"{'Tick':>6}  {'Sim Time (s)':>12}  {'Speed (km/h)':>12}")
        log("-" * 55)

        for tick_num in range(num_ticks):

            # ── Advance the simulation by one physics step ─────────────────────
            # This is the most important line in the whole project.
            # Nothing in CARLA moves until you call this.
            world.tick()

            # ── Read vehicle state ─────────────────────────────────────────────
            speed_kmh = get_vehicle_speed_kmh(vehicle)

            # Simulated time = tick number × delta_seconds
            # This is NOT wall-clock time — it is internal simulation time.
            sim_time = (tick_num + 1) * delta_seconds

            # Print every 10 ticks to avoid flooding the terminal
            if tick_num % 10 == 0:
                log(f"{tick_num:>6}  {sim_time:>12.2f}  {speed_kmh:>11.2f}")

        log("-" * 55)
        log("Tick loop complete.")

    except KeyboardInterrupt:
        # Ctrl+C is a normal way to stop the script during development.
        # We handle it explicitly so cleanup still runs.
        warn("Interrupted by user (Ctrl+C). Cleaning up ...")

    except Exception as e:
        error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ── CLEANUP — this block ALWAYS runs, even after exceptions ───────────
        #
        # The finally block is critical. If we skip cleanup:
        # - The vehicle stays in the CARLA world as a ghost
        # - Synchronous mode stays on, freezing CARLA
        # - After a few crashes, the server is littered with ghost vehicles
        #
        # Rule: always put CARLA cleanup in a finally block.

        log("Cleaning up ...")

        # Destroy the vehicle first
        if vehicle is not None:
            vehicle.destroy()
            log(f"Vehicle destroyed (id={vehicle.id})")

        # Restore async mode — must happen even if vehicle destroy failed
        if world is not None:
            disable_sync_mode(world)

        log("Cleanup complete.")

    return True


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_spawn_test(
        host="localhost",
        port=2000,
        map_name="Town03",
        num_ticks=200,       # 200 × 0.05s = 10 simulated seconds
        delta_seconds=0.05,  # 20 FPS — standard for RL in CARLA
    )
