"""
verify_carla.py
---------------
Phase 1 — CARLA Connection Verification

Purpose:
    Confirm that the CARLA server is running, the Python API is reachable,
    and we can load a map and read basic world information.

    This script does NOT spawn any vehicles or sensors.
    It only tests the client-server connection.

How to run:
    1. Launch CARLA:  ./CarlaUE4.sh -quality-level=Low -fps=20
    2. Run this:      python scripts/verify_carla.py

Expected output:
    A series of [INFO] lines confirming each step succeeded,
    ending with "Verification complete. CARLA is ready."

What you learn from this script:
    - How a CARLA client is created (carla.Client)
    - What a "world" is in CARLA (the simulation state)
    - How to load a specific map
    - How to read basic world properties
    - How to disconnect cleanly
"""

import sys
import time


# ── Helper ────────────────────────────────────────────────────────────────────

def log(message: str) -> None:
    """Simple timestamped logger. We use print here on purpose —
    no dependencies, easy to read, no setup needed."""
    print(f"[INFO] {message}")


def error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)


# ── Main verification function ─────────────────────────────────────────────────

def verify_carla(
    host: str = "localhost",
    port: int = 2000,
    timeout: float = 10.0,
    map_name: str = "Town03",
) -> bool:
    """
    Connect to CARLA, load a map, print world info, and disconnect.

    Parameters
    ----------
    host     : IP address of the CARLA server (localhost for local runs)
    port     : Port number (CARLA default is 2000)
    timeout  : Seconds to wait for the server before giving up
    map_name : Which CARLA map to load (Town03 is good for lane keeping)

    Returns
    -------
    True if everything succeeded, False if any step failed.
    """

    # ── Step 1: Import the CARLA Python API ───────────────────────────────────
    # We import inside the function so that a missing CARLA egg gives a
    # clear, localized error message instead of a confusing top-level crash.
    try:
        import carla
    except ImportError:
        error("Could not import the 'carla' module.")
        error("Make sure you have added the CARLA egg to your PYTHONPATH.")
        error("See README.md — Setup Instructions, Step 4.")
        return False

    # ── Step 2: Create the client and connect ─────────────────────────────────
    # carla.Client(host, port) creates a client object.
    # It does NOT connect immediately — the connection happens on the first
    # API call (like get_server_version or get_world).
    log(f"Connecting to CARLA at {host}:{port} ...")

    try:
        client = carla.Client(host, port)

        # Timeout: if the server does not respond within this many seconds,
        # raise an exception. Without this, the script would hang forever.
        client.set_timeout(timeout)

        # get_server_version() is the lightest call we can make.
        # It forces the actual TCP connection and immediately tells us
        # if the server is alive.
        server_version = client.get_server_version()
        log(f"Connected. Server version: {server_version}")

    except Exception as e:
        error(f"Could not connect to CARLA server: {e}")
        error("Is CarlaUE4.sh running? Check that CARLA is open in another terminal.")
        return False

    # ── Step 3: Load a map ────────────────────────────────────────────────────
    # The "world" in CARLA is the object that holds everything:
    # the map geometry, all actors (vehicles, pedestrians), weather, etc.
    # load_world() replaces the current world with a fresh one on the chosen map.
    # This takes a few seconds — CARLA is loading the level on the server.
    log(f"Loading map: {map_name} ...")

    try:
        world = client.load_world(map_name)

        # Small pause to let the world fully initialize on the server side.
        # Without this, some API calls immediately after load_world can fail.
        time.sleep(2.0)

        log(f"Map loaded: {world.get_map().name}")

    except Exception as e:
        error(f"Failed to load map '{map_name}': {e}")
        error("Check that the map name is correct. Available maps can be listed below.")
        return False

    # ── Step 4: Print world information ───────────────────────────────────────
    # This section just reads information — nothing is created or modified.

    try:
        # List all maps installed on the server
        available_maps = client.get_available_maps()
        # Sort and clean up the paths for readability
        map_names = sorted([m.split("/")[-1] for m in available_maps])
        log(f"Available maps: {map_names}")

        # Spawn points are predefined positions on the map where vehicles
        # can safely be placed without spawning inside walls or other objects.
        spawn_points = world.get_map().get_spawn_points()
        log(f"Number of spawn points in {map_name}: {len(spawn_points)}")

        # Weather gives us a sense of what the simulation looks like.
        weather = world.get_weather()
        log(f"Current weather preset: {weather}")

        # The world settings object controls synchronous mode, delta time, etc.
        # We will modify this in Phase 2 when we enable synchronous mode.
        settings = world.get_settings()
        log(f"Synchronous mode: {settings.synchronous_mode}")
        log(f"Fixed delta seconds: {settings.fixed_delta_seconds}")

    except Exception as e:
        error(f"Failed to read world info: {e}")
        return False

    # ── Step 5: Clean disconnect ───────────────────────────────────────────────
    # CARLA does not require an explicit disconnect call — the TCP connection
    # closes when the client object goes out of scope. But it is good practice
    # to be explicit, especially in longer scripts that might reuse the client.
    log("Disconnecting ...")
    # Deleting the client object closes the connection gracefully.
    del client
    log("Disconnected.")

    # ── Done ──────────────────────────────────────────────────────────────────
    log("=" * 50)
    log("Verification complete. CARLA is ready.")
    log("Next step: Phase 2 — spawn a vehicle and tick the world.")
    log("=" * 50)

    return True


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    success = verify_carla(
        host="localhost",
        port=2000,
        timeout=10.0,
        map_name="Town03",
    )

    # Exit with code 0 on success, 1 on failure.
    # This lets you use the script in shell pipelines or CI checks.
    sys.exit(0 if success else 1)