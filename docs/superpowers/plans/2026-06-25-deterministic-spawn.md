# Deterministic Spawn Point + Spectator Follow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `CarlaLaneKeepingEnv` spawn the ego vehicle at a fixed, configurable spawn point every episode (for validating that an algorithm is learning) instead of always picking randomly, and make the CARLA spectator camera follow the vehicle so it's visible without manually flying the camera.

**Architecture:** `carla_env/env.py` gains two small pure helper functions (spawn-index range validation, spectator chase-camera transform math) that are unit-testable offline without a CARLA server, plus the instance-level wiring that calls them from `_connect()`, `_spawn_vehicle()`, and `reset()`. `agent/train.py` and `configs/config.yaml` get the one-line plumbing needed to actually pass the existing-but-previously-unused `spawn_index` config value through.

**Tech Stack:** Python 3.7.16, CARLA 0.9.15 Python API (`carla` module — importable and usable for constructing `Transform`/`Location`/`Rotation` objects without a live server connection, confirmed during planning), run via `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python`.

## Global Constraints

- Python 3.7 compatible syntax only — no `list[int]`-style built-in generics.
- All CARLA imports stay inside functions, never at module top level (existing `carla_env/env.py` pattern — e.g. `_spawn_vehicle()` does `import carla` inside itself; new functions follow the same pattern).
- Use `logger` (not `print`) inside `carla_env/` and `agent/` modules; use `print` inside `scripts/`.
- Never hardcode reward weights or hyperparameters in code — they live in `configs/config.yaml`. (Not directly touched by this plan, but `spawn_index` itself must stay config-driven, not hardcoded.)
- Switching between deterministic and random spawn is a `configs/config.yaml` edit only — no CLI flag (per approved design).
- Deterministic mode never silently falls back to a different spawn point on failure — it retries the same point, then raises.
- The spectator-follow feature must not change training behavior in any way — it only moves a camera, never affects the vehicle, observation, reward, or termination logic.

---

### Task 1: Deterministic spawn point validation and branching

**Files:**
- Modify: `carla_env/env.py:84-85` (add nothing here yet — see Task 2 for constants; this task only touches the validation function and `_connect`/`_spawn_vehicle`/docstring)
- Modify: `carla_env/env.py:97-106` (class docstring — add `spawn_index` to Parameters)
- Modify: `carla_env/env.py:205-211` (`_connect()` — call the new validation)
- Modify: `carla_env/env.py:250-279` (`_spawn_vehicle()` — branch on `self.spawn_index`)
- Create: `scripts/test_spawn.py`

**Interfaces:**
- Produces: `_check_spawn_index_in_range(spawn_index, num_spawn_points: int) -> None` — module-level function in `carla_env/env.py`, raises `ValueError` if `spawn_index` is not `None` and not in `[0, num_spawn_points)`. Imported directly by `scripts/test_spawn.py` and called internally by `_connect()`.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_spawn.py`:

```python
"""
test_spawn.py
--------------
Offline unit tests for deterministic spawn point validation and the
spectator chase-camera transform (no CARLA server required — these are
pure functions of their inputs; only the `carla` Python module itself
needs to be importable, which it is in the carla915 conda env).

Run from anywhere:
    python scripts/test_spawn.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carla_env.env import _check_spawn_index_in_range


def separator(title=""):
    print(f"\n{'─'*55}")
    if title:
        print(f"  {title}")
        print(f"{'─'*55}")


def test_spawn_index_none_is_always_valid():
    separator("1. spawn_index=None is always valid (random mode)")
    _check_spawn_index_in_range(None, num_spawn_points=372)
    _check_spawn_index_in_range(None, num_spawn_points=0)
    print("  ✓ PASSED")


def test_spawn_index_in_range():
    separator("2. spawn_index within range does not raise")
    _check_spawn_index_in_range(0, num_spawn_points=372)
    _check_spawn_index_in_range(371, num_spawn_points=372)
    print("  ✓ PASSED")


def test_spawn_index_out_of_range():
    separator("3. spawn_index out of range raises ValueError")
    for bad_index in [-1, 372, 99999]:
        try:
            _check_spawn_index_in_range(bad_index, num_spawn_points=372)
            raise AssertionError(f"Expected ValueError for spawn_index={bad_index}")
        except ValueError as e:
            print(f"  spawn_index={bad_index} raised ValueError as expected: {e}")
    print("  ✓ PASSED")


if __name__ == "__main__":
    test_spawn_index_none_is_always_valid()
    test_spawn_index_in_range()
    test_spawn_index_out_of_range()
    print(f"\n{'='*55}")
    print("  ALL TESTS PASSED")
    print(f"{'='*55}\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_spawn.py`
Expected: `ImportError: cannot import name '_check_spawn_index_in_range' from 'carla_env.env'`

- [ ] **Step 3: Add the validation function to `carla_env/env.py`**

In `carla_env/env.py`, immediately before the `# ── Main environment class ─────` comment (currently at line 87), add:

```python
def _check_spawn_index_in_range(spawn_index, num_spawn_points: int) -> None:
    """
    Validate that spawn_index (if set) is a valid index into the map's
    spawn points list. Raises ValueError if out of range.

    Pure function (no CARLA connection needed) so it can be unit tested
    offline — called from _connect() once the real spawn points list is
    known, so a misconfigured spawn_index fails fast at environment
    construction time instead of deep inside a training run.
    """
    if spawn_index is None:
        return
    if not (0 <= spawn_index < num_spawn_points):
        raise ValueError(
            f"spawn_index={spawn_index} is out of range — "
            f"{num_spawn_points} spawn points available "
            f"(valid range: 0..{num_spawn_points - 1})."
        )


```

(Keep the blank line before the existing `# ── Main environment class ─────` comment.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_spawn.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Wire the validation into `_connect()`**

In `carla_env/env.py`, change:

```python
        self._carla_map   = self._world.get_map()
        self._spawn_points = self._carla_map.get_spawn_points()
        logger.info(f"Map loaded. Spawn points: {len(self._spawn_points)}")

        # Enable synchronous mode once — stays on for the entire training run.
        # We only disable it in close().
        self._enable_sync_mode()
```

to:

```python
        self._carla_map   = self._world.get_map()
        self._spawn_points = self._carla_map.get_spawn_points()
        logger.info(f"Map loaded. Spawn points: {len(self._spawn_points)}")

        _check_spawn_index_in_range(self.spawn_index, len(self._spawn_points))

        # Enable synchronous mode once — stays on for the entire training run.
        # We only disable it in close().
        self._enable_sync_mode()
```

- [ ] **Step 6: Branch `_spawn_vehicle()` on `self.spawn_index`**

In `carla_env/env.py`, change:

```python
    def _spawn_vehicle(self):
        """
        Spawn the ego vehicle at a random spawn point.

        Returns carla.Vehicle.
        Raises RuntimeError if all spawn attempts fail.
        """
        import carla

        bp = self._world.get_blueprint_library().find("vehicle.tesla.model3")
        if bp.has_attribute("color"):
            bp.set_attribute("color", "255,0,0")   # red for visibility

        # Try up to 5 random spawn points before giving up.
        # Some spawn points may be occupied if the world has traffic.
        for attempt in range(5):
            transform = self._rng.choice(self._spawn_points)
            vehicle   = self._world.try_spawn_actor(bp, transform)
            if vehicle is not None:
                logger.debug(
                    f"Vehicle spawned at attempt {attempt+1}: "
                    f"x={transform.location.x:.1f}, "
                    f"y={transform.location.y:.1f}"
                )
                return vehicle

        raise RuntimeError(
            "Failed to spawn vehicle after 5 attempts. "
            "All chosen spawn points were occupied."
        )
```

to:

```python
    def _spawn_vehicle(self):
        """
        Spawn the ego vehicle.

        If self.spawn_index is set, always spawns at that exact spawn
        point (deterministic — for validating that an algorithm is
        learning, since episode-to-episode progress is only comparable
        from a fixed start). Retries the same point up to 5 times on
        transient occupation, then raises — never falls back to a
        different point, which would silently break the "always the
        same start" guarantee.

        If self.spawn_index is None, picks a random point (existing
        behavior, unchanged) — for general training once validated.

        Returns carla.Vehicle.
        Raises RuntimeError if all spawn attempts fail.
        """
        import carla

        bp = self._world.get_blueprint_library().find("vehicle.tesla.model3")
        if bp.has_attribute("color"):
            bp.set_attribute("color", "255,0,0")   # red for visibility

        if self.spawn_index is not None:
            transform = self._spawn_points[self.spawn_index]
            for attempt in range(5):
                vehicle = self._world.try_spawn_actor(bp, transform)
                if vehicle is not None:
                    logger.debug(
                        f"Vehicle spawned at fixed spawn_index={self.spawn_index} "
                        f"(attempt {attempt+1}): "
                        f"x={transform.location.x:.1f}, y={transform.location.y:.1f}"
                    )
                    return vehicle
            raise RuntimeError(
                f"Failed to spawn vehicle at fixed spawn_index={self.spawn_index} "
                f"after 5 attempts. That spawn point stayed occupied."
            )

        # Random mode (unchanged): try up to 5 random spawn points before
        # giving up. Some spawn points may be occupied if the world has traffic.
        for attempt in range(5):
            transform = self._rng.choice(self._spawn_points)
            vehicle   = self._world.try_spawn_actor(bp, transform)
            if vehicle is not None:
                logger.debug(
                    f"Vehicle spawned at attempt {attempt+1}: "
                    f"x={transform.location.x:.1f}, "
                    f"y={transform.location.y:.1f}"
                )
                return vehicle

        raise RuntimeError(
            "Failed to spawn vehicle after 5 attempts. "
            "All chosen spawn points were occupied."
        )
```

- [ ] **Step 7: Document `spawn_index` in the class docstring**

In `carla_env/env.py`, change:

```python
    seed          : random seed for spawn point selection
    verbose       : if True, log step-level info (slow — use for debugging)
```

to:

```python
    seed          : random seed for spawn point selection (random mode only)
    spawn_index   : if set, always spawn at this exact spawn point index
                    (deterministic — for validating learning); if None,
                    spawn randomly (default)
    verbose       : if True, log step-level info (slow — use for debugging)
```

- [ ] **Step 8: Run the offline test suite to confirm no regressions**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_spawn.py && /home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_action.py && /home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_reward.py`
Expected: all three print `ALL TESTS PASSED` / `All tests passed.` / `All reward tests passed.` with no errors.

- [ ] **Step 9: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add carla_env/env.py scripts/test_spawn.py
git commit -m "Add deterministic spawn_index support to CarlaLaneKeepingEnv"
```

---

### Task 2: Spectator chase-camera follow

**Files:**
- Modify: `carla_env/env.py:84-85` (add `SPECTATOR_*` constants)
- Modify: `carla_env/env.py` (add `_compute_spectator_transform()` module-level function, near `_check_spawn_index_in_range` from Task 1)
- Modify: `carla_env/env.py` (add `_snap_spectator_to_vehicle()` instance method)
- Modify: `carla_env/env.py:311-315` (`reset()` — call the new method)
- Modify: `scripts/test_spawn.py` (append spectator-transform tests)

**Interfaces:**
- Consumes: nothing from Task 1 (independent helper), but lands in the same files.
- Produces: `_compute_spectator_transform(vehicle_transform, distance_m: float, height_m: float, pitch_deg: float)` — module-level function in `carla_env/env.py`, returns a `carla.Transform`. `_snap_spectator_to_vehicle(self) -> None` — instance method, no return value, called from `reset()`.

- [ ] **Step 1: Write the failing test**

Append to `scripts/test_spawn.py`. Add `import carla` to the top imports (needed only to construct `Transform`/`Location`/`Rotation` for the test — no server connection, consistent with this file's offline-testing intent), and update the `carla_env.env` import line:

```python
import carla

from carla_env.env import _check_spawn_index_in_range, _compute_spectator_transform
```

Add these test functions before the `if __name__ == "__main__":` block:

```python
def test_spectator_transform_is_behind_and_above():
    separator("4. Spectator transform is behind and above the vehicle")
    vehicle_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=0.0),
        carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),
    )
    cam_transform = _compute_spectator_transform(
        vehicle_transform, distance_m=8.0, height_m=4.0, pitch_deg=-15.0,
    )
    print(f"  vehicle location: {vehicle_transform.location}")
    print(f"  camera location:  {cam_transform.location}")

    assert abs(cam_transform.location.z - 4.0) < 1e-3, \
        "Camera should be 4m above the vehicle"

    forward = vehicle_transform.get_forward_vector()
    displacement_x = cam_transform.location.x - vehicle_transform.location.x
    displacement_y = cam_transform.location.y - vehicle_transform.location.y
    dot = displacement_x * forward.x + displacement_y * forward.y
    assert dot < 0, \
        "Camera should be displaced behind the vehicle (opposite its forward vector)"

    assert cam_transform.rotation.pitch == -15.0
    assert cam_transform.rotation.yaw == vehicle_transform.rotation.yaw
    print("  ✓ PASSED")


def test_spectator_transform_distance():
    separator("5. Spectator transform respects the configured distance")
    vehicle_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=0.0),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )
    cam_transform = _compute_spectator_transform(
        vehicle_transform, distance_m=10.0, height_m=0.0, pitch_deg=0.0,
    )
    horizontal_distance = (
        cam_transform.location.x ** 2 + cam_transform.location.y ** 2
    ) ** 0.5
    print(f"  horizontal distance from vehicle: {horizontal_distance:.2f}m")
    assert abs(horizontal_distance - 10.0) < 1e-2
    print("  ✓ PASSED")
```

Update the `__main__` block:

```python
if __name__ == "__main__":
    test_spawn_index_none_is_always_valid()
    test_spawn_index_in_range()
    test_spawn_index_out_of_range()
    test_spectator_transform_is_behind_and_above()
    test_spectator_transform_distance()
    print(f"\n{'='*55}")
    print("  ALL TESTS PASSED")
    print(f"{'='*55}\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_spawn.py`
Expected: `ImportError: cannot import name '_compute_spectator_transform' from 'carla_env.env'`

- [ ] **Step 3: Add the `SPECTATOR_*` constants**

In `carla_env/env.py`, change:

```python
SETTLE_TICKS    = 10     # ticks to wait after spawning before returning obs
                         # gives the physics engine time to stabilize the vehicle
DELTA_SECONDS   = 0.05   # fixed timestep: 0.05s = 20 FPS
DEFAULT_MAP     = "Town03"
DEFAULT_HOST    = "localhost"
DEFAULT_PORT    = 2000
DEFAULT_TIMEOUT = 10.0
```

to:

```python
SETTLE_TICKS    = 10     # ticks to wait after spawning before returning obs
                         # gives the physics engine time to stabilize the vehicle
DELTA_SECONDS   = 0.05   # fixed timestep: 0.05s = 20 FPS
DEFAULT_MAP     = "Town03"
DEFAULT_HOST    = "localhost"
DEFAULT_PORT    = 2000
DEFAULT_TIMEOUT = 10.0

SPECTATOR_DISTANCE_M = 8.0    # meters behind the vehicle
SPECTATOR_HEIGHT_M   = 4.0    # meters above the vehicle
SPECTATOR_PITCH_DEG  = -15.0  # degrees, looking down toward the vehicle
```

- [ ] **Step 4: Add `_compute_spectator_transform()`**

In `carla_env/env.py`, immediately after the `_check_spawn_index_in_range()` function added in Task 1 (still before the `# ── Main environment class ─────` comment), add:

```python
def _compute_spectator_transform(vehicle_transform, distance_m: float, height_m: float, pitch_deg: float):
    """
    Compute a chase-camera carla.Transform positioned behind and above
    vehicle_transform, pitched down toward it.

    Pure function of its inputs (only needs the carla module's Location/
    Rotation/Transform classes, not a live connection) so it can be unit
    tested offline against a manually constructed vehicle transform.
    """
    import carla

    forward = vehicle_transform.get_forward_vector()
    cam_location = (
        vehicle_transform.location
        - forward * distance_m
        + carla.Location(z=height_m)
    )
    cam_rotation = carla.Rotation(
        pitch=pitch_deg,
        yaw=vehicle_transform.rotation.yaw,
    )
    return carla.Transform(cam_location, cam_rotation)


```

- [ ] **Step 5: Run test to verify it passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_spawn.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 6: Add the `_snap_spectator_to_vehicle()` instance method**

In `carla_env/env.py`, immediately after the `_spawn_vehicle()` method (which Task 1 modified — it ends right before the `# ── Gymnasium interface ─────` comment), add:

```python
    def _snap_spectator_to_vehicle(self) -> None:
        """
        Move the CARLA spectator camera to a chase view of the ego vehicle.

        Purely a visualization aid — CARLA's spectator never moves on its
        own, so without this there's no way to see the car in the CARLA
        window without manually flying the free-look camera to find it.
        Has no effect on training; safe to call every reset.
        """
        cam_transform = _compute_spectator_transform(
            self._vehicle.get_transform(),
            distance_m=SPECTATOR_DISTANCE_M,
            height_m=SPECTATOR_HEIGHT_M,
            pitch_deg=SPECTATOR_PITCH_DEG,
        )
        self._world.get_spectator().set_transform(cam_transform)

```

- [ ] **Step 7: Call it from `reset()`**

In `carla_env/env.py`, change:

```python
        # ── Spawn new vehicle ──────────────────────────────────────────────────
        self._vehicle = self._spawn_vehicle()

        # ── Attach collision sensor ────────────────────────────────────────────
        self._collision_sensor = CollisionSensor(self._world, self._vehicle)
```

to:

```python
        # ── Spawn new vehicle ──────────────────────────────────────────────────
        self._vehicle = self._spawn_vehicle()

        # ── Move spectator camera to follow the vehicle ────────────────────────
        # Visualization aid only — CARLA's spectator never moves on its own.
        self._snap_spectator_to_vehicle()

        # ── Attach collision sensor ────────────────────────────────────────────
        self._collision_sensor = CollisionSensor(self._world, self._vehicle)
```

- [ ] **Step 8: Run the offline test suite to confirm no regressions**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_spawn.py && /home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_action.py && /home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_reward.py`
Expected: all pass with no errors.

- [ ] **Step 9: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add carla_env/env.py scripts/test_spawn.py
git commit -m "Add spectator chase-camera follow on vehicle spawn"
```

---

### Task 3: Wire `spawn_index` through `agent/train.py` and clarify `config.yaml`

**Files:**
- Modify: `agent/train.py:110-118`
- Modify: `configs/config.yaml:25`

**Interfaces:**
- Consumes: `CarlaLaneKeepingEnv.__init__`'s existing `spawn_index` parameter (already present before this plan; Task 1 makes it meaningful).

- [ ] **Step 1: Pass `spawn_index` through in `make_env()`**

In `agent/train.py`, change:

```python
    env = CarlaLaneKeepingEnv(
        host          = env_cfg["host"],
        port          = env_cfg["port"],
        map_name      = env_cfg["map_name"],
        max_steps     = env_cfg["max_steps"],
        reward_config = rc,
        action_smooth = env_cfg["action_smooth"],
        seed          = env_cfg["seed"] + seed,
        verbose       = False,
    )
```

to:

```python
    env = CarlaLaneKeepingEnv(
        host          = env_cfg["host"],
        port          = env_cfg["port"],
        map_name      = env_cfg["map_name"],
        max_steps     = env_cfg["max_steps"],
        reward_config = rc,
        action_smooth = env_cfg["action_smooth"],
        seed          = env_cfg["seed"] + seed,
        spawn_index   = env_cfg.get("spawn_index"),
        verbose       = False,
    )
```

Note: `.get("spawn_index")` (not `["spawn_index"]`) so the field can be omitted from `config.yaml` entirely to mean "random" — not just set to `null`.

- [ ] **Step 2: Clarify the `config.yaml` comment**

In `configs/config.yaml`, change:

```yaml
  spawn_index:    0
```

to:

```yaml
  spawn_index:    0             # int = always spawn here (deterministic,
                                 #   for validating learning); null = random
```

- [ ] **Step 3: Sanity-check the script still imports cleanly**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python -c "import sys; sys.path.insert(0, '.'); import agent.train"`
Expected: no output, exit code 0.

- [ ] **Step 4: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add agent/train.py configs/config.yaml
git commit -m "Wire spawn_index through agent/train.py's make_env()"
```

---

### Task 4: Live verification against a running CARLA server

**Files:** none (verification only — no source changes expected unless a bug surfaces, in which case fix it in `carla_env/env.py` or `agent/train.py` and re-run)

**Interfaces:** Exercises the full `agent/train.py` CLI end-to-end against a running CARLA server.

- [ ] **Step 1: Confirm CARLA is reachable**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/verify_carla.py`
Expected: connects successfully.

- [ ] **Step 2: Verify deterministic spawn — same point every episode**

`configs/config.yaml` already has `env.spawn_index: 0`. Run a short training session that produces multiple episodes:

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python agent/train.py --algo ppo --timesteps 4000
```

Expected: completes without traceback. In the output, every `[INFO] carla_env.env: Vehicle spawned at fixed spawn_index=0` debug-level log (enable with `verbose=True` if not visible at default log level, or check via the per-episode `x=`/`y=` coordinates if logged at a visible level) shows the identical `x=`/`y=` coordinates across all episodes in this run. (PPO's `n_steps=2048` means a 4000-timestep run produces 2+ episodes at `max_steps=1000`, enough to compare.)

- [ ] **Step 3: Verify the spectator camera follows the vehicle**

While the CARLA window is visible (same machine/display as the CARLA server), re-run the same command from Step 2, or run:

```bash
/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_env.py
```

Expected: watching the CARLA window, the spectator camera is positioned behind/above the vehicle immediately after each reset, without any manual camera movement.

- [ ] **Step 4: Verify out-of-range `spawn_index` fails fast with a clear error**

Temporarily edit `configs/config.yaml`'s `env.spawn_index` to `99999`, then run:

```bash
/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python agent/train.py --algo ppo --timesteps 300
```

Expected: fails immediately (during environment construction, before any training output) with `ValueError: spawn_index=99999 is out of range — 372 spawn points available (valid range: 0..371).` (the exact spawn-point count depends on the configured map). Revert `configs/config.yaml`'s `env.spawn_index` back to `0` afterward.

- [ ] **Step 5: Verify random mode still works**

Temporarily edit `configs/config.yaml`'s `env.spawn_index` to `null`, then run:

```bash
/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python agent/train.py --algo ppo --timesteps 4000
```

Expected: completes without traceback; spawn coordinates differ across episodes (random mode, as before this plan); spectator camera still follows the vehicle. Revert `configs/config.yaml`'s `env.spawn_index` back to `0` afterward (matching the project's "fixed spawn point during early training" documented default).

- [ ] **Step 6: Record results**

No commit needed for this task if nothing required a fix (no source files changed). If a fix was needed, it should be committed as part of Task 1, 2, or 3 — amend here only if a genuinely new bug was found and fixed during this verification:

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add -A
git commit -m "Fix issue found during live deterministic-spawn verification"
```

---

## Plan Self-Review Notes

- **Spec coverage:** Goal 1 (deterministic spawn) → Task 1. Goal 2 (random unchanged) → Task 1 (explicit else-branch, unchanged code). Goal 3 (config-only switch) → Task 3 (`.get()`, no CLI flag added anywhere). Goal 4 (spectator follow) → Task 2. Validation/error behavior from the spec's "Testing" section item 2 → Task 1 Step 3/5 + Task 4 Step 4. All spec sections have a corresponding task.
- **Type consistency:** `_check_spawn_index_in_range(spawn_index, num_spawn_points: int) -> None` and `_compute_spectator_transform(vehicle_transform, distance_m, height_m, pitch_deg)` signatures are identical everywhere they're defined, called (from `_connect()`/`_snap_spectator_to_vehicle()`), and tested (`scripts/test_spawn.py`).
- **Placeholder scan:** no TBD/TODO; every step has runnable code or an exact command with expected output.
