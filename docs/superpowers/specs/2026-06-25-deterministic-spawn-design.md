# Deterministic Spawn Point + Spectator Follow — Design

## Problem

`CarlaLaneKeepingEnv._spawn_vehicle()` (`carla_env/env.py:250-279`) always
picks a spawn point via `self._rng.choice(self._spawn_points)`. The class
already accepts a `spawn_index` constructor parameter (`carla_env/env.py:131`)
and `configs/config.yaml` already has an `env.spawn_index: 0` field
(`configs/config.yaml:24`), but neither is wired to anything —
`_spawn_vehicle()` never reads `self.spawn_index`, and `agent/train.py`'s
`make_env()` (`agent/train.py:79-121`) never even passes `spawn_index` to
the `CarlaLaneKeepingEnv(...)` constructor call. The field is vestigial.

This blocks a real need: visually validating that an algorithm is learning
requires starting every episode from the same point, so episode-to-episode
progress (how far the car gets before failing) is comparable. Today every
episode starts from a different, effectively-random point.

Separately, CARLA's spectator camera never moves on its own. Even with a
fixed spawn point, there's no way to see the car in the CARLA window
without manually flying the free-look camera to find it.

## Goal

1. When `spawn_index` is set (an integer), every episode spawns the vehicle
   at that exact spawn point — deterministic, for validating learning.
2. When `spawn_index` is `null` (or omitted), spawning is unchanged
   (current seeded-random behavior) — for general training after
   validation.
3. Switching between the two is a one-line edit to `configs/config.yaml`,
   no code change and no CLI flag.
4. Whenever a vehicle spawns (either mode), the CARLA spectator camera
   snaps to a chase view of it, so it's visible in the CARLA window
   without manual camera navigation.

## Changes

### 1. `carla_env/env.py` — `_connect()`: validate `spawn_index` once

After `self._spawn_points` is populated (`carla_env/env.py:206`), if
`self.spawn_index is not None`, validate it's in range:

```python
if self.spawn_index is not None:
    if not (0 <= self.spawn_index < len(self._spawn_points)):
        raise ValueError(
            f"spawn_index={self.spawn_index} is out of range — "
            f"map '{self.map_name}' has {len(self._spawn_points)} "
            f"spawn points (valid range: 0..{len(self._spawn_points)-1})."
        )
```

Fails fast at environment construction time, not on the first `reset()`
deep inside a training run.

### 2. `carla_env/env.py` — `_spawn_vehicle()`: branch on `spawn_index`

Replace the single random-attempt loop with two modes:

```python
def _spawn_vehicle(self):
    """
    Spawn the ego vehicle.

    If self.spawn_index is set, always spawns at that exact spawn point
    (deterministic — for validating that an algorithm is learning, since
    episode-to-episode progress is only comparable from a fixed start).
    Retries the same point up to 5 times on transient occupation, then
    raises — never falls back to a different point, which would silently
    break the "always the same start" guarantee.

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
                f"x={transform.location.x:.1f}, y={transform.location.y:.1f}"
            )
            return vehicle

    raise RuntimeError(
        "Failed to spawn vehicle after 5 attempts. "
        "All chosen spawn points were occupied."
    )
```

### 3. `carla_env/env.py` — new method `_snap_spectator_to_vehicle()`

Called from `reset()` right after `self._vehicle = self._spawn_vehicle()`
(`carla_env/env.py:312`), for both spawn modes:

```python
def _snap_spectator_to_vehicle(self) -> None:
    """
    Move the CARLA spectator camera to a chase view of the ego vehicle.

    Purely a visualization aid — CARLA's spectator never moves on its
    own, so without this there's no way to see the car in the CARLA
    window without manually flying the free-look camera to find it.
    Has no effect on training; safe to call every reset.
    """
    import carla

    vehicle_transform = self._vehicle.get_transform()
    forward = vehicle_transform.get_forward_vector()

    cam_location = (
        vehicle_transform.location
        - forward * SPECTATOR_DISTANCE_M
        + carla.Location(z=SPECTATOR_HEIGHT_M)
    )
    cam_rotation = carla.Rotation(
        pitch=SPECTATOR_PITCH_DEG,
        yaw=vehicle_transform.rotation.yaw,
    )
    self._world.get_spectator().set_transform(
        carla.Transform(cam_location, cam_rotation)
    )
```

New module-level constants alongside the existing `SETTLE_TICKS` /
`DELTA_SECONDS` block (`carla_env/env.py:78-84`):

```python
SPECTATOR_DISTANCE_M = 8.0    # meters behind the vehicle
SPECTATOR_HEIGHT_M   = 4.0    # meters above the vehicle
SPECTATOR_PITCH_DEG  = -15.0  # degrees, looking down toward the vehicle
```

### 4. `agent/train.py` — `make_env()`: pass `spawn_index` through

In the `CarlaLaneKeepingEnv(...)` call (`agent/train.py:104-113`), add:

```python
spawn_index = env_cfg.get("spawn_index"),
```

Uses `.get()` (not `["spawn_index"]`) so the field can be omitted from
`config.yaml` entirely to mean "random" — not just set to `null`.

### 5. `configs/config.yaml` — clarify the comment

Change:
```yaml
spawn_index:    0
```
to:
```yaml
spawn_index:    0             # int = always spawn here (deterministic,
                               #   for validating learning); null = random
```

## Out of scope

- No CLI override (`--spawn-index`) — config-file edit only, per the
  approved design choice.
- No change to the existing random-mode RNG behavior
  (`self._rng = random.Random(seed)`, reused across episodes) — only
  the new deterministic branch is added alongside it.
- No change to `scripts/manual_drive.py` or other scripts — this only
  touches `CarlaLaneKeepingEnv` and the one `train.py` call site that
  constructs it from config.

## Testing

This logic lives entirely inside `_spawn_vehicle()` and `reset()`, which
need a real `carla.World` and spawn-points list — not reachable from the
offline suites (`scripts/test_action.py`, `scripts/test_reward.py`,
`scripts/test_algorithms.py`), consistent with the rest of
`carla_env/env.py` having no offline tests today.

Verification will be live, against a running CARLA server:
1. Set `spawn_index: 0`, run a short session (e.g.
   `python agent/train.py --algo ppo --timesteps 300` or
   `scripts/manual_drive.py`), confirm every episode reset spawns the
   vehicle at the same location and the spectator camera shows it
   immediately.
2. Set `spawn_index` to an out-of-range value (e.g. 99999), confirm a
   clear `ValueError` at environment construction, not a confusing
   failure deep in training.
3. Set `spawn_index: null` (or remove the line), confirm spawning
   returns to the previous random behavior and the spectator still
   follows.
