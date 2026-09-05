"""
env.py
------
Phase 6 — CARLA Lane Keeping Gym Environment

Purpose:
    Wrap the CARLA simulator in a gymnasium.Env interface so that any
    standard RL algorithm (PPO, SAC, etc.) can train on it without
    knowing anything about CARLA internals.

    The two methods PPO calls every episode:
        obs, info            = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)

    Everything else (CARLA connection, sync mode, vehicle spawning,
    sensors, observation, reward, termination) is handled internally.

Architecture:
    env.py calls into:
        carla_env/observation.py  — compute_observation()
        carla_env/action.py       — ActionProcessor
        carla_env/reward.py       — compute_reward(), check_termination()
        carla_env/sensors.py      — CollisionSensor

    env.py is called by:
        agent/train.py            — PPO training loop
        scripts/manual_drive.py   — human testing
        agent/evaluate.py         — evaluation

Episode lifecycle:
    reset() →
        destroy previous actors (if any)
        load/reuse world
        enable sync mode
        spawn vehicle at random spawn point
        attach collision sensor
        tick × SETTLE_TICKS (let physics stabilize)
        compute and return first observation

    step(action) →
        smooth action → apply VehicleControl
        world.tick()
        compute observation
        compute reward
        check termination
        return (obs, reward, terminated, truncated, info)

    close() →
        destroy all actors
        disable sync mode
        disconnect client
"""

import time
import random
import logging
import numpy as np
import gymnasium as gym

from carla_env.observation import compute_observation, get_observation_space
from carla_env.action      import ActionProcessor, get_action_space
from carla_env.reward      import (
    RewardConfig,
    StallDetector,
    compute_reward,
    check_termination,
)
from carla_env.sensors     import CollisionSensor
from carla_env.traffic_rules import get_traffic_light_affordance, RedLightViolationDetector

from carla_env.route_planner import RoutePlanner



# ── Module logger ──────────────────────────────────────────────────────────────
# Using Python's logging module instead of print() so the caller can
# control verbosity (e.g. suppress during training, enable during debugging).
logger = logging.getLogger(__name__)


# ── Environment constants ──────────────────────────────────────────────────────

SETTLE_TICKS    = 10     # ticks to wait after spawning before returning obs
                         # gives the physics engine time to stabilize the vehicle
DELTA_SECONDS   = 0.05   # fixed timestep: 0.05s = 20 FPS
DEFAULT_MAP     = "Town03"
DEFAULT_HOST    = "localhost"
DEFAULT_PORT    = 2000
DEFAULT_TIMEOUT = 10.0
DESTINATION_REACHED_M = 5.0
SPAWN_HEIGHT_OFFSET_M = 0.5   # lift above a route waypoint's raw road-surface
                              # z when spawning there, to avoid a ground-
                              # clipping spawn collision (see reset())

# Route line visualization: redrawn periodically with a short life_time
# rather than once per episode with a life_time covering the whole
# episode. CARLA has no API to explicitly clear a debug shape — it only
# disappears when its own life_time expires — so a single long-lived
# draw would leave a short episode's route still visible on screen after
# the next episode's reset() draws a new one on top, showing two
# overlapping routes at once. Periodic redraw with a short life_time
# means a finished episode's route fades within ROUTE_DRAW_LIFETIME_S
# regardless of when it actually ended.
ROUTE_REDRAW_INTERVAL_STEPS = 20    # ~1s at 20Hz — cheap enough to redraw
                                    # a few-hundred-segment route this often
ROUTE_DRAW_LIFETIME_S       = 1.2   # slightly longer than the redraw
                                    # interval so the line never flickers
                                    # between redraws


SPECTATOR_DISTANCE_M = 8.0    # meters behind the vehicle
SPECTATOR_HEIGHT_M   = 4.0    # meters above the vehicle
SPECTATOR_PITCH_DEG  = -15.0  # degrees, looking down toward the vehicle


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


def _compute_effective_spawn_index(spawn_index: int, offset: int, num_spawn_points: int) -> int:
    """
    Compute the actual spawn point index to use, after applying a role
    offset (e.g. +1 for the eval environment) with wraparound.

    Pure function (no CARLA connection needed) so it can be unit tested
    offline. Exists because agent/train.py runs a train_env and an
    eval_env simultaneously against the same CARLA server; if both were
    configured with the identical spawn_index, whichever one currently
    has a live vehicle there would permanently block the other from
    ever spawning (confirmed via live testing). Offsetting the eval
    environment's effective index keeps both deterministic while
    avoiding the conflict.
    """
    return (spawn_index + offset) % num_spawn_points


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


# ── Main environment class ─────────────────────────────────────────────────────

class CarlaLaneKeepingEnv(gym.Env):
    """
    CARLA Lane Keeping environment following the Gymnasium interface.

    Observation space: Box(4,) — [lateral_dist, heading_err, speed, steer]
    Action space:      Box(2,) — [acceleration, steer]
    Reward:            dense, see carla_env/reward.py

    Parameters
    ----------
    host          : CARLA server IP
    port          : CARLA server port
    map_name      : which CARLA map to use
    max_steps     : episode length limit (steps before truncation)
    reward_config : RewardConfig instance (uses defaults if None)
    action_smooth : alpha for action smoother (0=heavy, 1=none)
    seed          : random seed for spawn point selection (random mode only)
    spawn_index   : if set, always spawn at this exact spawn point index
                    (deterministic — for validating learning); if None,
                    spawn randomly (default)
    spawn_index_offset : added to spawn_index (with wraparound) before
                    use; lets a second simultaneous environment (e.g.
                    agent/train.py's eval_env) avoid contending for the
                    identical spawn point as this one (default 0)
    verbose       : if True, log step-level info (slow — use for debugging)

    Example usage:
        env = CarlaLaneKeepingEnv()
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        env.close()
    """

    # Gymnasium metadata — tells wrappers and monitors what render modes exist
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str           = DEFAULT_HOST,
        port: int           = DEFAULT_PORT,
        map_name: str       = DEFAULT_MAP,
        max_steps: int      = 1000,
        reward_config       = None,
        action_smooth: float= 0.6,
        seed: int           = None,
        spawn_index: int     = None,
        spawn_index_offset: int = 0,
        verbose: bool       = False,
    ):
        super().__init__()

        # ── Store config ───────────────────────────────────────────────────────
        self.host          = host
        self.port          = port
        self.map_name      = map_name
        self.max_steps     = max_steps
        self.reward_config = reward_config or RewardConfig()
        self.spawn_index        = spawn_index
        self.spawn_index_offset = spawn_index_offset
        self.verbose       = verbose

        # ── Gymnasium spaces ───────────────────────────────────────────────────
        self.observation_space = get_observation_space()
        self.action_space      = get_action_space()

        # ── Internal state — all None until reset() is called ─────────────────
        self._client           = None   # carla.Client
        self._world            = None   # carla.World
        self._carla_map        = None   # carla.Map
        self._vehicle             = None   # carla.Vehicle (ego)
        self._collision_sensor    = None   # CollisionSensor wrapper
        self._spawn_points        = []     # list of carla.Transform
        self._last_spawn_transform = None  # carla.Transform vehicle was spawned at
        self._previous_raw_action  = np.zeros(2, dtype=np.float32)  # for smoothness reward

        # ── Episode tracking ───────────────────────────────────────────────────
        self._step_count       = 0
        self._episode_reward   = 0.0
        self._episode_count    = 0

        # ── Action processor (smoother + translator) ───────────────────────────
        self._action_processor = ActionProcessor(alpha=action_smooth)

        # ── Stall detector ─────────────────────────────────────────────────────
        self._stall_detector = StallDetector(self.reward_config)

        # ── Red light violation detector ───────────────────────────────────────
        self._red_light_detector = RedLightViolationDetector(self.reward_config)

        # ── Random number generator for spawn point selection ──────────────────
        self._rng = random.Random(seed)

        # ── Route planning ───────────────────────────────────────────────────── 
        self._route_planner = None
        self._route = []
        self._route_index = 0

        # ── Connect to CARLA once at init ──────────────────────────────────────
        # We connect here rather than in reset() so connection errors are
        # caught immediately at env creation, not mid-training.
        self._connect()

    # ── Connection and world setup ─────────────────────────────────────────────

    def _connect(self) -> None:
        """
        Connect to CARLA server and load the map.
        Called once at __init__. Not called again between episodes.
        """
        import carla

        logger.info(f"Connecting to CARLA at {self.host}:{self.port} ...")
        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(DEFAULT_TIMEOUT)

        version = self._client.get_server_version()
        logger.info(f"Connected. Server version: {version}")

        # Reuse the already-running world if it's already on the requested
        # map, instead of always calling load_world(). Reloading a map that's
        # already loaded is redundant and has been observed to crash the
        # CARLA server — this matters because train.py creates two
        # environments (train + eval) back to back, and the second one would
        # otherwise reload the map the first one just loaded.
        current_world     = self._client.get_world()
        current_map_name  = current_world.get_map().name.split("/")[-1]
        if current_map_name == self.map_name:
            logger.info(f"Map {self.map_name} already loaded, reusing world.")
            self._world = current_world
        else:
            logger.info(f"Loading map: {self.map_name} ...")
            self._world = self._client.load_world(self.map_name)
            time.sleep(2.0)   # let world initialize

        self._carla_map   = self._world.get_map()
        self._spawn_points = self._carla_map.get_spawn_points()
        logger.info(f"Map loaded. Spawn points: {len(self._spawn_points)}")

        _check_spawn_index_in_range(self.spawn_index, len(self._spawn_points))

        self._route_planner = RoutePlanner(self._carla_map, sampling_resolution=2.0)

        # Enable synchronous mode once — stays on for the entire training run.
        # We only disable it in close().
        self._enable_sync_mode()

    def _enable_sync_mode(self) -> None:
        """Switch CARLA to synchronous mode with fixed timestep."""
        settings = self._world.get_settings()
        settings.synchronous_mode  = True
        settings.fixed_delta_seconds = DELTA_SECONDS
        self._world.apply_settings(settings)
        self._world.tick()   # acknowledge new settings
        logger.info(f"Sync mode ON  (delta={DELTA_SECONDS}s)")

    def _disable_sync_mode(self) -> None:
        """Restore asynchronous mode. Called only in close()."""
        if self._world is not None:
            settings = self._world.get_settings()
            settings.synchronous_mode  = False
            settings.fixed_delta_seconds = None
            self._world.apply_settings(settings)
            logger.info("Sync mode OFF")

    # ── Actor management ───────────────────────────────────────────────────────

    def _destroy_actors(self) -> None:
        """
        Destroy the collision sensor and vehicle from the previous episode.

        Order matters: always destroy sensors before the vehicle.
        If you destroy the vehicle first, the sensor loses its parent
        and CARLA may crash or leave orphaned actors.
        """
        if self._collision_sensor is not None:
            self._collision_sensor.destroy()
            self._collision_sensor = None

        if self._vehicle is not None:
            if self._vehicle.is_alive:
                self._vehicle.destroy()
            self._vehicle = None

    def _choose_spawn_transform(self):
        """
        Choose the candidate origin transform used to seed route
        planning — NOT where the vehicle is ultimately spawned.

        If self.spawn_index is set, always returns that exact spawn
        point (deterministic — for validating that an algorithm is
        learning, since episode-to-episode progress is only comparable
        from a fixed start). If None, picks a random spawn point.

        Why the vehicle doesn't spawn here directly: route planning
        needs a location before any route (or actor) exists, but the
        actual spawn transform should be the route's own first waypoint
        (see reset()) — RoutePlanner.plan_route()'s underlying
        trace_route() snaps its start to the nearest topology node,
        which is usually this exact candidate point but not always
        (short routes, awkward junctions). Spawning at the raw candidate
        instead of the route's real start produces an instant, spurious
        lateral-distance blowup the moment the episode begins (confirmed
        via live testing — occurred in ~1 in 10 episodes on Town10).
        """
        if self.spawn_index is not None:
            effective_index = _compute_effective_spawn_index(
                self.spawn_index, self.spawn_index_offset, len(self._spawn_points)
            )
            return self._spawn_points[effective_index]
        return self._rng.choice(self._spawn_points)

    def _spawn_vehicle_at(self, transform):
        """
        Spawn the ego vehicle at an exact transform.

        Retries the same point up to 5 times on transient occupation,
        then raises — never falls back to a different point. By the
        time this is called, the caller has already committed to this
        exact transform (the route's own first waypoint); silently
        spawning somewhere else would break the "vehicle starts exactly
        on its route" guarantee this method exists to provide.

        Returns carla.Vehicle.
        Raises RuntimeError if all spawn attempts fail.
        """
        bp = self._world.get_blueprint_library().find("vehicle.tesla.model3")
        if bp.has_attribute("color"):
            bp.set_attribute("color", "255,0,0")   # red for visibility

        for attempt in range(5):
            vehicle = self._world.try_spawn_actor(bp, transform)
            if vehicle is not None:
                logger.debug(
                    f"Vehicle spawned (attempt {attempt+1}): "
                    f"x={transform.location.x:.1f}, y={transform.location.y:.1f}"
                )
                self._last_spawn_transform = transform
                return vehicle

        raise RuntimeError(
            f"Failed to spawn vehicle at x={transform.location.x:.1f}, "
            f"y={transform.location.y:.1f} after 5 attempts. "
            f"That point stayed occupied."
        )

    def _snap_spectator_to_vehicle(self, vehicle_transform) -> None:
        """
        Move the CARLA spectator camera to a chase view of the vehicle.

        Purely a visualization aid — CARLA's spectator never moves on its
        own, so without this there's no way to see the car in the CARLA
        window without manually flying the free-look camera to find it.
        Has no effect on training; safe to call every reset and step.

        Takes vehicle_transform as a parameter rather than always calling
        self._vehicle.get_transform() itself, because the right source
        differs by caller:
          - reset() passes self._last_spawn_transform (the transform
            _spawn_vehicle_at() requested). CARLA's client-side actor cache
            does not reflect a freshly spawned actor's real transform
            until at least one world.tick() has elapsed — querying
            get_transform() right after spawning returns a stale
            (0,0,0)-at-yaw-0 placeholder (confirmed via live testing).
          - step() passes self._vehicle.get_transform() directly, which
            is reliable by then (at least one tick has already happened
            since spawn) and lets the camera continuously track the
            moving vehicle rather than staying fixed at its spawn point.
        """
        cam_transform = _compute_spectator_transform(
            vehicle_transform,
            distance_m=SPECTATOR_DISTANCE_M,
            height_m=SPECTATOR_HEIGHT_M,
            pitch_deg=SPECTATOR_PITCH_DEG,
        )
        self._world.get_spectator().set_transform(cam_transform)

    def _destination_reached(self) -> bool:
        """Return True when the vehicle is close enough to the route destination."""
        if not self._route:
            return False

        destination = self._route[-1].waypoint.transform.location
        vehicle_location = self._vehicle.get_transform().location

        return (
            self._route_planner.distance(
                vehicle_location,
                destination,
            )
            <= DESTINATION_REACHED_M
        )

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Start a new episode.

        Called by PPO at the beginning of every episode.
        Also called manually to start the first episode.

        Returns
        -------
        obs  : np.ndarray of shape (4,) — initial observation
        info : dict — episode metadata
        """
        super().reset(seed=seed)

        # Log previous episode summary
        if self._episode_count > 0:
            logger.info(
                f"Episode {self._episode_count} ended. "
                f"Steps: {self._step_count}  "
                f"Total reward: {self._episode_reward:.2f}"
            )

        # ── Clean up previous episode actors ──────────────────────────────────
        self._destroy_actors()

        # ── Tick once to let destructions propagate ────────────────────────────
        self._world.tick()

        # ── Plan the route before spawning ──────────────────────────────────────
        # Route planning only needs a location, not a live actor — and the
        # vehicle should ultimately spawn on the route's own first waypoint
        # (see below), so the route has to exist first.
        candidate_transform = self._choose_spawn_transform()
        start_location = candidate_transform.location

        # Choose a destination different from the spawn point.
        if self.spawn_index is not None:
            effective_spawn_index = _compute_effective_spawn_index(
                self.spawn_index,
                self.spawn_index_offset,
                len(self._spawn_points),
            )

            destination_index = (
                effective_spawn_index + 20
            ) % len(self._spawn_points)
        else:
            destination_index = self._rng.randrange(len(self._spawn_points))

        destination_transform = self._spawn_points[destination_index]
        destination_location = destination_transform.location

        self._route = self._route_planner.plan_route(
            start_location,
            destination_location,
        )

        self._route_index = 0

        logger.info(
            f"Route generated: {len(self._route)} waypoints, "
            f"destination=({destination_location.x:.1f}, "
            f"{destination_location.y:.1f})"
        )

        # ── Spawn the vehicle exactly on the route's first waypoint ─────────────
        # Not at candidate_transform directly — trace_route() snaps its start
        # to the nearest topology node, which usually matches the candidate
        # but not always. Spawning at route[0] instead guarantees the vehicle
        # starts exactly where the route begins (position AND heading),
        # eliminating that mismatch by construction rather than in the
        # common case only. Falls back to the candidate if plan_route()
        # somehow returned an empty route (e.g. start == destination).
        #
        # Route waypoints sit exactly at road-surface height, unlike
        # carla_map.get_spawn_points()'s entries, which carry a small
        # built-in vertical offset specifically to avoid a ground-clipping
        # spawn collision. Spawning at the waypoint's raw z reliably failed
        # all 5 retry attempts in live testing ("stayed occupied" — not
        # real occupation, since nothing else exists in this world) — so
        # lift it the same way _compute_spectator_transform and
        # RoutePlanner.draw_route already lift waypoint locations for
        # their own purposes.
        import carla
        if self._route:
            wp_transform = self._route[0].waypoint.transform
            spawn_transform = carla.Transform(
                wp_transform.location + carla.Location(z=SPAWN_HEIGHT_OFFSET_M),
                wp_transform.rotation,
            )
        else:
            spawn_transform = candidate_transform
        self._vehicle = self._spawn_vehicle_at(spawn_transform)

        # ── Draw the route in the CARLA world ───────────────────────────────────
        # Visualization aid only — CARLA's debug draw has no effect on
        # training. Short life_time, redrawn periodically in step() — see
        # ROUTE_DRAW_LIFETIME_S's comment for why not a single long-lived draw.
        self._route_planner.draw_route(
            self._world, self._route, life_time=ROUTE_DRAW_LIFETIME_S
        )

        # ── Move spectator camera to follow the vehicle ────────────────────────
        # Visualization aid only — CARLA's spectator never moves on its own.
        self._snap_spectator_to_vehicle(self._last_spawn_transform)

        # ── Attach collision sensor ────────────────────────────────────────────
        self._collision_sensor = CollisionSensor(self._world, self._vehicle)

        # ── Tick to register actors in physics engine ──────────────────────────
        self._world.tick()

        # ── Settle ticks — let physics stabilize ──────────────────────────────
        # Without this, the vehicle may have residual velocity or be slightly
        # clipping the ground. A few ticks at zero throttle fixes this.
        for _ in range(SETTLE_TICKS):
            self._world.tick()

        # ── Reset episode tracking ─────────────────────────────────────────────
        self._step_count     = 0
        self._episode_reward = 0.0
        self._episode_count += 1

        # ── Reset action smoother ──────────────────────────────────────────────
        # Critical: without this, the smoother carries state from the
        # last episode into the new one.
        self._action_processor.reset()

        # ── Reset stall detector ───────────────────────────────────────────────
        self._stall_detector.reset()

        # ── Reset red light violation detector ─────────────────────────────────
        self._red_light_detector.reset()

        # ── Reset smoothness-reward tracking ────────────────────────────────────
        # Same reasoning as the action smoother: without this, the first
        # action of a new episode would be compared against the last
        # action of the *previous* episode.
        self._previous_raw_action = np.zeros(2, dtype=np.float32)

        # ── Compute initial observation ────────────────────────────────────────
        # lookahead=0: the closest route waypoint, not one further ahead.
        # lateral_distance/heading_error are measured against this point, so
        # it needs to reflect where the car actually is right now — a point
        # further down the route can already be mid-curve, producing a large
        # apparent lateral_distance that has nothing to do with real lane
        # position (confirmed via live testing: caused spurious off-road
        # terminations on routes with an early turn or a short route where
        # lookahead overshot almost to the destination).
        target_waypoint = self._route_planner.get_target_waypoint(
            self._route, self._route_index, lookahead=0
        )
        self._route_planner.draw_target_waypoint(
            self._world, target_waypoint, life_time=DELTA_SECONDS * 2
        )

        traffic_light = get_traffic_light_affordance(self._vehicle)

        obs_array, obs_data = compute_observation(
            self._vehicle,
            self._carla_map,
            route_waypoint=target_waypoint,
            traffic_light=traffic_light,
        )

        info = {
            "episode":          self._episode_count,
            "lateral_distance": obs_data.lateral_distance_m,
            "heading_error":    obs_data.heading_error_rad,
            "speed_kmh":        obs_data.speed_kmh,
            "traffic_light_state":     obs_data.traffic_light_state,
            "traffic_light_must_stop": obs_data.traffic_light_must_stop,
            "route_index": self._route_index,
            "route_length": len(self._route),
            "remaining_distance": self._route_planner.remaining_distance(
                self._route,
                self._route_index,
            ),
        }

        return obs_array, info

    def step(self, action: np.ndarray):
        """
        Apply one action and advance the simulation by one timestep.

        Called by PPO at every step of the episode.

        Parameters
        ----------
        action : np.ndarray of shape (2,)
                 action[0] = acceleration ∈ [-1, 1]
                 action[1] = steer        ∈ [-1, 1]

        Returns
        -------
        obs        : np.ndarray (4,) — new observation
        reward     : float
        terminated : bool — True if agent failed (collision, off-road)
        truncated  : bool — True if episode hit step limit
        info       : dict — step metadata for logging
        """
        assert self._vehicle is not None, \
            "step() called before reset(). Call env.reset() first."

        # ── Apply action to vehicle ────────────────────────────────────────────
        control = self._action_processor.process(action)
        self._vehicle.apply_control(control)

        # ── Advance simulation by one physics step ─────────────────────────────
        self._world.tick()

        # ── Move spectator camera to follow the moving vehicle ─────────────────
        # Visualization aid only — see _snap_spectator_to_vehicle()'s docstring
        # for why step() (unlike reset()) can safely use get_transform() directly.
        self._snap_spectator_to_vehicle(self._vehicle.get_transform())

        # ── Update route tracking ──────────────────────────────────────────────
        current_location = self._vehicle.get_transform().location

        self._route_index = self._route_planner.get_closest_waypoint_index(
            self._route,
            current_location,
            start_index=self._route_index,
        )

        # lookahead=0 — see the matching comment in reset() for why.
        target_waypoint = self._route_planner.get_target_waypoint(
            self._route,
            self._route_index,
            lookahead=0,
        )
        self._route_planner.draw_target_waypoint(
            self._world, target_waypoint, life_time=DELTA_SECONDS * 2
        )

        # ── Periodically refresh the route line ─────────────────────────────────
        # See ROUTE_DRAW_LIFETIME_S's comment in the constants section for why
        # this is a periodic short-lived redraw rather than a single
        # episode-long one.
        if self._step_count % ROUTE_REDRAW_INTERVAL_STEPS == 0:
            self._route_planner.draw_route(
                self._world, self._route, life_time=ROUTE_DRAW_LIFETIME_S
            )

        # ── Read traffic light affordance ──────────────────────────────────────
        traffic_light = get_traffic_light_affordance(self._vehicle)

        # ── Compute new observation ────────────────────────────────────────────
        obs_array, obs_data = compute_observation(
            self._vehicle,
            self._carla_map,
            route_waypoint=target_waypoint,
            traffic_light=traffic_light,
        )

        # ── Check whether destination was reached ──────────────────────────────
        destination_reached = self._destination_reached()

        # ── Check termination ──────────────────────────────────────────────────
        collision_flag = self._collision_sensor.has_collided
        self._step_count += 1
        stall_flag = self._stall_detector.update(
            obs_data.speed_kmh, must_stop=obs_data.traffic_light_must_stop
        )
        red_light_violation = self._red_light_detector.update(
            traffic_light, obs_data.speed_kmh
        )

        terminated, truncated, term_reason = check_termination(
            obs_data             = obs_data,
            collision_flag       = collision_flag,
            step_count           = self._step_count,
            max_steps            = self.max_steps,
            max_lateral_m        = self.reward_config.max_lateral_m,
            stall_flag           = stall_flag,
            red_light_violation  = red_light_violation,
        )

        # Reaching the destination is a successful termination.
        if destination_reached and not terminated:
            terminated = True
            truncated = False
            term_reason = "destination_reached"


        # Terminal penalty only on agent failure, not on timeout
        is_terminal_for_reward = terminated   # not truncated

        # ── Compute action delta for the smoothness reward ─────────────────────
        action_array = np.asarray(action, dtype=np.float32)
        action_delta = action_array - self._previous_raw_action
        self._previous_raw_action = action_array.copy()

        # ── Compute reward ─────────────────────────────────────────────────────
        reward, reward_info = compute_reward(
            obs_data     = obs_data,
            is_terminal  = is_terminal_for_reward,
            action_delta = action_delta,
            cfg          = self.reward_config,
        )

        # ── Track episode totals ───────────────────────────────────────────────
        self._episode_reward += reward

        # ── Build info dict ────────────────────────────────────────────────────
        # SB3 logs scalar values from this dict automatically if you use
        # the right callback. Keep keys consistent — they become your
        # TensorBoard metric names.
        info = {
            # Step state
            "step":             self._step_count,
            "lateral_distance": obs_data.lateral_distance_m,
            "heading_error_deg":np.degrees(obs_data.heading_error_rad),
            "speed_kmh":        obs_data.speed_kmh,
            "steering":         obs_data.steering,
            "traffic_light_state":     obs_data.traffic_light_state,
            "traffic_light_must_stop": obs_data.traffic_light_must_stop,
            # Reward breakdown
            "reward_total":      reward_info.total,
            "reward_centering":  reward_info.r_centering,
            "reward_speed":      reward_info.r_speed,
            "reward_heading":    reward_info.r_heading,
            "reward_smoothness": reward_info.r_smoothness,
            "reward_terminal":   reward_info.r_terminal,
            # Episode
            "episode_reward":   self._episode_reward,
            "collision":        collision_flag,
            "red_light_violation": red_light_violation,
            "termination_reason": term_reason,
            # Route
            "route_index": self._route_index,
            "route_length": len(self._route),
            "destination_reached": destination_reached,
            "remaining_distance": self._route_planner.remaining_distance(
                self._route,
                self._route_index,
            ),
        }

        if self.verbose:
            logger.debug(
                f"Step {self._step_count:4d} | "
                f"lat={obs_data.lateral_distance_m:+.2f}m | "
                f"spd={obs_data.speed_kmh:.1f}km/h | "
                f"r={reward:+.3f} | "
                f"{term_reason if term_reason else 'ok'}"
            )

        return obs_array, reward, terminated, truncated, info

    def close(self) -> None:
        """
        Clean up all CARLA resources.

        Called automatically by Gymnasium when the environment is garbage
        collected, or manually with env.close(). Always call this when
        you are done with the environment — leaving sync mode on crashes
        CARLA.
        """
        logger.info("Closing environment ...")
        self._destroy_actors()
        self._disable_sync_mode()

        # Tick once after restoring async mode so CARLA acknowledges it
        if self._world is not None:
            try:
                self._world.tick()
            except Exception:
                pass   # world may already be gone if CARLA was closed

        self._client = None
        self._world  = None
        logger.info("Environment closed.")

    # ── Properties for external access ────────────────────────────────────────

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def episode_reward(self) -> float:
        return self._episode_reward
