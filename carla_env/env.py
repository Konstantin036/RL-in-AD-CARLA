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
    compute_reward,
    check_termination,
)
from carla_env.sensors     import CollisionSensor


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
    seed          : random seed for spawn point selection
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
        verbose: bool       = False,
    ):
        super().__init__()

        # ── Store config ───────────────────────────────────────────────────────
        self.host          = host
        self.port          = port
        self.map_name      = map_name
        self.max_steps     = max_steps
        self.reward_config = reward_config or RewardConfig()
        self.spawn_index = spawn_index
        self.verbose       = verbose

        # ── Gymnasium spaces ───────────────────────────────────────────────────
        self.observation_space = get_observation_space()
        self.action_space      = get_action_space()

        # ── Internal state — all None until reset() is called ─────────────────
        self._client           = None   # carla.Client
        self._world            = None   # carla.World
        self._carla_map        = None   # carla.Map
        self._vehicle          = None   # carla.Vehicle (ego)
        self._collision_sensor = None   # CollisionSensor wrapper
        self._spawn_points     = []     # list of carla.Transform

        # ── Episode tracking ───────────────────────────────────────────────────
        self._step_count       = 0
        self._episode_reward   = 0.0
        self._episode_count    = 0

        # ── Action processor (smoother + translator) ───────────────────────────
        self._action_processor = ActionProcessor(alpha=action_smooth)

        # ── Random number generator for spawn point selection ──────────────────
        self._rng = random.Random(seed)

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

        # ── Spawn new vehicle ──────────────────────────────────────────────────
        self._vehicle = self._spawn_vehicle()

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

        # ── Compute initial observation ────────────────────────────────────────
        obs_array, obs_data = compute_observation(self._vehicle, self._carla_map)

        info = {
            "episode":          self._episode_count,
            "lateral_distance": obs_data.lateral_distance_m,
            "heading_error":    obs_data.heading_error_rad,
            "speed_kmh":        obs_data.speed_kmh,
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

        # ── Compute new observation ────────────────────────────────────────────
        obs_array, obs_data = compute_observation(self._vehicle, self._carla_map)

        # ── Check termination ──────────────────────────────────────────────────
        collision_flag = self._collision_sensor.has_collided
        self._step_count += 1

        terminated, truncated, term_reason = check_termination(
            obs_data         = obs_data,
            collision_flag   = collision_flag,
            step_count       = self._step_count,
            max_steps        = self.max_steps,
            max_lateral_m    = self.reward_config.max_lateral_m,
        )

        # Terminal penalty only on agent failure, not on timeout
        is_terminal_for_reward = terminated   # not truncated

        # ── Compute reward ─────────────────────────────────────────────────────
        reward, reward_info = compute_reward(
            obs_data    = obs_data,
            is_terminal = is_terminal_for_reward,
            cfg         = self.reward_config,
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
            # Reward breakdown
            "reward_total":     reward_info.total,
            "reward_centering": reward_info.r_centering,
            "reward_speed":     reward_info.r_speed,
            "reward_heading":   reward_info.r_heading,
            "reward_terminal":  reward_info.r_terminal,
            # Episode
            "episode_reward":   self._episode_reward,
            "collision":        collision_flag,
            "termination_reason": term_reason,
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
