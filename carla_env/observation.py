"""
observation.py
--------------
Phase 3 — Observation Space

Purpose:
    Given a CARLA vehicle actor and world, compute the 4-element
    observation vector that the RL agent receives at every step.

    obs = [lateral_distance, heading_error, speed_norm, steering]

    All values are normalized to roughly [-1, 1] so the neural network
    gets consistently scaled inputs. Raw physical units are also available
    for logging and debugging.

Why this is its own file:
    The observation logic is independent of the reward, action, and
    training code. Keeping it isolated means:
    - You can unit-test it without running CARLA
    - You can swap or extend it without touching env.py
    - It is easy to explain in a thesis: "Section 3.2 — Observation Design"

Observation vector layout:
    Index  Name              Range (normalized)   Physical meaning
    -----  ----------------  -------------------  --------------------------------
      0    lateral_distance  -1.0 … +1.0          meters from lane center
      1    heading_error     -1.0 … +1.0          radians off road direction
      2    speed             0.0 … 1.0            vehicle speed in km/h
      3    steering          -1.0 … +1.0          current steering wheel angle

Sign conventions:
    lateral_distance > 0  → vehicle is to the RIGHT of lane center
    lateral_distance < 0  → vehicle is to the LEFT of lane center
    heading_error    > 0  → vehicle is pointing RIGHT of road direction
    heading_error    < 0  → vehicle is pointing LEFT of road direction
"""

import math
import numpy as np


# ── Normalization constants ───────────────────────────────────────────────────
# These define the physical range we expect during lane keeping.
# Values outside these ranges get clipped to [-1, 1].
# Tune these if you change the task or map.

MAX_LATERAL_DISTANCE_M = 3.5    # Half lane-width. Beyond this = off road.
MAX_HEADING_ERROR_RAD  = math.pi  # Full 180° (worst case heading)
MAX_SPEED_KMH          = 80.0   # We will cap the vehicle to ~50 km/h in training,
                                 # but normalize against 80 so there is headroom.


# ── Data class for raw (un-normalized) values ─────────────────────────────────
# We return both the normalized obs array (for the agent) and the raw
# ObservationData object (for logging and reward computation).

class ObservationData:
    """
    Holds the raw, physical-unit values computed from CARLA.
    Used for logging, reward computation, and debugging.
    Not passed directly to the agent — the agent gets the normalized array.
    """
    def __init__(
        self,
        lateral_distance_m: float,   # meters, signed
        heading_error_rad:  float,   # radians, signed
        speed_kmh:          float,   # km/h, always >= 0
        steering:           float,   # -1.0 to 1.0 (CARLA native)
        waypoint=None,               # carla.Waypoint (for debugging/rendering)
    ):
        self.lateral_distance_m = lateral_distance_m
        self.heading_error_rad  = heading_error_rad
        self.speed_kmh          = speed_kmh
        self.steering           = steering
        self.waypoint           = waypoint

    def __repr__(self) -> str:
        return (
            f"ObsData("
            f"lat={self.lateral_distance_m:+.2f}m, "
            f"hdg={math.degrees(self.heading_error_rad):+.1f}°, "
            f"spd={self.speed_kmh:.1f}km/h, "
            f"steer={self.steering:+.2f})"
        )


# ── Core computation functions ────────────────────────────────────────────────

def get_lateral_distance(vehicle, waypoint) -> float:
    """
    Compute the signed lateral distance from the vehicle to the lane center.

    How it works:
        1. Get the vector from waypoint (lane center) to vehicle
        2. Project it onto the waypoint's RIGHT axis
        3. The scalar projection is the signed lateral displacement

    Why the right axis?
        The waypoint gives us the road's forward direction (yaw angle).
        Rotating that 90° clockwise gives the road's RIGHT direction.
        Projecting the offset onto RIGHT tells us: is the vehicle to the
        right (+) or left (-) of center, and by how much?

    Args:
        vehicle:  carla.Vehicle actor
        waypoint: carla.Waypoint (nearest to the vehicle)

    Returns:
        float: signed distance in meters
    """
    # Vehicle position
    veh_loc = vehicle.get_location()

    # Waypoint position (lane center)
    wp_loc = waypoint.transform.location

    # Vector from waypoint to vehicle
    dx = veh_loc.x - wp_loc.x
    dy = veh_loc.y - wp_loc.y

    # Waypoint yaw in radians (road forward direction)
    # CARLA yaw: 0° = East, 90° = South (left-hand coordinate system)
    wp_yaw_rad = math.radians(waypoint.transform.rotation.yaw)

    # Right-axis vector: rotate forward by -90° (clockwise = rightward)
    # In a left-hand system (CARLA): right = (sin(yaw), -cos(yaw))
    right_x = math.sin(wp_yaw_rad)
    right_y = -math.cos(wp_yaw_rad)

    # Signed lateral distance = dot product of offset with right axis
    lateral_distance = dx * right_x + dy * right_y

    return lateral_distance


def get_heading_error(vehicle, waypoint) -> float:
    """
    Compute the signed heading error between the vehicle and the road.

    The heading error is the angular difference between:
    - Where the vehicle is pointing (vehicle yaw)
    - Where the road is pointing (waypoint yaw)

    A heading error of 0 means the vehicle is aligned with the road.
    Positive means the vehicle points right of the road direction.
    Negative means the vehicle points left.

    We wrap the result to [-π, π] to avoid discontinuities at ±180°.

    Args:
        vehicle:  carla.Vehicle actor
        waypoint: carla.Waypoint

    Returns:
        float: heading error in radians, in [-π, π]
    """
    veh_yaw_rad = math.radians(vehicle.get_transform().rotation.yaw)
    wp_yaw_rad  = math.radians(waypoint.transform.rotation.yaw)

    # Raw difference
    error = veh_yaw_rad - wp_yaw_rad

    # Wrap to [-π, π] to handle the 0°/360° boundary
    # math.atan2(sin(x), cos(x)) is the standard wrap trick
    error = math.atan2(math.sin(error), math.cos(error))

    return error


def get_speed_kmh(vehicle) -> float:
    """
    Compute the vehicle's scalar speed in km/h.

    CARLA gives velocity as a 3D vector (vx, vy, vz) in m/s.
    We compute the Euclidean magnitude and convert.

    Args:
        vehicle: carla.Vehicle actor

    Returns:
        float: speed in km/h (always >= 0)
    """
    v = vehicle.get_velocity()
    speed_ms = math.sqrt(v.x**2 + v.y**2 + v.z**2)
    return speed_ms * 3.6


def get_steering(vehicle) -> float:
    """
    Read the current steering angle from the vehicle's control state.

    CARLA's steer is already in [-1.0, +1.0]:
        -1.0 = full left
         0.0 = straight
        +1.0 = full right

    Args:
        vehicle: carla.Vehicle actor

    Returns:
        float: steering in [-1.0, 1.0]
    """
    control = vehicle.get_control()
    return control.steer


# ── Normalization helper ───────────────────────────────────────────────────────

def normalize_clip(value: float, max_abs: float) -> float:
    """
    Normalize a value to [-1, 1] and clip it to that range.

    Formula: normalized = clip(value / max_abs, -1, 1)

    Why clip?
        If the vehicle goes far off road, the raw value may exceed max_abs.
        Clipping prevents NaN or extreme values from reaching the neural net.
        The agent will already be getting a penalty for being far off road,
        so the clipped value is fine for training.

    Args:
        value:   raw physical value
        max_abs: the maximum expected absolute value (defines the scale)

    Returns:
        float in [-1.0, 1.0]
    """
    return float(np.clip(value / max_abs, -1.0, 1.0))


# ── Main observation builder ──────────────────────────────────────────────────

def compute_observation(vehicle, carla_map) -> tuple:
    """
    Build the full observation for one step.

    This is the function called by env.py at every step() and reset().

    Args:
        vehicle:    carla.Vehicle actor (the ego vehicle)
        carla_map:  carla.Map  (from world.get_map())

    Returns:
        obs_array:  np.ndarray of shape (4,), dtype float32
                    normalized values ready for the neural network
        obs_data:   ObservationData with raw physical values
                    used for reward computation and logging
    """

    # ── Get the nearest waypoint ───────────────────────────────────────────────
    # project_to_road=True: snap to the nearest point ON the road
    # lane_type=Driving:    ignore sidewalks and shoulders
    waypoint = carla_map.get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,  # noqa (carla imported in env.py)
    )

    # ── Compute raw values ────────────────────────────────────────────────────
    lateral_distance = get_lateral_distance(vehicle, waypoint)
    heading_error    = get_heading_error(vehicle, waypoint)
    speed_kmh        = get_speed_kmh(vehicle)
    steering         = get_steering(vehicle)

    # ── Build raw data object (for reward + logging) ──────────────────────────
    obs_data = ObservationData(
        lateral_distance_m=lateral_distance,
        heading_error_rad=heading_error,
        speed_kmh=speed_kmh,
        steering=steering,
        waypoint=waypoint,
    )

    # ── Build normalized array (for the neural network) ───────────────────────
    obs_array = np.array([
        normalize_clip(lateral_distance, MAX_LATERAL_DISTANCE_M),   # obs[0]
        normalize_clip(heading_error,    MAX_HEADING_ERROR_RAD),     # obs[1]
        normalize_clip(speed_kmh,        MAX_SPEED_KMH),             # obs[2]
        float(np.clip(steering, -1.0, 1.0)),                         # obs[3]
    ], dtype=np.float32)

    return obs_array, obs_data


# ── Observation space definition (for Gymnasium) ──────────────────────────────

def get_observation_space():
    """
    Return the Gymnasium observation space definition.

    This tells the RL algorithm the shape and bounds of what the agent sees.
    We use a Box space: a 4D continuous vector, all values in [-1, 1].

    Called once during environment initialization (env.py __init__).
    """
    import gymnasium as gym
    return gym.spaces.Box(
        low=np.array([-1.0, -1.0,  0.0, -1.0], dtype=np.float32),
        high=np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
        dtype=np.float32,
    )


# ── Import guard ───────────────────────────────────────────────────────────────
# carla is imported here only for the LaneType enum used in compute_observation.
# This allows the rest of the functions (geometry math) to be tested
# without CARLA running.
try:
    import carla
except ImportError:
    carla = None   # Will fail at runtime if compute_observation is called
                   # without CARLA available — that is the correct behavior.
