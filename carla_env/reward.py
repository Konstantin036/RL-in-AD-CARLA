"""
reward.py
---------
Phase 5 — Reward Function

Purpose:
    Compute the scalar reward signal at each step of the episode.
    The reward tells the agent how well it is performing lane keeping.

Reward formula:
    r = w_center * r_centering
      + w_speed  * r_speed
      + w_heading * r_heading
      + r_terminal  (only on terminal steps)

Where:
    r_centering = 1.0 - |lateral_distance| / MAX_LATERAL_DISTANCE
    r_speed     = exp(-((speed - target_speed)^2) / (2 * sigma^2))
    r_heading   = 1.0 - |heading_error| / pi
    r_terminal  = TERMINAL_PENALTY  (large negative, episode-ending events)

Design goals:
    1. All non-terminal terms are in [0, 1] — easy to reason about weights
    2. Smooth and dense — agent gets a useful signal every single step
    3. No sparse rewards — the agent never has to guess why it got punished
    4. Tunable via RewardConfig — all magic numbers live in one place

Thesis note:
    This reward function can be written as a single equation in your thesis.
    Each term has a clear physical interpretation and a corresponding weight
    that can be reported in your hyperparameter table.
"""

import math
import numpy as np
from dataclasses import dataclass


# ── Reward configuration ───────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """
    All reward hyperparameters in one place.

    These are the values you will report in your thesis hyperparameter table.
    Change them here and the effect propagates everywhere automatically.

    Weights control the relative importance of each term.
    The defaults are a reasonable starting point — you may tune them
    during training if the agent develops bad habits.

    Common tuning guidance:
        Agent stands still     → increase w_speed or target_speed
        Agent drives too fast  → decrease target_speed or sigma_speed
        Agent cuts corners     → increase w_heading
        Agent hugs one side    → increase w_center
        Agent steers jerkily   → increase w_smooth (0.0 disables it)
    """

    # ── Centering term ─────────────────────────────────────────────────────────
    w_center: float = 1.0           # weight for centering reward
    max_lateral_m: float = 3.5      # lateral distance at which r_centering = 0

    # ── Speed term ─────────────────────────────────────────────────────────────
    w_speed: float = 0.5            # weight for speed reward
    target_speed_kmh: float = 30.0  # desired cruising speed
    sigma_speed: float = 10.0       # Gaussian width; larger = more lenient

    # ── Heading term ───────────────────────────────────────────────────────────
    w_heading: float = 0.5          # weight for heading alignment reward

    # ── Smoothness term ────────────────────────────────────────────────────────
    # Penalizes large step-to-step changes in the raw action (acceleration
    # and steering both). Set to 0.0 to disable.
    w_smooth: float = 0.5           # weight for action smoothness penalty

    # ── Terminal penalty ────────────────────────────────────────────────────────
    terminal_penalty: float = -10.0  # reward given when episode ends badly
                                     # (collision or going too far off road)

    # ── Step penalty ────────────────────────────────────────────────────────────
    # Small constant penalty per step. Encourages the agent to complete
    # the task efficiently rather than dawdling.
    # Set to 0.0 to disable.
    step_penalty: float = -0.05


# ── Reward component data class ────────────────────────────────────────────────

@dataclass
class RewardInfo:
    """
    Holds the breakdown of each reward component.
    Returned alongside the scalar reward for logging and debugging.

    In your thesis you can log these separately to show which component
    drives learning at different stages of training.
    """
    total: float         # final scalar reward sent to the agent
    r_centering: float   # centering component (before weighting)
    r_speed: float       # speed component (before weighting)
    r_heading: float     # heading component (before weighting)
    r_smoothness: float  # smoothness component (before weighting)
    r_terminal: float    # terminal penalty (0 unless episode ended)
    r_step: float        # step penalty
    is_terminal: bool    # True if episode ended this step

    def __repr__(self) -> str:
        return (
            f"Reward(total={self.total:+.3f} | "
            f"center={self.r_centering:.3f} "
            f"speed={self.r_speed:.3f} "
            f"heading={self.r_heading:.3f} "
            f"smooth={self.r_smoothness:.3f} "
            f"terminal={self.r_terminal:.1f})"
        )


# ── Individual reward component functions ─────────────────────────────────────

def compute_centering_reward(lateral_distance_m: float, max_lateral_m: float) -> float:
    """
    Reward for staying close to the lane center.

    Formula: 1.0 - |lateral_distance| / max_lateral
    Range:   [0.0, 1.0]
    Peak:    1.0 at lateral_distance = 0 (perfectly centered)
    Zero:    0.0 at |lateral_distance| >= max_lateral (at lane edge)

    Why linear and not quadratic?
        Linear gives the agent a constant gradient to follow — it always
        knows that moving toward center improves reward by the same amount
        regardless of current position. Quadratic would make small errors
        feel nearly identical and only penalize large ones heavily, which
        can slow learning near the lane edge.
    """
    normalized = abs(lateral_distance_m) / max_lateral_m
    return float(max(0.0, 1.0 - normalized))


def compute_speed_reward(speed_kmh: float, target_speed_kmh: float, sigma: float) -> float:
    """
    Reward for driving near the target speed.

    Formula: exp(-((speed - target)^2) / (2 * sigma^2))
    Range:   (0.0, 1.0]
    Peak:    1.0 at speed == target_speed
    Shape:   Gaussian bell curve centered at target_speed

    Why Gaussian?
        It is symmetric around the target and falls off smoothly.
        With sigma=10 km/h, the agent gets >0.6 reward anywhere in
        [20, 40] km/h, giving it a comfortable operating range rather
        than a knife-edge target.

        If we used |speed - target| (linear), the agent would get equal
        penalty for being 5 km/h too slow as being 5 km/h too fast,
        which is physically reasonable. The Gaussian is slightly more
        forgiving and works well in practice.
    """
    diff = speed_kmh - target_speed_kmh
    return float(math.exp(-(diff ** 2) / (2.0 * sigma ** 2)))


def compute_heading_reward(heading_error_rad: float) -> float:
    """
    Reward for being aligned with the road direction.

    Formula: 1.0 - |heading_error| / pi
    Range:   [0.0, 1.0]
    Peak:    1.0 at heading_error = 0 (perfectly aligned)
    Zero:    0.0 at |heading_error| = pi (pointing backwards)

    Why include heading separately from centering?
        A vehicle can be perfectly centered (lateral=0) while pointing
        sideways — it will leave the lane on the next tick regardless
        of steering. The heading term penalizes misalignment *before*
        it causes a lateral deviation, giving the agent an earlier signal.
    """
    normalized = abs(heading_error_rad) / math.pi
    return float(max(0.0, 1.0 - normalized))


def compute_smoothness_reward(action_delta: np.ndarray) -> float:
    """
    Reward for smooth (low-jitter) control inputs.

    Formula: 1.0 - sum(|action_delta|) / 4.0
    Range:   [0.0, 1.0]
    Peak:    1.0 when action_delta is exactly zero (no change since
             last step)
    Zero:    0.0 when both action dimensions swing the full range in
             one step (e.g. acceleration -1 -> +1 AND steer -1 -> +1
             simultaneously: |Δ|=2.0 each, sum=4.0)

    Why penalize the raw action instead of the smoothed/applied one?
        carla_env/action.py's ActionSmoother (alpha=0.6) already limits
        how much jitter reaches the vehicle, but the policy itself can
        still rely on that filter to absorb jitter it didn't need to
        output in the first place. Penalizing the raw action teaches
        the policy to be smooth on its own.
    """
    delta_magnitude = float(np.abs(action_delta).sum())
    return max(0.0, 1.0 - delta_magnitude / 4.0)


# ── Main reward function ───────────────────────────────────────────────────────

def compute_reward(
    obs_data,           # ObservationData from observation.py
    is_terminal: bool,  # True if the episode is ending this step
    cfg: RewardConfig = None,
) -> tuple:
    """
    Compute the full reward for one step.

    This is called by env.py at every step(), after the world ticks
    and the new observation is computed.

    Args:
        obs_data:    ObservationData (lateral_distance_m, heading_error_rad,
                     speed_kmh, steering) — from observation.py
        is_terminal: True if the episode ends after this step
                     (collision, off-road, or timeout)
        cfg:         RewardConfig — defaults to RewardConfig() if None

    Returns:
        reward:      float — scalar reward for the agent
        info:        RewardInfo — breakdown for logging
    """
    if cfg is None:
        cfg = RewardConfig()

    # ── Compute individual components ──────────────────────────────────────────
    r_centering = compute_centering_reward(
        obs_data.lateral_distance_m,
        cfg.max_lateral_m,
    )

    r_speed = compute_speed_reward(
        obs_data.speed_kmh,
        cfg.target_speed_kmh,
        cfg.sigma_speed,
    )

    r_heading = compute_heading_reward(obs_data.heading_error_rad)

    # ── Terminal penalty ────────────────────────────────────────────────────────
    # Only applied on the step where the episode ends badly.
    # Not applied on timeout (episode ran for max steps without crashing).
    r_terminal = cfg.terminal_penalty if is_terminal else 0.0

    # ── Step penalty ────────────────────────────────────────────────────────────
    r_step = cfg.step_penalty

    # ── Weighted sum ────────────────────────────────────────────────────────────
    total = (
        cfg.w_center  * r_centering
        + cfg.w_speed   * r_speed
        + cfg.w_heading * r_heading
        + r_terminal
        + r_step
    )

    info = RewardInfo(
        total=total,
        r_centering=r_centering,
        r_speed=r_speed,
        r_heading=r_heading,
        r_terminal=r_terminal,
        r_step=r_step,
        is_terminal=is_terminal,
    )

    return float(total), info


# ── Termination condition checker ─────────────────────────────────────────────

def check_termination(
    obs_data,
    collision_flag: bool,
    step_count: int,
    max_steps: int = 1000,
    max_lateral_m: float = 3.5,
    max_heading_deg: float = 90.0,
) -> tuple:
    """
    Decide whether the current episode should end.

    Returns (terminated, truncated, reason) following Gymnasium convention:
        terminated: True if the agent failed (collision, off-road)
        truncated:  True if the episode hit the step limit (timeout)
        reason:     string describing why the episode ended (for logging)

    Gymnasium distinguishes terminated vs truncated:
        terminated = agent did something wrong → apply terminal penalty
        truncated  = ran out of time → no terminal penalty (not agent's fault)

    Termination conditions (agent's fault):
        1. Collision detected by CARLA collision sensor
        2. Lateral distance exceeds max_lateral_m (off road)
        3. Heading error exceeds max_heading_deg (pointing wrong way)

    Truncation condition (timeout):
        4. step_count >= max_steps

    Args:
        obs_data:       ObservationData from current step
        collision_flag: True if collision sensor fired this step
        step_count:     current step number in this episode
        max_steps:      episode length limit
        max_lateral_m:  lateral distance threshold for off-road
        max_heading_deg: heading error threshold in degrees
    """
    # ── Collision ──────────────────────────────────────────────────────────────
    if collision_flag:
        return True, False, "collision"

    # ── Off road ───────────────────────────────────────────────────────────────
    if abs(obs_data.lateral_distance_m) >= max_lateral_m:
        return True, False, "off_road"

    # ── Wrong heading ──────────────────────────────────────────────────────────
    max_heading_rad = math.radians(max_heading_deg)
    if abs(obs_data.heading_error_rad) >= max_heading_rad:
        return True, False, "wrong_heading"

    # ── Timeout ────────────────────────────────────────────────────────────────
    if step_count >= max_steps:
        return False, True, "timeout"

    # ── Continue ───────────────────────────────────────────────────────────────
    return False, False, ""
