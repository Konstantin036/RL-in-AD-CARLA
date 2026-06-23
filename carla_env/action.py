"""
action.py
---------
Phase 4 — Action Space

Purpose:
    Define what the RL agent can do, and translate agent outputs into
    CARLA VehicleControl commands.

    The agent outputs a 2D continuous action:
        action[0] = acceleration  ∈ [-1.0, +1.0]
        action[1] = steer         ∈ [-1.0, +1.0]

    We map this to CARLA's 3-value control:
        acceleration > 0  →  throttle = acceleration,  brake = 0
        acceleration < 0  →  throttle = 0,             brake = -acceleration
        steer             →  steer directly (same scale)

    We also apply action smoothing to reduce jerky oscillations.

Why 2D instead of 3D (throttle, brake, steer)?
    A 3D action allows the agent to press throttle and brake at the same time,
    which wastes energy and creates unstable physics. The 1D acceleration
    axis encodes the intent (speed up / slow down) cleanly.

Why action smoothing?
    PPO with a Gaussian policy can output large action changes between steps.
    Without smoothing, the steering jumps around and the vehicle oscillates.
    Smoothing acts like a low-pass filter on the control signal.
    It is a small addition that dramatically improves training stability.

Action smoothing formula:
    smoothed = alpha * new_action + (1 - alpha) * previous_action
    alpha = 0.6 means 60% new, 40% old — a gentle filter.
"""

import numpy as np
import gymnasium as gym


# ── Action space definition ────────────────────────────────────────────────────

def get_action_space() -> gym.spaces.Box:
    """
    Return the Gymnasium action space.

    Shape:  (2,)
    Index   Name          Range        Meaning
    -----   -----------   ----------   -------------------------------
      0     acceleration  [-1, +1]     negative=brake, positive=throttle
      1     steer         [-1, +1]     negative=left,  positive=right

    Called once during environment initialization in env.py __init__.
    """
    return gym.spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0,  1.0], dtype=np.float32),
        dtype=np.float32,
    )


# ── Action smoother ────────────────────────────────────────────────────────────

class ActionSmoother:
    """
    Applies exponential moving average smoothing to actions.

    This is a stateful object — it remembers the last action and blends
    new actions toward it. Reset it at the start of each episode.

    Usage:
        smoother = ActionSmoother(alpha=0.6)
        smoother.reset()
        for step in episode:
            smoothed_action = smoother.smooth(raw_action)
            apply_to_vehicle(smoothed_action)

    Parameters
    ----------
    alpha : float in (0, 1]
        Weight given to the new action.
        alpha = 1.0  → no smoothing (raw action passed through)
        alpha = 0.6  → 60% new, 40% previous (default, good balance)
        alpha = 0.3  → heavy smoothing (very slow steering response)
    """

    def __init__(self, alpha: float = 0.6):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._last_action = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        """Call at the start of each episode to clear the smoothing state."""
        self._last_action = np.zeros(2, dtype=np.float32)

    def smooth(self, action: np.ndarray) -> np.ndarray:
        """
        Blend the new action with the previous smoothed action.

        Args:
            action: np.ndarray of shape (2,), values in [-1, 1]

        Returns:
            smoothed: np.ndarray of shape (2,), values in [-1, 1]
        """
        smoothed = self.alpha * action + (1.0 - self.alpha) * self._last_action
        self._last_action = smoothed.copy()
        return smoothed


# ── Action → CARLA control translator ─────────────────────────────────────────

def action_to_control(action: np.ndarray):
    """
    Convert a 2D agent action to a CARLA VehicleControl object.

    This function is the bridge between the RL world (numpy arrays) and
    the CARLA world (VehicleControl objects).

    Args:
        action: np.ndarray of shape (2,)
                action[0] = acceleration ∈ [-1, +1]
                action[1] = steer        ∈ [-1, +1]

    Returns:
        carla.VehicleControl with throttle, brake, steer set appropriately.

    Mapping:
        acceleration ∈ [0,  1] → throttle = acceleration,  brake = 0.0
        acceleration ∈ [-1, 0) → throttle = 0.0,           brake = -acceleration
        steer is passed directly to CARLA (same [-1, 1] scale)
    """
    import carla

    acceleration = float(action[0])
    steer        = float(action[1])

    # Split acceleration into throttle and brake
    if acceleration >= 0.0:
        throttle = acceleration
        brake    = 0.0
    else:
        throttle = 0.0
        brake    = -acceleration   # acceleration is negative, brake must be positive

    # Clip to valid CARLA ranges (should already be in range, but safety first)
    throttle = float(np.clip(throttle, 0.0, 1.0))
    brake    = float(np.clip(brake,    0.0, 1.0))
    steer    = float(np.clip(steer,   -1.0, 1.0))

    control = carla.VehicleControl(
        throttle=throttle,
        steer=steer,
        brake=brake,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
    )

    return control


# ── Convenience: combine smoothing + translation ───────────────────────────────

class ActionProcessor:
    """
    Combines smoothing and translation in one object.
    This is what env.py will instantiate and use.

    Usage:
        processor = ActionProcessor(alpha=0.6)
        processor.reset()                    # at episode start
        control = processor.process(action)  # at each step
        vehicle.apply_control(control)
    """

    def __init__(self, alpha: float = 0.6):
        self.smoother = ActionSmoother(alpha=alpha)

    def reset(self) -> None:
        """Reset smoothing state at the start of each episode."""
        self.smoother.reset()

    def process(self, action: np.ndarray):
        """
        Smooth the action and convert to CARLA VehicleControl.

        Args:
            action: raw agent action, shape (2,)

        Returns:
            carla.VehicleControl
        """
        smoothed = self.smoother.smooth(action)
        control  = action_to_control(smoothed)
        return control

    def process_raw(self, action: np.ndarray) -> np.ndarray:
        """
        Return the smoothed action array without converting to CARLA control.
        Used for logging.
        """
        return self.smoother.smooth(action)
