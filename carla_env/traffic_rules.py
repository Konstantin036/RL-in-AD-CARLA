"""
traffic_rules.py
-----------------
Traffic-light compliance — ground-truth affordance + violation detection.

Purpose:
    Give the agent a signal for "must I stop right now?" and detect when
    it drives through a red/yellow light instead of stopping.

Why a separate file:
    Same reasoning as route_planner.py — this is one clearly-scoped
    concern (traffic-light state) kept out of env.py, observation.py and
    reward.py so each of those stays about what it already does.

Ground truth today, swappable later:
    get_traffic_light_affordance() currently reads CARLA's own
    ground-truth API (vehicle.is_at_traffic_light() /
    get_traffic_light_state()). When a real light/sign detector is added
    later, it only needs to produce the same TrafficLightAffordance shape
    and be called from env.py's single call site — nothing in
    observation.py, reward.py, or RedLightViolationDetector needs to
    change, since they only depend on the affordance fields, not on how
    it was obtained.

    Stop signs are intentionally out of scope for this first pass — they
    would be a sibling get_stop_sign_affordance() / StopSignViolationDetector
    pair added to this same file later, following the same shape.
"""

from dataclasses import dataclass


# ── Affordance ──────────────────────────────────────────────────────────────

@dataclass
class TrafficLightAffordance:
    """
    Ground-truth traffic-light state as it applies to the ego vehicle
    right now. Mirrors carla_env.observation.ObservationData's pattern:
    a small bundle of raw values, not yet normalized for the network.
    """
    is_at_light: bool   # True while inside the controlling light's trigger volume
    must_stop: bool     # True if is_at_light and the light is red or yellow
    state: str          # human-readable state, for logging/debugging


def get_traffic_light_affordance(vehicle) -> TrafficLightAffordance:
    """
    Read the traffic light currently affecting the vehicle, if any.

    carla.Vehicle.get_traffic_light_state() returns Green when the
    vehicle isn't affected by any light, so must_stop is gated on
    is_at_traffic_light() to avoid ever stopping for a light that isn't
    actually controlling this lane.
    """
    import carla

    is_at_light = vehicle.is_at_traffic_light()
    state = vehicle.get_traffic_light_state()

    must_stop = is_at_light and state in (
        carla.TrafficLightState.Red,
        carla.TrafficLightState.Yellow,
    )

    return TrafficLightAffordance(
        is_at_light=is_at_light,
        must_stop=must_stop,
        state=str(state),
    )


# ── Violation detector ───────────────────────────────────────────────────────

class RedLightViolationDetector:
    """
    Detects driving through a red/yellow light instead of stopping.

    Owned by CarlaLaneKeepingEnv, same lifecycle as StallDetector in
    reward.py: instantiated once, reset() at the start of every episode,
    update() called every step.

    Why "exiting the zone while still moving" and not "moving while
    must_stop is True":
        The vehicle can't brake to a stop instantly the moment a light
        turns red — flagging every step where must_stop is True and
        speed > 0 would punish normal braking distance, not just actual
        violations. Waiting until the vehicle leaves the trigger volume
        tells us what it actually did: if it was still moving above a
        near-stationary threshold at that point, it drove through
        without stopping. If it slowed below the threshold at any point
        while must_stop was True (i.e. it waited for green before
        must_stop went False), no violation is flagged.
    """

    def __init__(self, cfg):
        self.enabled          = cfg.red_light_enabled
        self.stop_speed_kmh   = cfg.red_light_stop_speed_kmh
        self._was_at_light    = False
        self._was_must_stop   = False

    def reset(self):
        self._was_at_light  = False
        self._was_must_stop = False

    def update(self, affordance: TrafficLightAffordance, speed_kmh: float) -> bool:
        """
        Call every step with the current affordance and speed.
        Returns True on the step a violation is detected.
        """
        if not self.enabled:
            return False

        violated = (
            self._was_at_light
            and not affordance.is_at_light
            and self._was_must_stop
            and speed_kmh > self.stop_speed_kmh
        )

        self._was_at_light  = affordance.is_at_light
        self._was_must_stop = affordance.must_stop

        return violated
