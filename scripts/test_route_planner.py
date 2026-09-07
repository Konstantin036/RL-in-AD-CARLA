"""
test_route_planner.py
----------------------
RoutePlanner geometry unit tests (no CARLA required)

RoutePlanner.__init__() needs CARLA's GlobalRoutePlanner, but the
geometry helpers (get_closest_waypoint_index, get_target_waypoint,
distance, remaining_distance) only operate on plain (x, y, z) values, so
they're tested here against lightweight stand-ins for carla.Waypoint —
same offline-testing approach as scripts/test_reward.py.

Run from project root:
    python scripts/test_route_planner.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carla_env.route_planner import RoutePlanner, RouteWaypoint


def sep(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


# ── Lightweight stand-ins for carla.Waypoint ───────────────────────────────────

class FakeLocation:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class FakeTransform:
    def __init__(self, x, y, z=0.0):
        self.location = FakeLocation(x, y, z)


class FakeWaypoint:
    def __init__(self, x, y, z=0.0, is_junction=False):
        self.transform = FakeTransform(x, y, z)
        self.is_junction = is_junction


class FakeLaneWaypoint:
    """
    Stand-in for a carla.Waypoint with lane topology, for
    count_same_direction_lanes(). left/right are FakeLaneWaypoint or None.
    """
    def __init__(self, lane_id, lane_type, left=None, right=None):
        import carla
        self.lane_id = lane_id
        self.lane_type = lane_type if lane_type is not None else carla.LaneType.Driving
        self._left = left
        self._right = right

    def get_left_lane(self):
        return self._left

    def get_right_lane(self):
        return self._right


def make_straight_route(length_m=100.0, spacing_m=2.0):
    """A route running along the x-axis from (0,0) to (length_m,0)."""
    n = int(length_m / spacing_m) + 1
    return [
        RouteWaypoint(waypoint=FakeWaypoint(i * spacing_m, 0.0))
        for i in range(n)
    ]


# We only need the geometry helpers, which are plain methods — no need to
# go through __init__ (which requires a live carla.Map for GlobalRoutePlanner).
def make_planner():
    return RoutePlanner.__new__(RoutePlanner)


# ── Test 1: closest index advances normally along a straight route ────────────

def test_closest_index_advances_normally():
    sep("1. Closest waypoint index advances as the vehicle moves forward")
    planner = make_planner()
    route = make_straight_route()

    index = planner.get_closest_waypoint_index(route, FakeLocation(0.0, 0.0), start_index=0)
    ok1 = index == 0

    index = planner.get_closest_waypoint_index(route, FakeLocation(10.0, 0.0), start_index=index)
    ok2 = index == 5   # 10m / 2m spacing

    index = planner.get_closest_waypoint_index(route, FakeLocation(20.1, 0.0), start_index=index)
    ok3 = index == 10

    print(f"  indices: {[0, 5, 10]}  got: matches={ok1 and ok2 and ok3}")
    ok = ok1 and ok2 and ok3
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 2: a closer parallel-street point beyond the search window is ignored ─

def test_ignores_distant_parallel_street():
    sep("2. A physically-closer point far down the route is NOT selected "
        "(prevents jumping to an unrelated parallel street)")
    planner = make_planner()
    route = make_straight_route(length_m=200.0)

    # Insert a point 100 route-samples ahead (200m along the route) that
    # happens to sit right next to the vehicle's actual location — like a
    # parallel street a block over crossing close to the current position.
    route[100] = RouteWaypoint(waypoint=FakeWaypoint(10.1, 0.05))

    # Vehicle is actually at (10, 0) — right at index ~5 on the current
    # street. Without the search-ahead cap, index 100 (distance ~0.11m)
    # would win over index 5 (distance ~0.1m off from the real position);
    # exaggerate the parallel point's proximity to make the failure mode
    # unambiguous if the cap isn't applied.
    route[100] = RouteWaypoint(waypoint=FakeWaypoint(10.0, 0.001))

    index = planner.get_closest_waypoint_index(
        route, FakeLocation(10.0, 0.0), start_index=0, max_search_ahead=30
    )
    ok = index == 5
    print(f"  expected index=5 (local), got index={index}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 3: a route that loops back near itself is still followed in order ─────

def test_respects_order_when_route_loops_near_itself():
    sep("3. A route curving back close to itself is still tracked in order "
        "(not jumped ahead to a later, spatially-closer point)")
    planner = make_planner()

    # Outbound leg: (0,0) to (10,0) at indices 0..10.
    route = make_straight_route(length_m=10.0, spacing_m=1.0)

    # Extend with a "return leg" that loops back and, at index 15, passes
    # extremely close to the outbound leg's index 5 — like a tight hairpin
    # or a road that curves around near itself. Filler points in between
    # just continue away from the outbound leg so they can't interfere.
    for i in range(11, 21):
        route.append(RouteWaypoint(waypoint=FakeWaypoint(10.0 - (i - 10), 5.0)))
    route[15] = RouteWaypoint(waypoint=FakeWaypoint(5.0, 0.03))

    # Vehicle is exactly on top of index 15 (distance 0) but only 0.03m
    # from index 5 — index 15 is the true global-nearest point, but it's
    # badly out of sequence (10 waypoints further down the route).
    # Tracking was already at index 5 last step.
    vehicle_location = FakeLocation(5.0, 0.03)

    index = planner.get_closest_waypoint_index(
        route, vehicle_location, start_index=5, max_search_ahead=30
    )
    ok = index == 5
    print(f"  index 15 is the true nearest point (distance ~0), "
          f"but expected index=5 (in-order): got index={index}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 4: recovers from the vehicle drifting slightly behind start_index ─────

def test_recovers_from_backward_drift():
    sep("4. A vehicle that drifts slightly behind its last tracked index "
        "is still found (e.g. rolling back on an incline while stopped)")
    planner = make_planner()
    route = make_straight_route(length_m=20.0, spacing_m=1.0)

    # Tracking was at index 10 last step; the vehicle has since rolled
    # back 2m (e.g. gravity while stopped at a red light on a slope).
    index = planner.get_closest_waypoint_index(
        route, FakeLocation(8.0, 0.0), start_index=10
    )
    ok = index == 8
    print(f"  vehicle rolled back to x=8.0, start_index=10: got index={index}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 5: a single outlier waypoint doesn't halt the search prematurely ──────

def test_tolerates_single_outlier_waypoint():
    sep("5. A single irregular waypoint doesn't halt the search one step early")
    planner = make_planner()
    route = make_straight_route(length_m=20.0, spacing_m=1.0)

    # Index 3 is a wild outlier (e.g. a glitchy point from GlobalRoutePlanner
    # at a junction) — a purely greedy "stop at the first non-improving step"
    # search would halt at index 2 and never reach the real closest point.
    route[3] = RouteWaypoint(waypoint=FakeWaypoint(100.0, 0.0))

    index = planner.get_closest_waypoint_index(
        route, FakeLocation(9.0, 0.0), start_index=2
    )
    ok = index == 9
    print(f"  outlier at index 3, vehicle actually at x=9.0: got index={index}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 6: get_target_waypoint with lookahead=0 returns the closest point ─────

def test_target_waypoint_lookahead_zero():
    sep("6. get_target_waypoint(lookahead=0) returns the closest waypoint itself")
    planner = make_planner()
    route = make_straight_route()

    target = planner.get_target_waypoint(route, current_index=7, lookahead=0)
    ok = target is route[7]
    print(f"  target is route[7]: {ok}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 7: get_target_waypoint clamps lookahead at the route's end ───────────

def test_target_waypoint_clamps_at_end():
    sep("7. get_target_waypoint clamps lookahead at the end of a short route")
    planner = make_planner()
    route = make_straight_route(length_m=8.0)   # 5 waypoints

    target = planner.get_target_waypoint(route, current_index=0, lookahead=5)
    ok = target is route[-1]
    print(f"  route length={len(route)}, target is route[-1]: {ok}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 7b: count_same_direction_lanes counts only same-direction lanes ──────

def test_count_same_direction_lanes():
    sep("7b. count_same_direction_lanes() stops at the opposite-direction "
        "side and ignores non-Driving lanes")
    import carla
    planner = make_planner()

    # Layout, left to right: opposite lane (-1) — shoulder (non-Driving) —
    # lane1 (self) — lane2 — lane3 (rightmost). Construct all nodes first,
    # then wire left/right explicitly so the linkage is unambiguous.
    opposite = FakeLaneWaypoint(lane_id=-1, lane_type=carla.LaneType.Driving)
    shoulder = FakeLaneWaypoint(lane_id=2, lane_type=carla.LaneType.Shoulder)
    lane1 = FakeLaneWaypoint(lane_id=1, lane_type=carla.LaneType.Driving)
    lane2 = FakeLaneWaypoint(lane_id=2, lane_type=carla.LaneType.Driving)
    lane3 = FakeLaneWaypoint(lane_id=3, lane_type=carla.LaneType.Driving)

    opposite._right, shoulder._left = shoulder, opposite
    shoulder._right, lane1._left    = lane1, shoulder
    lane1._right,    lane2._left    = lane2, lane1
    lane2._right,    lane3._left    = lane3, lane2

    count = planner.count_same_direction_lanes(lane1)
    # Same-direction lanes: lane1 (self) + lane2 + lane3 = 3. The shoulder
    # (non-Driving) and the opposite-direction lane must NOT be counted.
    ok = count == 3
    print(f"  expected 3 same-direction lanes (self + 2 more), got {count}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 8: remaining_distance decreases monotonically along the route ────────

def test_remaining_distance_monotonic():
    sep("8. remaining_distance decreases as current_index advances")
    planner = make_planner()
    route = make_straight_route(length_m=50.0)

    d0 = planner.remaining_distance(route, 0)
    d10 = planner.remaining_distance(route, 10)
    d_end = planner.remaining_distance(route, len(route) - 1)

    ok = d0 > d10 > d_end == 0.0
    print(f"  remaining_distance(0)={d0:.1f}  (10)={d10:.1f}  (end)={d_end:.1f}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 9: precompute_remaining_distances matches remaining_distance() ───────

def test_precompute_remaining_distances_matches():
    sep("9. precompute_remaining_distances() matches remaining_distance() "
        "at every index")
    planner = make_planner()
    route = make_straight_route(length_m=37.0, spacing_m=2.0)

    precomputed = planner.precompute_remaining_distances(route)
    ok = all(
        abs(precomputed[i] - planner.remaining_distance(route, i)) < 1e-9
        for i in range(len(route))
    )
    print(f"  {len(route)} indices checked, all match: {ok}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


def main():
    print("=" * 60)
    print("  RoutePlanner Geometry Unit Tests")
    print("  (No CARLA connection required)")
    print("=" * 60)

    test_closest_index_advances_normally()
    test_ignores_distant_parallel_street()
    test_respects_order_when_route_loops_near_itself()
    test_recovers_from_backward_drift()
    test_tolerates_single_outlier_waypoint()
    test_target_waypoint_lookahead_zero()
    test_target_waypoint_clamps_at_end()
    test_count_same_direction_lanes()
    test_remaining_distance_monotonic()
    test_precompute_remaining_distances_matches()

    print(f"\n{'='*60}")
    print("  All route planner tests passed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
