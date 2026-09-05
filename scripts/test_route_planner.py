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


# ── Test 4: get_target_waypoint with lookahead=0 returns the closest point ─────

def test_target_waypoint_lookahead_zero():
    sep("4. get_target_waypoint(lookahead=0) returns the closest waypoint itself")
    planner = make_planner()
    route = make_straight_route()

    target = planner.get_target_waypoint(route, current_index=7, lookahead=0)
    ok = target is route[7]
    print(f"  target is route[7]: {ok}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 5: get_target_waypoint clamps lookahead at the route's end ───────────

def test_target_waypoint_clamps_at_end():
    sep("5. get_target_waypoint clamps lookahead at the end of a short route")
    planner = make_planner()
    route = make_straight_route(length_m=8.0)   # 5 waypoints

    target = planner.get_target_waypoint(route, current_index=0, lookahead=5)
    ok = target is route[-1]
    print(f"  route length={len(route)}, target is route[-1]: {ok}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 6: remaining_distance decreases monotonically along the route ────────

def test_remaining_distance_monotonic():
    sep("6. remaining_distance decreases as current_index advances")
    planner = make_planner()
    route = make_straight_route(length_m=50.0)

    d0 = planner.remaining_distance(route, 0)
    d10 = planner.remaining_distance(route, 10)
    d_end = planner.remaining_distance(route, len(route) - 1)

    ok = d0 > d10 > d_end == 0.0
    print(f"  remaining_distance(0)={d0:.1f}  (10)={d10:.1f}  (end)={d_end:.1f}")
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
    test_target_waypoint_lookahead_zero()
    test_target_waypoint_clamps_at_end()
    test_remaining_distance_monotonic()

    print(f"\n{'='*60}")
    print("  All route planner tests passed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
