"""
route_planner.py
----------------
Route planner for CARLA lane-keeping.

Responsibilities:
    - Generate a route between two CARLA locations.
    - Follow CARLA's road topology.
    - Provide waypoints and high-level road options.
    - Stay independent from the RL agent/reward function.

The environment can later use this module to:
    1. Generate a route at reset().
    2. Find the next target waypoint.
    3. Calculate route progress.
    4. Detect upcoming turns/junctions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

# CARLA is only referenced here in type hints, which `from __future__ import
# annotations` (above) defers to strings — so it's never evaluated at
# runtime. Importing it only under TYPE_CHECKING keeps this module (and
# anything that imports it, like carla_env/env.py) importable without CARLA
# installed, per this project's "CARLA imports never at module top level"
# rule (see CLAUDE.md) — it's what lets scripts/test_*.py run offline.
if TYPE_CHECKING:
    import carla


# Vertical offsets for drawing route waypoints, which sit exactly at
# road-surface height — lifted purely so the debug-draw shapes render
# above the road mesh instead of clipping into it. Deliberately smaller
# than env.py's SPAWN_HEIGHT_OFFSET_M: that one needs enough clearance to
# avoid a ground-clipping *physics* spawn collision, a different and
# stricter requirement than "visible above the road surface."
ROUTE_LINE_HEIGHT_M     = 0.3
TARGET_MARKER_HEIGHT_M  = 0.5


@dataclass
class RouteWaypoint:
    """A waypoint belonging to the planned route."""

    waypoint: carla.Waypoint
    road_option: int = 4  # RoadOption.LANEFOLLOW


class RoutePlanner:
    """
    Simple CARLA route planner.

    Uses CARLA's road topology to build a route from a start
    location to a destination location.

    This is intentionally kept simple for the first integration step.
    More advanced routing/cost functions can be added later.
    """

    def __init__(
        self,
        world_map: carla.Map,
        sampling_resolution: float = 2.0,
    ):
        """
        Args:
            world_map:
                CARLA map returned by world.get_map().

            sampling_resolution:
                Distance in meters between route waypoints.
        """
        self.world_map = world_map
        self.sampling_resolution = sampling_resolution

        self._planner = None
        self._build_planner()

    def _build_planner(self):
        """Build CARLA's GlobalRoutePlanner."""
        try:
            from agents.navigation.global_route_planner import (
                GlobalRoutePlanner,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import CARLA GlobalRoutePlanner. "
                "Make sure the CARLA PythonAPI agents package is installed "
                "and available in PYTHONPATH."
            ) from exc

        self._planner = GlobalRoutePlanner(
            self.world_map,
            self.sampling_resolution,
        )

    def plan_route(
        self,
        start: carla.Location,
        destination: carla.Location,
    ) -> List[RouteWaypoint]:
        """
        Calculate a route from start to destination.

        Returns:
            List of RouteWaypoint objects ordered from start
            to destination.
        """
        route = self._planner.trace_route(start, destination)

        return [
            RouteWaypoint(
                waypoint=waypoint,
                road_option=int(road_option),
            )
            for waypoint, road_option in route
        ]

    @staticmethod
    def distance(
        location_a: carla.Location,
        location_b: carla.Location,
    ) -> float:
        """Euclidean distance between two CARLA locations."""
        dx = location_a.x - location_b.x
        dy = location_a.y - location_b.y
        dz = location_a.z - location_b.z

        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def get_closest_waypoint_index(
        self,
        route: List[RouteWaypoint],
        location: carla.Location,
        start_index: int = 0,
        max_search_ahead: int = 30,
        max_search_behind: int = 5,
        patience: int = 3,
    ) -> int:
        """
        Find the route waypoint closest to the vehicle, respecting the
        route's own order.

        Walks forward from start_index, tracking the running closest
        point, and stops once distance has failed to improve for
        `patience` consecutive waypoints (i.e. a local minimum). This is
        deliberately NOT "the closest point within the search window" —
        it never looks far past the point where it stopped improving, so
        it can't skip ahead to a different, out-of-sequence point on the
        route no matter how spatially close that point is. That matters
        whenever the route curves back near itself (a tight turn, a
        loop, two lanes running close together): the waypoints must be
        visited in the order the route actually defines, based on the
        planned start and destination — not in whichever order happens
        to be nearest in raw distance. A pure global-minimum search over
        a search window (an earlier version of this method) does not
        have this guarantee: a later, out-of-sequence point that happens
        to be closer still wins if it's inside the window.

        Also checks a small bounded window behind start_index. A purely
        forward-only search (an earlier version of this method) can
        never recover once the vehicle ends up behind its last tracked
        index — e.g. rolling back a little on an incline while stopped
        at a red light, or a near-stall wobble — route_index would
        freeze there permanently even as the vehicle resumes driving
        forward normally afterward (confirmed via code review).

        start_index can be used to avoid searching the entire route
        every simulation step.

        max_search_ahead bounds how far the forward walk is allowed to
        go before giving up (default 30, i.e. 60m at the 2.0m
        sampling_resolution this project uses) — a defensive limit for
        the case where distance never stops decreasing within a
        reasonable range (e.g. start_index is badly out of sync), not
        the primary mechanism that keeps tracking correct.

        max_search_behind bounds the backward check (default 5, i.e.
        10m) — small on purpose, just enough to recover from minor
        drift, not enough to reintroduce order violations from that
        direction.

        patience (default 3) tolerates a few consecutive non-improving
        waypoints before concluding the local minimum has been passed,
        so a single irregular/outlier waypoint from GlobalRoutePlanner
        (e.g. at a junction or lane-merge boundary) can't prematurely
        halt the search one step too early (confirmed via code review) —
        still bounded, so it doesn't reopen the out-of-sequence-jump
        problem the forward-only version was built to close.
        """
        if not route:
            raise ValueError("Route is empty.")

        start_index = max(0, min(start_index, len(route) - 1))

        def dist_at(i: int) -> float:
            return self.distance(location, route[i].waypoint.transform.location)

        best_index = start_index
        best_distance = dist_at(start_index)

        # Small bounded backward check — see max_search_behind's docstring.
        behind_bound = max(0, start_index - max_search_behind)
        for i in range(start_index - 1, behind_bound - 1, -1):
            d = dist_at(i)
            if d < best_distance:
                best_distance = d
                best_index = i
            else:
                break

        # Forward walk, tolerating up to `patience` non-improving steps
        # in a row before concluding the local minimum has been passed.
        search_end = min(start_index + max_search_ahead, len(route))
        stale_steps = 0
        for i in range(start_index + 1, search_end):
            d = dist_at(i)
            if d < best_distance:
                best_distance = d
                best_index = i
                stale_steps = 0
            else:
                stale_steps += 1
                if stale_steps >= patience:
                    break

        return best_index

    def get_target_waypoint(
        self,
        route: List[RouteWaypoint],
        current_index: int,
        lookahead: int = 5,
    ) -> Optional[RouteWaypoint]:
        """
        Return a waypoint ahead of the vehicle.

        Args:
            route: planned route.
            current_index: current closest waypoint index.
            lookahead: number of route waypoints to look ahead.
        """
        if not route:
            return None

        target_index = min(
            current_index + lookahead,
            len(route) - 1,
        )

        return route[target_index]

    @staticmethod
    def count_same_direction_lanes(waypoint, max_lanes_each_side: int = 4) -> int:
        """
        Count Driving lanes going the same direction as `waypoint`'s lane,
        including itself.

        Why: a car drifting into an adjacent same-direction lane on a
        multi-lane one-way road hasn't left the road — it's just in a
        different lane of the same one — but the off-road termination
        threshold was sized for a single lane, treating that drift as a
        road-departure failure. Confirmed via live testing: this is
        disproportionately likely right at intersections, where dedicated
        turn lanes commonly add extra same-direction lanes. env.py widens
        the termination threshold by this count (not the centering
        reward, which should still gently encourage staying in the
        route's own lane).

        Uses CARLA's lane_id sign convention: lanes on the same side of a
        road's centerline (same direction) share the sign of lane_id;
        get_left_lane()/get_right_lane() cross to the opposite-direction
        side once you pass the centerline, which the sign check below
        detects and stops at. max_lanes_each_side bounds the walk
        defensively (CARLA never has this many real lanes) rather than
        relying on it always terminating naturally.
        """
        import carla

        def is_same_direction(wp) -> bool:
            return (
                wp is not None
                and wp.lane_type == carla.LaneType.Driving
                and (wp.lane_id > 0) == (waypoint.lane_id > 0)
            )

        count = 1

        wp = waypoint.get_left_lane()
        steps = 0
        while is_same_direction(wp) and steps < max_lanes_each_side:
            count += 1
            wp = wp.get_left_lane()
            steps += 1

        wp = waypoint.get_right_lane()
        steps = 0
        while is_same_direction(wp) and steps < max_lanes_each_side:
            count += 1
            wp = wp.get_right_lane()
            steps += 1

        return count

    @staticmethod
    def remaining_distance(
        route: List[RouteWaypoint],
        current_index: int,
    ) -> float:
        """
        Calculate approximate remaining route distance from scratch.

        O(route length - current_index) — fine for a one-off query, but
        env.py calls this every single step purely to populate a logging
        field, which turns into real repeated cost on Town10's long
        random routes over up to 1000 steps/episode (confirmed via code
        review). env.py uses precompute_remaining_distances() instead for
        that O(1)-per-step case; this method stays for one-off queries
        and is what precompute_remaining_distances() is checked against
        in scripts/test_route_planner.py.
        """
        if not route or current_index >= len(route) - 1:
            return 0.0

        distance = 0.0

        for i in range(current_index, len(route) - 1):
            distance += RoutePlanner.distance(
                route[i].waypoint.transform.location,
                route[i + 1].waypoint.transform.location,
            )

        return distance

    @staticmethod
    def precompute_remaining_distances(route: List[RouteWaypoint]) -> List[float]:
        """
        Precompute remaining_distance() for every index in the route at
        once, in a single O(route length) backward pass — call this once
        per episode (right after plan_route()) so env.py's per-step
        lookup is O(1) instead of recomputing the whole remaining route
        from scratch on every single step (see remaining_distance()'s
        docstring for why that matters).

        Returns a list `remaining` where `remaining[i]` equals
        `remaining_distance(route, i)`.
        """
        n = len(route)
        if n == 0:
            return []

        remaining = [0.0] * n
        for i in range(n - 2, -1, -1):
            remaining[i] = remaining[i + 1] + RoutePlanner.distance(
                route[i].waypoint.transform.location,
                route[i + 1].waypoint.transform.location,
            )
        return remaining

    @staticmethod
    def draw_route(world, route: List[RouteWaypoint], life_time: float = 0.0) -> None:
        """
        Draw the planned route as a green line strip in the CARLA world,
        plus a red marker at the destination — purely a visualization
        aid, no effect on training. Call once per episode, right after
        plan_route(); life_time (seconds) should cover the whole episode
        so it stays visible without needing to be redrawn every tick.

        Why this belongs on RoutePlanner and not env.py: it only needs
        route data and CARLA's debug-draw API, the same dependencies
        plan_route() already has — keeping it here means env.py just
        makes one call instead of knowing anything about how routes are
        rendered.
        """
        import carla

        if not route:
            return

        for i in range(len(route) - 1):
            loc_a = route[i].waypoint.transform.location + carla.Location(z=ROUTE_LINE_HEIGHT_M)
            loc_b = route[i + 1].waypoint.transform.location + carla.Location(z=ROUTE_LINE_HEIGHT_M)
            world.debug.draw_line(
                loc_a, loc_b,
                thickness=0.15,
                color=carla.Color(0, 255, 0),
                life_time=life_time,
            )

        destination = route[-1].waypoint.transform.location + carla.Location(z=ROUTE_LINE_HEIGHT_M)
        world.debug.draw_point(
            destination,
            size=0.2,
            color=carla.Color(255, 0, 0),
            life_time=life_time,
        )

    @staticmethod
    def draw_target_waypoint(world, route_waypoint: Optional[RouteWaypoint], life_time: float) -> None:
        """
        Draw a marker at the waypoint the environment is currently
        tracking (env.py's target_waypoint) — lets you see what the car
        is actually steering toward, updated every step. life_time
        should be a bit more than one tick's duration so it refreshes
        cleanly without leaving a trail.
        """
        import carla

        if route_waypoint is None:
            return

        location = route_waypoint.waypoint.transform.location + carla.Location(z=TARGET_MARKER_HEIGHT_M)
        world.debug.draw_point(
            location,
            size=0.15,
            color=carla.Color(255, 255, 0),
            life_time=life_time,
        )
