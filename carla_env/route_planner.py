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

    # CARLA GlobalRoutePlanner RoadOption values.
    # Keeping these locally avoids coupling the rest of the project
    # to a particular enum implementation.
    LANEFOLLOW = 4
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    CHANGE_LANE_LEFT = 5
    CHANGE_LANE_RIGHT = 6
    VOID = 0

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

    def plan_route_from_waypoints(
        self,
        start_waypoint: carla.Waypoint,
        destination_waypoint: carla.Waypoint,
    ) -> List[RouteWaypoint]:
        """Convenience wrapper using CARLA waypoints."""
        return self.plan_route(
            start_waypoint.transform.location,
            destination_waypoint.transform.location,
        )

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
    ) -> int:
        """
        Find the route waypoint closest to the vehicle.

        start_index can be used to avoid searching the entire route
        every simulation step.

        max_search_ahead caps how many waypoints past start_index are
        considered (default 30, i.e. 60m at the 2.0m sampling_resolution
        this project uses). Without this cap, searching all the way to
        the end of the route lets a spatially-close-but-topologically-
        distant point win — on a dense grid (e.g. Town10), a parallel
        street a block over can be physically closer than the correct
        next waypoint on the current street, snapping route_index far
        ahead to an unrelated segment pointing a different direction.
        That produces a sudden fake heading/lateral error and a sharp,
        spurious steering correction — confirmed via live testing (car
        visibly yanking the wheel on a straight stretch of road).
        """
        if not route:
            raise ValueError("Route is empty.")

        start_index = max(0, min(start_index, len(route) - 1))
        search_end = min(start_index + max_search_ahead, len(route))

        distances = [
            self.distance(
                location,
                route[i].waypoint.transform.location,
            )
            for i in range(start_index, search_end)
        ]

        return start_index + distances.index(min(distances))

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
    def is_junction(route_waypoint: RouteWaypoint) -> bool:
        """Return True if this route waypoint belongs to a junction."""
        return route_waypoint.waypoint.is_junction

    @classmethod
    def road_option_name(cls, road_option: int) -> str:
        """Convert a RoadOption integer into a readable name."""
        names = {
            cls.VOID: "VOID",
            cls.LEFT: "LEFT",
            cls.RIGHT: "RIGHT",
            cls.STRAIGHT: "STRAIGHT",
            cls.LANEFOLLOW: "LANEFOLLOW",
            cls.CHANGE_LANE_LEFT: "CHANGE_LANE_LEFT",
            cls.CHANGE_LANE_RIGHT: "CHANGE_LANE_RIGHT",
        }

        return names.get(road_option, "UNKNOWN")

    @staticmethod
    def remaining_distance(
        route: List[RouteWaypoint],
        current_index: int,
    ) -> float:
        """Calculate approximate remaining route distance."""
        if not route or current_index >= len(route) - 1:
            return 0.0

        distance = 0.0

        for i in range(current_index, len(route) - 1):
            a = route[i].waypoint.transform.location
            b = route[i + 1].waypoint.transform.location

            dx = a.x - b.x
            dy = a.y - b.y
            dz = a.z - b.z

            distance += (dx * dx + dy * dy + dz * dz) ** 0.5

        return distance

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
            loc_a = route[i].waypoint.transform.location + carla.Location(z=0.3)
            loc_b = route[i + 1].waypoint.transform.location + carla.Location(z=0.3)
            world.debug.draw_line(
                loc_a, loc_b,
                thickness=0.15,
                color=carla.Color(0, 255, 0),
                life_time=life_time,
            )

        destination = route[-1].waypoint.transform.location + carla.Location(z=0.3)
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

        location = route_waypoint.waypoint.transform.location + carla.Location(z=0.5)
        world.debug.draw_point(
            location,
            size=0.15,
            color=carla.Color(255, 255, 0),
            life_time=life_time,
        )
