"""
RRT* (Rapidly-exploring Random Tree Star) Path Planning Algorithm

RRT* is an asymptotically optimal extension of RRT that continuously rewires
the tree to find shorter paths. It operates in continuous 2D space and avoids
circular NFZ obstacles.

Key characteristics:
- Very fast at finding initial solutions
- Paths are often zigzag due to random sampling nature
- Asymptotically optimal (path improves with more iterations)
"""

import time
import math
import random
from typing import List, Tuple, Optional, Dict

try:
    from .base import PathPlanner, PathResult
    from .utils import euclidean_distance, line_obstacles_intersection, is_point_in_obstacles
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from algorithms.base import PathPlanner, PathResult
    from algorithms.utils import euclidean_distance, line_obstacles_intersection, is_point_in_obstacles


class RRTNode:
    """Node in the RRT* tree"""
    __slots__ = ('x', 'y', 'parent', 'cost')

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent: Optional['RRTNode'] = None
        self.cost: float = 0.0

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class RRTStar(PathPlanner):
    """
    RRT* path planner for continuous 2D space with circular obstacles.

    Parameters
    ----------
    max_iterations : int
        Maximum tree expansion iterations.
    step_size : float
        Maximum distance between two consecutive nodes (steer step).
    goal_sample_rate : float
        Probability [0,1] of sampling the goal directly.
    search_radius : float
        Rewiring neighbourhood radius.  If None, set automatically.
    goal_tolerance : float
        Distance within which a node is considered to have reached the goal.
    """

    def __init__(
        self,
        obstacles: List[Tuple[float, float, float]],
        width: int,
        height: int,
        max_iterations: int = 3000,
        step_size: float = 3.0,
        goal_sample_rate: float = 0.10,
        search_radius: Optional[float] = None,
        goal_tolerance: float = 1.5,
    ):
        super().__init__(obstacles, width, height)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.goal_tolerance = goal_tolerance

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def plan(self, start: Tuple[float, float],
             goal: Tuple[float, float]) -> PathResult:
        start_time = time.time()

        if not self.is_valid_point(start):
            return PathResult([], False, time.time() - start_time,
                              0.0, float('inf'), 0,
                              "Start point is in obstacle or out of bounds")

        if not self.is_valid_point(goal):
            return PathResult([], False, time.time() - start_time,
                              0.0, float('inf'), 0,
                              "Goal point is in obstacle or out of bounds")

        # Adaptive search radius: γ * sqrt(log(n)/n), floor at step_size
        gamma = 3.0 * math.sqrt(self.width * self.height)

        root = RRTNode(start[0], start[1])
        nodes: List[RRTNode] = [root]

        best_goal_node: Optional[RRTNode] = None

        for iteration in range(self.max_iterations):
            # ── 1. Sample random point ───────────────────────────────────
            rand_pt = self._sample(goal)

            # ── 2. Nearest node ──────────────────────────────────────────
            nearest = self._nearest(nodes, rand_pt)

            # ── 3. Steer toward sample ───────────────────────────────────
            new_pt = self._steer(nearest.as_tuple(), rand_pt)

            if not self.is_valid_point(new_pt):
                continue
            if not self.is_valid_line(nearest.as_tuple(), new_pt):
                continue

            # ── 4. Create new node ───────────────────────────────────────
            new_node = RRTNode(new_pt[0], new_pt[1])
            new_node.cost = nearest.cost + euclidean_distance(nearest.as_tuple(), new_pt)
            new_node.parent = nearest

            # ── 5. Find nearby nodes for rewiring ────────────────────────
            n = len(nodes) + 1
            r = self.search_radius if self.search_radius is not None \
                else min(gamma * math.sqrt(math.log(n) / n), self.step_size * 4)

            nearby = self._near(nodes, new_pt, r)

            # ── 6. Choose best parent among nearby nodes ─────────────────
            for candidate in nearby:
                if not self.is_valid_line(candidate.as_tuple(), new_pt):
                    continue
                c = candidate.cost + euclidean_distance(candidate.as_tuple(), new_pt)
                if c < new_node.cost:
                    new_node.cost = c
                    new_node.parent = candidate

            nodes.append(new_node)

            # ── 7. Rewire nearby nodes through new_node ───────────────────
            for candidate in nearby:
                if candidate is new_node.parent:
                    continue
                if not self.is_valid_line(new_pt, candidate.as_tuple()):
                    continue
                c = new_node.cost + euclidean_distance(new_pt, candidate.as_tuple())
                if c < candidate.cost:
                    candidate.parent = new_node
                    candidate.cost = c

            # ── 8. Check goal ─────────────────────────────────────────────
            dist_to_goal = euclidean_distance(new_pt, goal)
            if dist_to_goal <= self.goal_tolerance:
                if self.is_valid_line(new_pt, goal):
                    total_cost = new_node.cost + dist_to_goal
                    if best_goal_node is None or total_cost < best_goal_node.cost:
                        goal_node = RRTNode(goal[0], goal[1])
                        goal_node.parent = new_node
                        goal_node.cost = total_cost
                        best_goal_node = goal_node

        if best_goal_node is None:
            return PathResult([], False, time.time() - start_time,
                              0.0, float('inf'), len(nodes),
                              "RRT* failed to reach goal within max iterations")

        path = self._extract_path(best_goal_node)
        path_length = sum(euclidean_distance(path[i], path[i + 1])
                          for i in range(len(path) - 1))

        return PathResult(
            path=path,
            success=True,
            computation_time=time.time() - start_time,
            path_length=path_length,
            cost=best_goal_node.cost,
            num_nodes_explored=len(nodes),
            message="RRT* path found",
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _sample(self, goal: Tuple[float, float]) -> Tuple[float, float]:
        """Sample a random point, biased toward the goal."""
        if random.random() < self.goal_sample_rate:
            return goal
        x = random.uniform(0.0, float(self.width - 1))
        y = random.uniform(0.0, float(self.height - 1))
        return (x, y)

    def _nearest(self, nodes: List[RRTNode],
                 pt: Tuple[float, float]) -> RRTNode:
        """Return the node closest to pt."""
        return min(nodes, key=lambda n: euclidean_distance(n.as_tuple(), pt))

    def _steer(self, from_pt: Tuple[float, float],
               to_pt: Tuple[float, float]) -> Tuple[float, float]:
        """Move from from_pt toward to_pt by at most step_size."""
        d = euclidean_distance(from_pt, to_pt)
        if d <= self.step_size:
            return to_pt
        ratio = self.step_size / d
        x = from_pt[0] + ratio * (to_pt[0] - from_pt[0])
        y = from_pt[1] + ratio * (to_pt[1] - from_pt[1])
        # Clamp to map bounds
        x = max(0.0, min(float(self.width - 1), x))
        y = max(0.0, min(float(self.height - 1), y))
        return (x, y)

    def _near(self, nodes: List[RRTNode],
              pt: Tuple[float, float],
              radius: float) -> List[RRTNode]:
        """Return all nodes within radius of pt."""
        return [n for n in nodes
                if euclidean_distance(n.as_tuple(), pt) <= radius]

    def _extract_path(self, goal_node: RRTNode) -> List[Tuple[float, float]]:
        """Walk parent pointers from goal back to root."""
        path = []
        node: Optional[RRTNode] = goal_node
        while node is not None:
            path.append(node.as_tuple())
            node = node.parent
        path.reverse()
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys
    import os

    parser = argparse.ArgumentParser(description="Run RRT* on a UAV map")
    parser.add_argument(
        "--map",
        default=os.path.join("maps", "destinations", "m1_50x50.json"),
        help="Path to the map JSON file (default: maps/destinations/m1_50x50.json)",
    )
    parser.add_argument("--iterations", type=int,   default=3000, help="Max RRT* iterations (default: 3000)")
    parser.add_argument("--step",       type=float, default=3.0,  help="Steer step size (default: 3.0)")
    parser.add_argument("--goal-rate",  type=float, default=0.10, help="Goal sample rate 0-1 (default: 0.10)")
    parser.add_argument("--tolerance",  type=float, default=1.5,  help="Goal tolerance distance (default: 1.5)")
    args = parser.parse_args()

    if not os.path.exists(args.map):
        print(f"Error: map file '{args.map}' not found.")
        print("Run map_generator.py first to generate maps.")
        sys.exit(1)

    with open(args.map) as f:
        data = json.load(f)

    obstacles = [tuple(obs) for obs in data["obstacles"]]
    width     = data["width"]
    height    = data["height"]
    start     = tuple(data["start"])
    goal      = tuple(data["goal"])

    print("=" * 60)
    print("  RRT* (Rapidly-exploring Random Tree Star)")
    print("=" * 60)
    print(f"  Map        : {args.map}")
    print(f"  Size       : {width} × {height}")
    print(f"  NFZ count  : {len(obstacles)} circles")
    print(f"  Start      : {start}")
    print(f"  Goal       : {goal}")
    print(f"  Iterations : {args.iterations}")
    print(f"  Step size  : {args.step}")
    print(f"  Goal rate  : {args.goal_rate}")
    print()

    planner = RRTStar(
        obstacles, width, height,
        max_iterations=args.iterations,
        step_size=args.step,
        goal_sample_rate=args.goal_rate,
        goal_tolerance=args.tolerance,
    )

    result = planner.plan(start, goal)

    print(f"  Success      : {'Yes' if result.success else 'No'}")
    if result.success:
        print(f"  Length       : {result.path_length:.4f}")
        print(f"  Waypoints    : {len(result.path)}")
        print(f"  Time (s)     : {result.computation_time:.4f}")
        print(f"  Nodes in tree: {result.num_nodes_explored}")
        print(f"  Message      : {result.message}")
        print()
        print("  Path:")
        for i, pt in enumerate(result.path):
            tag = "  ← start" if i == 0 else ("  ← goal" if i == len(result.path) - 1 else "")
            print(f"    [{i:>3}]  ({pt[0]:.2f}, {pt[1]:.2f}){tag}")
    else:
        print(f"  Message: {result.message}")
    print("=" * 60)
