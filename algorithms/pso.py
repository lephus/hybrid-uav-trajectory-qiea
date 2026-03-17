"""
PSO (Particle Swarm Optimization) Path Planning Algorithm

PSO treats path planning as a continuous optimization problem.  Each particle
encodes a sequence of intermediate waypoints between start and goal.  The
swarm collectively searches for the shortest collision-free path.

Key characteristics:
- Operates fully in continuous space (no grid discretisation)
- Global optimum tendency, but may stagnate in NFZ-dense environments
- NFZs (circular no-fly zones) are handled via a large penalty term in fitness
"""

import time
import math
import random
import copy
from typing import List, Tuple, Optional

try:
    from .base import PathPlanner, PathResult
    from .utils import (
        euclidean_distance,
        line_obstacles_intersection,
        is_point_in_obstacles,
        calculate_path_length,
    )
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from algorithms.base import PathPlanner, PathResult
    from algorithms.utils import (
        euclidean_distance,
        line_obstacles_intersection,
        is_point_in_obstacles,
        calculate_path_length,
    )


class PSO(PathPlanner):
    """
    PSO path planner for continuous 2D environments with circular NFZ obstacles.

    Each particle is a flat vector of 2*n_waypoints values:
        [x0, y0, x1, y1, ..., x_{k-1}, y_{k-1}]
    representing k intermediate waypoints.  The full path is:
        start → w0 → w1 → … → w_{k-1} → goal

    Parameters
    ----------
    n_waypoints : int
        Number of intermediate waypoints per particle.
    n_particles : int
        Swarm size.
    max_iter : int
        Maximum number of PSO iterations.
    w : float
        Inertia weight.
    c1 : float
        Cognitive coefficient (pull toward personal best).
    c2 : float
        Social coefficient (pull toward global best).
    collision_penalty : float
        Penalty added to fitness for each unit of path that crosses an NFZ.
    """

    def __init__(
        self,
        obstacles: List[Tuple[float, float, float]],
        width: int,
        height: int,
        n_waypoints: int = 5,
        n_particles: int = 60,
        max_iter: int = 300,
        w: float = 0.6,
        c1: float = 1.5,
        c2: float = 2.0,
        collision_penalty: float = 500.0,
    ):
        super().__init__(obstacles, width, height)
        self.n_waypoints = n_waypoints
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.collision_penalty = collision_penalty
        self._dim = 2 * n_waypoints  # dimensionality of each particle

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

        # Velocity clamping bounds
        v_max = max(self.width, self.height) * 0.1

        # ── Initialise swarm ──────────────────────────────────────────────
        positions = self._init_positions(start, goal)
        velocities = [[random.uniform(-v_max, v_max)
                       for _ in range(self._dim)]
                      for _ in range(self.n_particles)]

        personal_best_pos = [p[:] for p in positions]
        personal_best_fit = [self._fitness(p, start, goal)
                             for p in positions]

        gbest_idx = personal_best_fit.index(min(personal_best_fit))
        global_best_pos = personal_best_pos[gbest_idx][:]
        global_best_fit = personal_best_fit[gbest_idx]

        # ── Main PSO loop ─────────────────────────────────────────────────
        for iteration in range(self.max_iter):
            # Linearly decay inertia weight for better convergence
            w_curr = self.w - (self.w - 0.4) * iteration / self.max_iter

            for i in range(self.n_particles):
                # Update velocity
                for d in range(self._dim):
                    r1 = random.random()
                    r2 = random.random()
                    velocities[i][d] = (
                        w_curr * velocities[i][d]
                        + self.c1 * r1 * (personal_best_pos[i][d] - positions[i][d])
                        + self.c2 * r2 * (global_best_pos[d] - positions[i][d])
                    )
                    # Clamp velocity
                    velocities[i][d] = max(-v_max, min(v_max, velocities[i][d]))

                # Update position
                for d in range(self._dim):
                    positions[i][d] += velocities[i][d]

                # Clamp positions to map bounds
                for k in range(self.n_waypoints):
                    positions[i][2 * k] = max(0.0, min(float(self.width - 1),
                                                        positions[i][2 * k]))
                    positions[i][2 * k + 1] = max(0.0, min(float(self.height - 1),
                                                             positions[i][2 * k + 1]))

                # Evaluate fitness
                fit = self._fitness(positions[i], start, goal)

                # Update personal best
                if fit < personal_best_fit[i]:
                    personal_best_fit[i] = fit
                    personal_best_pos[i] = positions[i][:]

                # Update global best
                if fit < global_best_fit:
                    global_best_fit = fit
                    global_best_pos = positions[i][:]

        # ── Build result path from global best ───────────────────────────
        path = self._decode_path(global_best_pos, start, goal)
        path_length = calculate_path_length(path)

        # Check if the best path has any collision
        has_collision = self._path_has_collision(path)

        success = not has_collision
        message = "PSO path found" if success else \
                  "PSO converged but path still intersects NFZ obstacles"

        return PathResult(
            path=path,
            success=success,
            computation_time=time.time() - start_time,
            path_length=path_length,
            cost=global_best_fit,
            num_nodes_explored=self.n_particles * self.max_iter,
            message=message,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _init_positions(self, start: Tuple[float, float],
                        goal: Tuple[float, float]) -> List[List[float]]:
        """
        Initialise particles by interpolating between start and goal with
        random perturbations.  This biases the swarm toward plausible paths.
        """
        positions = []
        for _ in range(self.n_particles):
            particle = []
            for k in range(1, self.n_waypoints + 1):
                # Linear interpolation fraction
                t = k / (self.n_waypoints + 1)
                # Base interpolated point
                bx = start[0] + t * (goal[0] - start[0])
                by = start[1] + t * (goal[1] - start[1])
                # Random perturbation
                perturb = max(self.width, self.height) * 0.25
                x = bx + random.uniform(-perturb, perturb)
                y = by + random.uniform(-perturb, perturb)
                x = max(0.0, min(float(self.width - 1), x))
                y = max(0.0, min(float(self.height - 1), y))
                particle.extend([x, y])
            positions.append(particle)
        return positions

    def _decode_path(self, particle: List[float],
                     start: Tuple[float, float],
                     goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Convert flat particle vector to a list of (x, y) waypoints."""
        path = [start]
        for k in range(self.n_waypoints):
            x = particle[2 * k]
            y = particle[2 * k + 1]
            path.append((x, y))
        path.append(goal)
        return path

    def _fitness(self, particle: List[float],
                 start: Tuple[float, float],
                 goal: Tuple[float, float]) -> float:
        """
        Fitness = path_length + collision_penalty * total_collision_depth.

        Collision depth for a segment is measured as the fraction of the
        segment length that lies inside obstacles, approximated by sampling.
        """
        path = self._decode_path(particle, start, goal)
        length = calculate_path_length(path)

        # Penalty: for each waypoint inside an obstacle add a large term
        penalty = 0.0
        for pt in path[1:-1]:  # intermediate waypoints only
            if is_point_in_obstacles(pt, self.obstacles):
                penalty += self.collision_penalty

        # Penalty: for each segment that crosses an obstacle
        for i in range(len(path) - 1):
            if line_obstacles_intersection(path[i], path[i + 1], self.obstacles):
                seg_len = euclidean_distance(path[i], path[i + 1])
                penalty += self.collision_penalty * (seg_len / max(1.0, length))

        return length + penalty

    def _path_has_collision(self, path: List[Tuple[float, float]]) -> bool:
        """True if any waypoint or segment violates obstacle constraints."""
        for pt in path:
            if is_point_in_obstacles(pt, self.obstacles):
                return True
        for i in range(len(path) - 1):
            if line_obstacles_intersection(path[i], path[i + 1], self.obstacles):
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys
    import os

    parser = argparse.ArgumentParser(description="Run PSO on a UAV map")
    parser.add_argument(
        "--map",
        default=os.path.join("maps", "destinations", "m1_50x50.json"),
        help="Path to the map JSON file (default: maps/destinations/m1_50x50.json)",
    )
    parser.add_argument("--waypoints",  type=int,   default=5,     help="Intermediate waypoints per particle (default: 5)")
    parser.add_argument("--particles",  type=int,   default=60,    help="Swarm size (default: 60)")
    parser.add_argument("--iterations", type=int,   default=300,   help="Max PSO iterations (default: 300)")
    parser.add_argument("--penalty",    type=float, default=500.0, help="NFZ collision penalty (default: 500)")
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
    print("  PSO (Particle Swarm Optimization)")
    print("=" * 60)
    print(f"  Map        : {args.map}")
    print(f"  Size       : {width} × {height}")
    print(f"  NFZ count  : {len(obstacles)} circles")
    print(f"  Start      : {start}")
    print(f"  Goal       : {goal}")
    print(f"  Waypoints  : {args.waypoints}")
    print(f"  Particles  : {args.particles}")
    print(f"  Iterations : {args.iterations}")
    print()

    planner = PSO(
        obstacles, width, height,
        n_waypoints=args.waypoints,
        n_particles=args.particles,
        max_iter=args.iterations,
        collision_penalty=args.penalty,
    )

    result = planner.plan(start, goal)

    print(f"  Success      : {'Yes' if result.success else 'No  ← path crosses NFZ (known PSO limitation)'}")
    if result.path:
        print(f"  Length       : {result.path_length:.4f}")
        print(f"  Waypoints    : {len(result.path)}")
        print(f"  Time (s)     : {result.computation_time:.4f}")
        print(f"  Best fitness : {result.cost:.4f}")
        print(f"  Evaluations  : {result.num_nodes_explored}")
        print(f"  Message      : {result.message}")
        print()
        print("  Path:")
        for i, pt in enumerate(result.path):
            tag = "  ← start" if i == 0 else ("  ← goal" if i == len(result.path) - 1 else "")
            print(f"    [{i:>3}]  ({pt[0]:.2f}, {pt[1]:.2f}){tag}")
    else:
        print(f"  Message: {result.message}")
    print("=" * 60)
