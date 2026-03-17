"""
Test script for all path planning algorithms.

Measures three metrics for every algorithm on the 50×50 map:
  • Length    – total Euclidean path length
  • Waypoints – number of nodes / waypoints in the returned path
  • Time      – wall-clock computation time (seconds)
"""

import json
import time
from pathlib import Path
from algorithms import (
    AStar, ThetaStar, Dijkstra,
    AStarQIEA, ThetaStarQIEA, DijkstraQIEA,
    RRTStar, PSO,
)


# ─────────────────────────────────────────────────────────────────────────────
# Map loading
# ─────────────────────────────────────────────────────────────────────────────

def load_map(map_file: str) -> dict:
    with open(map_file, 'r') as f:
        data = json.load(f)
    return {
        'width':     data['width'],
        'height':    data['height'],
        'obstacles': [tuple(obs) for obs in data['obstacles']],
        'start':     tuple(data['start']),
        'goal':      tuple(data['goal']),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-algorithm runner
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm(planner_class, map_data: dict, algorithm_name: str):
    """Instantiate planner, run plan(), return PathResult."""
    planner = planner_class(
        map_data['obstacles'],
        map_data['width'],
        map_data['height'],
    )
    t0 = time.perf_counter()
    result = planner.plan(map_data['start'], map_data['goal'])
    wall_time = time.perf_counter() - t0

    # Use wall_time if the algorithm didn't record its own
    if result.computation_time == 0.0:
        result.computation_time = wall_time

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHMS = [
    ("A*",            AStar),
    ("Theta*",        ThetaStar),
    ("Dijkstra",      Dijkstra),
    ("A*+QIEA",       AStarQIEA),
    ("Theta*+QIEA",   ThetaStarQIEA),
    ("Dijkstra+QIEA", DijkstraQIEA),
    ("RRT*",          RRTStar),
    ("PSO",           PSO),
]


def benchmark_map(map_file: str):
    """Run all algorithms on one map and print a 3-metric comparison table."""
    print(f"\n{'#' * 70}")
    print(f"  Map: {map_file}")
    print(f"{'#' * 70}")

    map_data = load_map(map_file)
    print(f"\n  Size      : {map_data['width']} × {map_data['height']}")
    print(f"  Obstacles : {len(map_data['obstacles'])} NFZ circles")
    print(f"  Start     : {map_data['start']}")
    print(f"  Goal      : {map_data['goal']}")

    results = {}

    for name, cls in ALGORITHMS:
        print(f"\n  Running {name} …", end=" ", flush=True)
        result = run_algorithm(cls, map_data, name)
        results[name] = result
        status = "OK" if result.success else "FAIL"
        print(status)

    # ── Summary table ──────────────────────────────────────────────────────
    col_alg  = 18
    col_ok   = 8
    col_len  = 14
    col_wp   = 12
    col_time = 12

    header = (f"  {'Algorithm':<{col_alg}} {'OK?':<{col_ok}}"
              f" {'Length':>{col_len}} {'Waypoints':>{col_wp}}"
              f" {'Time (s)':>{col_time}}")
    sep = "  " + "-" * (col_alg + col_ok + col_len + col_wp + col_time + 6)

    print(f"\n{'=' * 70}")
    print("  BENCHMARK RESULTS  –  3 Metrics Comparison")
    print(f"{'=' * 70}")
    print(header)
    print(sep)

    for name, result in results.items():
        ok_str   = "✓" if result.success else "✗"
        len_str  = f"{result.path_length:.2f}" if result.success else "N/A"
        wp_str   = f"{len(result.path)}"       if result.success else "N/A"
        time_str = f"{result.computation_time:.4f}"

        print(f"  {name:<{col_alg}} {ok_str:<{col_ok}}"
              f" {len_str:>{col_len}} {wp_str:>{col_wp}}"
              f" {time_str:>{col_time}}")

    print(sep)

    # ── Quick analysis ─────────────────────────────────────────────────────
    successful = {n: r for n, r in results.items() if r.success}
    if successful:
        best_len  = min(successful, key=lambda n: successful[n].path_length)
        fewest_wp = min(successful, key=lambda n: len(successful[n].path))
        fastest   = min(successful, key=lambda n: successful[n].computation_time)

        print(f"\n  Best path length   : {best_len}"
              f"  ({successful[best_len].path_length:.2f})")
        print(f"  Fewest waypoints   : {fewest_wp}"
              f"  ({len(successful[fewest_wp].path)} pts)")
        print(f"  Fastest computation: {fastest}"
              f"  ({successful[fastest].computation_time:.4f} s)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  UAV Path Planning – Algorithm Comparison")
    print("  Metrics: Length | Waypoints | Time")
    print("=" * 70)

    maps_dir = Path("maps/destinations")
    test_map = maps_dir / "m1_50x50.json"

    if not test_map.exists():
        print(f"\nError: {test_map} not found. Run map_generator.py first.")
        return

    benchmark_map(str(test_map))

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
