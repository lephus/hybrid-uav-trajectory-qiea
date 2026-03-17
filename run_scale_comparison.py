"""
Scale Comparison: A* + QIEA vs Theta* + QIEA across map sizes
Generates 50×50, 100×100, and 200×200 environments with proportional
obstacle density (same circle coverage %) then benchmarks both hybrid
algorithms on every size.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from map_generator_nn import NNDestinationMapGenerator
from algorithms import AStarQIEA, ThetaStarQIEA, AStar, ThetaStar
from algorithms.base import PathResult


# ─────────────────────────────────────────────────────────────────────────────
# Map sizes and UAV count
# ─────────────────────────────────────────────────────────────────────────────
SIZES = [50, 100, 200]
NUM_UAVS = 3          # keep constant across sizes for a fair comparison
MAP_TYPE = "m1"       # sparse – low obstacle density
RANDOM_SEED = 42      # reproducible layouts

# ── Obstacle density parameters ───────────────────────────────────────────
# To keep the same circle-count density (circles per unit area) across sizes:
#   n_circles = BASE_CIRCLES × (size/BASE_SIZE)²
# Example:  50×50→10, 100×100→40, 200×200→160  (exactly 4× each doubling)
BASE_SIZE    = 50
BASE_CIRCLES = 10          # circles on the 50×50 reference map
# Circle radii are FIXED for all map sizes so the physical size of each NFZ
# does not change – only the count scales with area.
CIRCLE_MIN_R = 1.5         # grid units
CIRCLE_MAX_R = 3.5         # grid units

# Algorithms to benchmark
ALGORITHMS = {
    "AStar":         AStar,
    "ThetaStar":     ThetaStar,
    "AStarQIEA":     AStarQIEA,
    "ThetaStarQIEA": ThetaStarQIEA,
}


# ─────────────────────────────────────────────────────────────────────────────
# Map generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_or_load_map(size: int, num_uavs: int = NUM_UAVS,
                         map_type: str = MAP_TYPE,
                         force_regenerate: bool = False) -> dict:
    """
    Generate (or load from cache) a map of the requested size.

    Circle-count density is maintained by scaling the number of circles
    quadratically with map size while keeping each circle's physical radius
    fixed (CIRCLE_MIN_R … CIRCLE_MAX_R):

        n_circles = BASE_CIRCLES × (size / BASE_SIZE)²

    Examples with BASE_CIRCLES=10:
        50×50   →  10 circles   (ratio 1×)
        100×100 →  40 circles   (ratio 4×)
        200×200 → 160 circles   (ratio 16×)
    """
    output_dir = Path("maps/nn_destinations")
    output_dir.mkdir(parents=True, exist_ok=True)

    area_ratio  = (size / BASE_SIZE) ** 2
    n_circles   = int(BASE_CIRCLES * area_ratio)
    filename    = f"{map_type}_{size}x{size}_{num_uavs}nn_scale.json"
    map_path    = output_dir / filename

    if map_path.exists() and not force_regenerate:
        print(f"  Loading cached map: {map_path}")
        with open(map_path) as f:
            data = json.load(f)
        return _parse_map_data(data)

    print(f"  Generating {map_type} {size}×{size} map with {num_uavs} UAVs …")
    print(f"    Target circles : {n_circles}  "
          f"(= {BASE_CIRCLES} × ({size}/{BASE_SIZE})²)")
    np.random.seed(RANDOM_SEED + size)   # distinct seed per size, reproducible

    gen = NNDestinationMapGenerator(size, size)
    success = gen.generate_m1_sparse(
        num_uavs,
        num_obstacles=n_circles,
        min_radius=CIRCLE_MIN_R,
        max_radius=CIRCLE_MAX_R,
    )

    if not success:
        raise RuntimeError(f"Failed to generate {size}×{size} map")

    gen.save_to_file(str(map_path))

    density   = gen.get_obstacle_density()
    actual_n  = len(gen.obstacles)
    print(f"    ✓  Saved  {map_path.name}")
    print(f"       Circles (NFZs) placed : {actual_n}  (target {n_circles})")
    print(f"       Grid cell coverage    : {density:.2f} %")
    print(f"       Starts                : {gen.starts}")
    print(f"       Goals                 : {gen.goals}")

    with open(map_path) as f:
        data = json.load(f)
    return _parse_map_data(data)


def _parse_map_data(data: dict) -> dict:
    if "assignments" in data:
        assignments = {uav_id: goal_id for uav_id, goal_id in data["assignments"]}
    else:
        assignments = {i: i for i in range(len(data["starts"]))}

    return {
        "width":       data["width"],
        "height":      data["height"],
        "obstacles":   [tuple(obs) for obs in data["obstacles"]],
        "starts":      [tuple(s) for s in data["starts"]],
        "goals":       [tuple(g) for g in data["goals"]],
        "assignments": assignments,
        "num_uavs":    len(data["starts"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single run helper
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm(alg_class, map_data: dict) -> Tuple[List[PathResult], float]:
    """Run one algorithm on all UAV–goal pairs; return (results, total_wall_time)."""
    results = []
    t0 = time.perf_counter()

    for uav_id, goal_id in map_data["assignments"].items():
        start = map_data["starts"][uav_id]
        goal  = map_data["goals"][goal_id]

        planner = alg_class(
            map_data["obstacles"],
            map_data["width"],
            map_data["height"],
        )
        result = planner.plan(start, goal)
        results.append(result)

    wall_time = time.perf_counter() - t0
    return results, wall_time


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str, width: int = 72):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def summarise_results(results: List[PathResult], num_uavs: int) -> dict:
    successful   = [r for r in results if r.success]
    success_rate = len(successful) / num_uavs
    total_len    = sum(r.path_length for r in successful)
    avg_len      = total_len / len(successful) if successful else 0.0
    total_time   = sum(r.computation_time for r in results)
    avg_time     = total_time / num_uavs
    return {
        "success_rate": success_rate,
        "successful":   len(successful),
        "total_length": total_len,
        "avg_length":   avg_len,
        "total_time":   total_time,
        "avg_time":     avg_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(force_regenerate: bool = False):
    print("\n" + "╔" + "═" * 70 + "╗")
    print("║  SCALE COMPARISON: A* + QIEA  vs  Theta* + QIEA" + " " * 21 + "║")
    print("║  Map sizes: 50×50  |  100×100  |  200×200" + " " * 28 + "║")
    print("╚" + "═" * 70 + "╝")

    # ── Step 1 : generate / load maps ───────────────────────────────────────
    _section("STEP 1 – Map Generation (obstacle density ≈ 5 % cell coverage)")

    maps: Dict[int, dict] = {}
    for size in SIZES:
        print(f"\n[{size}×{size}]")
        maps[size] = generate_or_load_map(size, force_regenerate=force_regenerate)

    # Verify density scaling
    print("\n  Obstacle counts (should scale as 1 : 4 : 16):")
    ref_n    = len(maps[SIZES[0]]["obstacles"])
    ref_area = SIZES[0] ** 2
    ref_den  = ref_n / ref_area
    for size in SIZES:
        n     = len(maps[size]["obstacles"])
        ratio = n / ref_n if ref_n else 0
        area  = size * size
        den   = n / area
        print(f"    {size:>3}×{size:<3}  circles={n:>5}   ratio≈{ratio:.1f}x"
              f"   area={area:>6}   circles/area≈{den:.4f}"
              f"  (expected {int(BASE_CIRCLES * (size/BASE_SIZE)**2):>4})")

    # ── Step 2 : run algorithms ──────────────────────────────────────────────
    _section("STEP 2 – Algorithm Benchmarks")

    # all_stats[alg_name][size] = summary_dict
    all_stats: Dict[str, Dict[int, dict]] = {name: {} for name in ALGORITHMS}

    for size in SIZES:
        map_data = maps[size]
        print(f"\n[{size}×{size}]  {map_data['num_uavs']} UAVs"
              f"  |  {len(map_data['obstacles'])} NFZ circles")
        print(f"  {'Algorithm':<20} {'Success':<10} {'Avg Length':>12} "
              f"{'Total Time (s)':>16} {'Avg Time (s)':>14}")
        print(f"  {'-'*72}")

        for alg_name, alg_class in ALGORITHMS.items():
            results, wall_time = run_algorithm(alg_class, map_data)
            stats = summarise_results(results, map_data["num_uavs"])
            all_stats[alg_name][size] = stats

            sr = f"{stats['successful']}/{map_data['num_uavs']}"
            print(f"  {alg_name:<20} {sr:<10} {stats['avg_length']:>12.2f} "
                  f"{stats['total_time']:>16.4f} {stats['avg_time']:>14.4f}")

    # ── Step 3 : cross-size comparison table ────────────────────────────────
    _section("STEP 3 – Cross-Size Comparison (focus: A* + QIEA  &  Theta* + QIEA)")

    focus_algs = ["AStarQIEA", "ThetaStarQIEA"]

    for alg_name in focus_algs:
        print(f"\n  ▶  {alg_name}")
        print(f"  {'Map size':<12} {'NFZ circles':>12} {'Success':>9} "
              f"{'Avg Length':>12} {'Total Time (s)':>16} {'Avg Time (s)':>14}")
        print(f"  {'-'*75}")

        for size in SIZES:
            n_circles = len(maps[size]["obstacles"])
            s = all_stats[alg_name][size]
            sr = f"{s['successful']}/{maps[size]['num_uavs']}"
            print(f"  {size}×{size:<8}  {n_circles:>12} {sr:>9} "
                  f"{s['avg_length']:>12.2f} {s['total_time']:>16.4f} {s['avg_time']:>14.4f}")

    # ── Step 4 : A*QIEA vs Theta*QIEA head-to-head ──────────────────────────
    _section("STEP 4 – Head-to-Head: AStarQIEA vs ThetaStarQIEA")

    print(f"\n  {'Map size':<12} {'Metric':<22} {'AStarQIEA':>14} "
          f"{'ThetaStarQIEA':>15} {'Winner':>10}")
    print(f"  {'-'*75}")

    for size in SIZES:
        sa = all_stats["AStarQIEA"][size]
        st = all_stats["ThetaStarQIEA"][size]
        tag = f"{size}×{size}"

        # Avg path length (lower is better)
        winner_len = ("AStarQIEA" if sa["avg_length"] < st["avg_length"]
                      else "ThetaStarQIEA" if st["avg_length"] < sa["avg_length"]
                      else "tie")
        print(f"  {tag:<12} {'avg_length (lower ✓)':<22} "
              f"{sa['avg_length']:>14.2f} {st['avg_length']:>15.2f} {winner_len:>10}")

        # Avg computation time (lower is better)
        winner_time = ("AStarQIEA" if sa["avg_time"] < st["avg_time"]
                       else "ThetaStarQIEA" if st["avg_time"] < sa["avg_time"]
                       else "tie")
        print(f"  {'':<12} {'avg_time   (lower ✓)':<22} "
              f"{sa['avg_time']:>14.4f} {st['avg_time']:>15.4f} {winner_time:>10}")

    print("\n" + "═" * 72)
    print("  Done.")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scale comparison: A*+QIEA vs Theta*+QIEA on 50×50, 100×100, 200×200")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force re-generate all maps even if cached files exist")
    args = parser.parse_args()

    main(force_regenerate=args.regenerate)
