"""
Test script for multi-UAV path planning algorithms
Tests all 6 multi-path planners (3 base + 3 QIEA-optimized)
"""

import json
from pathlib import Path
from algorithms import (
    MultiPathAStar, MultiPathThetaStar, MultiPathDijkstra,
    MultiPathAStarQIEA, MultiPathThetaStarQIEA, MultiPathDijkstraQIEA
)
import time
import random
from visualize_multi_paths import visualize_multi_paths, visualize_comparison


def load_map(map_file: str):
    """Load map from JSON file (supports both single start and multi-UAV formats)"""
    with open(map_file, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: single start or multiple starts
    if 'starts' in data:
        # Multi-UAV format: use first start as reference (for compatibility)
        starts_list = [tuple(s) for s in data['starts']]
        start_ref = starts_list[0] if starts_list else (0, 0)
    elif 'start' in data:
        # Single start format
        start_ref = tuple(data['start'])
        starts_list = [start_ref]
    else:
        raise ValueError("Map file must contain either 'start' or 'starts' field")
    
    return {
        'width': data['width'],
        'height': data['height'],
        'obstacles': [tuple(obs) for obs in data['obstacles']],
        'start': start_ref,  # For backward compatibility
        'starts': starts_list,  # Multi-UAV format
        'goal': tuple(data['goal'])
    }


def generate_starts(num_uavs: int, map_data: dict, goal: tuple) -> list:
    """
    Generate start positions for multiple UAVs
    
    If map has pre-defined starts, use all of them (ignore num_uavs).
    Otherwise, generate new starts.
    """
    # Check if map already has starts defined - use all of them
    if 'starts' in map_data and len(map_data['starts']) > 0:
        # Use all pre-defined starts from multi-UAV map
        return map_data['starts']
    
    # Otherwise, generate new starts (for single-start maps)
    """
    Generate start positions for multiple UAVs
    
    Strategy: Place starts around the map perimeter, avoiding obstacles
    """
    width = map_data['width']
    height = map_data['height']
    obstacles = map_data['obstacles']
    
    starts = []
    attempts = 0
    max_attempts = 1000
    
    # Try to place starts in different regions
    regions = [
        (0, 0, width * 0.3, height * 0.3),  # Top-left
        (width * 0.7, 0, width, height * 0.3),  # Top-right
        (0, height * 0.7, width * 0.3, height),  # Bottom-left
        (width * 0.7, height * 0.7, width, height),  # Bottom-right
        (width * 0.3, 0, width * 0.7, height * 0.3),  # Top-center
        (0, height * 0.3, width * 0.3, height * 0.7),  # Left-center
        (width * 0.7, height * 0.3, width, height * 0.7),  # Right-center
        (width * 0.3, height * 0.7, width * 0.7, height),  # Bottom-center
    ]
    
    # Check if point is valid (not in obstacles)
    def is_valid_point(x, y):
        if not (0 <= x < width and 0 <= y < height):
            return False
        for cx, cy, radius in obstacles:
            dist_sq = (x - cx)**2 + (y - cy)**2
            if dist_sq <= radius**2:
                return False
        return True
    
    # Generate starts
    used_regions = set()
    while len(starts) < num_uavs and attempts < max_attempts:
        attempts += 1
        
        # Try different regions
        region_idx = len(starts) % len(regions)
        if region_idx in used_regions:
            region_idx = random.randint(0, len(regions) - 1)
        
        x_min, y_min, x_max, y_max = regions[region_idx]
        
        # Try random point in region
        for _ in range(10):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            
            if is_valid_point(x, y):
                # Check distance from existing starts
                too_close = False
                for sx, sy in starts:
                    if ((x - sx)**2 + (y - sy)**2) < 4.0:  # Min 2 units apart
                        too_close = True
                        break
                
                if not too_close:
                    starts.append((x, y))
                    used_regions.add(region_idx)
                    break
    
    # If we don't have enough, fill with random valid points
    while len(starts) < num_uavs and attempts < max_attempts:
        attempts += 1
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        
        if is_valid_point(x, y):
            too_close = False
            for sx, sy in starts:
                if ((x - sx)**2 + (y - sy)**2) < 4.0:
                    too_close = True
                    break
            
            if not too_close:
                starts.append((x, y))
    
    return starts


def test_multi_path_planner(planner_class, map_data, num_uavs: int, algorithm_name: str):
    """Test a single multi-path planner"""
    print(f"\n{'='*60}")
    print(f"Testing: {algorithm_name} with {num_uavs} UAVs")
    print(f"{'='*60}")
    
    # Generate start positions (will use all from map if available)
    goal = map_data['goal']
    starts = generate_starts(num_uavs, map_data, goal)
    
    # Update num_uavs to actual number of starts found
    actual_num_uavs = len(starts)
    if actual_num_uavs == 0:
        print(f"  Error: No valid start positions found")
        return None
    
    # Update num_uavs if we got different number from map
    if actual_num_uavs != num_uavs:
        print(f"  Note: Using {actual_num_uavs} UAVs (from map) instead of requested {num_uavs}")
        num_uavs = actual_num_uavs
    
    print(f"  Start positions: {starts[:3]}..." if len(starts) > 3 else f"  Start positions: {starts}")
    print(f"  Goal: {goal}")
    
    # Create planner - Base planners don't need num_uavs parameter
    if 'QIEA' in algorithm_name:
        # Hybrid planners need num_uavs
        planner = planner_class(
            map_data['obstacles'],
            map_data['width'],
            map_data['height'],
            num_uavs=num_uavs
        )
    else:
        # Base planners don't need num_uavs
        planner = planner_class(
            map_data['obstacles'],
            map_data['width'],
            map_data['height'],
            uav_speed=1.0
        )
    
    # Plan paths
    start_time = time.time()
    result = planner.plan(starts, goal)
    total_time = time.time() - start_time
    
    print(f"  Success: {result.success}")
    print(f"  Paths found: {len(result.paths)}/{num_uavs} UAVs")
    
    if result.success:
        print(f"  Total path length: {result.total_cost:.2f}")
        print(f"  Makespan: {result.makespan:.2f} seconds")
        print(f"  Computation time: {result.computation_time:.4f} seconds")
        print(f"  Message: {result.message}")
        
        # Path statistics
        for uav_id, path in result.paths.items():
            path_length = sum(
                ((path[i][0] - path[i+1][0])**2 + (path[i][1] - path[i+1][1])**2)**0.5
                for i in range(len(path) - 1)
            )
            print(f"    UAV {uav_id}: {len(path)} waypoints, length: {path_length:.2f}")
        
        # Skip conflicts display (user requested to ignore conflicts)
    else:
        print(f"  Message: {result.message}")
        if result.paths:
            print(f"  Partial paths found: {len(result.paths)}/{num_uavs} UAVs")
            for uav_id, path in result.paths.items():
                print(f"    UAV {uav_id}: {len(path)} waypoints")
    
    return result


def test_all_multi_path_planners(map_file: str, num_uavs: int = None, visualize: bool = False, save_viz: bool = False):
    """Test all multi-path planners on a single map
    
    Args:
        map_file: Path to map file
        num_uavs: Number of UAVs (if None, auto-detect from map's starts)
        visualize: Whether to show visualization
        save_viz: Whether to save visualization images
    """
    # Load map first
    map_data = load_map(map_file)
    
    # Auto-detect num_uavs from map if not provided
    if num_uavs is None:
        if 'starts' in map_data and len(map_data['starts']) > 0:
            num_uavs = len(map_data['starts'])
        else:
            raise ValueError("Cannot determine number of UAVs: map has no 'starts' field and num_uavs not provided")
    
    print(f"\n{'#'*60}")
    print(f"Testing Multi-Path Planners on: {map_file}")
    print(f"Number of UAVs: {num_uavs} (auto-detected from map)")
    print(f"{'#'*60}")
    
    print(f"\nMap Info:")
    print(f"  Size: {map_data['width']}x{map_data['height']}")
    print(f"  Obstacles: {len(map_data['obstacles'])}")
    print(f"  Goal: {map_data['goal']}")
    if 'starts' in map_data:
        print(f"  Pre-defined starts: {len(map_data['starts'])}")
    
    results = {}
    
    # Test base multi-path planners
    print(f"\n{'='*60}")
    print("BASE MULTI-PATH PLANNERS")
    print(f"{'='*60}")
    results['MultiPathAStar'] = test_multi_path_planner(
        MultiPathAStar, map_data, num_uavs, "MultiPathAStar"
    )
    results['MultiPathThetaStar'] = test_multi_path_planner(
        MultiPathThetaStar, map_data, num_uavs, "MultiPathThetaStar"
    )
    results['MultiPathDijkstra'] = test_multi_path_planner(
        MultiPathDijkstra, map_data, num_uavs, "MultiPathDijkstra"
    )
    
    # Test hybrid multi-path planners (QIEA-optimized)
    print(f"\n{'='*60}")
    print("HYBRID MULTI-PATH PLANNERS (QIEA-OPTIMIZED)")
    print(f"{'='*60}")
    print("Note: QIEA-optimized planners may take longer...")
    
    results['MultiPathAStarQIEA'] = test_multi_path_planner(
        MultiPathAStarQIEA, map_data, num_uavs, "MultiPathAStarQIEA"
    )
    results['MultiPathThetaStarQIEA'] = test_multi_path_planner(
        MultiPathThetaStarQIEA, map_data, num_uavs, "MultiPathThetaStarQIEA"
    )
    results['MultiPathDijkstraQIEA'] = test_multi_path_planner(
        MultiPathDijkstraQIEA, map_data, num_uavs, "MultiPathDijkstraQIEA"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Algorithm':<25} {'Success':<10} {'Total Length':<15} {'Makespan':<12} {'Time (s)':<12}")
    print("-"*60)
    
    for name, result in results.items():
        if result is None:
            continue
        success_str = "✓" if result.success else "✗"
        length_str = f"{result.total_cost:.2f}" if result.success else "N/A"
        makespan_str = f"{result.makespan:.2f}" if result.success else "N/A"
        time_str = f"{result.computation_time:.4f}" if result.success else "N/A"
        
        print(f"{name:<25} {success_str:<10} {length_str:<15} {makespan_str:<12} {time_str:<12}")
    
    # Visualize results if requested
    if visualize:
        print(f"\n{'='*60}")
        print("VISUALIZING RESULTS")
        print(f"{'='*60}")
        
        # Visualize comparison of all successful results
        successful_results = {name: r for name, r in results.items() 
                            if r is not None and r.success}
        
        if successful_results:
            if save_viz:
                viz_dir = Path("maps/visualizations")
                viz_dir.mkdir(parents=True, exist_ok=True)
                map_name = Path(map_file).stem
                save_path = viz_dir / f"{map_name}_comparison_{num_uavs}uavs.png"
            else:
                save_path = None
            
            visualize_comparison(
                successful_results,
                map_data,
                save_path=save_path,
                show=True
            )
        else:
            print("  No successful results to visualize")
        
        # Also visualize individual results if save requested
        if save_viz:
            viz_dir = Path("maps/visualizations")
            viz_dir.mkdir(parents=True, exist_ok=True)
            map_name = Path(map_file).stem
            
            for name, result in results.items():
                if result is not None and result.success:
                    save_path = viz_dir / f"{map_name}_{name}_{num_uavs}uavs.png"
                    visualize_multi_paths(
                        result,
                        map_data,
                        name,
                        save_path=str(save_path),
                        show=False
                    )
                    print(f"  Saved: {save_path}")
    
    return results


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test multi-UAV path planning algorithms')
    parser.add_argument('--map', type=str, default=None,
                       help='Path to map file (default: maps/multi_uav/m1_50x50_3uavs.json)')
    parser.add_argument('--uavs', type=int, nargs='+', default=None,
                       help='Number of UAVs to test (default: auto-detect from map)')
    parser.add_argument('--viz', action='store_true',
                       help='Show visualization after testing')
    parser.add_argument('--save-viz', action='store_true',
                       help='Save visualization images to maps/visualizations/')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization (overrides --viz)')
    
    args = parser.parse_args()
    
    # Determine map file
    if args.map:
        test_map = Path(args.map)
    else:
        maps_dir = Path("maps/multi_uav")
        test_map = maps_dir / "m1_50x50_3uavs.json"
    
    if not test_map.exists():
        print(f"Error: Test map {test_map} not found!")
        print("Please run map_generator_multi_uav.py first to generate maps.")
        return
    
    # Determine visualization setting
    visualize = args.viz and not args.no_viz
    
    print("Multi-UAV Path Planning Algorithms Test")
    print("="*60)
    print("\nThis test compares:")
    print("  - Base multi-path planners: MultiPathAStar, MultiPathThetaStar, MultiPathDijkstra")
    print("  - QIEA-optimized multi-path planners: MultiPathAStarQIEA, MultiPathThetaStarQIEA, MultiPathDijkstraQIEA")
    print("\nAll planners optimize paths for n UAVs → 1 goal simultaneously.")
    print("\nNote: Number of UAVs will be auto-detected from map's 'starts' field")
    print("      (unless --uavs is explicitly provided)")
    if visualize:
        print("\nVisualization: ENABLED")
    if args.save_viz:
        print("Save visualization: ENABLED")
    print("="*60)
    
    # Determine UAV counts to test
    if args.uavs:
        # User specified UAV counts
        uav_counts = args.uavs
    else:
        # Auto-detect from map
        try:
            map_data = load_map(str(test_map))
            if 'starts' in map_data and len(map_data['starts']) > 0:
                uav_counts = [len(map_data['starts'])]
            else:
                print("Error: Map has no 'starts' field. Please use --uavs to specify number of UAVs.")
                return
        except Exception as e:
            print(f"Error loading map: {e}")
            return
    
    # Test with different numbers of UAVs
    for num_uavs in uav_counts:
        print(f"\n\n{'#'*60}")
        print(f"TESTING WITH {num_uavs} UAVs")
        print(f"{'#'*60}")
        results = test_all_multi_path_planners(
            str(test_map), 
            num_uavs,
            visualize=visualize,
            save_viz=args.save_viz
        )
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)


if __name__ == "__main__":
    main()
