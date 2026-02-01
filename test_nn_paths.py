"""
Test script for n-UAV to n-Destination path planning
Tests all path planning algorithms on each UAV-destination pair
"""

import json
from pathlib import Path
from algorithms import (
    AStar, ThetaStar, Dijkstra,
    AStarQIEA, ThetaStarQIEA, DijkstraQIEA
)
from algorithms.qiea import QIEA
import time
from visualize_nn_paths import visualize_nn_paths, visualize_nn_comparison
from typing import Dict, List, Tuple
from algorithms.base import PathResult


def load_nn_map(map_file: str):
    """Load n-n map from JSON file"""
    with open(map_file, 'r') as f:
        data = json.load(f)
    
    if 'starts' not in data or 'goals' not in data:
        raise ValueError("Map file must contain 'starts' and 'goals' fields")
    
    # Load assignments
    if 'assignments' in data:
        assignments = {uav_id: goal_id for uav_id, goal_id in data['assignments']}
    else:
        # Default: sequential assignment
        assignments = {i: i for i in range(len(data['starts']))}
    
    return {
        'width': data['width'],
        'height': data['height'],
        'obstacles': [tuple(obs) for obs in data['obstacles']],
        'starts': [tuple(s) for s in data['starts']],
        'goals': [tuple(g) for g in data['goals']],
        'assignments': assignments,
        'num_uavs': len(data['starts'])
    }


def test_single_path_planner(planner_class, map_data: dict, 
                            start: Tuple[float, float], 
                            goal: Tuple[float, float],
                            algorithm_name: str) -> PathResult:
    """Test a single path planner on one start-goal pair"""
    planner = planner_class(
        map_data['obstacles'],
        map_data['width'],
        map_data['height']
    )
    
    result = planner.plan(start, goal)
    return result


def test_all_planners_on_nn_map(map_file: str, visualize: bool = False, save_viz: bool = False):
    """
    Test all path planning algorithms on n-n map
    
    For each UAV-destination pair, test all algorithms and collect results
    """
    # Load map
    map_data = load_nn_map(map_file)
    
    print(f"\n{'#'*60}")
    print(f"Testing Path Planners on n-n Map: {map_file}")
    print(f"Number of UAVs: {map_data['num_uavs']}")
    print(f"{'#'*60}")
    
    print(f"\nMap Info:")
    print(f"  Size: {map_data['width']}x{map_data['height']}")
    print(f"  Obstacles: {len(map_data['obstacles'])}")
    print(f"  Assignments: {map_data['assignments']}")
    
    # Define all algorithms to test
    algorithms = {
        'AStar': AStar,
        'ThetaStar': ThetaStar,
        'Dijkstra': Dijkstra,
        'AStarQIEA': AStarQIEA,
        'ThetaStarQIEA': ThetaStarQIEA,
        'DijkstraQIEA': DijkstraQIEA,
        'QIEA': QIEA
    }
    
    # Store results: {algorithm_name: {uav_id: PathResult}}
    all_results: Dict[str, Dict[int, PathResult]] = {}
    
    # Test each algorithm
    for alg_name, alg_class in algorithms.items():
        print(f"\n{'='*60}")
        print(f"Testing: {alg_name}")
        print(f"{'='*60}")
        
        algorithm_results = {}
        total_time = 0.0
        total_length = 0.0
        successful_paths = 0
        
        # Test on each UAV-destination pair
        for uav_id, goal_id in map_data['assignments'].items():
            start = map_data['starts'][uav_id]
            goal = map_data['goals'][goal_id]
            
            print(f"  UAV {uav_id} → Goal {goal_id}: ", end="")
            
            result = test_single_path_planner(
                alg_class, map_data, start, goal, alg_name
            )
            
            algorithm_results[uav_id] = result
            total_time += result.computation_time
            
            if result.success:
                successful_paths += 1
                total_length += result.path_length
                path_length = result.path_length
                print(f"✓ Success (length: {path_length:.2f}, time: {result.computation_time:.4f}s)")
            else:
                print(f"✗ Failed: {result.message}")
        
        all_results[alg_name] = algorithm_results
        
        # Summary for this algorithm
        print(f"\n  Summary for {alg_name}:")
        print(f"    Successful paths: {successful_paths}/{map_data['num_uavs']}")
        print(f"    Total path length: {total_length:.2f}")
        print(f"    Average path length: {total_length/successful_paths:.2f}" if successful_paths > 0 else "    Average path length: N/A")
        print(f"    Total computation time: {total_time:.4f}s")
        print(f"    Average computation time: {total_time/map_data['num_uavs']:.4f}s")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Algorithm':<20} {'Success Rate':<15} {'Avg Length':<15} {'Total Time (s)':<15}")
    print("-"*60)
    
    for alg_name, results in all_results.items():
        successful = sum(1 for r in results.values() if r.success)
        success_rate = f"{successful}/{map_data['num_uavs']}"
        
        if successful > 0:
            total_length = sum(r.path_length for r in results.values() if r.success)
            avg_length = total_length / successful
        else:
            avg_length = 0.0
        
        total_time = sum(r.computation_time for r in results.values())
        
        print(f"{alg_name:<20} {success_rate:<15} {avg_length:<15.2f} {total_time:<15.4f}")
    
    # Visualize results if requested
    if visualize or save_viz:
        print(f"\n{'='*60}")
        print("VISUALIZING RESULTS")
        print(f"{'='*60}")
        
        # Filter successful results only
        successful_results = {}
        for alg_name, results in all_results.items():
            # Check if all paths succeeded
            if all(r.success for r in results.values()):
                successful_results[alg_name] = results
        
        if successful_results:
            if save_viz:
                viz_dir = Path("maps/visualizations")
                viz_dir.mkdir(parents=True, exist_ok=True)
                map_name = Path(map_file).stem
                save_path = viz_dir / f"{map_name}_nn_comparison.png"
            else:
                save_path = None
            
            visualize_nn_comparison(
                successful_results,
                map_data,
                save_path=save_path,
                show=visualize
            )
            
            # Also save individual algorithm visualizations
            if save_viz:
                for alg_name, results in successful_results.items():
                    save_path = viz_dir / f"{map_name}_{alg_name}_nn.png"
                    visualize_nn_paths(
                        results,
                        map_data,
                        alg_name,
                        save_path=str(save_path),
                        show=False
                    )
                    print(f"  Saved: {save_path}")
        else:
            print("  No fully successful results to visualize")
    
    return all_results


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test path planning algorithms on n-n maps')
    parser.add_argument('--map', type=str, default=None,
                       help='Path to n-n map file (default: maps/nn_destinations/m1_50x50_5nn.json)')
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
        maps_dir = Path("maps/nn_destinations")
        test_map = maps_dir / "m1_50x50_5nn.json"
    
    if not test_map.exists():
        print(f"Error: Test map {test_map} not found!")
        print("Please run map_generator_nn.py first to generate maps.")
        return
    
    # Determine visualization setting
    visualize = args.viz and not args.no_viz
    
    print("n-UAV to n-Destination Path Planning Test")
    print("="*60)
    print("\nThis test runs the following algorithms on each UAV-destination pair:")
    print("  - AStar")
    print("  - ThetaStar")
    print("  - Dijkstra")
    print("  - AStarQIEA (A* + QIEA)")
    print("  - ThetaStarQIEA (Theta* + QIEA)")
    print("  - DijkstraQIEA (Dijkstra + QIEA)")
    print("  - QIEA")
    print("\nEach algorithm is tested independently on each UAV-destination pair.")
    if visualize:
        print("\nVisualization: ENABLED")
    if args.save_viz:
        print("Save visualization: ENABLED")
    print("="*60)
    
    # Run tests
    results = test_all_planners_on_nn_map(
        str(test_map),
        visualize=visualize,
        save_viz=args.save_viz
    )
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)


if __name__ == "__main__":
    main()
