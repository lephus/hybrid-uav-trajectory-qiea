"""
Test script for all path planning algorithms
"""

import json
from pathlib import Path
from algorithms import (
    AStar, ThetaStar, Dijkstra,
    AStarQIEA, ThetaStarQIEA, DijkstraQIEA
)
import time


def load_map(map_file: str):
    """Load map from JSON file"""
    with open(map_file, 'r') as f:
        data = json.load(f)
    
    return {
        'width': data['width'],
        'height': data['height'],
        'obstacles': [tuple(obs) for obs in data['obstacles']],
        'start': tuple(data['start']),
        'goal': tuple(data['goal'])
    }


def test_algorithm(planner_class, map_data, algorithm_name: str):
    """Test a single algorithm"""
    print(f"\n{'='*60}")
    print(f"Testing: {algorithm_name}")
    print(f"{'='*60}")
    
    planner = planner_class(
        map_data['obstacles'],
        map_data['width'],
        map_data['height']
    )
    
    start_time = time.time()
    result = planner.plan(map_data['start'], map_data['goal'])
    total_time = time.time() - start_time
    
    print(f"Success: {result.success}")
    if result.success:
        print(f"Path length: {result.path_length:.2f}")
        print(f"Cost: {result.cost:.2f}")
        print(f"Number of waypoints: {len(result.path)}")
        print(f"Computation time: {result.computation_time:.4f} seconds")
        print(f"Nodes explored: {result.num_nodes_explored}")
    else:
        print(f"Message: {result.message}")
    
    return result


def test_all_algorithms(map_file: str):
    """Test all algorithms on a single map"""
    print(f"\n{'#'*60}")
    print(f"Testing algorithms on: {map_file}")
    print(f"{'#'*60}")
    
    # Load map
    map_data = load_map(map_file)
    print(f"\nMap Info:")
    print(f"  Size: {map_data['width']}x{map_data['height']}")
    print(f"  Obstacles: {len(map_data['obstacles'])}")
    print(f"  Start: {map_data['start']}")
    print(f"  Goal: {map_data['goal']}")
    
    results = {}
    
    # Test classical algorithms
    results['A*'] = test_algorithm(AStar, map_data, "A*")
    results['Theta*'] = test_algorithm(ThetaStar, map_data, "Theta*")
    results['Dijkstra'] = test_algorithm(Dijkstra, map_data, "Dijkstra")
    
    # Test hybrid algorithms (QIEA-optimized)
    print(f"\nNote: Hybrid algorithms use QIEA for optimization and may take longer...")
    results['A*+QIEA'] = test_algorithm(AStarQIEA, map_data, "A* + QIEA")
    results['Theta*+QIEA'] = test_algorithm(ThetaStarQIEA, map_data, "Theta* + QIEA")
    results['Dijkstra+QIEA'] = test_algorithm(DijkstraQIEA, map_data, "Dijkstra + QIEA")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Algorithm':<20} {'Success':<10} {'Path Length':<15} {'Time (s)':<12} {'Nodes':<10}")
    print("-"*60)
    
    for name, result in results.items():
        success_str = "✓" if result.success else "✗"
        length_str = f"{result.path_length:.2f}" if result.success else "N/A"
        time_str = f"{result.computation_time:.4f}" if result.success else "N/A"
        nodes_str = f"{result.num_nodes_explored}" if result.success else "N/A"
        
        print(f"{name:<20} {success_str:<10} {length_str:<15} {time_str:<12} {nodes_str:<10}")
    
    return results


def main():
    """Main test function"""
    maps_dir = Path("maps")
    
    # Test on a small map first
    test_map = maps_dir / "m1_20x20.json"
    
    if not test_map.exists():
        print(f"Error: Test map {test_map} not found!")
        print("Please run map_generator.py first to generate maps.")
        return
    
    print("Path Planning Algorithms Test")
    print("="*60)
    print("\nThis test compares:")
    print("  - Classical algorithms: A*, Theta*, Dijkstra")
    print("  - QIEA-optimized variants: A*+QIEA, Theta*+QIEA, Dijkstra+QIEA")
    print("\nQIEA (Quantum-Inspired Evolutionary Algorithm) is used to optimize")
    print("paths found by classical algorithms, demonstrating the improvement.")
    print("="*60)
    
    # Test all algorithms on the test map
    results = test_all_algorithms(str(test_map))
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)


if __name__ == "__main__":
    main()

