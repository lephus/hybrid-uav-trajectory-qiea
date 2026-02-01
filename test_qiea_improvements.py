"""
Test script to demonstrate improvements from seeding and adaptive rotation in QIEA
"""

import json
import time
from pathlib import Path
from algorithms import AStar, ThetaStar, Dijkstra
from algorithms import AStarQIEA, ThetaStarQIEA, DijkstraQIEA


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
    planner = planner_class(
        map_data['obstacles'],
        map_data['width'],
        map_data['height']
    )
    
    start_time = time.time()
    result = planner.plan(map_data['start'], map_data['goal'])
    total_time = time.time() - start_time
    
    return {
        'name': algorithm_name,
        'success': result.success,
        'path_length': result.path_length if result.success else 0.0,
        'cost': result.cost if result.success else float('inf'),
        'computation_time': result.computation_time,
        'num_waypoints': len(result.path) if result.success else 0,
        'nodes_explored': result.num_nodes_explored
    }


def main():
    """Test improvements"""
    print("QIEA Improvements Test: Seeding + Adaptive Rotation")
    print("=" * 70)
    
    # Test on m1_20x20
    map_file = Path("maps/m1_20x20.json")
    if not map_file.exists():
        print(f"Error: {map_file} not found!")
        return
    
    map_data = load_map(str(map_file))
    print(f"\nMap: {map_file.name}")
    print(f"  Size: {map_data['width']}x{map_data['height']}")
    print(f"  Obstacles: {len(map_data['obstacles'])}")
    print(f"  Start: {map_data['start']}, Goal: {map_data['goal']}")
    
    print("\n" + "=" * 70)
    print("Testing Algorithms")
    print("=" * 70)
    
    results = {}
    
    # Test classical algorithms
    print("\nClassical Algorithms:")
    results['A*'] = test_algorithm(AStar, map_data, "A*")
    results['Theta*'] = test_algorithm(ThetaStar, map_data, "Theta*")
    results['Dijkstra'] = test_algorithm(Dijkstra, map_data, "Dijkstra")
    
    # Test hybrid algorithms (with seeding and adaptive rotation)
    print("\nHybrid Algorithms (with Seeding + Adaptive Rotation):")
    results['A*+QIEA'] = test_algorithm(AStarQIEA, map_data, "A*+QIEA")
    results['Theta*+QIEA'] = test_algorithm(ThetaStarQIEA, map_data, "Theta*+QIEA")
    results['Dijkstra+QIEA'] = test_algorithm(DijkstraQIEA, map_data, "Dijkstra+QIEA")
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<20} {'Success':<10} {'Path Length':<15} {'Cost':<15} {'Time (s)':<12} {'Waypoints':<10}")
    print("-" * 70)
    
    for name in ['A*', 'Theta*', 'Dijkstra', 'A*+QIEA', 'Theta*+QIEA', 'Dijkstra+QIEA']:
        r = results[name]
        success_str = "✓" if r['success'] else "✗"
        length_str = f"{r['path_length']:.2f}" if r['success'] else "N/A"
        cost_str = f"{r['cost']:.2f}" if r['success'] else "N/A"
        time_str = f"{r['computation_time']:.4f}" if r['success'] else "N/A"
        waypoints_str = f"{r['num_waypoints']}" if r['success'] else "N/A"
        
        print(f"{name:<20} {success_str:<10} {length_str:<15} {cost_str:<15} {time_str:<12} {waypoints_str:<10}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    # Compare A* vs A*+QIEA
    if results['A*']['success'] and results['A*+QIEA']['success']:
        length_improvement = ((results['A*']['path_length'] - results['A*+QIEA']['path_length']) / 
                             results['A*']['path_length'] * 100)
        cost_improvement = ((results['A*']['cost'] - results['A*+QIEA']['cost']) / 
                           results['A*']['cost'] * 100)
        
        print(f"\nA* vs A*+QIEA:")
        print(f"  Path length: {results['A*']['path_length']:.2f} -> {results['A*+QIEA']['path_length']:.2f} "
              f"({length_improvement:+.2f}%)")
        print(f"  Cost: {results['A*']['cost']:.2f} -> {results['A*+QIEA']['cost']:.2f} "
              f"({cost_improvement:+.2f}%)")
        print(f"  Computation time: {results['A*']['computation_time']:.4f}s -> "
              f"{results['A*+QIEA']['computation_time']:.4f}s")
        print(f"  Waypoints: {results['A*']['num_waypoints']} -> {results['A*+QIEA']['num_waypoints']}")
    
    # Compare Theta* vs Theta*+QIEA
    if results['Theta*']['success'] and results['Theta*+QIEA']['success']:
        length_improvement = ((results['Theta*']['path_length'] - results['Theta*+QIEA']['path_length']) / 
                             results['Theta*']['path_length'] * 100)
        cost_improvement = ((results['Theta*']['cost'] - results['Theta*+QIEA']['cost']) / 
                           results['Theta*']['cost'] * 100)
        
        print(f"\nTheta* vs Theta*+QIEA:")
        print(f"  Path length: {results['Theta*']['path_length']:.2f} -> {results['Theta*+QIEA']['path_length']:.2f} "
              f"({length_improvement:+.2f}%)")
        print(f"  Cost: {results['Theta*']['cost']:.2f} -> {results['Theta*+QIEA']['cost']:.2f} "
              f"({cost_improvement:+.2f}%)")
        print(f"  Computation time: {results['Theta*']['computation_time']:.4f}s -> "
              f"{results['Theta*+QIEA']['computation_time']:.4f}s")
        print(f"  Waypoints: {results['Theta*']['num_waypoints']} -> {results['Theta*+QIEA']['num_waypoints']}")
    
    print("\n" + "=" * 70)
    print("KEY IMPROVEMENTS:")
    print("=" * 70)
    print("1. Seeding: QIEA population is initialized with path from classical algorithm")
    print("   - First individual: encoded seed path")
    print("   - Next 20%: variations of seed path")
    print("   - Remaining: random initialization")
    print("   - Benefit: Faster convergence, better initial solutions")
    print()
    print("2. Adaptive Rotation Angle:")
    print("   - Decreases over generations (exploration -> exploitation)")
    print("   - Scales with fitness gap (larger gap = larger rotation)")
    print("   - Benefit: Better balance between exploration and exploitation")
    print("=" * 70)


if __name__ == "__main__":
    main()

