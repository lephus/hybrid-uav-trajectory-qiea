"""
Test Strict LOS on all maps (m1, m2, m3) with different sizes
"""

import json
from pathlib import Path
from algorithms import AStar, Dijkstra
from test_strict_los_detailed import validate_path_strict_los


def test_map(map_file):
    """Test a single map"""
    print(f"\n{'='*60}")
    print(f"Testing: {map_file.name}")
    print(f"{'='*60}")
    
    with open(map_file, 'r') as f:
        data = json.load(f)
    
    obstacles = [tuple(obs) for obs in data['obstacles']]
    start = tuple(data['start'])
    goal = tuple(data['goal'])
    
    print(f"  Size: {data['width']}x{data['height']}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Start: {start}, Goal: {goal}")
    
    # Test A*
    print("\n  Testing A*...")
    astar = AStar(obstacles, data['width'], data['height'])
    astar_result = astar.plan(start, goal)
    
    if astar_result.success:
        is_valid = validate_path_strict_los(astar_result.path, obstacles, "A*")
        if not is_valid:
            print(f"    Path length: {astar_result.path_length:.4f}")
            print(f"    Waypoints: {len(astar_result.path)}")
    else:
        print(f"    ✗ A*: Failed to find path")
        is_valid = False
    
    # Test Dijkstra
    print("\n  Testing Dijkstra...")
    dijkstra = Dijkstra(obstacles, data['width'], data['height'])
    dijkstra_result = dijkstra.plan(start, goal)
    
    if dijkstra_result.success:
        is_valid_d = validate_path_strict_los(dijkstra_result.path, obstacles, "Dijkstra")
        if not is_valid_d:
            print(f"    Path length: {dijkstra_result.path_length:.4f}")
            print(f"    Waypoints: {len(dijkstra_result.path)}")
    else:
        print(f"    ✗ Dijkstra: Failed to find path")
        is_valid_d = False
    
    return is_valid and is_valid_d


def main():
    maps_dir = Path("maps")
    
    if not maps_dir.exists():
        print("Error: maps directory not found!")
        return
    
    # Find all map files
    map_files = sorted(maps_dir.glob("m*.json"))
    
    if not map_files:
        print("No map files found!")
        return
    
    print("STRICT LOS VALIDATION ON ALL MAPS")
    print("="*60)
    
    results = {}
    
    for map_file in map_files:
        try:
            is_valid = test_map(map_file)
            results[map_file.name] = is_valid
        except Exception as e:
            print(f"\n  ✗ Error testing {map_file.name}: {e}")
            results[map_file.name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_valid = True
    for map_name, is_valid in results.items():
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"  {status}: {map_name}")
        if not is_valid:
            all_valid = False
    
    print("\n" + "="*60)
    if all_valid:
        print("✓ All maps: Strict LOS compliant")
    else:
        print("✗ Some maps have Strict LOS violations")
    print("="*60)


if __name__ == "__main__":
    main()

