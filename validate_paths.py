"""
Validate paths to ensure they don't cross obstacles
"""

import json
import sys
from pathlib import Path
from algorithms import (
    AStar, ThetaStar, Dijkstra,
    AStarQIEA, ThetaStarQIEA, DijkstraQIEA
)
from algorithms.utils import line_obstacles_intersection, is_point_in_obstacles


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


def validate_path(path, obstacles, algorithm_name):
    """Validate a path and report any issues"""
    if not path or len(path) < 2:
        print(f"  ✗ {algorithm_name}: Invalid path (empty or too short)")
        return False
    
    issues = []
    
    # Check each point
    for i, point in enumerate(path):
        if is_point_in_obstacles(point, obstacles):
            issues.append(f"  Point {i} ({point}) is inside an obstacle")
    
    # Check each segment
    for i in range(len(path) - 1):
        if line_obstacles_intersection(path[i], path[i+1], obstacles):
            issues.append(f"  Segment {i}->{i+1} ({path[i]} to {path[i+1]}) crosses an obstacle")
    
    if issues:
        print(f"  ✗ {algorithm_name}: Found {len(issues)} issue(s):")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"    {issue}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more issues")
        return False
    else:
        print(f"  ✓ {algorithm_name}: Path is valid (no obstacles crossed)")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 validate_paths.py <map_file>")
        print("Example: python3 validate_paths.py maps/m1_20x20.json")
        return
    
    map_file = sys.argv[1]
    map_path = Path(map_file)
    
    if not map_path.exists():
        print(f"Error: Map file {map_file} not found!")
        return
    
    print(f"Loading map: {map_path.name}")
    map_data = load_map(str(map_path))
    
    print(f"\nMap Info:")
    print(f"  Size: {map_data['width']}x{map_data['height']}")
    print(f"  Obstacles: {len(map_data['obstacles'])}")
    print(f"  Start: {map_data['start']}, Goal: {map_data['goal']}")
    
    algorithms = {
        'A*': AStar,
        'Theta*': ThetaStar,
        'Dijkstra': Dijkstra,
        'A*+QIEA': AStarQIEA,
        'Theta*+QIEA': ThetaStarQIEA,
        'Dijkstra+QIEA': DijkstraQIEA,
    }
    
    print("\n" + "="*60)
    print("PATH VALIDATION")
    print("="*60)
    
    all_valid = True
    
    for name, planner_class in algorithms.items():
        planner = planner_class(
            map_data['obstacles'],
            map_data['width'],
            map_data['height']
        )
        result = planner.plan(map_data['start'], map_data['goal'])
        
        if result.success:
            is_valid = validate_path(result.path, map_data['obstacles'], name)
            if not is_valid:
                all_valid = False
        else:
            print(f"  ✗ {name}: Failed to find path - {result.message}")
            all_valid = False
    
    print("\n" + "="*60)
    if all_valid:
        print("✓ All paths are valid!")
    else:
        print("✗ Some paths have issues!")
    print("="*60)


if __name__ == "__main__":
    main()

