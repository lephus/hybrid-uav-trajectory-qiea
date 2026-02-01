"""
Detailed test to check if A* and Dijkstra paths violate Strict LOS
Checks both points and line segments with high precision
"""

import json
import math
from pathlib import Path
from algorithms import AStar, Dijkstra
from algorithms.utils import (
    euclidean_distance,
    line_circle_intersection,
    is_point_in_circle
)


def check_point_strict_los(point, obstacles, tolerance=1e-6):
    """
    Check if a point violates Strict LOS
    Returns: (is_violation, details)
    """
    for cx, cy, radius in obstacles:
        dist = euclidean_distance(point, (cx, cy))
        if dist <= radius + tolerance:
            return True, {
                'point': point,
                'obstacle_center': (cx, cy),
                'radius': radius,
                'distance': dist,
                'violation': 'on_or_inside' if dist <= radius else 'too_close'
            }
    return False, None


def check_segment_strict_los(p1, p2, obstacles, num_samples=50, tolerance=1e-6):
    """
    Check if a line segment violates Strict LOS by sampling points along it
    Returns: (is_violation, details)
    """
    # Check endpoints
    violation1, details1 = check_point_strict_los(p1, obstacles, tolerance)
    if violation1:
        return True, {'type': 'endpoint', 'point': p1, **details1}
    
    violation2, details2 = check_point_strict_los(p2, obstacles, tolerance)
    if violation2:
        return True, {'type': 'endpoint', 'point': p2, **details2}
    
    # Check line segment intersection
    for cx, cy, radius in obstacles:
        if line_circle_intersection(p1, p2, (cx, cy), radius):
            # Calculate minimum distance from line to circle center
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            fx = cx - p1[0]
            fy = cy - p1[1]
            
            cross_product = abs(dy * fx - dx * fy)
            line_length = math.sqrt(dx * dx + dy * dy)
            
            if line_length > 1e-10:
                distance_to_line = cross_product / line_length
                
                # Check closest point on segment
                t = (dx * fx + dy * fy) / (dx * dx + dy * dy)
                t = max(0.0, min(1.0, t))
                closest_point = (p1[0] + t * dx, p1[1] + t * dy)
                dist_to_closest = euclidean_distance((cx, cy), closest_point)
                
                if dist_to_closest <= radius + tolerance:
                    return True, {
                        'type': 'line_segment',
                        'segment': (p1, p2),
                        'obstacle_center': (cx, cy),
                        'radius': radius,
                        'distance_to_line': distance_to_line,
                        'distance_to_closest': dist_to_closest,
                        'closest_point': closest_point,
                        'violation': 'tangent' if abs(dist_to_closest - radius) < tolerance else 'intersects'
                    }
    
    # Sample points along the segment for extra safety
    for i in range(num_samples + 1):
        t = i / num_samples
        sample_point = (
            p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1])
        )
        violation, details = check_point_strict_los(sample_point, obstacles, tolerance)
        if violation:
            return True, {'type': 'sampled_point', 't': t, **details}
    
    return False, None


def validate_path_strict_los(path, obstacles, algorithm_name):
    """Validate path with Strict LOS checking"""
    if not path or len(path) < 2:
        print(f"  ✗ {algorithm_name}: Invalid path")
        return False
    
    violations = []
    
    # Check all points
    for i, point in enumerate(path):
        violation, details = check_point_strict_los(point, obstacles)
        if violation:
            violations.append({
                'type': 'point',
                'index': i,
                'details': details
            })
    
    # Check all segments
    for i in range(len(path) - 1):
        violation, details = check_segment_strict_los(path[i], path[i+1], obstacles)
        if violation:
            violations.append({
                'type': 'segment',
                'index': i,
                'details': details
            })
    
    if violations:
        print(f"\n  ✗ {algorithm_name}: Found {len(violations)} Strict LOS violation(s):")
        for v in violations[:10]:  # Show first 10
            if v['type'] == 'point':
                d = v['details']
                print(f"    Point {v['index']} {d['point']}: dist={d['distance']:.6f} <= radius={d['radius']:.6f}")
            elif v['type'] == 'segment':
                d = v['details']
                if d['type'] == 'line_segment':
                    print(f"    Segment {v['index']}: dist_to_closest={d['distance_to_closest']:.6f} <= radius={d['radius']:.6f}")
                else:
                    print(f"    Segment {v['index']}: {d['violation']} at t={d.get('t', 0):.3f}")
        if len(violations) > 10:
            print(f"    ... and {len(violations) - 10} more violations")
        return False
    else:
        print(f"  ✓ {algorithm_name}: Path is Strict LOS compliant")
        return True


def main():
    # Test on m1_20x20
    map_file = Path("maps/m1_20x20.json")
    
    if not map_file.exists():
        print(f"Error: Map file {map_file} not found!")
        return
    
    print("Loading map...")
    with open(map_file, 'r') as f:
        data = json.load(f)
    
    obstacles = [tuple(obs) for obs in data['obstacles']]
    start = tuple(data['start'])
    goal = tuple(data['goal'])
    
    print(f"\nMap: {map_file.name}")
    print(f"  Size: {data['width']}x{data['height']}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Start: {start}, Goal: {goal}")
    
    print("\n" + "="*60)
    print("STRICT LOS VALIDATION")
    print("="*60)
    
    # Test A*
    print("\nTesting A*...")
    astar = AStar(obstacles, data['width'], data['height'])
    astar_result = astar.plan(start, goal)
    
    if astar_result.success:
        validate_path_strict_los(astar_result.path, obstacles, "A*")
    else:
        print(f"  ✗ A*: Failed to find path")
    
    # Test Dijkstra
    print("\nTesting Dijkstra...")
    dijkstra = Dijkstra(obstacles, data['width'], data['height'])
    dijkstra_result = dijkstra.plan(start, goal)
    
    if dijkstra_result.success:
        validate_path_strict_los(dijkstra_result.path, obstacles, "Dijkstra")
    else:
        print(f"  ✗ Dijkstra: Failed to find path")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

