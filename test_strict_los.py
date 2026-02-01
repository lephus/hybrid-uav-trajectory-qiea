"""
Test script to verify Strict LOS implementation
Strict LOS: 
1. Distance from circle center to line must be > radius (strictly greater)
2. Distance from point to circle center must be > radius (strictly greater)
Tangent lines (distance == radius) and points on boundary (dist == radius) should be considered collisions.
"""

import math
from algorithms.utils import (
    line_circle_intersection,
    line_circle_intersection_precise,
    is_point_in_circle,
    is_point_in_obstacles,
    euclidean_distance
)


def test_strict_los():
    """Test Strict LOS implementation"""
    print("Testing Strict LOS Implementation")
    print("=" * 60)
    
    # Test case 1: Line tangent to circle (distance == radius) - should be collision
    print("\nTest 1: Line tangent to circle (distance == radius)")
    circle_center = (5.0, 5.0)
    radius = 2.0
    
    # Create a line that is tangent to the circle
    # Point on circle: (5, 7) - distance from center = 2 = radius
    # Tangent line: horizontal line at y = 7
    p1 = (0.0, 7.0)
    p2 = (10.0, 7.0)
    
    result1 = line_circle_intersection(p1, p2, circle_center, radius)
    result1_precise = line_circle_intersection_precise(p1, p2, circle_center, radius)
    
    # Calculate actual distance
    # Distance from (5,5) to line y=7 is |7-5| = 2 = radius
    distance = abs(7.0 - 5.0)
    
    print(f"  Circle center: {circle_center}, radius: {radius}")
    print(f"  Line: {p1} to {p2}")
    print(f"  Distance from center to line: {distance:.6f} (should be == {radius})")
    print(f"  line_circle_intersection result: {result1} (should be True - collision)")
    print(f"  line_circle_intersection_precise result: {result1_precise} (should be True - collision)")
    
    assert result1 == True, "Tangent line should be collision (Strict LOS)"
    assert result1_precise == True, "Tangent line should be collision (Strict LOS precise)"
    print("  ✓ PASSED: Tangent line correctly identified as collision")
    
    # Test case 2: Line strictly outside circle (distance > radius) - should be safe
    print("\nTest 2: Line strictly outside circle (distance > radius)")
    # Line at y = 8, distance from center (5,5) = 3 > radius (2)
    p1 = (0.0, 8.0)
    p2 = (10.0, 8.0)
    
    result2 = line_circle_intersection(p1, p2, circle_center, radius)
    result2_precise = line_circle_intersection_precise(p1, p2, circle_center, radius)
    
    distance = abs(8.0 - 5.0)
    print(f"  Line: {p1} to {p2}")
    print(f"  Distance from center to line: {distance:.6f} (should be > {radius})")
    print(f"  line_circle_intersection result: {result2} (should be False - safe)")
    print(f"  line_circle_intersection_precise result: {result2_precise} (should be False - safe)")
    
    assert result2 == False, "Line outside circle should be safe (Strict LOS)"
    assert result2_precise == False, "Line outside circle should be safe (Strict LOS precise)"
    print("  ✓ PASSED: Line outside circle correctly identified as safe")
    
    # Test case 3: Line inside circle (distance < radius) - should be collision
    print("\nTest 3: Line inside circle (distance < radius)")
    # Line at y = 6, distance from center (5,5) = 1 < radius (2)
    p1 = (0.0, 6.0)
    p2 = (10.0, 6.0)
    
    result3 = line_circle_intersection(p1, p2, circle_center, radius)
    result3_precise = line_circle_intersection_precise(p1, p2, circle_center, radius)
    
    distance = abs(6.0 - 5.0)
    print(f"  Line: {p1} to {p2}")
    print(f"  Distance from center to line: {distance:.6f} (should be < {radius})")
    print(f"  line_circle_intersection result: {result3} (should be True - collision)")
    print(f"  line_circle_intersection_precise result: {result3_precise} (should be True - collision)")
    
    assert result3 == True, "Line inside circle should be collision"
    assert result3_precise == True, "Line inside circle should be collision"
    print("  ✓ PASSED: Line inside circle correctly identified as collision")
    
    # Test case 4: Endpoint on circle boundary (distance == radius) - should be collision
    print("\nTest 4: Endpoint on circle boundary (distance == radius)")
    # Endpoint at (5, 7) which is on circle boundary
    p1 = (0.0, 5.0)
    p2 = (5.0, 7.0)  # This point is on circle: dist = sqrt((5-5)^2 + (7-5)^2) = 2 = radius
    
    result4 = line_circle_intersection(p1, p2, circle_center, radius)
    result4_precise = line_circle_intersection_precise(p1, p2, circle_center, radius)
    
    dist_p2 = euclidean_distance(p2, circle_center)
    print(f"  Line: {p1} to {p2}")
    print(f"  Endpoint p2 distance from center: {dist_p2:.6f} (should be == {radius})")
    print(f"  line_circle_intersection result: {result4} (should be True - collision)")
    print(f"  line_circle_intersection_precise result: {result4_precise} (should be True - collision)")
    
    assert result4 == True, "Endpoint on boundary should be collision (Strict LOS)"
    assert result4_precise == True, "Endpoint on boundary should be collision (Strict LOS precise)"
    print("  ✓ PASSED: Endpoint on boundary correctly identified as collision")
    
    # Test case 5: Endpoint strictly outside (distance > radius) - should be safe if line is also safe
    print("\nTest 5: Endpoint strictly outside circle")
    # Endpoint at (5, 8) which is outside circle: dist = sqrt((5-5)^2 + (8-5)^2) = 3 > radius
    p1 = (0.0, 5.0)
    p2 = (5.0, 8.0)
    
    result5 = line_circle_intersection(p1, p2, circle_center, radius)
    result5_precise = line_circle_intersection_precise(p1, p2, circle_center, radius)
    
    dist_p2 = euclidean_distance(p2, circle_center)
    print(f"  Line: {p1} to {p2}")
    print(f"  Endpoint p2 distance from center: {dist_p2:.6f} (should be > {radius})")
    print(f"  line_circle_intersection result: {result5} (should be False - safe)")
    print(f"  line_circle_intersection_precise result: {result5_precise} (should be False - safe)")
    
    assert result5 == False, "Endpoint outside circle should be safe if line is safe"
    assert result5_precise == False, "Endpoint outside circle should be safe if line is safe"
    print("  ✓ PASSED: Endpoint outside circle correctly identified as safe")
    
    # Test case 6: Point on circle boundary (distance == radius) - should be collision
    print("\nTest 6: Point on circle boundary (distance == radius)")
    circle_center = (5.0, 5.0)
    radius = 2.0
    
    # Point at (5, 7) which is on circle boundary: dist = sqrt((5-5)^2 + (7-5)^2) = 2 = radius
    point_on_boundary = (5.0, 7.0)
    
    result6 = is_point_in_circle(point_on_boundary, circle_center, radius)
    dist = euclidean_distance(point_on_boundary, circle_center)
    
    print(f"  Circle center: {circle_center}, radius: {radius}")
    print(f"  Point: {point_on_boundary}")
    print(f"  Distance from point to center: {dist:.6f} (should be == {radius})")
    print(f"  is_point_in_circle result: {result6} (should be True - collision)")
    
    assert result6 == True, "Point on boundary should be collision (Strict LOS)"
    print("  ✓ PASSED: Point on boundary correctly identified as collision")
    
    # Test case 7: Point strictly outside circle (distance > radius) - should be safe
    print("\nTest 7: Point strictly outside circle (distance > radius)")
    point_outside = (5.0, 8.0)  # dist = sqrt((5-5)^2 + (8-5)^2) = 3 > radius
    
    result7 = is_point_in_circle(point_outside, circle_center, radius)
    dist = euclidean_distance(point_outside, circle_center)
    
    print(f"  Point: {point_outside}")
    print(f"  Distance from point to center: {dist:.6f} (should be > {radius})")
    print(f"  is_point_in_circle result: {result7} (should be False - safe)")
    
    assert result7 == False, "Point outside circle should be safe (Strict LOS)"
    print("  ✓ PASSED: Point outside circle correctly identified as safe")
    
    # Test case 8: Point inside circle (distance < radius) - should be collision
    print("\nTest 8: Point inside circle (distance < radius)")
    point_inside = (5.0, 6.0)  # dist = sqrt((5-5)^2 + (6-5)^2) = 1 < radius
    
    result8 = is_point_in_circle(point_inside, circle_center, radius)
    dist = euclidean_distance(point_inside, circle_center)
    
    print(f"  Point: {point_inside}")
    print(f"  Distance from point to center: {dist:.6f} (should be < {radius})")
    print(f"  is_point_in_circle result: {result8} (should be True - collision)")
    
    assert result8 == True, "Point inside circle should be collision"
    print("  ✓ PASSED: Point inside circle correctly identified as collision")
    
    # Test case 9: Multiple obstacles - point must be outside all
    print("\nTest 9: Multiple obstacles - point must be outside all")
    obstacles = [
        (5.0, 5.0, 2.0),  # Circle 1
        (10.0, 10.0, 1.5),  # Circle 2
    ]
    
    # Point outside circle 1 but on boundary of circle 2
    point_test = (10.0, 11.5)  # dist to circle 2 = 1.5 = radius (on boundary)
    
    result9 = is_point_in_obstacles(point_test, obstacles)
    dist_to_circle1 = euclidean_distance(point_test, obstacles[0][:2])
    dist_to_circle2 = euclidean_distance(point_test, obstacles[1][:2])
    
    print(f"  Obstacles: {obstacles}")
    print(f"  Point: {point_test}")
    print(f"  Distance to circle 1: {dist_to_circle1:.6f}, radius: {obstacles[0][2]}")
    print(f"  Distance to circle 2: {dist_to_circle2:.6f}, radius: {obstacles[1][2]} (on boundary)")
    print(f"  is_point_in_obstacles result: {result9} (should be True - collision)")
    
    assert result9 == True, "Point on boundary of any obstacle should be collision (Strict LOS)"
    print("  ✓ PASSED: Point on boundary of any obstacle correctly identified as collision")
    
    print("\n" + "=" * 60)
    print("All Strict LOS tests PASSED! ✓")
    print("Strict LOS ensures:")
    print("  - Lines must be strictly outside circles (distance > radius)")
    print("  - Points must be strictly outside circles (distance > radius)")
    print("  - Tangent lines and boundary points are considered collisions")
    print("=" * 60)


if __name__ == "__main__":
    test_strict_los()

