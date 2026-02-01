"""
Utility functions for path planning algorithms
"""

import numpy as np
from typing import Tuple, List, Set
import math


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def manhattan_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Manhattan distance between two points"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def is_point_in_circle(point: Tuple[float, float], 
                      circle_center: Tuple[float, float], 
                      radius: float) -> bool:
    """
    Check if a point is inside or on the boundary of a circle.
    
    STRICT LOS: Point must be strictly outside circle.
    Distance from point to center must be > radius (strictly greater, not >=).
    Points on boundary (dist == radius) are considered collisions.
    
    Args:
        point: (x, y) coordinates
        circle_center: Center of circle (cx, cy)
        radius: Radius of circle
    
    Returns:
        True if point is on or inside circle (dist <= radius) - collision
        False if point is strictly outside circle (dist > radius) - safe
    """
    dist = euclidean_distance(point, circle_center)
    return dist <= radius  # Strict LOS: dist == radius is also collision


def is_point_in_obstacles(point: Tuple[float, float], 
                          obstacles: List[Tuple[float, float, float]]) -> bool:
    """
    Check if a point collides with any obstacle.
    
    STRICT LOS: Point must be strictly outside all obstacles.
    Distance from point to obstacle center must be > radius for all obstacles.
    Points on boundary (dist == radius) are considered collisions.
    
    Args:
        point: (x, y) coordinates
        obstacles: List of (center_x, center_y, radius)
    
    Returns:
        True if point is on or inside any obstacle (violates Strict LOS)
        False if point is strictly outside all obstacles (safe)
    """
    for cx, cy, radius in obstacles:
        if is_point_in_circle(point, (cx, cy), radius):
            return True
    return False


def line_circle_intersection(p1: Tuple[float, float], 
                             p2: Tuple[float, float],
                             circle_center: Tuple[float, float],
                             radius: float) -> bool:
    """
    Check if a line segment intersects with a circle (classical algorithms version).
    
    STRICT LOS: Line segment must be strictly outside circle.
    Distance from center to line must be > radius (strictly greater, not >=).
    Tangent lines (distance == radius) are considered collisions.
    
    Args:
        p1: Start point of line segment
        p2: End point of line segment
        circle_center: Center of circle
        radius: Radius of circle
    
    Returns:
        True if line segment touches or intersects circle (violates Strict LOS)
    """
    # Vector from p1 to p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Handle degenerate case: p1 == p2 (zero-length segment)
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        # Check if the point is inside or on the circle
        dist = euclidean_distance(p1, circle_center)
        return dist <= radius  # Point on or inside circle = collision
    
    # Check endpoints first - STRICT: if endpoint is on or inside circle, collision
    dist_p1 = euclidean_distance(p1, circle_center)
    dist_p2 = euclidean_distance(p2, circle_center)
    
    # STRICT LOS: endpoint on boundary (dist == radius) is collision
    if dist_p1 <= radius or dist_p2 <= radius:
        return True
    
    # Vector from p1 to circle center
    fx = circle_center[0] - p1[0]
    fy = circle_center[1] - p1[1]
    
    # Calculate distance from circle center to line segment using cross product
    cross_product = abs(dy * fx - dx * fy)
    line_length = math.sqrt(dx * dx + dy * dy)
    
    if line_length < 1e-10:
        return dist_p1 <= radius
    
    distance_to_line = cross_product / line_length
    
    # STRICT LOS: If distance > radius, line is strictly outside (safe)
    if distance_to_line > radius:
        return False
    
    # If distance <= radius, check closest point on segment
    # Parameter t where closest point is: p1 + t * (p2 - p1)
    t = (dx * fx + dy * fy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
    
    # Calculate closest point on line segment
    closest_point = (p1[0] + t * dx, p1[1] + t * dy)
    dist_to_closest = euclidean_distance(circle_center, closest_point)
    
    # STRICT LOS: if distance <= radius (including == radius), it's collision
    return dist_to_closest <= radius


def line_circle_intersection_precise(p1: Tuple[float, float], 
                                     p2: Tuple[float, float],
                                     circle_center: Tuple[float, float],
                                     radius: float) -> bool:
    """
    Check if a line segment intersects with a circle using precise mathematical formula.
    
    STRICT LOS: Line segment must be strictly outside circle.
    Distance from center to line must be > radius (strictly greater, not >=).
    Tangent lines (distance == radius) are considered collisions.
    
    Uses the distance from circle center to line segment method:
    d = |(dy*fx - dx*fy)| / sqrt(dx² + dy²)
    
    This precise version is used for Theta* which requires accurate line-of-sight
    checking to avoid cutting through circular obstacles.
    
    Args:
        p1: Start point of line segment
        p2: End point of line segment
        circle_center: Center of circle
        radius: Radius of circle
    
    Returns:
        True if line segment touches or intersects circle (violates Strict LOS)
    """
    # Vector from p1 to p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Handle degenerate case: p1 == p2 (zero-length segment)
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        # Check if the point is inside or on the circle
        dist = euclidean_distance(p1, circle_center)
        return dist <= radius  # Point on or inside circle = collision
    
    # Check endpoints first - STRICT: if endpoint is on or inside circle, collision
    dist_p1 = euclidean_distance(p1, circle_center)
    dist_p2 = euclidean_distance(p2, circle_center)
    
    # STRICT LOS: endpoint on boundary (dist == radius) is collision
    if dist_p1 <= radius or dist_p2 <= radius:
        return True
    
    # Vector from p1 to circle center
    fx = circle_center[0] - p1[0]
    fy = circle_center[1] - p1[1]
    
    # Calculate distance from circle center to the infinite line
    # Using cross product: |(dy*fx - dx*fy)| / |(dx, dy)|
    cross_product = abs(dy * fx - dx * fy)
    line_length = math.sqrt(dx * dx + dy * dy)
    
    if line_length < 1e-10:
        return dist_p1 <= radius
    
    distance_to_line = cross_product / line_length
    
    # STRICT LOS: If distance > radius, line is strictly outside (safe)
    if distance_to_line > radius:
        return False
    
    # If distance <= radius, find the closest point on the line segment
    # Parameter t where closest point is: p1 + t * (p2 - p1)
    t = (dx * fx + dy * fy) / (dx * dx + dy * dy)
    
    # Clamp t to [0, 1] to ensure the closest point is on the line segment
    t = max(0.0, min(1.0, t))
    
    # Calculate the closest point on the line segment
    closest_point = (p1[0] + t * dx, p1[1] + t * dy)
    
    # Distance from circle center to the closest point on the segment
    dist_to_closest = euclidean_distance(circle_center, closest_point)
    
    # STRICT LOS: if distance <= radius (including == radius), it's collision
    return dist_to_closest <= radius


def line_obstacles_intersection(p1: Tuple[float, float],
                               p2: Tuple[float, float],
                               obstacles: List[Tuple[float, float, float]],
                               use_precise: bool = False) -> bool:
    """
    Check if a line segment intersects with any obstacle
    
    Args:
        p1: Start point
        p2: End point
        obstacles: List of (center_x, center_y, radius)
        use_precise: If True, use precise distance method (for Theta*)
    
    Returns:
        True if line segment intersects any obstacle
    """
    if use_precise:
        for cx, cy, radius in obstacles:
            if line_circle_intersection_precise(p1, p2, (cx, cy), radius):
                return True
    else:
        for cx, cy, radius in obstacles:
            if line_circle_intersection(p1, p2, (cx, cy), radius):
                return True
    return False


def get_neighbors_8_connected(point: Tuple[int, int], 
                              width: int, 
                              height: int) -> List[Tuple[int, int]]:
    """
    Get 8-connected neighbors of a grid point
    
    Returns:
        List of valid neighbor coordinates
    """
    x, y = point
    neighbors = []
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
    
    return neighbors


def get_neighbors_4_connected(point: Tuple[int, int],
                              width: int,
                              height: int) -> List[Tuple[int, int]]:
    """
    Get 4-connected neighbors of a grid point
    
    Returns:
        List of valid neighbor coordinates
    """
    x, y = point
    neighbors = []
    
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbors.append((nx, ny))
    
    return neighbors


def smooth_path(path: List[Tuple[float, float]],
                obstacles: List[Tuple[float, float, float]],
                max_iterations: int = 100) -> List[Tuple[float, float]]:
    """
    Smooth a path by removing unnecessary waypoints
    
    Args:
        path: List of waypoints
        obstacles: List of obstacles
        max_iterations: Maximum iterations for smoothing
    
    Returns:
        Smoothed path
    """
    if len(path) <= 2:
        return path
    
    smoothed = [path[0]]
    i = 0
    
    while i < len(path) - 1:
        # Try to skip as many points as possible
        j = len(path) - 1
        found = False
        
        while j > i + 1:
            if not line_obstacles_intersection(path[i], path[j], obstacles):
                smoothed.append(path[j])
                i = j
                found = True
                break
            j -= 1
        
        if not found:
            smoothed.append(path[i + 1])
            i += 1
    
    return smoothed


def calculate_path_length(path: List[Tuple[float, float]]) -> float:
    """Calculate total length of a path"""
    if len(path) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(path) - 1):
        total_length += euclidean_distance(path[i], path[i + 1])
    
    return total_length


def calculate_path_cost(path: List[Tuple[float, float]],
                       obstacles: List[Tuple[float, float, float]],
                       weight_length: float = 1.0,
                       weight_energy: float = 0.5,
                       weight_safety: float = 2.0) -> float:
    """
    Calculate multi-objective cost of a path
    
    Args:
        path: List of waypoints
        obstacles: List of obstacles
        weight_length: Weight for path length
        weight_energy: Weight for energy consumption (related to turns)
        weight_safety: Weight for safety (distance from obstacles)
    
    Returns:
        Total cost
    """
    if len(path) < 2:
        return float('inf')
    
    # Path length cost
    length_cost = calculate_path_length(path)
    
    # Energy cost (related to number of turns and direction changes)
    energy_cost = 0.0
    for i in range(1, len(path) - 1):
        # Calculate angle change
        v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        
        # Normalize vectors
        len1 = euclidean_distance((0, 0), v1)
        len2 = euclidean_distance((0, 0), v2)
        
        if len1 > 0 and len2 > 0:
            cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
            angle = math.acos(cos_angle)
            energy_cost += angle  # Penalty for turns
    
    # Safety cost (distance from obstacles)
    safety_cost = 0.0
    for point in path:
        min_dist = float('inf')
        for cx, cy, radius in obstacles:
            dist = euclidean_distance(point, (cx, cy)) - radius
            min_dist = min(min_dist, dist)
        
        if min_dist < 0:
            safety_cost += abs(min_dist) * 10  # Heavy penalty for collision
        else:
            # Penalty for being too close
            safety_cost += max(0, 1.0 - min_dist) if min_dist < 1.0 else 0
    
    total_cost = (weight_length * length_cost + 
                  weight_energy * energy_cost + 
                  weight_safety * safety_cost)
    
    return total_cost

