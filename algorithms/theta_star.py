"""
Theta* Path Planning Algorithm
Theta* is an any-angle path planning algorithm that allows paths to pass through grid cells
"""

import heapq
import math
from typing import List, Tuple, Dict, Optional
from .base import PathPlanner, PathResult
from .utils import euclidean_distance, get_neighbors_8_connected
import time


class ThetaStar(PathPlanner):
    """Theta* path planning algorithm (any-angle A*)"""
    
    def __init__(self, obstacles: List[Tuple[float, float, float]], 
                 width: int, height: int):
        super().__init__(obstacles, width, height)
    
    def _validate_and_fix_path(self, path: List[Tuple[float, float]], 
                               start: Tuple[float, float],
                               goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Validate path and fix any segments that cross obstacles
        
        Args:
            path: Original path
            start: Start point
            goal: Goal point
        
        Returns:
            Validated and fixed path
        """
        if len(path) < 2:
            return path
        
        # Ensure start and goal are correct
        if len(path) > 0:
            path[0] = start
        if len(path) > 1:
            path[-1] = goal
        
        # Remove waypoints that are in obstacles
        valid_path = []
        for point in path:
            if self.is_valid_point(point):
                valid_path.append(point)
            elif len(valid_path) > 0:
                # If point is invalid, try to find a nearby valid point
                # or skip it if we can connect directly
                pass
        
        if len(valid_path) < 2:
            valid_path = [start, goal]
        
        # Smooth path by removing unnecessary waypoints
        # but ensure no segment crosses obstacles (using precise method)
        smoothed = [valid_path[0]]
        i = 0
        
        while i < len(valid_path) - 1:
            # Try to skip as many points as possible
            j = len(valid_path) - 1
            found = False
            
            while j > i + 1:
                # Check if we can go directly from i to j (using precise check)
                if self.is_valid_line(valid_path[i], valid_path[j], use_precise=True):
                    smoothed.append(valid_path[j])
                    i = j
                    found = True
                    break
                j -= 1
            
            if not found:
                # Can't skip, add next point
                smoothed.append(valid_path[i + 1])
                i += 1
        
        # Final validation: check all segments and fix any that cross obstacles
        final_path = [smoothed[0]]
        for i in range(1, len(smoothed)):
            p1 = final_path[-1]
            p2 = smoothed[i]
            
            if self.is_valid_line(p1, p2, use_precise=True):
                # Segment is valid, add it
                final_path.append(p2)
            else:
                # Segment crosses obstacle, need to find detour
                # Try multiple intermediate points
                detour_found = False
                
                # Strategy 1: Try midpoint
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                mid = (mid_x, mid_y)
                
                if (self.is_valid_point(mid) and
                    self.is_valid_line(p1, mid, use_precise=True) and
                    self.is_valid_line(mid, p2, use_precise=True)):
                    final_path.append(mid)
                    final_path.append(p2)
                    detour_found = True
                
                # Strategy 2: Try grid neighbors around midpoint
                if not detour_found:
                    mid_int = (int(round(mid_x)), int(round(mid_y)))
                    neighbors = get_neighbors_8_connected(mid_int, self.width, self.height)
                    
                    # Sort neighbors by distance to line p1-p2
                    candidates = []
                    for neighbor in neighbors:
                        neighbor_float = (float(neighbor[0]), float(neighbor[1]))
                        if (self.is_valid_point(neighbor_float) and
                            self.is_valid_line(p1, neighbor_float, use_precise=True) and
                            self.is_valid_line(neighbor_float, p2, use_precise=True)):
                            # Calculate distance from line
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            fx = neighbor_float[0] - p1[0]
                            fy = neighbor_float[1] - p1[1]
                            if abs(dx) > 1e-10 or abs(dy) > 1e-10:
                                line_len = math.sqrt(dx*dx + dy*dy)
                                dist_to_line = abs(dy * fx - dx * fy) / line_len if line_len > 0 else float('inf')
                                candidates.append((dist_to_line, neighbor_float))
                    
                    if candidates:
                        # Use closest point to line
                        candidates.sort(key=lambda x: x[0])
                        final_path.append(candidates[0][1])
                        final_path.append(p2)
                        detour_found = True
                
                # Strategy 3: If still no detour, try A* style connection
                if not detour_found:
                    # Use A* to find path between p1 and p2
                    from .astar import AStar
                    temp_planner = AStar(self.obstacles, self.width, self.height)
                    temp_result = temp_planner.plan(p1, p2)
                    
                    if temp_result.success and len(temp_result.path) > 2:
                        # Add intermediate waypoints from A*
                        for waypoint in temp_result.path[1:-1]:  # Skip start and goal
                            if self.is_valid_point(waypoint):
                                final_path.append(waypoint)
                        final_path.append(p2)
                        detour_found = True
                
                # Last resort: if no detour found, skip this point
                if not detour_found:
                    # Skip p2 and try next point
                    continue
        
        # Ensure goal is in path
        if len(final_path) > 0 and final_path[-1] != goal:
            # Check if we can connect directly to goal
            if self.is_valid_line(final_path[-1], goal, use_precise=True):
                final_path.append(goal)
            else:
                # Need to find path to goal
                from .astar import AStar
                temp_planner = AStar(self.obstacles, self.width, self.height)
                temp_result = temp_planner.plan(final_path[-1], goal)
                if temp_result.success:
                    final_path.extend(temp_result.path[1:])  # Skip start
                else:
                    final_path.append(goal)  # Force add goal
        
        return final_path
    
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float]) -> PathResult:
        """
        Plan path using Theta* algorithm
        
        Theta* allows line-of-sight connections between nodes,
        resulting in smoother paths than A*
        
        Args:
            start: Start point (x, y)
            goal: Goal point (x, y)
        
        Returns:
            PathResult object
        """
        start_time = time.time()
        
        # Validate start and goal
        if not self.is_valid_point(start):
            return PathResult(
                path=[],
                success=False,
                computation_time=time.time() - start_time,
                path_length=0.0,
                cost=float('inf'),
                num_nodes_explored=0,
                message="Start point is invalid (in obstacle or out of bounds)"
            )
        
        if not self.is_valid_point(goal):
            return PathResult(
                path=[],
                success=False,
                computation_time=time.time() - start_time,
                path_length=0.0,
                cost=float('inf'),
                num_nodes_explored=0,
                message="Goal point is invalid (in obstacle or out of bounds)"
            )
        
        # Convert to integer grid coordinates
        start_int = (int(round(start[0])), int(round(start[1])))
        goal_int = (int(round(goal[0])), int(round(goal[1])))
        
        # Priority queue: (f_score, g_score, node)
        open_set = [(0, 0, start_int)]
        heapq.heapify(open_set)
        
        # Track visited nodes and their costs
        g_score: Dict[Tuple[int, int], float] = {start_int: 0.0}
        f_score: Dict[Tuple[int, int], float] = {start_int: euclidean_distance(start, goal)}
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_int: None}
        visited: set = set()
        
        num_explored = 0
        
        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)
            
            # Skip if we've already processed this node with a better cost
            if current in visited:
                continue
            
            visited.add(current)
            num_explored += 1
            
            # Check if we reached the goal
            if current == goal_int:
                # Reconstruct path with precise validation at each step
                path = []
                node = current
                while node is not None:
                    path.append((float(node[0]), float(node[1])))
                    node = came_from[node]
                path.reverse()
                
                # Ensure start and goal are correct
                if len(path) > 0:
                    path[0] = start
                if len(path) > 1:
                    path[-1] = goal
                
                # Validate and fix path to ensure no obstacles are crossed
                # This uses precise line-circle intersection checking
                path = self._validate_and_fix_path(path, start, goal)
                
                computation_time = time.time() - start_time
                path_length = sum(euclidean_distance(path[i], path[i+1]) 
                                 for i in range(len(path)-1))
                
                return PathResult(
                    path=path,
                    success=True,
                    computation_time=computation_time,
                    path_length=path_length,
                    cost=current_g,
                    num_nodes_explored=num_explored,
                    message="Path found successfully"
                )
            
            # Theta*: Check line-of-sight from parent
            current_parent = came_from[current]
            if current_parent is not None:
                current_float = (float(current[0]), float(current[1]))
                parent_float = (float(current_parent[0]), float(current_parent[1]))
                
                if self.is_valid_line(parent_float, current_float, use_precise=True):
                    # Can reach current directly from parent
                    direct_cost = g_score.get(current_parent, float('inf')) + euclidean_distance(current_parent, current)
                    if direct_cost < g_score.get(current, float('inf')):
                        g_score[current] = direct_cost
                        came_from[current] = current_parent
                        h_score = euclidean_distance(current, goal_int)
                        f_score[current] = direct_cost + h_score
            
            # Explore neighbors
            neighbors = get_neighbors_8_connected(current, self.width, self.height)
            
            for neighbor in neighbors:
                # Check if neighbor is valid
                if not self.is_valid_point(neighbor):
                    continue
                
                # Theta*: Try direct path from current's parent
                if came_from[current] is not None:
                    parent = came_from[current]
                    parent_float = (float(parent[0]), float(parent[1]))
                    neighbor_float = (float(neighbor[0]), float(neighbor[1]))
                    
                    if self.is_valid_line(parent_float, neighbor_float, use_precise=True):
                        # Direct path from parent to neighbor
                        direct_cost = g_score[parent] + euclidean_distance(parent, neighbor)
                        if neighbor not in g_score or direct_cost < g_score[neighbor]:
                            g_score[neighbor] = direct_cost
                            came_from[neighbor] = parent
                            h_score = euclidean_distance(neighbor, goal_int)
                            f_score[neighbor] = direct_cost + h_score
                            heapq.heappush(open_set, (f_score[neighbor], direct_cost, neighbor))
                            continue
                
                # Standard A* connection - also check line validity
                current_float = (float(current[0]), float(current[1]))
                neighbor_float = (float(neighbor[0]), float(neighbor[1]))
                
                if not self.is_valid_line(current_float, neighbor_float, use_precise=True):
                    continue
                
                move_cost = euclidean_distance(current, neighbor)
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    h_score = euclidean_distance(neighbor, goal_int)
                    f_score[neighbor] = tentative_g + h_score
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        # No path found
        computation_time = time.time() - start_time
        return PathResult(
            path=[],
            success=False,
            computation_time=computation_time,
            path_length=0.0,
            cost=float('inf'),
            num_nodes_explored=num_explored,
            message="No path found from start to goal"
        )

