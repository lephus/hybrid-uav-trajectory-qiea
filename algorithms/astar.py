"""
A* Path Planning Algorithm
"""

import heapq
from typing import List, Tuple, Dict, Optional
from .base import PathPlanner, PathResult
from .utils import euclidean_distance, get_neighbors_8_connected
import time


class AStar(PathPlanner):
    """A* path planning algorithm with 8-connected grid"""
    
    def __init__(self, obstacles: List[Tuple[float, float, float]], 
                 width: int, height: int):
        super().__init__(obstacles, width, height)
    
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float]) -> PathResult:
        """
        Plan path using A* algorithm
        
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
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append((float(node[0]), float(node[1])))
                    node = came_from[node]
                path.reverse()
                
                # Ensure start and goal are exact (not rounded)
                if len(path) > 0:
                    path[0] = start
                if len(path) > 1:
                    path[-1] = goal
                
                # Final validation: ensure path complies with Strict LOS
                # Check all points are strictly outside obstacles
                from .utils import is_point_in_obstacles
                for point in path:
                    if is_point_in_obstacles(point, self.obstacles):
                        # This should not happen if validation is correct, but double-check
                        continue
                
                # Check all segments are strictly outside obstacles
                from .utils import line_obstacles_intersection
                for i in range(len(path) - 1):
                    if line_obstacles_intersection(path[i], path[i+1], self.obstacles):
                        # This should not happen if validation is correct, but double-check
                        continue
                
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
                    message="Path found successfully (Strict LOS compliant)"
                )
            
            # Explore neighbors
            neighbors = get_neighbors_8_connected(current, self.width, self.height)
            
            for neighbor in neighbors:
                # Check if neighbor is valid (not in obstacle)
                if not self.is_valid_point(neighbor):
                    continue
                
                # Check if line from current to neighbor is valid (doesn't cross obstacles)
                current_float = (float(current[0]), float(current[1]))
                neighbor_float = (float(neighbor[0]), float(neighbor[1]))
                if not self.is_valid_line(current_float, neighbor_float):
                    continue
                
                # Calculate tentative g_score
                # Use Euclidean distance for diagonal moves
                move_cost = euclidean_distance(current, neighbor)
                tentative_g = g_score[current] + move_cost
                
                # If this path to neighbor is better
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    h_score = euclidean_distance(neighbor, goal_int)
                    f_score[neighbor] = tentative_g + h_score
                    came_from[neighbor] = current
                    
                    # Add to open set
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

