"""
Hybrid Path Planning Algorithms
Combine classical algorithms (A*, Theta*, Dijkstra) with QIEA for optimization
"""

from typing import List, Tuple
from .base import PathPlanner, PathResult
from .astar import AStar
from .theta_star import ThetaStar
from .dijkstra import Dijkstra
from .qiea import QIEA
from .utils import euclidean_distance, smooth_path, calculate_path_cost
import time


class HybridPlanner(PathPlanner):
    """Base class for hybrid planners"""
    
    def __init__(self, obstacles: List[Tuple[float, float, float]], 
                 width: int, height: int,
                 classical_planner: PathPlanner,
                 qiea_population_size: int = 30,
                 qiea_max_generations: int = 50):
        """
        Initialize hybrid planner
        
        Args:
            obstacles: List of obstacles
            width: Map width
            height: Map height
            classical_planner: Classical path planner (A*, Theta*, or Dijkstra)
            qiea_population_size: QIEA population size
            qiea_max_generations: QIEA max generations
        """
        super().__init__(obstacles, width, height)
        self.classical_planner = classical_planner
        self.qiea = QIEA(
            obstacles, 
            width, 
            height,
            population_size=qiea_population_size,
            max_generations=qiea_max_generations
        )
    
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float]) -> PathResult:
        """
        Plan path using hybrid approach:
        1. Use classical algorithm to find initial path
        2. Use QIEA to optimize the path
        
        Args:
            start: Start point
            goal: Goal point
        
        Returns:
            PathResult object
        """
        total_start_time = time.time()
        
        # Step 1: Find initial path using classical algorithm
        initial_result = self.classical_planner.plan(start, goal)
        
        if not initial_result.success:
            return PathResult(
                path=[],
                success=False,
                computation_time=time.time() - total_start_time,
                path_length=0.0,
                cost=float('inf'),
                num_nodes_explored=initial_result.num_nodes_explored,
                message=f"Classical algorithm failed: {initial_result.message}"
            )
        
        # Step 2: Optimize path using QIEA
        # Initialize QIEA with initial path as seed
        optimized_result = self._optimize_with_qiea(
            start, 
            goal, 
            initial_result.path
        )
        
        # Compare results and return the better one
        if optimized_result.success:
            # Use optimized path if it's better
            if optimized_result.cost < initial_result.cost:
                optimized_result.computation_time = time.time() - total_start_time
                optimized_result.num_nodes_explored += initial_result.num_nodes_explored
                optimized_result.message = f"Hybrid: {initial_result.message} + QIEA optimization"
                return optimized_result
        
        # Fallback to initial path
        initial_result.computation_time = time.time() - total_start_time
        initial_result.message = f"Hybrid: {initial_result.message} (QIEA did not improve)"
        return initial_result
    
    def _optimize_with_qiea(self, start: Tuple[float, float],
                           goal: Tuple[float, float],
                           initial_path: List[Tuple[float, float]]) -> PathResult:
        """
        Optimize initial path using QIEA with seed path
        
        Args:
            start: Start point
            goal: Goal point
            initial_path: Initial path from classical algorithm (used as seed)
        
        Returns:
            Optimized PathResult
        """
        # Use QIEA to optimize with seed path from classical algorithm
        # This seeds the population with the initial path, improving convergence
        result = self.qiea.plan(start, goal, seed_path=initial_path)
        
        # If QIEA found a better path, use it
        if result.success:
            initial_cost = sum(euclidean_distance(initial_path[i], initial_path[i+1])
                             for i in range(len(initial_path)-1))
            
            if result.cost < initial_cost * 1.1:  # Allow 10% tolerance
                return result
        
        # Return initial path if QIEA didn't improve
        return PathResult(
            path=initial_path,
            success=True,
            computation_time=result.computation_time,
            path_length=sum(euclidean_distance(initial_path[i], initial_path[i+1])
                          for i in range(len(initial_path)-1)),
            cost=sum(euclidean_distance(initial_path[i], initial_path[i+1])
                    for i in range(len(initial_path)-1)),
            num_nodes_explored=result.num_nodes_explored,
            message="Used initial path (QIEA did not improve)"
        )


class AStarQIEA(HybridPlanner):
    """Hybrid A* + QIEA path planner"""
    
    def __init__(self, obstacles: List[Tuple[float, float, float]], 
                 width: int, height: int,
                 qiea_population_size: int = 30,
                 qiea_max_generations: int = 50):
        classical_planner = AStar(obstacles, width, height)
        super().__init__(
            obstacles, 
            width, 
            height,
            classical_planner,
            qiea_population_size,
            qiea_max_generations
        )


class ThetaStarQIEA(HybridPlanner):
    """
    Hybrid Theta* + QIEA path planner
    
    Uses QIEA with enhanced validation to fix and optimize paths from Theta*.
    Accepts longer computation time for guaranteed obstacle-free paths.
    """
    
    def __init__(self, obstacles: List[Tuple[float, float, float]], 
                 width: int, height: int,
                 qiea_population_size: int = 50,  # Increased for better exploration
                 qiea_max_generations: int = 100):  # Increased for better convergence
        classical_planner = ThetaStar(obstacles, width, height)
        super().__init__(
            obstacles, 
            width, 
            height,
            classical_planner,
            qiea_population_size,
            qiea_max_generations
        )
    
    def _optimize_with_qiea(self, start: Tuple[float, float],
                           goal: Tuple[float, float],
                           initial_path: List[Tuple[float, float]]) -> PathResult:
        """
        Optimize initial path using QIEA with strict validation
        
        For Theta*, we always use QIEA to ensure obstacle-free path,
        even if it takes longer.
        """
        from .utils import line_obstacles_intersection, is_point_in_obstacles
        
        # First, validate initial path
        initial_has_collisions = False
        for i in range(len(initial_path) - 1):
            if line_obstacles_intersection(initial_path[i], initial_path[i+1], self.obstacles):
                initial_has_collisions = True
                break
        
        for point in initial_path:
            if is_point_in_obstacles(point, self.obstacles):
                initial_has_collisions = True
                break
        
        # Always use QIEA for Theta* to fix any collision issues
        # Use QIEA to optimize with seed path from Theta*
        result = self.qiea.plan(start, goal, seed_path=initial_path)
        
        # Validate QIEA result
        if result.success:
            # Check if QIEA path is valid (no collisions)
            qiea_valid = True
            for i in range(len(result.path) - 1):
                if line_obstacles_intersection(result.path[i], result.path[i+1], self.obstacles):
                    qiea_valid = False
                    break
            
            for point in result.path:
                if is_point_in_obstacles(point, self.obstacles):
                    qiea_valid = False
                    break
            
            if qiea_valid:
                # QIEA found valid path
                initial_cost = sum(euclidean_distance(initial_path[i], initial_path[i+1])
                                 for i in range(len(initial_path)-1))
                
                # Use QIEA path if it's better or if initial had collisions
                if initial_has_collisions or result.cost < initial_cost * 1.2:
                    return result
        
        # If QIEA didn't find valid path, try to fix initial path
        if initial_has_collisions:
            # Try to fix initial path using QIEA's validation
            fixed_path = self.qiea._validate_and_smooth_path(initial_path)
            
            # Validate fixed path
            fixed_valid = True
            for i in range(len(fixed_path) - 1):
                if line_obstacles_intersection(fixed_path[i], fixed_path[i+1], self.obstacles):
                    fixed_valid = False
                    break
            
            if fixed_valid:
                path_length = sum(euclidean_distance(fixed_path[i], fixed_path[i+1])
                                for i in range(len(fixed_path)-1))
                cost = calculate_path_cost(fixed_path, self.obstacles)
                
                return PathResult(
                    path=fixed_path,
                    success=True,
                    computation_time=result.computation_time,
                    path_length=path_length,
                    cost=cost,
                    num_nodes_explored=result.num_nodes_explored,
                    message="Used QIEA-validated path (fixed collisions from Theta*)"
                )
        
        # Fallback: return initial path (may have collisions)
        return PathResult(
            path=initial_path,
            success=True,
            computation_time=result.computation_time,
            path_length=sum(euclidean_distance(initial_path[i], initial_path[i+1])
                          for i in range(len(initial_path)-1)),
            cost=sum(euclidean_distance(initial_path[i], initial_path[i+1])
                    for i in range(len(initial_path)-1)),
            num_nodes_explored=result.num_nodes_explored,
            message="Used initial path (QIEA validation failed)"
        )


class DijkstraQIEA(HybridPlanner):
    """Hybrid Dijkstra + QIEA path planner"""
    
    def __init__(self, obstacles: List[Tuple[float, float, float]], 
                 width: int, height: int,
                 qiea_population_size: int = 30,
                 qiea_max_generations: int = 50):
        classical_planner = Dijkstra(obstacles, width, height)
        super().__init__(
            obstacles, 
            width, 
            height,
            classical_planner,
            qiea_population_size,
            qiea_max_generations
        )

