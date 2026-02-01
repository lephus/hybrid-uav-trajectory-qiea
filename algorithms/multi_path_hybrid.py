"""
Hybrid Multi-Path Planners
Combine classical algorithms with single-path QIEA for individual UAV optimization
Simplified approach: plan each UAV separately to ensure obstacle-free paths
"""

from typing import List, Tuple, Dict, Optional
from .base import MultiPathResult
from .multi_path_base import (
    MultiPathBasePlanner,
    MultiPathAStar,
    MultiPathThetaStar,
    MultiPathDijkstra
)
from .qiea import QIEA
from .astar import AStar
from .theta_star import ThetaStar
from .dijkstra import Dijkstra
from .utils import euclidean_distance, line_obstacles_intersection, is_point_in_obstacles
import time


class MultiPathHybridPlanner:
    """
    Base class for hybrid multi-path planners
    
    Simplified approach: Plan each UAV separately using single-path QIEA
    This ensures obstacle-free paths and is more reliable
    """
    
    def __init__(self,
                 obstacles: List[Tuple[float, float, float]],
                 width: int,
                 height: int,
                 num_uavs: int,
                 base_planner: MultiPathBasePlanner,
                 classical_planner_class,  # Single-path planner class (AStar, ThetaStar, Dijkstra)
                 qiea_population_size: int = 50,
                 qiea_max_generations: int = 100,
                 uav_speed: float = 1.0,
                 min_separation: float = 2.0):
        """
        Args:
            obstacles: Static obstacles
            width, height: Map dimensions
            num_uavs: Number of UAVs
            base_planner: Base multi-path planner (for initial paths)
            classical_planner_class: Single-path planner class (AStar, ThetaStar, or Dijkstra)
            qiea_population_size: QIEA population size
            qiea_max_generations: QIEA max generations
            uav_speed: Constant speed of UAVs
            min_separation: Minimum separation between UAVs
        """
        self.obstacles = obstacles
        self.width = width
        self.height = height
        self.num_uavs = num_uavs
        self.base_planner = base_planner
        self.uav_speed = uav_speed
        self.min_separation = min_separation
        
        # Initialize single-path QIEA for individual UAV optimization
        self.qiea = QIEA(
            obstacles,
            width,
            height,
            population_size=qiea_population_size,
            max_generations=qiea_max_generations
        )
        
        # Initialize single-path classical planner
        self.classical_planner = classical_planner_class(obstacles, width, height)
    
    def _validate_path(self, path: List[Tuple[float, float]]) -> bool:
        """Validate that path is obstacle-free"""
        if len(path) < 2:
            return False
        
        # Check all points are valid
        for point in path:
            if not self.qiea.is_valid_point(point):
                return False
        
        # Check all segments are valid
        for i in range(len(path) - 1):
            if not self.qiea.is_valid_line(path[i], path[i+1]):
                return False
        
        return True
    
    def plan(self,
            starts: List[Tuple[float, float]],
            goal: Tuple[float, float],
            priorities: Optional[List[int]] = None) -> MultiPathResult:
        """
        Plan paths using simplified hybrid approach:
        1. Plan each UAV separately using classical algorithm
        2. Optimize each path individually with single-path QIEA
        3. Validate all paths are obstacle-free
        
        Args:
            starts: List of start positions
            goal: Common goal position
            priorities: Optional priority list
        
        Returns:
            Optimized MultiPathResult
        """
        total_start_time = time.time()
        
        if len(starts) != self.num_uavs:
            return MultiPathResult(
                paths={},
                timestamps={},
                success=False,
                computation_time=time.time() - total_start_time,
                total_cost=float('inf'),
                makespan=0,
                conflicts=[],
                message=f"Mismatch: {len(starts)} starts but {self.num_uavs} UAVs"
            )
        
        optimized_paths = {}
        total_path_length = 0.0
        
        # Plan each UAV separately
        for uav_id in range(self.num_uavs):
            start = starts[uav_id]
            
            # Step 1: Get initial path from classical algorithm
            initial_result = self.classical_planner.plan(start, goal)
            
            if not initial_result.success:
                # If classical fails, try base planner's path for this UAV
                base_result = self.base_planner.plan(starts, goal, priorities)
                if uav_id in base_result.paths:
                    optimized_paths[uav_id] = base_result.paths[uav_id]
                    continue
                else:
                    # Skip this UAV
                    continue
            
            # Step 2: Optimize with single-path QIEA
            qiea_result = self.qiea.plan(start, goal, seed_path=initial_result.path)
            
            # Step 3: Validate QIEA path is obstacle-free
            if qiea_result.success and self._validate_path(qiea_result.path):
                # Use QIEA path if it's valid and better
                initial_length = sum(
                    euclidean_distance(initial_result.path[i], initial_result.path[i+1])
                    for i in range(len(initial_result.path) - 1)
                )
                
                if qiea_result.cost < initial_length * 1.1:  # Allow 10% tolerance
                    optimized_paths[uav_id] = qiea_result.path
                    total_path_length += qiea_result.cost
                else:
                    # Use initial path if QIEA didn't improve much
                    optimized_paths[uav_id] = initial_result.path
                    total_path_length += initial_length
            else:
                # QIEA path is invalid, use initial path
                optimized_paths[uav_id] = initial_result.path
                initial_length = sum(
                    euclidean_distance(initial_result.path[i], initial_result.path[i+1])
                    for i in range(len(initial_result.path) - 1)
                )
                total_path_length += initial_length
        
        if len(optimized_paths) == 0:
            return MultiPathResult(
                paths={},
                timestamps={},
                success=False,
                computation_time=time.time() - total_start_time,
                total_cost=float('inf'),
                makespan=0,
                conflicts=[],
                message="Failed to plan paths for any UAV"
            )
        
        # Calculate timestamps
        timestamps = {}
        for uav_id, path in optimized_paths.items():
            ts = [0.0]
            current_time = 0.0
            for i in range(len(path) - 1):
                segment_time = euclidean_distance(path[i], path[i+1]) / self.uav_speed
                current_time += segment_time
                ts.append(current_time)
            timestamps[uav_id] = ts
        
        makespan = max(
            timestamps[uav_id][-1] if timestamps[uav_id] else 0.0
            for uav_id in timestamps
        )
        
        return MultiPathResult(
            paths=optimized_paths,
            timestamps=timestamps,
            success=len(optimized_paths) > 0,
            computation_time=time.time() - total_start_time,
            total_cost=total_path_length,
            makespan=makespan,
            conflicts=[],  # Conflicts ignored as per user request
            message=f"Hybrid: Planned {len(optimized_paths)}/{self.num_uavs} UAVs individually"
        )


class MultiPathAStarQIEA(MultiPathHybridPlanner):
    """Hybrid A* + QIEA planner (simplified: plan each UAV separately)"""
    
    def __init__(self,
                 obstacles: List[Tuple[float, float, float]],
                 width: int,
                 height: int,
                 num_uavs: int,
                 qiea_population_size: int = 50,
                 qiea_max_generations: int = 100,
                 uav_speed: float = 1.0,
                 min_separation: float = 2.0):
        base_planner = MultiPathAStar(obstacles, width, height, uav_speed)
        super().__init__(
            obstacles,
            width,
            height,
            num_uavs,
            base_planner,
            AStar,  # Single-path classical planner
            qiea_population_size,
            qiea_max_generations,
            uav_speed,
            min_separation
        )


class MultiPathThetaStarQIEA(MultiPathHybridPlanner):
    """Hybrid Theta* + QIEA planner (simplified: plan each UAV separately)"""
    
    def __init__(self,
                 obstacles: List[Tuple[float, float, float]],
                 width: int,
                 height: int,
                 num_uavs: int,
                 qiea_population_size: int = 50,
                 qiea_max_generations: int = 100,
                 uav_speed: float = 1.0,
                 min_separation: float = 2.0):
        base_planner = MultiPathThetaStar(obstacles, width, height, uav_speed)
        super().__init__(
            obstacles,
            width,
            height,
            num_uavs,
            base_planner,
            ThetaStar,  # Single-path classical planner
            qiea_population_size,
            qiea_max_generations,
            uav_speed,
            min_separation
        )


class MultiPathDijkstraQIEA(MultiPathHybridPlanner):
    """Hybrid Dijkstra + QIEA planner (simplified: plan each UAV separately)"""
    
    def __init__(self,
                 obstacles: List[Tuple[float, float, float]],
                 width: int,
                 height: int,
                 num_uavs: int,
                 qiea_population_size: int = 50,
                 qiea_max_generations: int = 100,
                 uav_speed: float = 1.0,
                 min_separation: float = 2.0):
        base_planner = MultiPathDijkstra(obstacles, width, height, uav_speed)
        super().__init__(
            obstacles,
            width,
            height,
            num_uavs,
            base_planner,
            Dijkstra,  # Single-path classical planner
            qiea_population_size,
            qiea_max_generations,
            uav_speed,
            min_separation
        )
