"""
Base classes for path planning algorithms
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class PathResult:
    """Result of path planning algorithm"""
    path: List[Tuple[float, float]]
    success: bool
    computation_time: float
    path_length: float
    cost: float
    num_nodes_explored: int
    message: str = ""
    
    def __post_init__(self):
        """Calculate path length if not provided"""
        if self.path_length == 0 and len(self.path) > 1:
            from .utils import calculate_path_length
            self.path_length = calculate_path_length(self.path)


@dataclass
class MultiPathResult:
    """Result of multi-UAV path planning algorithm"""
    paths: Dict[int, List[Tuple[float, float]]]  # uav_id -> path
    timestamps: Dict[int, List[float]]  # uav_id -> timestamps for each waypoint
    success: bool
    computation_time: float
    total_cost: float  # Combined fitness/cost
    makespan: float  # Time for all UAVs to complete
    conflicts: List[Tuple[int, int, float, Tuple[float, float]]]  # (uav1_id, uav2_id, time, location)
    message: str = ""
    
    def __post_init__(self):
        """Calculate makespan if not provided"""
        if self.makespan == 0 and self.timestamps:
            self.makespan = max(
                (ts[-1] if ts else 0.0) 
                for ts in self.timestamps.values()
            )


class PathPlanner(ABC):
    """Base class for all path planning algorithms"""
    
    def __init__(self, obstacles: List[Tuple[float, float, float]], 
                 width: int, height: int):
        """
        Initialize path planner
        
        Args:
            obstacles: List of (center_x, center_y, radius)
            width: Map width
            height: Map height
        """
        self.obstacles = obstacles
        self.width = width
        self.height = height
    
    @abstractmethod
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float]) -> PathResult:
        """
        Plan a path from start to goal
        
        Args:
            start: Start point (x, y)
            goal: Goal point (x, y)
        
        Returns:
            PathResult object
        """
        pass
    
    def is_valid_point(self, point: Tuple[float, float]) -> bool:
        """Check if a point is valid (not in obstacles)"""
        from .utils import is_point_in_obstacles
        x, y = point
        
        # Check bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        
        # Check obstacles
        return not is_point_in_obstacles(point, self.obstacles)
    
    def is_valid_line(self, p1: Tuple[float, float], 
                     p2: Tuple[float, float],
                     use_precise: bool = False) -> bool:
        """
        Check if a line segment is valid (doesn't intersect obstacles)
        
        Args:
            p1: Start point
            p2: End point
            use_precise: If True, use precise distance method (for Theta*)
        """
        from .utils import line_obstacles_intersection
        return not line_obstacles_intersection(p1, p2, self.obstacles, use_precise=use_precise)

