"""
Base Multi-Path Planners
Plan individual paths using classical algorithms and coordinate them with time-based scheduling
"""

from typing import List, Tuple, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import PathPlanner, MultiPathResult, PathResult
from .astar import AStar
from .theta_star import ThetaStar
from .dijkstra import Dijkstra
from .utils import euclidean_distance
import time


class MultiPathBasePlanner:
    """
    Base class for multi-path planners using classical algorithms
    
    Plans individual paths in parallel, then coordinates them with time-based scheduling
    """
    
    def __init__(self,
                 obstacles: List[Tuple[float, float, float]],
                 width: int,
                 height: int,
                 base_planner: PathPlanner,
                 uav_speed: float = 1.0,
                 min_goal_spacing: float = 1.5):
        """
        Args:
            obstacles: Static obstacles
            width, height: Map dimensions
            base_planner: Classical path planner (A*, Theta*, or Dijkstra)
            uav_speed: Constant speed of UAVs
            min_goal_spacing: Minimum time gap between arrivals at goal
        """
        self.obstacles = obstacles
        self.width = width
        self.height = height
        self.base_planner = base_planner
        self.uav_speed = uav_speed
        self.min_goal_spacing = min_goal_spacing
    
    def plan(self,
            starts: List[Tuple[float, float]],
            goal: Tuple[float, float],
            priorities: Optional[List[int]] = None) -> MultiPathResult:
        """
        Plan paths for n UAVs to 1 goal
        
        Args:
            starts: List of start positions
            goal: Common goal position
            priorities: Optional priority list (higher = more priority)
        
        Returns:
            MultiPathResult with coordinated paths
        """
        start_time = time.time()
        num_uavs = len(starts)
        
        if priorities is None:
            # Default: priority based on distance to goal (closer = higher priority)
            distances = [euclidean_distance(start, goal) for start in starts]
            priorities = self._distance_based_priorities(distances)
        
        # Step 1: Plan individual paths in parallel
        individual_paths = self._parallel_plan_paths(starts, goal, num_uavs)
        
        if not all(path.success for path in individual_paths.values()):
            failed = [i for i, p in individual_paths.items() if not p.success]
            return MultiPathResult(
                paths={},
                timestamps={},
                success=False,
                computation_time=time.time() - start_time,
                total_cost=float('inf'),
                makespan=0,
                conflicts=[],
                message=f"Failed to plan paths for UAVs: {failed}"
            )
        
        # Step 2: Coordinate paths with time-based scheduling
        coordinated_paths, timestamps = self._coordinate_paths(
            individual_paths, priorities, goal
        )
        
        # Step 3: Detect conflicts
        conflicts = self._detect_conflicts(coordinated_paths, timestamps)
        
        # Calculate metrics
        makespan = max(
            timestamps[uav_id][-1] if timestamps[uav_id] else 0.0
            for uav_id in timestamps
        )
        total_cost = sum(
            sum(euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1))
            for path in coordinated_paths.values()
        )
        
        # Success if all paths were found (conflicts can be resolved later)
        # Conflicts are reported but don't make the result a failure
        has_paths = len(coordinated_paths) == num_uavs and all(len(path) > 0 for path in coordinated_paths.values())
        conflict_msg = f" ({len(conflicts)} conflicts detected)" if conflicts else ""
        
        return MultiPathResult(
            paths=coordinated_paths,
            timestamps=timestamps,
            success=has_paths,  # Success if paths found, conflicts are separate concern
            computation_time=time.time() - start_time,
            total_cost=total_cost,
            makespan=makespan,
            conflicts=conflicts,
            message=f"Planned {num_uavs} UAVs to single goal{conflict_msg}"
        )
    
    def _parallel_plan_paths(self,
                             starts: List[Tuple[float, float]],
                             goal: Tuple[float, float],
                             num_uavs: int) -> Dict[int, PathResult]:
        """
        Plan individual paths in parallel using ThreadPoolExecutor
        """
        individual_paths = {}
        
        # Parallel planning
        with ThreadPoolExecutor(max_workers=min(num_uavs, 8)) as executor:
            future_to_uav = {
                executor.submit(self.base_planner.plan, starts[i], goal): i
                for i in range(num_uavs)
            }
            
            for future in as_completed(future_to_uav):
                uav_id = future_to_uav[future]
                try:
                    result = future.result()
                    individual_paths[uav_id] = result
                except Exception as e:
                    # Fallback: plan sequentially if parallel fails
                    individual_paths[uav_id] = self.base_planner.plan(
                        starts[uav_id], goal
                    )
        
        return individual_paths
    
    def _coordinate_paths(self,
                         individual_paths: Dict[int, PathResult],
                         priorities: List[int],
                         goal: Tuple[float, float]) -> Tuple[Dict[int, List[Tuple[float, float]]], Dict[int, List[float]]]:
        """
        Coordinate paths with time-based scheduling
        
        Strategy:
        1. Sort UAVs by priority (higher priority = start earlier)
        2. Assign timestamps to avoid conflicts at goal
        3. Space arrivals at goal
        """
        coordinated = {}
        timestamps = {}
        
        # Calculate path lengths and base arrival times
        path_info = {}
        for uav_id, result in individual_paths.items():
            path_length = result.path_length
            estimated_time = path_length / self.uav_speed
            path_info[uav_id] = {
                'path': result.path,
                'length': path_length,
                'time': estimated_time
            }
        
        # Sort by priority (descending)
        sorted_uavs = sorted(
            path_info.keys(),
            key=lambda x: (-priorities[x], path_info[x]['time'])
        )
        
        # Assign start times and timestamps
        goal_arrival_queue = []
        
        for uav_id in sorted_uavs:
            info = path_info[uav_id]
            path = info['path']
            path_length = info['length']
            
            # Base travel time
            base_time = path_length / self.uav_speed
            
            # Check goal arrival conflicts
            if goal_arrival_queue:
                last_arrival = max(goal_arrival_queue)
                required_start_delay = max(0, last_arrival + self.min_goal_spacing - base_time)
            else:
                required_start_delay = 0.0
            
            # Priority-based adjustment: higher priority starts earlier
            priority_delay = -priorities[uav_id] * 0.1  # Negative = earlier
            start_delay = max(0, required_start_delay + priority_delay)
            
            # Generate timed waypoints
            ts = []
            current_time = start_delay
            cumulative_distance = 0.0
            
            for i, pos in enumerate(path):
                if i > 0:
                    segment_time = euclidean_distance(path[i-1], path[i]) / self.uav_speed
                    current_time += segment_time
                    cumulative_distance += euclidean_distance(path[i-1], path[i])
                
                ts.append(current_time)
            
            arrival_time = current_time
            goal_arrival_queue.append(arrival_time)
            
            coordinated[uav_id] = path
            timestamps[uav_id] = ts
        
        return coordinated, timestamps
    
    def _detect_conflicts(self,
                         paths: Dict[int, List[Tuple[float, float]]],
                         timestamps: Dict[int, List[float]]) -> List[Tuple[int, int, float, Tuple[float, float]]]:
        """
        Detect conflicts between paths
        
        Conflict = 2 UAVs too close (< min_separation) at same time
        """
        conflicts = []
        min_separation = 2.0  # Default minimum separation
        uav_ids = list(paths.keys())
        
        # Check all pairs
        for i in range(len(uav_ids)):
            for j in range(i+1, len(uav_ids)):
                uav1_id = uav_ids[i]
                uav2_id = uav_ids[j]
                
                path1 = paths[uav1_id]
                path2 = paths[uav2_id]
                ts1 = timestamps[uav1_id]
                ts2 = timestamps[uav2_id]
                
                # Check conflicts with time sampling
                time_step = 0.1  # Check every 0.1s
                min_time = min(ts1[0] if ts1 else 0, ts2[0] if ts2 else 0)
                max_time = max(ts1[-1] if ts1 else 0, ts2[-1] if ts2 else 0)
                
                t = min_time
                while t <= max_time:
                    pos1 = self._interpolate_path_at_time(path1, ts1, t)
                    pos2 = self._interpolate_path_at_time(path2, ts2, t)
                    
                    if pos1 and pos2:
                        distance = euclidean_distance(pos1, pos2)
                        if distance < min_separation:
                            conflicts.append((uav1_id, uav2_id, t, pos1))
                    
                    t += time_step
        
        return conflicts
    
    def _interpolate_path_at_time(self,
                                  path: List[Tuple[float, float]],
                                  timestamps: List[float],
                                  time: float) -> Optional[Tuple[float, float]]:
        """Get position at specific time"""
        if not path or not timestamps or len(path) != len(timestamps):
            return None
        
        if time < timestamps[0]:
            return path[0]
        if time > timestamps[-1]:
            return path[-1]
        
        # Find segment
        for i in range(len(timestamps) - 1):
            t1 = timestamps[i]
            t2 = timestamps[i+1]
            
            if t1 <= time <= t2:
                if abs(t2 - t1) < 1e-6:
                    return path[i]
                
                # Linear interpolation
                alpha = (time - t1) / (t2 - t1)
                p1 = path[i]
                p2 = path[i+1]
                
                x = p1[0] + alpha * (p2[0] - p1[0])
                y = p1[1] + alpha * (p2[1] - p1[1])
                return (x, y)
        
        return path[-1]
    
    def _distance_based_priorities(self, distances: List[float]) -> List[int]:
        """Generate priorities based on distance (closer = higher priority)"""
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        priorities = [0] * len(distances)
        
        for rank, idx in enumerate(sorted_indices):
            priorities[idx] = len(distances) - rank  # Higher number = higher priority
        
        return priorities


class MultiPathAStar(MultiPathBasePlanner):
    """Multi-path planner using A* for initial planning"""
    
    def __init__(self,
                 obstacles: List[Tuple[float, float, float]],
                 width: int,
                 height: int,
                 uav_speed: float = 1.0,
                 min_goal_spacing: float = 1.5):
        base_planner = AStar(obstacles, width, height)
        super().__init__(obstacles, width, height, base_planner, uav_speed, min_goal_spacing)


class MultiPathThetaStar(MultiPathBasePlanner):
    """Multi-path planner using Theta* for initial planning"""
    
    def __init__(self,
                 obstacles: List[Tuple[float, float, float]],
                 width: int,
                 height: int,
                 uav_speed: float = 1.0,
                 min_goal_spacing: float = 1.5):
        base_planner = ThetaStar(obstacles, width, height)
        super().__init__(obstacles, width, height, base_planner, uav_speed, min_goal_spacing)


class MultiPathDijkstra(MultiPathBasePlanner):
    """Multi-path planner using Dijkstra for initial planning"""
    
    def __init__(self,
                 obstacles: List[Tuple[float, float, float]],
                 width: int,
                 height: int,
                 uav_speed: float = 1.0,
                 min_goal_spacing: float = 1.5):
        base_planner = Dijkstra(obstacles, width, height)
        super().__init__(obstacles, width, height, base_planner, uav_speed, min_goal_spacing)
