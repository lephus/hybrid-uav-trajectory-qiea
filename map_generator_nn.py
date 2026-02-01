"""
Map Generator for n-UAV to n-Destination Trajectory Planning
Generates maps with n start positions and n goal positions (one-to-one assignment)
Supports 4 types of maps:
- m1: Sparse environment with low obstacle density
- m2: Dense environment with high obstacle density
- m3: Maximum difficulty with only 1-2 possible paths
- m4: QIEA Challenge - Complex maze-like structure with multiple alternative paths,
      narrow passages, and local optima (designed to showcase QIEA's advantages)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Optional, Dict
import json
import os
from pathlib import Path


class NNDestinationMapGenerator:
    """Generate maps with circular obstacles for n-UAV to n-destination path planning"""
    
    def __init__(self, width: int, height: int):
        """
        Initialize map generator
        
        Args:
            width: Map width in grid units
            height: Map height in grid units
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=bool)  # True = obstacle, False = free
        self.obstacles = []  # List of (center_x, center_y, radius)
        self.starts = []  # List of (x, y) start positions for n UAVs
        self.goals = []  # List of (x, y) goal positions for n destinations
        self.assignments = {}  # Dict mapping uav_id -> goal_id (assignment)
    
    def _is_circle_in_bounds(self, cx: float, cy: float, radius: float) -> bool:
        """Check if circle is within map bounds"""
        return (cx - radius >= 0 and cx + radius < self.width and
                cy - radius >= 0 and cy + radius < self.height)
    
    def _circles_overlap(self, cx1: float, cy1: float, r1: float,
                         cx2: float, cy2: float, r2: float, 
                         min_distance: float = 0.0) -> bool:
        """Check if two circles overlap"""
        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        return distance < (r1 + r2 + min_distance)
    
    def _is_point_in_obstacle(self, x: float, y: float) -> bool:
        """Check if a point is inside any obstacle"""
        for cx, cy, radius in self.obstacles:
            dist_sq = (x - cx)**2 + (y - cy)**2
            if dist_sq <= radius**2:
                return True
        return False
    
    def _add_circle_to_grid(self, cx: float, cy: float, radius: float):
        """Add a circle obstacle to the grid"""
        y, x = np.ogrid[:self.height, :self.width]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        self.grid[mask] = True
        self.obstacles.append((cx, cy, radius))
    
    def _check_path_exists(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """
        Simple BFS to check if a path exists between start and goal
        """
        if self.grid[start[1], start[0]] or self.grid[goal[1], goal[0]]:
            return False
        
        visited = np.zeros_like(self.grid, dtype=bool)
        queue = [start]
        visited[start[1], start[0]] = True
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]  # 8-connected
        
        while queue:
            x, y = queue.pop(0)
            
            if (x, y) == goal:
                return True
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    not visited[ny, nx] and not self.grid[ny, nx]):
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        
        return False
    
    def _find_free_starts(self, num_uavs: int, 
                          min_separation: float = 3.0) -> List[Tuple[int, int]]:
        """
        Find n valid start positions in free space
        
        Strategy: Place starts in different regions of the map
        """
        starts = []
        max_attempts = 2000
        attempts = 0
        
        # Define regions for placing starts (distributed around map)
        regions = [
            (0, 0, self.width * 0.3, self.height * 0.3),  # Top-left
            (self.width * 0.7, 0, self.width, self.height * 0.3),  # Top-right
            (0, self.height * 0.7, self.width * 0.3, self.height),  # Bottom-left
            (self.width * 0.7, self.height * 0.7, self.width, self.height),  # Bottom-right
            (self.width * 0.3, 0, self.width * 0.7, self.height * 0.3),  # Top-center
            (0, self.height * 0.3, self.width * 0.3, self.height * 0.7),  # Left-center
            (self.width * 0.7, self.height * 0.3, self.width, self.height * 0.7),  # Right-center
            (self.width * 0.3, self.height * 0.7, self.width * 0.7, self.height),  # Bottom-center
        ]
        
        # Try to place starts in different regions
        used_regions = set()
        while len(starts) < num_uavs and attempts < max_attempts:
            attempts += 1
            
            # Select region
            if len(used_regions) < len(regions):
                region_idx = len(starts) % len(regions)
                if region_idx in used_regions:
                    available = [i for i in range(len(regions)) if i not in used_regions]
                    if available:
                        region_idx = available[np.random.randint(len(available))]
                    else:
                        region_idx = np.random.randint(len(regions))
            else:
                region_idx = np.random.randint(len(regions))
            
            x_min, y_min, x_max, y_max = regions[region_idx]
            
            # Try random point in region
            for _ in range(20):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                
                # Check if point is valid (not in obstacle)
                if self._is_point_in_obstacle(x, y):
                    continue
                
                # Check minimum separation from existing starts
                too_close = False
                for sx, sy in starts:
                    dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                    if dist < min_separation:
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                # Clamp to valid grid bounds
                x_int = max(0, min(self.width - 1, int(round(x))))
                y_int = max(0, min(self.height - 1, int(round(y))))
                start_int = (x_int, y_int)
                
                # Double-check point is not in obstacle after clamping
                if self._is_point_in_obstacle(x_int, y_int):
                    continue
                
                starts.append(start_int)
                used_regions.add(region_idx)
                break
        
        # If we don't have enough, try random positions
        while len(starts) < num_uavs and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            
            if self._is_point_in_obstacle(x, y):
                continue
            
            # Check separation
            too_close = False
            for sx, sy in starts:
                dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                if dist < min_separation:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Clamp to valid grid bounds
            x_int = max(0, min(self.width - 1, int(round(x))))
            y_int = max(0, min(self.height - 1, int(round(y))))
            start_int = (x_int, y_int)
            
            if self._is_point_in_obstacle(x_int, y_int):
                continue
            
            starts.append(start_int)
        
        return starts
    
    def _find_free_goals(self, num_goals: int, starts: List[Tuple[int, int]],
                        min_separation: float = 3.0) -> List[Tuple[int, int]]:
        """
        Find n valid goal positions in free space
        
        Strategy: Place goals in regions opposite to starts
        """
        goals = []
        max_attempts = 2000
        attempts = 0
        
        # Define regions for goals (opposite side from starts)
        # If starts are on left, place goals on right, etc.
        regions = [
            (self.width * 0.7, self.height * 0.7, self.width, self.height),  # Bottom-right
            (0, self.height * 0.7, self.width * 0.3, self.height),  # Bottom-left
            (self.width * 0.7, 0, self.width, self.height * 0.3),  # Top-right
            (0, 0, self.width * 0.3, self.height * 0.3),  # Top-left
            (self.width * 0.3, self.height * 0.7, self.width * 0.7, self.height),  # Bottom-center
            (self.width * 0.7, self.height * 0.3, self.width, self.height * 0.7),  # Right-center
            (0, self.height * 0.3, self.width * 0.3, self.height * 0.7),  # Left-center
            (self.width * 0.3, 0, self.width * 0.7, self.height * 0.3),  # Top-center
        ]
        
        # Try to place goals in different regions
        used_regions = set()
        while len(goals) < num_goals and attempts < max_attempts:
            attempts += 1
            
            # Select region
            if len(used_regions) < len(regions):
                region_idx = len(goals) % len(regions)
                if region_idx in used_regions:
                    available = [i for i in range(len(regions)) if i not in used_regions]
                    if available:
                        region_idx = available[np.random.randint(len(available))]
                    else:
                        region_idx = np.random.randint(len(regions))
            else:
                region_idx = np.random.randint(len(regions))
            
            x_min, y_min, x_max, y_max = regions[region_idx]
            
            # Try random point in region
            for _ in range(20):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                
                # Check if point is valid (not in obstacle)
                if self._is_point_in_obstacle(x, y):
                    continue
                
                # Check minimum separation from existing goals
                too_close = False
                for gx, gy in goals:
                    dist = np.sqrt((x - gx)**2 + (y - gy)**2)
                    if dist < min_separation:
                        too_close = True
                        break
                
                # Also check separation from starts
                for sx, sy in starts:
                    dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                    if dist < min_separation:
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                # Clamp to valid grid bounds
                x_int = max(0, min(self.width - 1, int(round(x))))
                y_int = max(0, min(self.height - 1, int(round(y))))
                goal_int = (x_int, y_int)
                
                # Double-check point is not in obstacle after clamping
                if self._is_point_in_obstacle(x_int, y_int):
                    continue
                
                goals.append(goal_int)
                used_regions.add(region_idx)
                break
        
        # If we don't have enough, try random positions
        while len(goals) < num_goals and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            
            if self._is_point_in_obstacle(x, y):
                continue
            
            # Check separation
            too_close = False
            for gx, gy in goals:
                dist = np.sqrt((x - gx)**2 + (y - gy)**2)
                if dist < min_separation:
                    too_close = True
                    break
            
            for sx, sy in starts:
                dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                if dist < min_separation:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Clamp to valid grid bounds
            x_int = max(0, min(self.width - 1, int(round(x))))
            y_int = max(0, min(self.height - 1, int(round(y))))
            goal_int = (x_int, y_int)
            
            if self._is_point_in_obstacle(x_int, y_int):
                continue
            
            goals.append(goal_int)
        
        return goals
    
    def _create_assignments(self, num_uavs: int) -> Dict[int, int]:
        """
        Create one-to-one assignments: UAV i -> Goal j
        
        Default: sequential assignment (UAV 0 -> Goal 0, UAV 1 -> Goal 1, etc.)
        Can be randomized if needed
        """
        assignments = {}
        for i in range(num_uavs):
            assignments[i] = i
        return assignments
    
    def generate_m1_sparse(self, num_uavs: int = 5,
                          num_obstacles: int = None, 
                          min_radius: float = None, 
                          max_radius: float = None) -> bool:
        """
        Generate m1: Sparse environment with low obstacle density
        
        Args:
            num_uavs: Number of UAVs (and destinations)
            num_obstacles: Number of obstacles (auto-calculated if None)
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
        
        Returns:
            True if successful, False otherwise
        """
        self.grid = np.zeros((self.height, self.width), dtype=bool)
        self.obstacles = []
        self.starts = []
        self.goals = []
        self.assignments = {}
        
        # Auto-calculate parameters based on map size
        if num_obstacles is None:
            area = self.width * self.height
            num_obstacles = int(area * 0.05)  # ~5% of cells as obstacles
        
        if min_radius is None:
            min_radius = min(self.width, self.height) * 0.02
        
        if max_radius is None:
            max_radius = min(self.width, self.height) * 0.08
        
        max_attempts = 1000
        attempts = 0
        
        while len(self.obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1
            
            # Random center and radius
            cx = np.random.uniform(max_radius, self.width - max_radius)
            cy = np.random.uniform(max_radius, self.height - max_radius)
            radius = np.random.uniform(min_radius, max_radius)
            
            # Check bounds
            if not self._is_circle_in_bounds(cx, cy, radius):
                continue
            
            # Check overlap with existing obstacles (allow small overlap)
            overlaps = False
            for ox, oy, orad in self.obstacles:
                if self._circles_overlap(cx, cy, radius, ox, oy, orad, min_distance=-radius*0.3):
                    overlaps = True
                    break
            
            if not overlaps:
                self._add_circle_to_grid(cx, cy, radius)
        
        # Find n start positions
        self.starts = self._find_free_starts(num_uavs)
        
        if len(self.starts) < num_uavs:
            return self.generate_m1_sparse(num_uavs, int(num_obstacles * 0.9), min_radius, max_radius)
        
        # Find n goal positions
        self.goals = self._find_free_goals(num_uavs, self.starts)
        
        if len(self.goals) < num_uavs:
            return self.generate_m1_sparse(num_uavs, int(num_obstacles * 0.9), min_radius, max_radius)
        
        # Create assignments
        self.assignments = self._create_assignments(num_uavs)
        
        # Verify all paths exist
        for uav_id, goal_id in self.assignments.items():
            start = self.starts[uav_id]
            goal = self.goals[goal_id]
            if not self._check_path_exists(start, goal):
                return self.generate_m1_sparse(num_uavs, int(num_obstacles * 0.9), min_radius, max_radius)
        
        return True
    
    def generate_m2_dense(self, num_uavs: int = 5,
                        num_obstacles: int = None,
                        min_radius: float = None, 
                        max_radius: float = None) -> bool:
        """
        Generate m2: Dense environment with high obstacle density
        
        Args:
            num_uavs: Number of UAVs (and destinations)
            num_obstacles: Number of obstacles (auto-calculated if None)
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
        
        Returns:
            True if successful, False otherwise
        """
        self.grid = np.zeros((self.height, self.width), dtype=bool)
        self.obstacles = []
        self.starts = []
        self.goals = []
        self.assignments = {}
        
        # Auto-calculate parameters - higher density than m1
        if num_obstacles is None:
            area = self.width * self.height
            num_obstacles = int(area * 0.15)  # ~15% of cells as obstacles
        
        if min_radius is None:
            min_radius = min(self.width, self.height) * 0.015
        
        if max_radius is None:
            max_radius = min(self.width, self.height) * 0.06
        
        max_attempts = 2000
        attempts = 0
        
        while len(self.obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1
            
            # Random center and radius
            cx = np.random.uniform(max_radius, self.width - max_radius)
            cy = np.random.uniform(max_radius, self.height - max_radius)
            radius = np.random.uniform(min_radius, max_radius)
            
            # Check bounds
            if not self._is_circle_in_bounds(cx, cy, radius):
                continue
            
            # Allow more overlap for dense environment
            overlaps_too_much = False
            overlap_count = 0
            for ox, oy, orad in self.obstacles:
                if self._circles_overlap(cx, cy, radius, ox, oy, orad, min_distance=-radius*0.5):
                    overlap_count += 1
                    if overlap_count > 3:  # Too many overlaps
                        overlaps_too_much = True
                        break
            
            if not overlaps_too_much:
                self._add_circle_to_grid(cx, cy, radius)
        
        # Find starts and goals
        self.starts = self._find_free_starts(num_uavs)
        
        if len(self.starts) < num_uavs:
            return self.generate_m2_dense(num_uavs, int(num_obstacles * 0.9), min_radius, max_radius)
        
        self.goals = self._find_free_goals(num_uavs, self.starts)
        
        if len(self.goals) < num_uavs:
            return self.generate_m2_dense(num_uavs, int(num_obstacles * 0.9), min_radius, max_radius)
        
        # Create assignments
        self.assignments = self._create_assignments(num_uavs)
        
        # Verify all paths exist
        for uav_id, goal_id in self.assignments.items():
            start = self.starts[uav_id]
            goal = self.goals[goal_id]
            if not self._check_path_exists(start, goal):
                return self.generate_m2_dense(num_uavs, int(num_obstacles * 0.9), min_radius, max_radius)
        
        return True
    
    def generate_m3_trap(self, num_uavs: int = 5) -> bool:
        """
        Generate m3: Maximum difficulty with only 1-2 possible paths
        Creates structured obstacles to form narrow passages (choke points)
        
        Args:
            num_uavs: Number of UAVs (and destinations)
        
        Returns:
            True if successful, False otherwise
        """
        self.grid = np.zeros((self.height, self.width), dtype=bool)
        self.obstacles = []
        self.starts = []
        self.goals = []
        self.assignments = {}
        
        # Strategy: Create U-shaped or corridor-like structures
        map_center_x = self.width / 2
        map_center_y = self.height / 2
        
        # Create a U-shaped trap in the middle
        u_width = self.width * 0.4
        u_height = self.height * 0.3
        u_thickness = min(self.width, self.height) * 0.08
        
        # Left wall of U
        self._add_circle_to_grid(
            map_center_x - u_width/2, 
            map_center_y, 
            u_thickness
        )
        
        # Right wall of U
        self._add_circle_to_grid(
            map_center_x + u_width/2, 
            map_center_y, 
            u_thickness
        )
        
        # Bottom of U (connecting left and right)
        num_bottom_circles = int(u_width / (u_thickness * 2))
        for i in range(num_bottom_circles):
            x = map_center_x - u_width/2 + (i * u_width / max(1, num_bottom_circles - 1))
            self._add_circle_to_grid(x, map_center_y + u_height/2, u_thickness)
        
        # Add additional obstacles to create narrow passages
        passage_width = min(self.width, self.height) * 0.1
        
        # Left passage
        passage_y = map_center_y - u_height/4
        num_blocking = int((u_width - passage_width) / (u_thickness * 2))
        for i in range(num_blocking):
            x = map_center_x - u_width/2 + passage_width/2 + (i * (u_width - passage_width) / max(1, num_blocking))
            if abs(x - map_center_x) > passage_width/2:
                self._add_circle_to_grid(x, passage_y, u_thickness * 0.8)
        
        # Add random obstacles to fill space
        area = self.width * self.height
        num_additional = int(area * 0.12)
        min_radius = min(self.width, self.height) * 0.02
        max_radius = min(self.width, self.height) * 0.05
        
        max_attempts = 1000
        attempts = 0
        
        while len(self.obstacles) < num_additional + 10 and attempts < max_attempts:
            attempts += 1
            
            cx = np.random.uniform(max_radius, self.width - max_radius)
            cy = np.random.uniform(max_radius, self.height - max_radius)
            radius = np.random.uniform(min_radius, max_radius)
            
            if not self._is_circle_in_bounds(cx, cy, radius):
                continue
            
            self._add_circle_to_grid(cx, cy, radius)
        
        # Find starts in top-left area
        self.starts = self._find_free_starts(num_uavs, min_separation=2.0)
        
        if len(self.starts) < num_uavs:
            return self.generate_m3_trap(num_uavs)
        
        # Find goals in bottom-right area (opposite side from starts)
        self.goals = self._find_free_goals(num_uavs, self.starts, min_separation=2.0)
        
        if len(self.goals) < num_uavs:
            return self.generate_m3_trap(num_uavs)
        
        # Create assignments
        self.assignments = self._create_assignments(num_uavs)
        
        # Verify all paths exist
        for uav_id, goal_id in self.assignments.items():
            start = self.starts[uav_id]
            goal = self.goals[goal_id]
            if not self._check_path_exists(start, goal):
                return self.generate_m3_trap(num_uavs)
        
        return True
    
    def generate_m4_qiea_challenge(self, num_uavs: int = 5) -> bool:
        """
        Generate m4: QIEA Challenge Map - Designed to showcase QIEA's advantages
        
        Features:
        - Multiple alternative paths with different lengths (local optima)
        - Narrow passages that require path smoothing
        - Maze-like structure with dead ends
        - Obstacles creating multiple route options
        - Longer paths that benefit from optimization
        
        This map is designed so that:
        - Classical algorithms may find suboptimal paths (stuck in local optima)
        - QIEA can explore multiple solutions and find shorter paths
        - Path smoothing is beneficial (narrow passages)
        
        Args:
            num_uavs: Number of UAVs (and destinations)
        
        Returns:
            True if successful, False otherwise
        """
        self.grid = np.zeros((self.height, self.width), dtype=bool)
        self.obstacles = []
        self.starts = []
        self.goals = []
        self.assignments = {}
        
        # Strategy: Create a complex maze-like structure with multiple routes
        map_center_x = self.width / 2
        map_center_y = self.height / 2
        
        # Create a complex obstacle pattern that creates multiple alternative paths
        # 1. Create vertical and horizontal barriers with gaps (creating multiple routes)
        
        # Vertical barriers (left side)
        barrier_thickness = min(self.width, self.height) * 0.06
        gap_size = min(self.width, self.height) * 0.12
        
        # Left vertical barrier with gaps
        num_gaps = 3
        barrier_length = self.height * 0.7
        barrier_start_y = self.height * 0.15
        gap_spacing = barrier_length / (num_gaps + 1)
        
        for i in range(num_gaps + 1):
            gap_y = barrier_start_y + (i + 1) * gap_spacing
            # Create barrier segments above and below gap
            if i == 0:
                # Bottom segment
                segment_height = gap_y - barrier_start_y
                num_circles = int(segment_height / (barrier_thickness * 2))
                for j in range(num_circles):
                    y = barrier_start_y + (j * segment_height / max(1, num_circles))
                    self._add_circle_to_grid(self.width * 0.25, y, barrier_thickness)
            else:
                # Middle segments
                prev_gap_y = barrier_start_y + i * gap_spacing
                segment_height = gap_y - prev_gap_y
                num_circles = int(segment_height / (barrier_thickness * 2))
                for j in range(num_circles):
                    y = prev_gap_y + (j * segment_height / max(1, num_circles))
                    self._add_circle_to_grid(self.width * 0.25, y, barrier_thickness)
            
            # Top segment (after last gap)
            if i == num_gaps:
                top_y = barrier_start_y + barrier_length
                segment_height = self.height - top_y
                num_circles = int(segment_height / (barrier_thickness * 2))
                for j in range(num_circles):
                    y = top_y + (j * segment_height / max(1, num_circles))
                    self._add_circle_to_grid(self.width * 0.25, y, barrier_thickness)
        
        # Right vertical barrier with gaps (mirror)
        for i in range(num_gaps + 1):
            gap_y = barrier_start_y + (i + 1) * gap_spacing
            if i == 0:
                segment_height = gap_y - barrier_start_y
                num_circles = int(segment_height / (barrier_thickness * 2))
                for j in range(num_circles):
                    y = barrier_start_y + (j * segment_height / max(1, num_circles))
                    self._add_circle_to_grid(self.width * 0.75, y, barrier_thickness)
            else:
                prev_gap_y = barrier_start_y + i * gap_spacing
                segment_height = gap_y - prev_gap_y
                num_circles = int(segment_height / (barrier_thickness * 2))
                for j in range(num_circles):
                    y = prev_gap_y + (j * segment_height / max(1, num_circles))
                    self._add_circle_to_grid(self.width * 0.75, y, barrier_thickness)
            
            if i == num_gaps:
                top_y = barrier_start_y + barrier_length
                segment_height = self.height - top_y
                num_circles = int(segment_height / (barrier_thickness * 2))
                for j in range(num_circles):
                    y = top_y + (j * segment_height / max(1, num_circles))
                    self._add_circle_to_grid(self.width * 0.75, y, barrier_thickness)
        
        # Horizontal barriers in the middle (creating maze-like structure)
        # Top horizontal barrier
        h_barrier_y = self.height * 0.4
        h_gap_size = self.width * 0.15
        h_barrier_start_x = self.width * 0.2
        h_barrier_end_x = self.width * 0.8
        h_barrier_length = h_barrier_end_x - h_barrier_start_x
        
        # Create horizontal barrier with 2 gaps
        num_h_gaps = 2
        h_gap_spacing = h_barrier_length / (num_h_gaps + 1)
        
        for i in range(num_h_gaps + 1):
            gap_x = h_barrier_start_x + (i + 1) * h_gap_spacing
            if i == 0:
                segment_length = gap_x - h_barrier_start_x
                num_circles = int(segment_length / (barrier_thickness * 2))
                for j in range(num_circles):
                    x = h_barrier_start_x + (j * segment_length / max(1, num_circles))
                    self._add_circle_to_grid(x, h_barrier_y, barrier_thickness)
            else:
                prev_gap_x = h_barrier_start_x + i * h_gap_spacing
                segment_length = gap_x - prev_gap_x
                num_circles = int(segment_length / (barrier_thickness * 2))
                for j in range(num_circles):
                    x = prev_gap_x + (j * segment_length / max(1, num_circles))
                    self._add_circle_to_grid(x, h_barrier_y, barrier_thickness)
            
            if i == num_h_gaps:
                end_x = h_barrier_start_x + h_barrier_length
                segment_length = h_barrier_end_x - end_x
                num_circles = int(segment_length / (barrier_thickness * 2))
                for j in range(num_circles):
                    x = end_x + (j * segment_length / max(1, num_circles))
                    self._add_circle_to_grid(x, h_barrier_y, barrier_thickness)
        
        # Bottom horizontal barrier
        h_barrier_y2 = self.height * 0.6
        for i in range(num_h_gaps + 1):
            gap_x = h_barrier_start_x + (i + 1) * h_gap_spacing
            if i == 0:
                segment_length = gap_x - h_barrier_start_x
                num_circles = int(segment_length / (barrier_thickness * 2))
                for j in range(num_circles):
                    x = h_barrier_start_x + (j * segment_length / max(1, num_circles))
                    self._add_circle_to_grid(x, h_barrier_y2, barrier_thickness)
            else:
                prev_gap_x = h_barrier_start_x + i * h_gap_spacing
                segment_length = gap_x - prev_gap_x
                num_circles = int(segment_length / (barrier_thickness * 2))
                for j in range(num_circles):
                    x = prev_gap_x + (j * segment_length / max(1, num_circles))
                    self._add_circle_to_grid(x, h_barrier_y2, barrier_thickness)
            
            if i == num_h_gaps:
                end_x = h_barrier_start_x + h_barrier_length
                segment_length = h_barrier_end_x - end_x
                num_circles = int(segment_length / (barrier_thickness * 2))
                for j in range(num_circles):
                    x = end_x + (j * segment_length / max(1, num_circles))
                    self._add_circle_to_grid(x, h_barrier_y2, barrier_thickness)
        
        # Add additional obstacles to create more complexity and local optima
        # These create "dead ends" and alternative longer routes
        area = self.width * self.height
        num_additional = int(area * 0.08)  # Moderate density
        min_radius = min(self.width, self.height) * 0.02
        max_radius = min(self.width, self.height) * 0.05
        
        max_attempts = 1500
        attempts = 0
        
        while len(self.obstacles) < num_additional + 50 and attempts < max_attempts:
            attempts += 1
            
            cx = np.random.uniform(max_radius, self.width - max_radius)
            cy = np.random.uniform(max_radius, self.height - max_radius)
            radius = np.random.uniform(min_radius, max_radius)
            
            if not self._is_circle_in_bounds(cx, cy, radius):
                continue
            
            # Avoid placing obstacles too close to barriers (to maintain passages)
            too_close_to_barrier = False
            barrier_x_positions = [self.width * 0.25, self.width * 0.75]
            barrier_y_positions = [self.height * 0.4, self.height * 0.6]
            
            for bx in barrier_x_positions:
                if abs(cx - bx) < gap_size * 0.8:
                    too_close_to_barrier = True
                    break
            
            for by in barrier_y_positions:
                if abs(cy - by) < gap_size * 0.8:
                    too_close_to_barrier = True
                    break
            
            if too_close_to_barrier:
                continue
            
            self._add_circle_to_grid(cx, cy, radius)
        
        # Find starts in top-left area (forcing paths through the maze)
        self.starts = self._find_free_starts(num_uavs, min_separation=2.0)
        
        if len(self.starts) < num_uavs:
            return self.generate_m4_qiea_challenge(num_uavs)
        
        # Find goals in bottom-right area (opposite side, forcing long paths)
        self.goals = self._find_free_goals(num_uavs, self.starts, min_separation=2.0)
        
        if len(self.goals) < num_uavs:
            return self.generate_m4_qiea_challenge(num_uavs)
        
        # Create assignments
        self.assignments = self._create_assignments(num_uavs)
        
        # Verify all paths exist
        for uav_id, goal_id in self.assignments.items():
            start = self.starts[uav_id]
            goal = self.goals[goal_id]
            if not self._check_path_exists(start, goal):
                return self.generate_m4_qiea_challenge(num_uavs)
        
        return True
    
    def visualize(self, save_path: str = None, show: bool = True):
        """Visualize the map with obstacles, n starts, n goals, and assignments"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Draw grid
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Draw obstacles as circles
        for cx, cy, radius in self.obstacles:
            circle = patches.Circle((cx, cy), radius, facecolor='red', alpha=0.25, 
                                   edgecolor='darkred', linewidth=1)
            ax.add_patch(circle)
        
        # Color palette for UAVs
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.starts)))
        
        # Draw start points (n UAVs) and their assigned goals
        for i, start in enumerate(self.starts):
            color = colors[i % len(colors)]
            
            # Draw start point
            ax.plot(start[0], start[1], 'o', markersize=12, 
                   color=color, markeredgecolor='black', markeredgewidth=2,
                   label=f'UAV {i} Start' if i < 5 else '')
            
            # Draw assigned goal
            if i in self.assignments:
                goal_id = self.assignments[i]
                if goal_id < len(self.goals):
                    goal = self.goals[goal_id]
                    ax.plot(goal[0], goal[1], '*', markersize=15, 
                           color=color, markeredgecolor='black', markeredgewidth=2,
                           label=f'Goal {goal_id} (UAV {i})' if i < 5 else '')
                    
                    # Draw assignment line (dashed)
                    ax.plot([start[0], goal[0]], [start[1], goal[1]], 
                           '--', color=color, alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X (grid units)', fontsize=12)
        ax.set_ylabel('Y (grid units)', fontsize=12)
        ax.set_title(f'n-n Map {self.width}x{self.height} - {len(self.starts)} UAVs → {len(self.goals)} Goals', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_to_file(self, filepath: str):
        """Save map to JSON file with n starts, n goals, and assignments"""
        # Convert numpy types to Python native types for JSON serialization
        obstacles_list = [[float(cx), float(cy), float(r)] for cx, cy, r in self.obstacles]
        starts_list = [[int(s[0]), int(s[1])] for s in self.starts]
        goals_list = [[int(g[0]), int(g[1])] for g in self.goals]
        assignments_list = [[int(uav_id), int(goal_id)] for uav_id, goal_id in self.assignments.items()]
        
        data = {
            'width': int(self.width),
            'height': int(self.height),
            'obstacles': obstacles_list,
            'starts': starts_list,  # List of n start positions
            'goals': goals_list,  # List of n goal positions
            'assignments': assignments_list,  # List of [uav_id, goal_id] pairs
            'num_uavs': len(self.starts),
            'num_goals': len(self.goals),
            'grid': self.grid.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load map from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.width = data['width']
        self.height = data['height']
        self.obstacles = [tuple(obs) for obs in data['obstacles']]
        self.starts = [tuple(s) for s in data['starts']]
        self.goals = [tuple(g) for g in data['goals']]
        
        # Load assignments
        if 'assignments' in data:
            self.assignments = {uav_id: goal_id for uav_id, goal_id in data['assignments']}
        else:
            # Default: sequential assignment
            self.assignments = {i: i for i in range(len(self.starts))}
        
        self.grid = np.array(data['grid'], dtype=bool)
    
    def get_obstacle_density(self) -> float:
        """Calculate obstacle density as percentage"""
        total_cells = self.width * self.height
        obstacle_cells = np.sum(self.grid)
        return (obstacle_cells / total_cells) * 100


def generate_all_nn_maps(num_uavs_list: List[int] = [3, 5, 10]):
    """
    Generate all n-n maps for all sizes, types, and UAV counts
    
    Args:
        num_uavs_list: List of number of UAVs (and destinations) to generate maps for
    """
    sizes = [20, 50, 100]
    map_types = ['m1', 'm2', 'm3', 'm4']
    
    # Create directories
    output_dir = Path('maps')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'nn_destinations').mkdir(exist_ok=True)
    
    results = []
    
    for size in sizes:
        for map_type in map_types:
            for num_uavs in num_uavs_list:
                print(f"\nGenerating {map_type} n-n map {size}x{size} with {num_uavs} UAVs...")
                
                generator = NNDestinationMapGenerator(size, size)
                
                success = False
                if map_type == 'm1':
                    success = generator.generate_m1_sparse(num_uavs)
                elif map_type == 'm2':
                    success = generator.generate_m2_dense(num_uavs)
                elif map_type == 'm3':
                    success = generator.generate_m3_trap(num_uavs)
                elif map_type == 'm4':
                    success = generator.generate_m4_qiea_challenge(num_uavs)
                
                if success:
                    # Save map
                    map_filename = f"{map_type}_{size}x{size}_{num_uavs}nn.json"
                    map_path = output_dir / 'nn_destinations' / map_filename
                    generator.save_to_file(str(map_path))
                    
                    # Save visualization
                    viz_filename = f"{map_type}_{size}x{size}_{num_uavs}nn.png"
                    viz_path = output_dir / 'visualizations' / viz_filename
                    generator.visualize(save_path=str(viz_path), show=False)
                    
                    density = generator.get_obstacle_density()
                    results.append({
                        'type': map_type,
                        'size': f"{size}x{size}",
                        'num_uavs': num_uavs,
                        'density': f"{density:.2f}%",
                        'num_obstacles': len(generator.obstacles),
                        'file': map_filename
                    })
                    
                    print(f"  ✓ Generated: {map_filename}")
                    print(f"    Obstacle density: {density:.2f}%")
                    print(f"    Number of obstacles: {len(generator.obstacles)}")
                    print(f"    Number of UAVs: {len(generator.starts)}")
                    print(f"    Number of Goals: {len(generator.goals)}")
                else:
                    print(f"  ✗ Failed to generate {map_type} {size}x{size} with {num_uavs} UAVs")
    
    # Print summary
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    print(f"{'Type':<10} {'Size':<15} {'UAVs':<10} {'Density':<15} {'Obstacles':<15} {'File':<30}")
    print("-"*70)
    for r in results:
        print(f"{r['type']:<10} {r['size']:<15} {r['num_uavs']:<10} {r['density']:<15} {r['num_obstacles']:<15} {r['file']:<30}")
    
    return results


def generate_single_nn_map(map_type: str = 'm1', size: int = 50, num_uavs: int = 5):
    """
    Generate a single n-n map for quick testing
    
    Args:
        map_type: 'm1', 'm2', or 'm3'
        size: Map size (width = height)
        num_uavs: Number of UAVs (and destinations)
    """
    print(f"Generating {map_type} n-n map {size}x{size} with {num_uavs} UAVs...")
    
    generator = NNDestinationMapGenerator(size, size)
    
    success = False
    if map_type == 'm1':
        success = generator.generate_m1_sparse(num_uavs)
    elif map_type == 'm2':
        success = generator.generate_m2_dense(num_uavs)
    elif map_type == 'm3':
        success = generator.generate_m3_trap(num_uavs)
    elif map_type == 'm4':
        success = generator.generate_m4_qiea_challenge(num_uavs)
    
    if success:
        # Save to maps/nn_destinations directory
        output_dir = Path('maps') / 'nn_destinations'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        map_filename = f"{map_type}_{size}x{size}_{num_uavs}nn.json"
        map_path = output_dir / map_filename
        generator.save_to_file(str(map_path))
        
        print(f"✓ Saved to: {map_path}")
        print(f"  Obstacles: {len(generator.obstacles)}")
        print(f"  UAVs: {len(generator.starts)}")
        print(f"  Goals: {len(generator.goals)}")
        print(f"  Assignments: {generator.assignments}")
        
        # Visualize
        generator.visualize(show=True)
        
        return generator
    else:
        print(f"✗ Failed to generate map")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate n-n maps for multi-UAV path planning')
    parser.add_argument('--all', action='store_true',
                       help='Generate all maps (all sizes, types, and UAV counts)')
    parser.add_argument('--type', type=str, choices=['m1', 'm2', 'm3', 'm4'], default='m1',
                       help='Map type: m1=sparse, m2=dense, m3=trap, m4=QIEA challenge (default: m1)')
    parser.add_argument('--size', type=int, default=50,
                       help='Map size (default: 50)')
    parser.add_argument('--uavs', type=int, default=5,
                       help='Number of UAVs (and destinations) (default: 5)')
    
    args = parser.parse_args()
    
    print("n-n Destination Map Generator")
    print("="*60)
    
    if args.all:
        generate_all_nn_maps(num_uavs_list=[3, 5, 10])
        print("\nAll maps generated successfully!")
    else:
        generate_single_nn_map(args.type, args.size, args.uavs)
