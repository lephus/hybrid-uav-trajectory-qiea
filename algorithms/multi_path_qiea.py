"""
Multi-Path Quantum-Inspired Evolutionary Algorithm (QIEA) for Multi-UAV Path Planning
Optimizes multiple UAV paths simultaneously in a single QIEA run
"""

import numpy as np
import random
import math
from typing import List, Tuple, Optional, Dict
from .base import PathPlanner, MultiPathResult
from .utils import (
    euclidean_distance,
    calculate_path_cost,
    smooth_path,
    get_neighbors_8_connected
)
import time


class MultiPathQIEA(PathPlanner):
    """
    Multi-Path QIEA: Optimize tất cả n paths đồng thời
    
    Key Innovation:
    - Encode tất cả n paths trong 1 chromosome
    - Multi-objective fitness: path costs + conflict penalties
    - Optimize đồng thời → tìm solution tốt nhất tổng thể
    """
    
    def __init__(self,
                 obstacles: List[Tuple[float, float, float]],
                 width: int,
                 height: int,
                 num_uavs: int,
                 population_size: int = 100,  # Increased for better exploration
                 max_generations: int = 200,  # Increased for better convergence
                 rotation_angle: float = 0.02 * math.pi,  # Increased from 0.01 to 0.02 for better exploration
                 mutation_rate: float = 0.15,  # Slightly increased for more exploration
                 uav_speed: float = 1.0,
                 min_separation: float = 2.0,
                 num_waypoints_per_path: int = 15):  # Increased for more path flexibility
        """
        Args:
            num_uavs: Số lượng UAVs
            num_waypoints_per_path: Số waypoints mỗi path
        """
        super().__init__(obstacles, width, height)
        self.num_uavs = num_uavs
        self.population_size = population_size
        self.max_generations = max_generations
        self.rotation_angle = rotation_angle
        self.mutation_rate = mutation_rate
        self.uav_speed = uav_speed
        self.min_separation = min_separation
        self.num_waypoints_per_path = num_waypoints_per_path
        
        # Chromosome size: n paths × waypoints × 2 (alpha, beta)
        self.chromosome_size = num_uavs * num_waypoints_per_path * 2
    
    def _encode_single_path_to_qbits(self,
                                    path: List[Tuple[float, float]],
                                    num_waypoints: int) -> np.ndarray:
        """Encode 1 path như QIEA hiện tại (direction vectors)"""
        chromosome = np.zeros(num_waypoints * 2, dtype=np.float64)
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        
        if len(path) < 2:
            return np.full(num_waypoints * 2, sqrt2_inv, dtype=np.float64)
        
        # Extract directions
        directions = []
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 1e-10:
                dx /= length
                dy /= length
                angle = math.atan2(dy, dx)
                directions.append(angle)
            else:
                if directions:
                    directions.append(directions[-1])
                else:
                    directions.append(0.0)
        
        # Interpolate/downsample to num_waypoints
        if len(directions) < num_waypoints:
            # Interpolate
            interpolated = []
            for i in range(num_waypoints):
                t = i / (num_waypoints - 1) if num_waypoints > 1 else 0
                idx = t * (len(directions) - 1) if len(directions) > 1 else 0
                idx_low = int(idx)
                idx_high = min(idx_low + 1, len(directions) - 1)
                
                if idx_low == idx_high or len(directions) == 1:
                    interpolated.append(directions[idx_low] if directions else 0.0)
                else:
                    angle_low = directions[idx_low]
                    angle_high = directions[idx_high]
                    local_t = idx - idx_low
                    interp_angle = angle_low + local_t * (angle_high - angle_low)
                    interpolated.append(interp_angle)
            
            directions = interpolated
        elif len(directions) > num_waypoints:
            step = len(directions) / num_waypoints
            directions = [directions[int(i * step)] for i in range(num_waypoints)]
        
        # Encode directions as Q-bits
        for i, angle in enumerate(directions[:num_waypoints]):
            normalized_angle = (angle + math.pi) / (2 * math.pi)
            q_angle = math.pi * normalized_angle
            chromosome[i * 2] = math.cos(q_angle)
            chromosome[i * 2 + 1] = math.sin(q_angle)
        
        # Normalize
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        for i in range(0, len(chromosome), 2):
            alpha, beta = chromosome[i], chromosome[i+1]
            norm = math.sqrt(alpha*alpha + beta*beta)
            if norm > 1e-10:
                chromosome[i] = alpha / norm
                chromosome[i+1] = beta / norm
            else:
                chromosome[i] = sqrt2_inv
                chromosome[i+1] = sqrt2_inv
        
        return chromosome
    
    def _encode_direction_to_qbits(self,
                                  direction_angle: float,
                                  num_waypoints: int) -> np.ndarray:
        """Encode một hướng duy nhất cho tất cả waypoints"""
        chromosome = np.zeros(num_waypoints * 2, dtype=np.float64)
        normalized_angle = (direction_angle + math.pi) / (2 * math.pi)
        q_angle = math.pi * normalized_angle
        
        cos_val = math.cos(q_angle)
        sin_val = math.sin(q_angle)
        
        for i in range(num_waypoints):
            chromosome[i * 2] = cos_val
            chromosome[i * 2 + 1] = sin_val
        
        return chromosome
    
    def _encode_multi_paths_to_qbits(self,
                                    paths: Dict[int, List[Tuple[float, float]]],
                                    starts: List[Tuple[float, float]],
                                    goal: Tuple[float, float]) -> np.ndarray:
        """
        Encode tất cả n paths thành 1 chromosome
        
        Structure: [path0_qbits, path1_qbits, ..., pathN_qbits]
        Mỗi path được encode như single-path QIEA
        """
        chromosome = np.zeros(self.chromosome_size, dtype=np.float64)
        
        for uav_id in range(self.num_uavs):
            if uav_id in paths and len(paths[uav_id]) > 1:
                # Encode path như single-path QIEA
                path_chromosome = self._encode_single_path_to_qbits(
                    paths[uav_id],
                    self.num_waypoints_per_path
                )
            else:
                # Empty path: encode direction từ start đến goal
                start = starts[uav_id]
                dx = goal[0] - start[0]
                dy = goal[1] - start[1]
                angle = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0.0
                path_chromosome = self._encode_direction_to_qbits(
                    angle,
                    self.num_waypoints_per_path
                )
            
            # Copy vào chromosome
            start_idx = uav_id * self.num_waypoints_per_path * 2
            end_idx = start_idx + len(path_chromosome)
            chromosome[start_idx:end_idx] = path_chromosome
        
        return chromosome
    
    def _measure_single_path(self,
                           path_chromosome: np.ndarray,
                           start: Tuple[float, float],
                           goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Decode 1 path từ chromosome (tương tự QIEA._measure)"""
        num_waypoints = len(path_chromosome) // 2
        path = [start]
        current_pos = start
        
        # First, check if direct path is possible (optimization for shorter paths)
        if self.is_valid_line(start, goal):
            # Direct path is possible - use it with minimal waypoints
            # Only add 1-2 intermediate waypoints if needed for smoothness
            direct_distance = euclidean_distance(start, goal)
            if direct_distance < 5.0:  # Very short distance, go directly
                path.append(goal)
                return path
            else:
                # Add 1 intermediate waypoint for longer direct paths
                mid_point = ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2)
                if self.is_valid_point(mid_point) and self.is_valid_line(start, mid_point) and self.is_valid_line(mid_point, goal):
                    path.append(mid_point)
                path.append(goal)
                return path
        
        total_distance = euclidean_distance(start, goal)
        # Increase base step size to allow larger steps and fewer waypoints
        # Use adaptive base step size: larger for longer distances
        if total_distance > 10:
            base_step_size = total_distance / max(5, num_waypoints // 2)  # Use fewer waypoints for long distances
        else:
            base_step_size = total_distance / (num_waypoints + 1) if num_waypoints > 0 else total_distance
        
        for i in range(num_waypoints):
            alpha = path_chromosome[i * 2]
            beta = path_chromosome[i * 2 + 1]
            
            # Decode direction angle
            q_angle = math.atan2(beta, alpha)
            if q_angle < 0:
                q_angle += 2 * math.pi
            
            normalized_angle = q_angle / math.pi
            if normalized_angle > 1.0:
                normalized_angle = normalized_angle - 1.0
            
            direction_angle = normalized_angle * 2 * math.pi - math.pi
            
            # Calculate step size - allow MUCH larger steps for better exploration
            remaining_distance = euclidean_distance(current_pos, goal)
            # Allow step size up to 5x base for better exploration (increased from 3x)
            # This allows QIEA to "jump" over obstacles to find shorter paths
            max_step_size = base_step_size * 5.0
            # Also allow direct steps towards goal if close enough
            if remaining_distance < base_step_size * 3:
                step_size = remaining_distance  # Can take direct step to goal
            else:
                # Adaptive step size: larger steps when far from goal
                adaptive_factor = 1.0 + (remaining_distance / total_distance) * 2.0  # Up to 3x for far distances
                step_size = min(max_step_size * adaptive_factor, remaining_distance / max(1, num_waypoints - i)) if (num_waypoints - i) > 0 else base_step_size
            
            # Early termination: if we can go directly to goal, do it
            if remaining_distance < step_size * 1.5 and self.is_valid_line(current_pos, goal):
                path.append(goal)
                return path
            
            # Add exploration - increased for better path discovery
            prob_0 = alpha * alpha
            prob_1 = beta * beta
            total_prob = prob_0 + prob_1
            
            if total_prob > 0:
                # Increased exploration factor from 0.1 to 0.4 for better exploration
                exploration_factor = 0.4 * (1.0 - abs(prob_0 - prob_1))
                direction_angle += random.uniform(-exploration_factor, exploration_factor)
                # Allow larger step size variation for better exploration
                step_size *= (1.0 + random.uniform(-0.3, 0.5))  # Increased from (-0.2, 0.2)
            
            # Generate waypoint
            dx = math.cos(direction_angle) * step_size
            dy = math.sin(direction_angle) * step_size
            new_waypoint = (
                max(0, min(self.width - 1, current_pos[0] + dx)),
                max(0, min(self.height - 1, current_pos[1] + dy))
            )
            
            if self.is_valid_point(new_waypoint):
                path.append(new_waypoint)
                current_pos = new_waypoint
            else:
                # Try to find nearby valid point - expanded search
                found_valid = False
                # Try more angles with finer granularity
                angle_offsets = [
                    -math.pi/2, -3*math.pi/8, -math.pi/4, -math.pi/8, -math.pi/16,
                    0, math.pi/16, math.pi/8, math.pi/4, 3*math.pi/8, math.pi/2
                ]
                # Also try with reduced step size
                step_sizes = [step_size, step_size * 0.7, step_size * 0.5]
                
                for step_size_adj in step_sizes:
                    for angle_offset in angle_offsets:
                        adjusted_angle = direction_angle + angle_offset
                        dx_adj = math.cos(adjusted_angle) * step_size_adj
                        dy_adj = math.sin(adjusted_angle) * step_size_adj
                        adjusted_waypoint = (
                            max(0, min(self.width - 1, current_pos[0] + dx_adj)),
                            max(0, min(self.height - 1, current_pos[1] + dy_adj))
                        )
                        
                        if self.is_valid_point(adjusted_waypoint):
                            path.append(adjusted_waypoint)
                            current_pos = adjusted_waypoint
                            found_valid = True
                            break
                    
                    if found_valid:
                        break
                
                if not found_valid:
                    # Last resort: try to move towards goal with smaller step
                    goal_direction = math.atan2(goal[1] - current_pos[1], goal[0] - current_pos[0])
                    small_step = step_size * 0.3
                    dx_goal = math.cos(goal_direction) * small_step
                    dy_goal = math.sin(goal_direction) * small_step
                    goal_waypoint = (
                        max(0, min(self.width - 1, current_pos[0] + dx_goal)),
                        max(0, min(self.height - 1, current_pos[1] + dy_goal))
                    )
                    if self.is_valid_point(goal_waypoint):
                        path.append(goal_waypoint)
                        current_pos = goal_waypoint
                    else:
                        continue
        
        # Ensure goal is in path
        if len(path) == 0 or path[-1] != goal:
            if len(path) > 0:
                # Check if we can go directly to goal from last waypoint
                if self.is_valid_line(path[-1], goal):
                    path.append(goal)
                else:
                    # Try to find a path to goal
                    # Check if we can reach goal with a few intermediate points
                    last_pos = path[-1]
                    # Try direct connection first
                    if euclidean_distance(last_pos, goal) < 10 and self.is_valid_line(last_pos, goal):
                        path.append(goal)
                    else:
                        # Add goal anyway (will be validated)
                        path.append(goal)
            else:
                path.append(goal)
        
        # Post-process: remove unnecessary waypoints if direct path exists
        path = self._optimize_path_waypoints(path)
        
        # Validate and smooth
        path = self._validate_and_smooth_path(path)
        return path
    
    def _optimize_path_waypoints(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Remove unnecessary waypoints if direct paths exist"""
        if len(path) <= 2:
            return path
        
        optimized = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Try to skip ahead as much as possible
            j = len(path) - 1
            found_direct = False
            
            # Check from furthest point backwards
            while j > i + 1:
                if self.is_valid_line(path[i], path[j]):
                    # Can skip all intermediate points
                    optimized.append(path[j])
                    i = j
                    found_direct = True
                    break
                j -= 1
            
            if not found_direct:
                # Can't skip, add next point
                i += 1
                if i < len(path):
                    optimized.append(path[i])
        
        return optimized if len(optimized) > 0 else path
    
    def _validate_and_smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Validate và smooth path (tương tự QIEA)"""
        if len(path) < 2:
            return path
        
        # Remove invalid waypoints
        valid_path = [p for p in path if self.is_valid_point(p)]
        
        if len(valid_path) < 2:
            return [path[0], path[-1]]
        
        # Ensure start and goal
        if valid_path[0] != path[0]:
            valid_path.insert(0, path[0])
        if valid_path[-1] != path[-1]:
            valid_path.append(path[-1])
        
        # Fix crossing segments
        fixed_path = [valid_path[0]]
        for i in range(len(valid_path) - 1):
            p1 = fixed_path[-1]
            p2 = valid_path[i + 1]
            
            if self.is_valid_line(p1, p2):
                fixed_path.append(p2)
            else:
                # Try midpoint
                mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                if (self.is_valid_point(mid) and
                    self.is_valid_line(p1, mid) and
                    self.is_valid_line(mid, p2)):
                    fixed_path.append(mid)
                    fixed_path.append(p2)
                else:
                    fixed_path.append(p2)  # Keep original
        
        return fixed_path
    
    def _measure_multi_paths(self,
                           chromosome: np.ndarray,
                           starts: List[Tuple[float, float]],
                           goal: Tuple[float, float]) -> Dict[int, List[Tuple[float, float]]]:
        """
        Decode chromosome thành n paths
        
        Tương tự _measure() của QIEA nhưng cho nhiều paths
        """
        paths = {}
        
        for uav_id in range(self.num_uavs):
            start_idx = uav_id * self.num_waypoints_per_path * 2
            end_idx = start_idx + self.num_waypoints_per_path * 2
            path_chromosome = chromosome[start_idx:end_idx]
            
            # Decode như single-path
            path = self._measure_single_path(
                path_chromosome,
                starts[uav_id],
                goal
            )
            paths[uav_id] = path
        
        return paths
    
    def _interpolate_path(self,
                         path: List[Tuple[float, float]],
                         time: float,
                         total_time: float) -> Optional[Tuple[float, float]]:
        """Interpolate position tại thời điểm time"""
        if time < 0:
            return path[0] if path else None
        if time > total_time or total_time <= 0:
            return path[-1] if path else None
        
        # Calculate cumulative distances
        cumulative_distances = [0.0]
        for i in range(len(path) - 1):
            dist = euclidean_distance(path[i], path[i+1])
            cumulative_distances.append(cumulative_distances[-1] + dist)
        
        total_distance = cumulative_distances[-1] if cumulative_distances else 0.0
        if total_distance <= 0:
            return path[0] if path else None
        
        target_distance = (time / total_time) * total_distance
        
        # Find segment
        for i in range(len(cumulative_distances) - 1):
            if cumulative_distances[i] <= target_distance <= cumulative_distances[i+1]:
                # Interpolate trong segment
                segment_start = cumulative_distances[i]
                segment_end = cumulative_distances[i+1]
                segment_length = segment_end - segment_start
                
                if segment_length < 1e-6:
                    return path[i]
                
                alpha = (target_distance - segment_start) / segment_length
                p1 = path[i]
                p2 = path[i+1]
                
                x = p1[0] + alpha * (p2[0] - p1[0])
                y = p1[1] + alpha * (p2[1] - p1[1])
                return (x, y)
        
        return path[-1] if path else None
    
    def _evaluate_multi_path_fitness(self,
                                    paths: Dict[int, List[Tuple[float, float]]],
                                    starts: List[Tuple[float, float]],
                                    goal: Tuple[float, float],
                                    allow_invalid: bool = False) -> Tuple[float, List]:
        """
        Multi-objective fitness cho tất cả paths
        
        Objectives:
        1. Sum of individual path costs (length, energy, safety)
        2. Conflict penalties (UAVs quá gần nhau)
        3. Makespan (thời gian hoàn thành tất cả)
        4. Goal congestion (quá nhiều UAV đến goal cùng lúc)
        """
        if len(paths) == 0:
            return float('inf'), []
        
        # 1. Individual path costs
        total_path_cost = 0.0
        total_collision_penalty = 0.0  # Track total collision penalty across all paths
        path_lengths = {}
        arrival_times = {}
        
        for uav_id, path in paths.items():
            if len(path) < 2:
                return float('inf'), []
            
            # Check collisions với obstacles
            collision_penalty = 0
            for i in range(len(path) - 1):
                if not self.is_valid_line(path[i], path[i+1]):
                    if allow_invalid:
                        # In exploration phase, allow invalid paths but with high penalty
                        collision_penalty += 500.0  # Reduced from 1000.0
                    else:
                        collision_penalty += 1000.0
            
            for point in path:
                if not self.is_valid_point(point):
                    if allow_invalid:
                        collision_penalty += 250.0  # Reduced from 500.0
                    else:
                        collision_penalty += 500.0
            
            # Accumulate collision penalty
            total_collision_penalty += collision_penalty
            
            # Only return inf if not in exploration phase and has collisions
            if collision_penalty > 0 and not allow_invalid:
                return float('inf'), []
            
            # Calculate path cost - prioritize path length for QIEA optimization
            path_cost = calculate_path_cost(
                path,
                self.obstacles,
                weight_length=2.0,  # Increased from 1.0 to prioritize shorter paths
                weight_energy=0.3,  # Reduced from 0.5
                weight_safety=1.0  # Reduced from 2.0 to allow more exploration
            )
            total_path_cost += path_cost
            
            # Calculate length và arrival time
            path_length = sum(
                euclidean_distance(path[i], path[i+1])
                for i in range(len(path)-1)
            )
            path_lengths[uav_id] = path_length
            arrival_times[uav_id] = path_length / self.uav_speed
        
        # 2. Conflict detection và penalties
        conflicts = []
        conflict_penalty = 0.0
        
        uav_ids = list(paths.keys())
        time_step = 0.1  # Check mỗi 0.1s
        
        for i in range(len(uav_ids)):
            for j in range(i+1, len(uav_ids)):
                uav1_id = uav_ids[i]
                uav2_id = uav_ids[j]
                
                path1 = paths[uav1_id]
                path2 = paths[uav2_id]
                
                # Sample paths theo thời gian
                max_time = max(arrival_times[uav1_id], arrival_times[uav2_id])
                t = 0.0
                
                while t <= max_time:
                    pos1 = self._interpolate_path(path1, t, arrival_times[uav1_id])
                    pos2 = self._interpolate_path(path2, t, arrival_times[uav2_id])
                    
                    if pos1 and pos2:
                        distance = euclidean_distance(pos1, pos2)
                        if distance < self.min_separation:
                            # Reduced penalty to allow QIEA to explore shorter paths
                            conflict_penalty += 50.0 * (self.min_separation - distance)  # Reduced from 100.0
                            conflicts.append((uav1_id, uav2_id, t, pos1))
                    
                    t += time_step
        
        # 3. Makespan penalty (khuyến khích hoàn thành sớm) - reduced weight
        makespan = max(arrival_times.values()) if arrival_times else 0
        makespan_penalty = makespan * 0.05  # Reduced from 0.1 to prioritize path length
        
        # 4. Goal congestion penalty - reduced weight
        goal_congestion_penalty = 0.0
        time_window = 2.0  # 2 seconds window
        sorted_arrivals = sorted(arrival_times.items(), key=lambda x: x[1])
        
        for i in range(len(sorted_arrivals)):
            uav1_id, time1 = sorted_arrivals[i]
            congestion_count = 1
            
            for j in range(i+1, len(sorted_arrivals)):
                uav2_id, time2 = sorted_arrivals[j]
                if time2 - time1 <= time_window:
                    congestion_count += 1
                else:
                    break
            
            if congestion_count > 1:
                goal_congestion_penalty += congestion_count * 5.0  # Reduced from 10.0
        
        # Total fitness - prioritize path length
        # Path length is the primary objective, penalties are secondary
        # Add collision penalty if in exploration phase
        total_fitness = (
            total_path_cost +           # Individual costs (path length prioritized)
            collision_penalty +         # Collision penalty (only in exploration phase)
            conflict_penalty * 0.5 +    # Reduced conflict penalty weight
            makespan_penalty +          # Makespan
            goal_congestion_penalty * 0.5  # Reduced goal congestion weight
        )
        
        return total_fitness, conflicts
    
    def _initialize_population(self,
                             starts: List[Tuple[float, float]],
                             goal: Tuple[float, float],
                             seed_paths: Optional[Dict[int, List[Tuple[float, float]]]] = None) -> List[np.ndarray]:
        """Initialize population với seed paths"""
        population = []
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        
        # Seed individual
        if seed_paths:
            seed_chromosome = self._encode_multi_paths_to_qbits(seed_paths, starts, goal)
            population.append(seed_chromosome)
        
        # Variations of seed - further reduced to allow more random exploration
        num_variations = int(self.population_size * 0.15)  # Reduced from 0.25 to 0.15 to reduce seed stickiness
        if seed_paths:
            for _ in range(num_variations):
                # Increased variation range for better exploration
                variation = seed_chromosome.copy() + np.random.normal(0, 0.3, len(seed_chromosome))  # Increased from 0.25 to 0.3
                # Normalize
                for j in range(0, len(variation), 2):
                    alpha, beta = variation[j], variation[j+1]
                    norm = math.sqrt(alpha*alpha + beta*beta)
                    if norm > 0:
                        variation[j] = alpha / norm
                        variation[j+1] = beta / norm
                    else:
                        variation[j] = sqrt2_inv
                        variation[j+1] = sqrt2_inv
                population.append(variation)
        
        # Random initialization - more diverse exploration with multiple strategies
        while len(population) < self.population_size:
            # Strategy 1: Random uniform (50% of remaining - increased for more exploration)
            if len(population) < self.population_size * 0.5:
                chromosome = np.random.uniform(-0.5, 0.5, self.chromosome_size).astype(np.float64)
            # Strategy 2: Random directions towards goal (30% of remaining)
            elif len(population) < self.population_size * 0.8:
                # Initialize with random directions that point roughly towards goal
                chromosome = np.zeros(self.chromosome_size, dtype=np.float64)
                for uav_id in range(self.num_uavs):
                    start = starts[uav_id]
                    dx = goal[0] - start[0]
                    dy = goal[1] - start[1]
                    base_angle = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0.0
                    # Add random variation
                    for j in range(self.num_waypoints_per_path):
                        angle = base_angle + random.uniform(-math.pi/2, math.pi/2)
                        normalized_angle = (angle + math.pi) / (2 * math.pi)
                        q_angle = math.pi * normalized_angle
                        idx = (uav_id * self.num_waypoints_per_path + j) * 2
                        chromosome[idx] = math.cos(q_angle)
                        chromosome[idx + 1] = math.sin(q_angle)
            # Strategy 3: Pure random (remaining 10%)
            else:
                chromosome = np.random.uniform(-1.0, 1.0, self.chromosome_size).astype(np.float64)
            # Normalize
            for j in range(0, len(chromosome), 2):
                alpha, beta = chromosome[j], chromosome[j+1]
                norm = math.sqrt(alpha*alpha + beta*beta)
                if norm > 0:
                    chromosome[j] = alpha / norm
                    chromosome[j+1] = beta / norm
                else:
                    chromosome[j] = sqrt2_inv
                    chromosome[j+1] = sqrt2_inv
            population.append(chromosome)
        
        return population
    
    def _quantum_rotation_gate(self,
                              q_chromosome: np.ndarray,
                              best_chromosome: np.ndarray,
                              current_fitness: float,
                              best_fitness: float,
                              generation: int) -> np.ndarray:
        """Quantum rotation gate (tương tự QIEA)"""
        updated = q_chromosome.copy()
        
        # Adaptive rotation angle - keep larger angle for better exploration
        # Reduce less aggressively to maintain exploration capability
        base_angle = self.rotation_angle * (1.0 - generation / self.max_generations * 0.3)  # Reduced from 0.5 to 0.3
        base_angle = max(base_angle, self.rotation_angle * 0.5)  # Increased minimum from 0.1 to 0.5
        
        if best_fitness > 0 and current_fitness != float('inf'):
            fitness_gap = abs(current_fitness - best_fitness)
            fitness_ratio = fitness_gap / best_fitness if best_fitness > 0 else 0
            gap_multiplier = 1.0 + min(fitness_ratio, 1.0)
            adaptive_angle = base_angle * gap_multiplier
        else:
            adaptive_angle = base_angle
        
        for i in range(0, len(q_chromosome), 2):
            alpha = q_chromosome[i]
            beta = q_chromosome[i + 1]
            best_alpha = best_chromosome[i]
            best_beta = best_chromosome[i + 1]
            
            if current_fitness > best_fitness:
                dot_product = alpha * best_alpha + beta * best_beta
                delta_theta = adaptive_angle if dot_product > 0 else -adaptive_angle
            else:
                delta_theta = -adaptive_angle * 0.3
            
            cos_theta = math.cos(delta_theta)
            sin_theta = math.sin(delta_theta)
            
            new_alpha = alpha * cos_theta - beta * sin_theta
            new_beta = alpha * sin_theta + beta * cos_theta
            
            norm = math.sqrt(new_alpha * new_alpha + new_beta * new_beta)
            if norm > 0:
                new_alpha /= norm
                new_beta /= norm
            else:
                sqrt2_inv = 1.0 / math.sqrt(2.0)
                new_alpha = sqrt2_inv
                new_beta = sqrt2_inv
            
            updated[i] = new_alpha
            updated[i + 1] = new_beta
        
        return updated
    
    def _mutate(self, q_chromosome: np.ndarray) -> np.ndarray:
        """Mutation (tương tự QIEA)"""
        mutated = q_chromosome.copy()
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = random.uniform(-1, 1)
        
        # Normalize
        for i in range(0, len(mutated), 2):
            alpha = mutated[i]
            beta = mutated[i+1]
            norm = math.sqrt(alpha * alpha + beta * beta)
            if norm > 0:
                mutated[i] = alpha / norm
                mutated[i+1] = beta / norm
            else:
                mutated[i] = sqrt2_inv
                mutated[i+1] = sqrt2_inv
        
        return mutated
    
    def _large_mutate(self, q_chromosome: np.ndarray) -> np.ndarray:
        """Large mutation to escape local optima - mutate entire path segments"""
        mutated = q_chromosome.copy()
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        
        # Mutate 20-30% of waypoints with larger changes
        num_mutations = max(1, int(len(q_chromosome) // 2 * random.uniform(0.2, 0.3)))
        mutation_indices = random.sample(range(0, len(q_chromosome), 2), num_mutations)
        
        for idx in mutation_indices:
            # Large mutation: random direction
            angle = random.uniform(0, 2 * math.pi)
            mutated[idx] = math.cos(angle)
            mutated[idx + 1] = math.sin(angle)
        
        # Normalize
        for i in range(0, len(mutated), 2):
            alpha = mutated[i]
            beta = mutated[i+1]
            norm = math.sqrt(alpha * alpha + beta * beta)
            if norm > 0:
                mutated[i] = alpha / norm
                mutated[i+1] = beta / norm
            else:
                mutated[i] = sqrt2_inv
                mutated[i+1] = sqrt2_inv
        
        return mutated
    
    def plan(self,
            starts: List[Tuple[float, float]],
            goal: Tuple[float, float],
            seed_paths: Optional[Dict[int, List[Tuple[float, float]]]] = None) -> MultiPathResult:
        """
        Plan paths cho tất cả n UAVs đến 1 goal
        
        Optimize đồng thời tất cả paths trong 1 QIEA run
        """
        start_time = time.time()
        
        if len(starts) != self.num_uavs:
            return MultiPathResult(
                paths={},
                timestamps={},
                success=False,
                computation_time=time.time() - start_time,
                total_cost=float('inf'),
                makespan=0,
                conflicts=[],
                message=f"Mismatch: {len(starts)} starts but {self.num_uavs} UAVs"
            )
        
        # Validate starts và goal
        for start in starts:
            if not self.is_valid_point(start):
                return MultiPathResult(
                    paths={},
                    timestamps={},
                    success=False,
                    computation_time=time.time() - start_time,
                    total_cost=float('inf'),
                    makespan=0,
                    conflicts=[],
                    message="Invalid start point"
                )
        
        if not self.is_valid_point(goal):
            return MultiPathResult(
                paths={},
                timestamps={},
                success=False,
                computation_time=time.time() - start_time,
                total_cost=float('inf'),
                makespan=0,
                conflicts=[],
                message="Invalid goal point"
            )
        
        # Initialize population
        population = self._initialize_population(starts, goal, seed_paths)
        
        # Evolution
        best_chromosome = None
        best_fitness = float('inf')
        best_paths = None
        best_conflicts = []
        
        # Cache fitness values to avoid double evaluation
        fitness_cache = {}
        
        for generation in range(self.max_generations):
            # Exploration phase: first 30% of generations focus on exploration
            is_exploration_phase = generation < self.max_generations * 0.3
            
            # Evaluate all individuals (only once per generation)
            for i, chromosome in enumerate(population):
                # Check cache first
                cache_key = tuple(chromosome.flatten()[:10])  # Use first 10 values as key
                if cache_key not in fitness_cache:
                    paths = self._measure_multi_paths(chromosome, starts, goal)
                    fitness, conflicts = self._evaluate_multi_path_fitness(paths, starts, goal, 
                                                                          allow_invalid=is_exploration_phase)
                    fitness_cache[cache_key] = (fitness, paths, conflicts)
                else:
                    fitness, paths, conflicts = fitness_cache[cache_key]
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_chromosome = chromosome.copy()
                    best_paths = paths
                    best_conflicts = conflicts
            
            # Update population
            for i, chromosome in enumerate(population):
                if best_chromosome is not None:
                    # Get fitness from cache (already evaluated above)
                    cache_key = tuple(chromosome.flatten()[:10])
                    if cache_key in fitness_cache:
                        fitness, _, _ = fitness_cache[cache_key]
                    else:
                        paths = self._measure_multi_paths(chromosome, starts, goal)
                        fitness, _, _ = self._evaluate_multi_path_fitness(paths, starts, goal,
                                                                          allow_invalid=is_exploration_phase)
                    
                    # In exploration phase, add more randomness
                    if is_exploration_phase:
                        # More aggressive exploration: sometimes rotate away from best
                        if random.random() < 0.3:  # 30% chance to explore opposite direction
                            # Create "anti-best" chromosome for exploration
                            anti_best = best_chromosome.copy()
                            for j in range(0, len(anti_best), 2):
                                # Rotate 180 degrees
                                alpha, beta = anti_best[j], anti_best[j+1]
                                anti_best[j] = -alpha
                                anti_best[j+1] = -beta
                            
                            population[i] = self._quantum_rotation_gate(
                                chromosome,
                                anti_best,  # Rotate towards anti-best for exploration
                                fitness,
                                best_fitness * 1.5,  # Pretend anti-best is worse
                                generation
                            )
                        else:
                            population[i] = self._quantum_rotation_gate(
                                chromosome,
                                best_chromosome,
                                fitness,
                                best_fitness,
                                generation
                            )
                    else:
                        # Normal rotation towards best
                        population[i] = self._quantum_rotation_gate(
                            chromosome,
                            best_chromosome,
                            fitness,
                            best_fitness,
                            generation
                        )
                    
                    # Mutation - higher rate in exploration phase
                    mutation_rate_adj = self.mutation_rate * (1.5 if is_exploration_phase else 1.0)
                    if random.random() < mutation_rate_adj:
                        population[i] = self._mutate(population[i])
                    
                    # Large mutation - higher chance in exploration phase
                    large_mutate_chance = 0.1 if is_exploration_phase else 0.05
                    if random.random() < large_mutate_chance:
                        population[i] = self._large_mutate(population[i])
            
            # Clear cache periodically to prevent memory issues
            if generation % 10 == 0:
                fitness_cache.clear()
            
            # Early stopping
            if best_fitness < 1e-6:
                break
        
        computation_time = time.time() - start_time
        
        if best_paths is None:
            return MultiPathResult(
                paths={},
                timestamps={},
                success=False,
                computation_time=computation_time,
                total_cost=float('inf'),
                makespan=0,
                conflicts=[],
                message="No valid solution found"
            )
        
        # Calculate timestamps
        timestamps = {}
        for uav_id, path in best_paths.items():
            path_length = sum(
                euclidean_distance(path[i], path[i+1])
                for i in range(len(path)-1)
            )
            arrival_time = path_length / self.uav_speed
            
            # Generate timestamps
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
            paths=best_paths,
            timestamps=timestamps,
            success=True,
            computation_time=computation_time,
            total_cost=best_fitness,
            makespan=makespan,
            conflicts=best_conflicts,
            message=f"Optimized {self.num_uavs} paths simultaneously"
        )
