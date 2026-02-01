"""
Quantum-Inspired Evolutionary Algorithm (QIEA) for Path Planning
"""

import numpy as np
import random
import math
from typing import List, Tuple, Optional
from .base import PathPlanner, PathResult
from .utils import (
    euclidean_distance, 
    calculate_path_cost,
    smooth_path,
    get_neighbors_8_connected
)
import time


class QIEA(PathPlanner):
    """
    Quantum-Inspired Evolutionary Algorithm for multi-objective path planning
    
    Uses quantum bits (Q-bits) to represent solutions and quantum rotation gates
    to evolve the population towards better solutions.
    """
    
    def __init__(self, obstacles: List[Tuple[float, float, float]], 
                 width: int, height: int,
                 population_size: int = 50,
                 max_generations: int = 100,
                 rotation_angle: float = 0.01 * math.pi,
                 mutation_rate: float = 0.1):
        """
        Initialize QIEA
        
        Args:
            obstacles: List of obstacles
            width: Map width
            height: Map height
            population_size: Size of quantum population
            max_generations: Maximum number of generations
            rotation_angle: Angle for quantum rotation gate
            mutation_rate: Probability of mutation
        """
        super().__init__(obstacles, width, height)
        self.population_size = population_size
        self.max_generations = max_generations
        self.rotation_angle = rotation_angle
        self.mutation_rate = mutation_rate
    
    def _encode_path_to_qbits(self, path: List[Tuple[float, float]], 
                              num_waypoints: int) -> np.ndarray:
        """
        Encode a classical path into Q-bit chromosome using direction vectors
        
        Improved encoding: Instead of encoding absolute positions, we encode
        direction vectors. This makes rotation gates more effective as they
        rotate directions rather than positions.
        
        Args:
            path: List of waypoints
            num_waypoints: Target number of waypoints in chromosome
        
        Returns:
            Q-bit chromosome encoding direction vectors
        """
        chromosome = np.zeros(num_waypoints * 2, dtype=np.float64)
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        
        if len(path) < 2:
            # Empty path, return superposition state
            return np.full(num_waypoints * 2, sqrt2_inv, dtype=np.float64)
        
        # Extract direction vectors from path segments
        directions = []
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 1e-10:
                # Normalize direction vector
                dx /= length
                dy /= length
                # Calculate angle: atan2(dy, dx) in range [-π, π]
                angle = math.atan2(dy, dx)
                directions.append(angle)
            else:
                # Zero-length segment, use previous direction or default
                if directions:
                    directions.append(directions[-1])
                else:
                    # Default direction towards goal
                    goal_dir = math.atan2(path[-1][1] - path[0][1], 
                                         path[-1][0] - path[0][0])
                    directions.append(goal_dir)
        
        # If we have fewer directions than waypoints, interpolate
        if len(directions) < num_waypoints:
            # Interpolate directions
            interpolated_directions = []
            for i in range(num_waypoints):
                t = i / (num_waypoints - 1) if num_waypoints > 1 else 0
                idx = t * (len(directions) - 1)
                idx_low = int(idx)
                idx_high = min(idx_low + 1, len(directions) - 1)
                
                if idx_low == idx_high:
                    interpolated_directions.append(directions[idx_low])
                else:
                    # Interpolate angle (handle wrap-around)
                    angle_low = directions[idx_low]
                    angle_high = directions[idx_high]
                    # Normalize angles to [0, 2π] for interpolation
                    angle_low_norm = angle_low + math.pi if angle_low < 0 else angle_low + math.pi
                    angle_high_norm = angle_high + math.pi if angle_high < 0 else angle_high + math.pi
                    
                    local_t = idx - idx_low
                    interp_angle_norm = angle_low_norm + local_t * (angle_high_norm - angle_low_norm)
                    interp_angle = interp_angle_norm - math.pi
                    interpolated_directions.append(interp_angle)
            
            directions = interpolated_directions
        elif len(directions) > num_waypoints:
            # Downsample directions
            step = len(directions) / num_waypoints
            directions = [directions[int(i * step)] for i in range(num_waypoints)]
        
        # Encode each direction angle as Q-bit
        # Map angle from [-π, π] to [0, 2π], then normalize to [0, 1]
        for i, angle in enumerate(directions[:num_waypoints]):
            # Normalize angle to [0, 2π]
            normalized_angle = (angle + math.pi) / (2 * math.pi)
            
            # Encode as Q-bit: alpha = cos(π * normalized_angle), beta = sin(π * normalized_angle)
            # This maps direction to Q-bit representation
            q_angle = math.pi * normalized_angle
            chromosome[i * 2] = math.cos(q_angle)
            chromosome[i * 2 + 1] = math.sin(q_angle)
        
        # Normalize all Q-bits (should already be normalized, but ensure)
        for i in range(0, len(chromosome), 2):
            alpha, beta = chromosome[i], chromosome[i+1]
            norm = math.sqrt(alpha*alpha + beta*beta)
            if norm > 1e-10:
                chromosome[i] = alpha / norm
                chromosome[i+1] = beta / norm
            else:
                # Fallback to superposition
                chromosome[i] = sqrt2_inv
                chromosome[i+1] = sqrt2_inv
        
        return chromosome
    
    def _initialize_population(self, start: Tuple[float, float], 
                              goal: Tuple[float, float],
                              seed_path: Optional[List[Tuple[float, float]]] = None) -> List[np.ndarray]:
        """
        Initialize quantum population with optional seed path
        
        Each individual is a Q-bit chromosome representing a path
        Q-bits are initialized to superposition state: (1/√2, 1/√2)
        
        If seed_path is provided:
        - First individual: encoded seed path
        - Next 20%: variations of seed path
        - Remaining: random initialization
        
        Args:
            start: Start point
            goal: Goal point
            seed_path: Optional initial path to seed the population
        
        Returns:
            List of Q-bit chromosomes
        """
        population = []
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        
        # Estimate path length (heuristic)
        estimated_length = euclidean_distance(start, goal)
        num_waypoints = max(5, int(estimated_length / 5))  # Adaptive number of waypoints
        
        # First individual: seed path (if provided)
        if seed_path is not None and len(seed_path) > 2:
            seed_chromosome = self._encode_path_to_qbits(seed_path, num_waypoints)
            population.append(seed_chromosome)
        
        # Next 20%: variations of seed path
        num_variations = int(self.population_size * 0.2)
        if seed_path is not None and len(seed_path) > 2:
            for _ in range(num_variations):
                # Add Gaussian noise to seed chromosome
                variation = seed_chromosome.copy() + np.random.normal(0, 0.15, len(seed_chromosome))
                
                # Normalize Q-bits
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
        
        # Remaining: random initialization (superposition state)
        while len(population) < self.population_size:
            chromosome = np.full(num_waypoints * 2, sqrt2_inv, dtype=np.float64)
            population.append(chromosome)
        
        return population
    
    def _measure(self, q_chromosome: np.ndarray, 
                start: Tuple[float, float],
                goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Measure Q-bit chromosome to get a classical path from direction vectors
        
        Improved measurement: Decode Q-bits as direction angles, then generate
        waypoints by moving in those directions. This maintains path continuity
        and makes better use of the direction-based encoding.
        
        Args:
            q_chromosome: Q-bit chromosome encoding direction vectors
            start: Start point
            goal: Goal point
        
        Returns:
            List of waypoints (path)
        """
        num_waypoints = len(q_chromosome) // 2
        path = [start]
        current_pos = start
        
        # Calculate adaptive step size based on remaining distance
        total_estimated_distance = euclidean_distance(start, goal)
        base_step_size = total_estimated_distance / (num_waypoints + 1)
        
        # Generate waypoints by following decoded directions
        for i in range(num_waypoints):
            alpha = q_chromosome[i * 2]
            beta = q_chromosome[i * 2 + 1]
            
            # Decode direction angle from Q-bit
            # Q-bit was encoded as: (cos(π*normalized_angle), sin(π*normalized_angle))
            # where normalized_angle = (original_angle + π) / (2π) is in [0, 1]
            # So: q_angle = atan2(beta, alpha) is in [0, 2π] range
            # We need to recover: original_angle = normalized_angle * 2π - π
            
            q_angle = math.atan2(beta, alpha)  # This gives angle in [-π, π] range
            
            # Normalize q_angle to [0, 2π]
            if q_angle < 0:
                q_angle += 2 * math.pi
            
            # q_angle now represents π * normalized_angle
            # So normalized_angle = q_angle / π (in [0, 2] range)
            normalized_angle = q_angle / math.pi
            
            # Map back to original angle range [-π, π]
            # original_angle = normalized_angle * 2π - π
            # But since normalized_angle is in [0, 2], we need to map it to [0, 1] first
            if normalized_angle > 1.0:
                normalized_angle = normalized_angle - 1.0  # Wrap to [0, 1]
            
            # Now convert to actual direction angle [-π, π]
            direction_angle = normalized_angle * 2 * math.pi - math.pi
            
            # Calculate step size (adaptive based on remaining distance)
            remaining_distance = euclidean_distance(current_pos, goal)
            step_size = min(base_step_size, remaining_distance / (num_waypoints - i + 1))
            
            # Add some exploration based on Q-bit probabilities
            # Use |alpha|^2 and |beta|^2 to add randomness
            prob_0 = alpha * alpha
            prob_1 = beta * beta
            total_prob = prob_0 + prob_1
            
            if total_prob > 0:
                # Add small random variation based on Q-bit state
                exploration_factor = 0.1 * (1.0 - abs(prob_0 - prob_1))  # More exploration if balanced
                direction_angle += random.uniform(-exploration_factor, exploration_factor)
                step_size *= (1.0 + random.uniform(-0.2, 0.2))  # Vary step size slightly
            
            # Generate waypoint by moving in decoded direction
            dx = math.cos(direction_angle) * step_size
            dy = math.sin(direction_angle) * step_size
            
            new_waypoint = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Clamp to bounds
            new_waypoint = (
                max(0, min(self.width - 1, new_waypoint[0])),
                max(0, min(self.height - 1, new_waypoint[1]))
            )
            
            # Only add if valid (not in obstacle)
            if self.is_valid_point(new_waypoint):
                path.append(new_waypoint)
                current_pos = new_waypoint
            else:
                # Try to find nearby valid point or adjust direction
                # Try 8 directions around the intended direction
                found_valid = False
                for angle_offset in [-math.pi/4, -math.pi/8, 0, math.pi/8, math.pi/4]:
                    adjusted_angle = direction_angle + angle_offset
                    dx_adj = math.cos(adjusted_angle) * step_size
                    dy_adj = math.sin(adjusted_angle) * step_size
                    adjusted_waypoint = (
                        max(0, min(self.width - 1, current_pos[0] + dx_adj)),
                        max(0, min(self.height - 1, current_pos[1] + dy_adj))
                    )
                    
                    if self.is_valid_point(adjusted_waypoint):
                        path.append(adjusted_waypoint)
                        current_pos = adjusted_waypoint
                        found_valid = True
                        break
                
                if not found_valid:
                    # Skip this waypoint if can't find valid position
                    continue
        
        # Ensure goal is in path
        if len(path) == 0 or path[-1] != goal:
            # Check if we can connect directly to goal
            if len(path) > 0 and self.is_valid_line(path[-1], goal):
                path.append(goal)
            else:
                # Try to find path to goal
                path.append(goal)
        
        # Smooth and validate path
        path = self._validate_and_smooth_path(path)
        return path
    
    def _validate_and_smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Validate and smooth a path with strict obstacle avoidance
        
        This function ensures the path never crosses obstacles by:
        1. Removing waypoints inside obstacles
        2. Adding intermediate waypoints if segments cross obstacles
        3. Smoothing while maintaining obstacle-free guarantee
        """
        if len(path) < 2:
            return path
        
        # Step 1: Remove waypoints that are inside obstacles
        valid_path = []
        for point in path:
            if self.is_valid_point(point):
                valid_path.append(point)
            elif len(valid_path) > 0:
                # Point is invalid, try to find nearby valid point
                # or skip if we can connect directly
                pass
        
        if len(valid_path) < 2:
            # Not enough valid points, return start and goal only
            return [path[0], path[-1]]
        
        # Ensure start and goal are in path
        if valid_path[0] != path[0]:
            valid_path.insert(0, path[0])
        if valid_path[-1] != path[-1]:
            valid_path.append(path[-1])
        
        # Step 2: Fix segments that cross obstacles by adding intermediate waypoints
        fixed_path = [valid_path[0]]
        
        for i in range(len(valid_path) - 1):
            p1 = fixed_path[-1]
            p2 = valid_path[i + 1]
            
            if self.is_valid_line(p1, p2):
                # Segment is valid, add p2
                fixed_path.append(p2)
            else:
                # Segment crosses obstacle, need to find detour
                detour = self._find_detour(p1, p2)
                if detour:
                    fixed_path.extend(detour)
                    fixed_path.append(p2)
                else:
                    # Can't find detour, try to add intermediate point
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2
                    mid = (mid_x, mid_y)
                    
                    if self.is_valid_point(mid):
                        if self.is_valid_line(p1, mid) and self.is_valid_line(mid, p2):
                            fixed_path.append(mid)
                            fixed_path.append(p2)
                        else:
                            # Still can't connect, keep original (will be penalized)
                            fixed_path.append(p2)
                    else:
                        # Mid point invalid, try grid neighbors
                        mid_int = (int(round(mid_x)), int(round(mid_y)))
                        from .utils import get_neighbors_8_connected
                        neighbors = get_neighbors_8_connected(mid_int, self.width, self.height)
                        
                        added = False
                        for neighbor in neighbors:
                            neighbor_float = (float(neighbor[0]), float(neighbor[1]))
                            if (self.is_valid_point(neighbor_float) and
                                self.is_valid_line(p1, neighbor_float) and
                                self.is_valid_line(neighbor_float, p2)):
                                fixed_path.append(neighbor_float)
                                fixed_path.append(p2)
                                added = True
                                break
                        
                        if not added:
                            # Fallback: keep original point
                            fixed_path.append(p2)
        
        # Step 3: Smooth path while maintaining obstacle-free guarantee
        try:
            smoothed = smooth_path(fixed_path, self.obstacles, max_iterations=100)
            # Final validation of smoothed path
            final_path = [smoothed[0]]
            for i in range(1, len(smoothed)):
                if self.is_valid_line(final_path[-1], smoothed[i]):
                    final_path.append(smoothed[i])
                else:
                    # Can't smooth this segment, keep original
                    final_path.append(smoothed[i])
            return final_path
        except:
            return fixed_path
    
    def _find_detour(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Find a detour path between p1 and p2 that avoids obstacles
        
        Uses simple strategy: try points around the midpoint
        """
        from .utils import get_neighbors_8_connected, euclidean_distance
        
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        mid_int = (int(round(mid_x)), int(round(mid_y)))
        
        # Try neighbors of midpoint
        neighbors = get_neighbors_8_connected(mid_int, self.width, self.height)
        
        # Sort by distance to line p1-p2
        candidates = []
        for neighbor in neighbors:
            neighbor_float = (float(neighbor[0]), float(neighbor[1]))
            if self.is_valid_point(neighbor_float):
                # Check if this creates valid segments
                if (self.is_valid_line(p1, neighbor_float) and
                    self.is_valid_line(neighbor_float, p2)):
                    # Calculate distance from line
                    dist_to_line = abs((p2[1] - p1[1]) * neighbor_float[0] - 
                                      (p2[0] - p1[0]) * neighbor_float[1] + 
                                      p2[0] * p1[1] - p2[1] * p1[0]) / euclidean_distance(p1, p2)
                    candidates.append((dist_to_line, neighbor_float))
        
        if candidates:
            # Return the closest point to the line
            candidates.sort(key=lambda x: x[0])
            return [candidates[0][1]]
        
        return None
    
    def _evaluate_fitness(self, path: List[Tuple[float, float]]) -> float:
        """
        Evaluate fitness of a path (lower is better)
        
        Uses multi-objective cost function with heavy penalty for obstacles
        """
        if len(path) < 2:
            return float('inf')
        
        # STRICT validation: Check if path is valid
        # Heavy penalty for any segment crossing obstacles
        collision_penalty = 0
        for i in range(len(path) - 1):
            if not self.is_valid_line(path[i], path[i + 1]):
                # Heavy penalty for crossing obstacles
                collision_penalty += 1000.0
        
        # Check if any waypoint is inside obstacle
        for point in path:
            if not self.is_valid_point(point):
                collision_penalty += 500.0
        
        # If there are collisions, return very high cost
        if collision_penalty > 0:
            return float('inf')
        
        # Calculate multi-objective cost (only if path is valid)
        cost = calculate_path_cost(
            path, 
            self.obstacles,
            weight_length=1.0,
            weight_energy=0.5,
            weight_safety=2.0
        )
        
        return cost
    
    def _quantum_rotation_gate(self, q_chromosome: np.ndarray, 
                              best_chromosome: np.ndarray,
                              current_fitness: float,
                              best_fitness: float,
                              generation: int = 0) -> np.ndarray:
        """
        Apply quantum rotation gate to update Q-bits with adaptive rotation angle
        
        Adaptive rotation angle based on:
        1. Generation number (decreases over time for fine-tuning)
        2. Fitness gap (larger gap = larger rotation for faster convergence)
        
        Rotates Q-bits towards the best solution
        
        Args:
            q_chromosome: Current Q-bit chromosome
            best_chromosome: Best Q-bit chromosome found so far
            current_fitness: Fitness of current chromosome
            best_fitness: Best fitness found so far
            generation: Current generation number (for adaptive angle)
        
        Returns:
            Updated Q-bit chromosome
        """
        updated = q_chromosome.copy()
        
        # Adaptive rotation angle
        # Base angle decreases over generations (exploration -> exploitation)
        base_angle = self.rotation_angle * (1.0 - generation / self.max_generations * 0.5)
        base_angle = max(base_angle, self.rotation_angle * 0.1)  # Minimum 10% of original
        
        # Adaptive based on fitness gap
        # Larger gap means we need larger rotation to converge faster
        if best_fitness > 0 and current_fitness != float('inf'):
            fitness_gap = abs(current_fitness - best_fitness)
            fitness_ratio = fitness_gap / best_fitness if best_fitness > 0 else 0
            
            # Scale rotation angle based on gap (1.0 to 2.0x)
            gap_multiplier = 1.0 + min(fitness_ratio, 1.0)  # Cap at 2x
            adaptive_angle = base_angle * gap_multiplier
        else:
            adaptive_angle = base_angle
        
        for i in range(0, len(q_chromosome), 2):
            alpha = q_chromosome[i]
            beta = q_chromosome[i + 1]
            best_alpha = best_chromosome[i]
            best_beta = best_chromosome[i + 1]
            
            # Determine rotation direction
            if current_fitness > best_fitness:
                # Rotate towards best solution
                # Check if current and best are in same direction
                dot_product = alpha * best_alpha + beta * best_beta
                if dot_product > 0:
                    delta_theta = adaptive_angle
                else:
                    delta_theta = -adaptive_angle
            else:
                # Current is better or equal, rotate slightly away to maintain diversity
                delta_theta = -adaptive_angle * 0.3
            
            # Apply rotation gate
            cos_theta = math.cos(delta_theta)
            sin_theta = math.sin(delta_theta)
            
            new_alpha = alpha * cos_theta - beta * sin_theta
            new_beta = alpha * sin_theta + beta * cos_theta
            
            # Normalize
            norm = math.sqrt(new_alpha * new_alpha + new_beta * new_beta)
            if norm > 0:
                new_alpha /= norm
                new_beta /= norm
            else:
                # Fallback to superposition
                sqrt2_inv = 1.0 / math.sqrt(2.0)
                new_alpha = sqrt2_inv
                new_beta = sqrt2_inv
            
            updated[i] = new_alpha
            updated[i + 1] = new_beta
        
        return updated
    
    def _mutate(self, q_chromosome: np.ndarray) -> np.ndarray:
        """Apply mutation to Q-bit chromosome"""
        mutated = q_chromosome.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Random mutation
                mutated[i] = random.uniform(-1, 1)
        
        # Normalize Q-bits
        for i in range(0, len(mutated), 2):
            alpha = mutated[i]
            beta = mutated[i + 1]
            norm = math.sqrt(alpha * alpha + beta * beta)
            if norm > 0:
                mutated[i] = alpha / norm
                mutated[i + 1] = beta / norm
        
        return mutated
    
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float],
             seed_path: Optional[List[Tuple[float, float]]] = None) -> PathResult:
        """
        Plan path using QIEA
        
        Args:
            start: Start point (x, y)
            goal: Goal point (x, y)
            seed_path: Optional initial path to seed the population (from classical algorithms)
        
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
                message="Start point is invalid"
            )
        
        if not self.is_valid_point(goal):
            return PathResult(
                path=[],
                success=False,
                computation_time=time.time() - start_time,
                path_length=0.0,
                cost=float('inf'),
                num_nodes_explored=0,
                message="Goal point is invalid"
            )
        
        # Initialize population with optional seed path
        population = self._initialize_population(start, goal, seed_path=seed_path)
        
        # Evaluate initial population
        best_chromosome = None
        best_fitness = float('inf')
        best_path = None
        
        num_explored = 0
        
        for generation in range(self.max_generations):
            # Evaluate all individuals
            for i, q_chromosome in enumerate(population):
                path = self._measure(q_chromosome, start, goal)
                fitness = self._evaluate_fitness(path)
                num_explored += 1
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_chromosome = q_chromosome.copy()
                    best_path = path
            
            # Update population using quantum rotation gates
            for i, q_chromosome in enumerate(population):
                if best_chromosome is not None:
                    path = self._measure(q_chromosome, start, goal)
                    fitness = self._evaluate_fitness(path)
                    
                    # Apply rotation gate with adaptive angle (pass generation number)
                    population[i] = self._quantum_rotation_gate(
                        q_chromosome, 
                        best_chromosome,
                        fitness,
                        best_fitness,
                        generation=generation
                    )
                    
                    # Apply mutation
                    if random.random() < self.mutation_rate:
                        population[i] = self._mutate(population[i])
            
            # Early stopping if good solution found
            if best_fitness < 1e-6:
                break
        
        computation_time = time.time() - start_time
        
        if best_path is None or len(best_path) < 2:
            return PathResult(
                path=[],
                success=False,
                computation_time=computation_time,
                path_length=0.0,
                cost=float('inf'),
                num_nodes_explored=num_explored,
                message="No valid path found"
            )
        
        path_length = sum(euclidean_distance(best_path[i], best_path[i+1]) 
                         for i in range(len(best_path)-1))
        
        return PathResult(
            path=best_path,
            success=True,
            computation_time=computation_time,
            path_length=path_length,
            cost=best_fitness,
            num_nodes_explored=num_explored,
            message=f"Path found after {generation + 1} generations"
        )

