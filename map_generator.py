"""
Map Generator for UAV Trajectory Planning
Generates 3 types of maps with circular obstacles:
- m1: Sparse environment with low obstacle density
- m2: Dense environment with high obstacle density
- m3: Maximum difficulty with only 1-2 possible paths
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List
import json
import os
from pathlib import Path


class MapGenerator:
    """Generate maps with circular obstacles for UAV path planning"""
    
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
        self.start = None
        self.goal = None
    
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
    
    def _find_free_start_goal(self, min_distance: float = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Find a valid start and goal pair in free space"""
        if min_distance is None:
            min_distance = min(self.width, self.height) * 0.3
        
        max_attempts = 1000
        for _ in range(max_attempts):
            # Try random positions
            free_cells = np.argwhere(~self.grid)
            if len(free_cells) < 2:
                break
            
            indices = np.random.choice(len(free_cells), 2, replace=False)
            start = tuple(free_cells[indices[0]][::-1])  # (x, y)
            goal = tuple(free_cells[indices[1]][::-1])
            
            distance = np.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
            if distance >= min_distance:
                return start, goal
        
        # Fallback: use corners if available
        corners = [(0, 0), (self.width-1, 0), (0, self.height-1), (self.width-1, self.height-1)]
        free_corners = [(x, y) for x, y in corners if not self.grid[y, x]]
        
        if len(free_corners) >= 2:
            return free_corners[0], free_corners[-1]
        
        # Last resort: find any two free cells
        free_cells = np.argwhere(~self.grid)
        if len(free_cells) >= 2:
            return tuple(free_cells[0][::-1]), tuple(free_cells[-1][::-1])
        
        return (0, 0), (self.width-1, self.height-1)
    
    def generate_m1_sparse(self, num_obstacles: int = None,
                          min_radius: float = None, max_radius: float = None,
                          allow_denser_packing: bool = False) -> bool:
        """
        Generate m1: Sparse environment with low obstacle density

        Args:
            num_obstacles: Number of obstacles (auto-calculated if None)
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            allow_denser_packing: If True, allow more overlap to reach high obstacle counts (e.g. scaled maps)

        Returns:
            True if successful, False otherwise
        """
        self.grid = np.zeros((self.height, self.width), dtype=bool)
        self.obstacles = []

        # Auto-calculate parameters based on map size
        if num_obstacles is None:
            area = self.width * self.height
            num_obstacles = int(area * 0.05)  # ~5% of cells as obstacles

        if min_radius is None:
            min_radius = min(self.width, self.height) * 0.02

        if max_radius is None:
            max_radius = min(self.width, self.height) * 0.08

        max_attempts = max(1000, num_obstacles * 3)  # scale for large maps
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

            # Overlap check: allow more overlap when allow_denser_packing (for scaled maps)
            overlap_margin = -radius * 0.8 if allow_denser_packing else -radius * 0.3
            if allow_denser_packing:
                overlap_count = sum(
                    1 for (ox, oy, orad) in self.obstacles
                    if self._circles_overlap(cx, cy, radius, ox, oy, orad, min_distance=overlap_margin)
                )
                if overlap_count <= 12:  # allow overlaps to reach scaled NFZ count (e.g. 1280 for 200x200)
                    self._add_circle_to_grid(cx, cy, radius)
            else:
                overlaps = any(
                    self._circles_overlap(cx, cy, radius, ox, oy, orad, min_distance=overlap_margin)
                    for ox, oy, orad in self.obstacles
                )
                if not overlaps:
                    self._add_circle_to_grid(cx, cy, radius)

        # Find start and goal
        self.start, self.goal = self._find_free_start_goal()

        # Verify path exists
        if not self._check_path_exists(self.start, self.goal):
            # Try to remove some obstacles near the path
            return self.generate_m1_sparse(int(num_obstacles * 0.9), min_radius, max_radius, allow_denser_packing)

        return True
    
    def generate_m2_dense(self, num_obstacles: int = None,
                         min_radius: float = None, max_radius: float = None) -> bool:
        """
        Generate m2: Dense environment with high obstacle density
        
        Args:
            num_obstacles: Number of obstacles (auto-calculated if None)
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
        
        Returns:
            True if successful, False otherwise
        """
        self.grid = np.zeros((self.height, self.width), dtype=bool)
        self.obstacles = []
        
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
        
        # Find start and goal
        self.start, self.goal = self._find_free_start_goal()
        
        # Verify path exists
        if not self._check_path_exists(self.start, self.goal):
            # Try with fewer obstacles
            return self.generate_m2_dense(int(num_obstacles * 0.9), min_radius, max_radius)
        
        return True
    
    def generate_m3_trap(self) -> bool:
        """
        Generate m3: Maximum difficulty with only 1-2 possible paths
        Creates structured obstacles to form narrow passages (choke points)
        
        Returns:
            True if successful, False otherwise
        """
        self.grid = np.zeros((self.height, self.width), dtype=bool)
        self.obstacles = []
        
        # Strategy: Create U-shaped or corridor-like structures
        # This creates narrow passages that limit path options
        
        # Create large blocking obstacles
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
        # Create 1-2 narrow passages
        passage_width = min(self.width, self.height) * 0.1
        
        # Left passage
        passage_y = map_center_y - u_height/4
        num_blocking = int((u_width - passage_width) / (u_thickness * 2))
        for i in range(num_blocking):
            x = map_center_x - u_width/2 + passage_width/2 + (i * (u_width - passage_width) / max(1, num_blocking))
            if abs(x - map_center_x) > passage_width/2:
                self._add_circle_to_grid(x, passage_y, u_thickness * 0.8)
        
        # Add random obstacles to fill space and create complexity
        area = self.width * self.height
        num_additional = int(area * 0.12)  # Additional obstacles
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
            
            # Allow overlap but not complete blocking
            self._add_circle_to_grid(cx, cy, radius)
        
        # Find start and goal on opposite sides
        # Place start in top-left area, goal in bottom-right area
        start_area = (0, 0, self.width * 0.3, self.height * 0.3)
        goal_area = (self.width * 0.7, self.height * 0.7, self.width, self.height)
        
        # Find free cells in start area
        start_candidates = []
        for y in range(int(start_area[1]), int(start_area[3])):
            for x in range(int(start_area[0]), int(start_area[2])):
                if not self.grid[y, x]:
                    start_candidates.append((x, y))
        
        goal_candidates = []
        for y in range(int(goal_area[1]), int(goal_area[3])):
            for x in range(int(goal_area[0]), int(goal_area[2])):
                if not self.grid[y, x]:
                    goal_candidates.append((x, y))
        
        if start_candidates and goal_candidates:
            self.start = start_candidates[np.random.randint(len(start_candidates))]
            self.goal = goal_candidates[np.random.randint(len(goal_candidates))]
        else:
            self.start, self.goal = self._find_free_start_goal()
        
        # Verify path exists (should be very limited)
        if not self._check_path_exists(self.start, self.goal):
            # Regenerate with slightly fewer obstacles
            return self.generate_m3_trap()
        
        return True
    
    def visualize(self, save_path: str = None, show: bool = True):
        """Visualize the map with obstacles, start, and goal"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Draw grid
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match image coordinates
        
        # Draw obstacles as circles
        for cx, cy, radius in self.obstacles:
            circle = patches.Circle((cx, cy), radius, color='red', alpha=0.25)
            ax.add_patch(circle)
        
        # Draw start point
        if self.start:
            ax.plot(self.start[0], self.start[1], 'go', markersize=15, label='Start', markeredgecolor='black', markeredgewidth=2)
        
        # Draw goal point
        if self.goal:
            ax.plot(self.goal[0], self.goal[1], 'b*', markersize=20, label='Goal', markeredgecolor='black', markeredgewidth=2)
        
        ax.set_xlabel('X (grid units)')
        ax.set_ylabel('Y (grid units)')
        ax.set_title(f'Map {self.width}x{self.height}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_to_file(self, filepath: str):
        """Save map to JSON file"""
        # Convert numpy types to Python native types for JSON serialization
        obstacles_list = [[float(cx), float(cy), float(r)] for cx, cy, r in self.obstacles]
        start_list = [int(self.start[0]), int(self.start[1])] if self.start else None
        goal_list = [int(self.goal[0]), int(self.goal[1])] if self.goal else None
        
        data = {
            'width': int(self.width),
            'height': int(self.height),
            'obstacles': obstacles_list,
            'start': start_list,
            'goal': goal_list,
            'grid': self.grid.tolist()  # Convert numpy array to list
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
        self.start = tuple(data['start'])
        self.goal = tuple(data['goal'])
        self.grid = np.array(data['grid'], dtype=bool)
    
    def get_obstacle_density(self) -> float:
        """Calculate obstacle density as percentage"""
        total_cells = self.width * self.height
        obstacle_cells = np.sum(self.grid)
        return (obstacle_cells / total_cells) * 100


def generate_all_maps():
    """Generate all maps for all sizes and types"""
    sizes = [20, 50, 100]
    map_types = ['m1', 'm2', 'm3']
    
    # Create directories
    output_dir = Path('maps')
    output_dir.mkdir(exist_ok=True)
    
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    
    results = []
    
    for size in sizes:
        for map_type in map_types:
            print(f"\nGenerating {map_type} map {size}x{size}...")
            
            generator = MapGenerator(size, size)
            
            success = False
            if map_type == 'm1':
                success = generator.generate_m1_sparse()
            elif map_type == 'm2':
                success = generator.generate_m2_dense()
            elif map_type == 'm3':
                success = generator.generate_m3_trap()
            
            if success:
                # Save map
                map_filename = f"{map_type}_{size}x{size}.json"
                map_path = output_dir / map_filename
                generator.save_to_file(str(map_path))
                
                # Save visualization
                viz_filename = f"{map_type}_{size}x{size}.png"
                viz_path = output_dir / 'visualizations' / viz_filename
                generator.visualize(save_path=str(viz_path), show=False)
                
                density = generator.get_obstacle_density()
                results.append({
                    'type': map_type,
                    'size': f"{size}x{size}",
                    'density': f"{density:.2f}%",
                    'num_obstacles': len(generator.obstacles),
                    'file': map_filename
                })
                
                print(f"  ✓ Generated: {map_filename}")
                print(f"    Obstacle density: {density:.2f}%")
                print(f"    Number of obstacles: {len(generator.obstacles)}")
            else:
                print(f"  ✗ Failed to generate {map_type} {size}x{size}")
    
    # Print summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"{'Type':<10} {'Size':<15} {'Density':<15} {'Obstacles':<15} {'File':<25}")
    print("-"*60)
    for r in results:
        print(f"{r['type']:<10} {r['size']:<15} {r['density']:<15} {r['num_obstacles']:<15} {r['file']:<25}")
    
    return results


def generate_one_destination_maps():
    """
    Generate m1 one-destination maps for 100x100 and 200x200 with the same
    obstacle density as the 50x50 map (maps/one_destinations/m1_50x50.json has 80 NFZs).
    Density scaling: (size/50)^2 → 100x100 has 4x obstacles, 200x200 has 16x.
    """
    base_size = 50
    base_num_obstacles = 80  # from m1_50x50.json
    one_dest_dir = Path('maps') / 'one_destinations'
    one_dest_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for size in [100, 200]:
        scale = (size / base_size) ** 2
        num_obstacles = int(base_num_obstacles * scale)
        print(f"\nGenerating m1 one-destination map {size}x{size} ({num_obstacles} NFZs)...")
        generator = MapGenerator(size, size)
        success = generator.generate_m1_sparse(num_obstacles=num_obstacles, allow_denser_packing=True)
        if success:
            path = one_dest_dir / f"m1_{size}x{size}.json"
            generator.save_to_file(str(path))
            density = generator.get_obstacle_density()
            results.append({
                'size': f"{size}x{size}",
                'num_obstacles': len(generator.obstacles),
                'density': f"{density:.2f}%",
                'file': str(path),
            })
            print(f"  ✓ Saved {path} (density: {density:.2f}%, obstacles: {len(generator.obstacles)})")
        else:
            print(f"  ✗ Failed to generate m1 {size}x{size}")
    return results


if __name__ == "__main__":
    print("UAV Map Generator")
    print("="*60)
    generate_all_maps()
    print("\n--- One-destination maps (m1 100x100, 200x200) ---")
    generate_one_destination_maps()
    print("\nAll maps generated successfully!")

