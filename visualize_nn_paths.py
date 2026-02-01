"""
Visualization tool for n-UAV to n-Destination path planning results
Shows all UAV paths with their assigned destinations on the same map
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from algorithms.base import PathResult


def load_nn_map(map_file: str):
    """Load n-n map from JSON file"""
    with open(map_file, 'r') as f:
        data = json.load(f)
    
    if 'starts' not in data or 'goals' not in data:
        raise ValueError("Map file must contain 'starts' and 'goals' fields")
    
    # Load assignments
    if 'assignments' in data:
        assignments = {uav_id: goal_id for uav_id, goal_id in data['assignments']}
    else:
        # Default: sequential assignment
        assignments = {i: i for i in range(len(data['starts']))}
    
    return {
        'width': data['width'],
        'height': data['height'],
        'obstacles': [tuple(obs) for obs in data['obstacles']],
        'starts': [tuple(s) for s in data['starts']],
        'goals': [tuple(g) for g in data['goals']],
        'assignments': assignments,
        'num_uavs': len(data['starts'])
    }


def visualize_nn_paths(results: Dict[int, PathResult],
                       map_data: dict,
                       algorithm_name: str,
                       save_path: Optional[str] = None,
                       show: bool = True):
    """
    Visualize n-n paths on map
    
    Args:
        results: Dict of {uav_id: PathResult} for each UAV-destination pair
        map_data: Map data with obstacles, width, height, starts, goals, assignments
        algorithm_name: Name of algorithm for title
        save_path: Optional path to save figure
        show: Whether to display figure
    """
    # Check if all paths succeeded
    if not all(r.success for r in results.values()):
        failed = [uav_id for uav_id, r in results.items() if not r.success]
        print(f"Cannot visualize: Some paths failed for UAVs {failed}")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    width = map_data['width']
    height = map_data['height']
    obstacles = map_data['obstacles']
    starts = map_data['starts']
    goals = map_data['goals']
    assignments = map_data['assignments']
    
    # Set up plot
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_xlabel('X (grid units)', fontsize=12)
    ax.set_ylabel('Y (grid units)', fontsize=12)
    ax.set_title(f'{algorithm_name} - {len(results)} UAVs → {len(goals)} Destinations', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Draw obstacles
    for cx, cy, radius in obstacles:
        circle = patches.Circle((cx, cy), radius, color='red', alpha=0.25, 
                               edgecolor='darkred', linewidth=1)
        ax.add_patch(circle)
    
    # Color palette for different UAVs
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Draw paths for each UAV
    total_length = 0.0
    total_time = 0.0
    
    for uav_id, result in results.items():
        color = colors[uav_id % len(colors)]
        path = result.path
        
        # Draw path as line
        if len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 
                   color=color, linewidth=2, alpha=0.7, 
                   label=f'UAV {uav_id}')
            
            # Draw waypoints
            ax.scatter(path_array[:, 0], path_array[:, 1],
                      color=color, s=50, alpha=0.8, zorder=5)
        
        # Draw start point
        if path:
            start = starts[uav_id]
            ax.plot(start[0], start[1], 'o', 
                   color=color, markersize=12, markeredgecolor='black', 
                   markeredgewidth=2, label=f'UAV {uav_id} Start' if uav_id == 0 else '')
        
        # Draw assigned goal
        if uav_id in assignments:
            goal_id = assignments[uav_id]
            if goal_id < len(goals):
                goal = goals[goal_id]
                ax.plot(goal[0], goal[1], '*', markersize=18, 
                       color=color, markeredgecolor='black', markeredgewidth=2,
                       label=f'Goal {goal_id} (UAV {uav_id})' if uav_id == 0 else '',
                       zorder=10)
        
        total_length += result.path_length
        total_time += result.computation_time
    
    # Add legend
    ax.legend(loc='upper right', fontsize=9)
    
    # Add info text
    info_text = (
        f"Total Path Length: {total_length:.2f}\n"
        f"Average Path Length: {total_length/len(results):.2f}\n"
        f"Total Computation Time: {total_time:.4f}s\n"
        f"Average Computation Time: {total_time/len(results):.4f}s"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_nn_comparison(results: Dict[str, Dict[int, PathResult]],
                           map_data: dict,
                           save_path: Optional[str] = None,
                           show: bool = True):
    """
    Visualize comparison of multiple algorithms side-by-side for n-n paths
    
    Args:
        results: Dict of {algorithm_name: {uav_id: PathResult}}
        map_data: Map data
        save_path: Optional path to save figure
        show: Whether to display figure
    """
    num_algorithms = len(results)
    if num_algorithms == 0:
        return
    
    # Create subplots
    cols = min(3, num_algorithms)
    rows = (num_algorithms + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if num_algorithms == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    width = map_data['width']
    height = map_data['height']
    obstacles = map_data['obstacles']
    starts = map_data['starts']
    goals = map_data['goals']
    assignments = map_data['assignments']
    
    for idx, (algorithm_name, algorithm_results) in enumerate(results.items()):
        ax = axes[idx]
        
        # Set up subplot
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.set_title(algorithm_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if idx >= len(axes) - cols:
            ax.set_xlabel('X (grid units)', fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel('Y (grid units)', fontsize=10)
        
        # Check if all paths succeeded
        if not all(r.success for r in algorithm_results.values()):
            ax.text(0.5, 0.5, 'Some paths failed',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            continue
        
        # Draw obstacles
        for cx, cy, radius in obstacles:
            circle = patches.Circle((cx, cy), radius, color='red', alpha=0.25, 
                                   edgecolor='darkred', linewidth=1)
            ax.add_patch(circle)
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithm_results)))
        
        # Draw paths
        total_length = 0.0
        for uav_id, result in algorithm_results.items():
            color = colors[uav_id % len(colors)]
            path = result.path
            
            if len(path) > 1:
                path_array = np.array(path)
                ax.plot(path_array[:, 0], path_array[:, 1],
                       color=color, linewidth=1.5, alpha=0.7)
                ax.scatter(path_array[:, 0], path_array[:, 1],
                          color=color, s=30, alpha=0.8, zorder=5)
            
            # Draw start
            if path:
                start = starts[uav_id]
                ax.plot(start[0], start[1], 'o',
                       color=color, markersize=8, markeredgecolor='black',
                       markeredgewidth=1.5)
            
            # Draw assigned goal
            if uav_id in assignments:
                goal_id = assignments[uav_id]
                if goal_id < len(goals):
                    goal = goals[goal_id]
                    ax.plot(goal[0], goal[1], '*', markersize=12,
                           color=color, markeredgecolor='black',
                           markeredgewidth=1.5, zorder=10)
            
            total_length += result.path_length
        
        # Add info
        avg_length = total_length / len(algorithm_results) if algorithm_results else 0.0
        info_text = (
            f"Avg Length: {avg_length:.1f}\n"
            f"Total Length: {total_length:.1f}"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Hide unused subplots
    for idx in range(num_algorithms, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Saved comparison visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize n-n path planning results')
    parser.add_argument('--map', type=str, required=True,
                       help='Path to n-n map JSON file')
    parser.add_argument('--algorithm', type=str,
                       choices=['AStar', 'ThetaStar', 'Dijkstra', 'AStarQIEA', 'ThetaStarQIEA', 'DijkstraQIEA', 'QIEA'],
                       help='Algorithm to visualize (if not provided, will test all)')
    parser.add_argument('--save', type=str,
                       help='Path to save visualization (PNG/PDF)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot (useful when saving)')
    
    args = parser.parse_args()
    
    # Load map
    map_data = load_nn_map(args.map)
    
    print("n-n Path Visualization Tool")
    print("="*60)
    print(f"Map: {args.map}")
    print(f"Number of UAVs: {map_data['num_uavs']}")
    print("\nNote: This tool visualizes PathResult objects for each UAV-destination pair.")
    print("Run test_nn_paths.py first to generate results, then load them here.")
    print("="*60)


if __name__ == "__main__":
    main()
