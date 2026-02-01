"""
Visualization tool for multi-UAV path planning results
Shows all UAV paths on the same map with timestamps and conflict highlighting
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from algorithms import MultiPathResult


def load_map(map_file: str):
    """Load map from JSON file (supports both single start and multi-UAV formats)"""
    with open(map_file, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: single start or multiple starts
    if 'starts' in data:
        # Multi-UAV format: use first start as reference (for compatibility)
        starts_list = [tuple(s) for s in data['starts']]
        start_ref = starts_list[0] if starts_list else (0, 0)
    elif 'start' in data:
        # Single start format
        start_ref = tuple(data['start'])
        starts_list = [start_ref]
    else:
        raise ValueError("Map file must contain either 'start' or 'starts' field")
    
    return {
        'width': data['width'],
        'height': data['height'],
        'obstacles': [tuple(obs) for obs in data['obstacles']],
        'start': start_ref,  # For backward compatibility
        'starts': starts_list,  # Multi-UAV format
        'goal': tuple(data['goal'])
    }


def visualize_multi_paths(result: MultiPathResult,
                          map_data: dict,
                          algorithm_name: str,
                          save_path: Optional[str] = None,
                          show: bool = True):
    """
    Visualize multi-UAV paths on map
    
    Args:
        result: MultiPathResult from planner
        map_data: Map data with obstacles, width, height
        algorithm_name: Name of algorithm for title
        save_path: Optional path to save figure
        show: Whether to display figure
    """
    if not result.success:
        print(f"Cannot visualize: {result.message}")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    width = map_data['width']
    height = map_data['height']
    obstacles = map_data['obstacles']
    
    # Set up plot
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_xlabel('X (grid units)', fontsize=12)
    ax.set_ylabel('Y (grid units)', fontsize=12)
    ax.set_title(f'{algorithm_name} - {len(result.paths)} UAVs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Draw obstacles
    for cx, cy, radius in obstacles:
        circle = patches.Circle((cx, cy), radius, color='red', alpha=0.25, edgecolor='darkred', linewidth=1)
        ax.add_patch(circle)
    
    # Color palette for different UAVs
    colors = plt.cm.tab10(np.linspace(0, 1, len(result.paths)))
    
    # Draw paths for each UAV
    for uav_id, path in result.paths.items():
        color = colors[uav_id % len(colors)]
        
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
            ax.plot(path[0][0], path[0][1], 'o', 
                   color=color, markersize=10, markeredgecolor='black', 
                   markeredgewidth=2, label=f'UAV {uav_id} Start' if uav_id == 0 else '')
    
    # Draw goal (all UAVs share same goal)
    goal = None
    if result.paths:
        # Get goal from first path
        first_path = list(result.paths.values())[0]
        if first_path:
            goal = first_path[-1]
    
    if goal:
        ax.plot(goal[0], goal[1], 'b*', markersize=20, 
               markeredgecolor='black', markeredgewidth=2, 
               label='Goal (shared)', zorder=10)
    
    # Skip conflicts visualization (user requested to ignore conflicts)
    # if result.conflicts:
    #     conflict_color = 'red'
    #     for uav1_id, uav2_id, conflict_time, conflict_location in result.conflicts:
    #         ax.plot(conflict_location[0], conflict_location[1], 'X',
    #                color=conflict_color, markersize=15, markeredgecolor='darkred',
    #                markeredgewidth=2, zorder=15)
    #         # Add annotation
    #         ax.annotate(f'Conflict\nUAV{uav1_id}-UAV{uav2_id}\nt={conflict_time:.1f}s',
    #                    xy=conflict_location, xytext=(10, 10),
    #                    textcoords='offset points', fontsize=8,
    #                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
    #                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add legend
    ax.legend(loc='upper right', fontsize=9)
    
    # Add info text
    info_text = (
        f"Total Path Length: {result.total_cost:.2f}\n"
        f"Makespan: {result.makespan:.2f}s\n"
        f"Computation Time: {result.computation_time:.4f}s"
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


def visualize_comparison(results: Dict[str, MultiPathResult],
                        map_data: dict,
                        save_path: Optional[str] = None,
                        show: bool = True):
    """
    Visualize comparison of multiple algorithms side-by-side
    
    Args:
        results: Dict of {algorithm_name: MultiPathResult}
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
    
    for idx, (algorithm_name, result) in enumerate(results.items()):
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
        
        if not result.success:
            ax.text(0.5, 0.5, f'Failed:\n{result.message}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            continue
        
        # Draw obstacles
        for cx, cy, radius in obstacles:
            circle = patches.Circle((cx, cy), radius, color='red', alpha=0.25, 
                                   edgecolor='darkred', linewidth=1)
            ax.add_patch(circle)
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(result.paths)))
        
        # Draw paths
        for uav_id, path in result.paths.items():
            color = colors[uav_id % len(colors)]
            
            if len(path) > 1:
                path_array = np.array(path)
                ax.plot(path_array[:, 0], path_array[:, 1],
                       color=color, linewidth=1.5, alpha=0.7)
                ax.scatter(path_array[:, 0], path_array[:, 1],
                          color=color, s=30, alpha=0.8, zorder=5)
            
            if path:
                ax.plot(path[0][0], path[0][1], 'o',
                       color=color, markersize=8, markeredgecolor='black',
                       markeredgewidth=1.5)
        
        # Draw goal
        if result.paths:
            first_path = list(result.paths.values())[0]
            if first_path:
                goal = first_path[-1]
                ax.plot(goal[0], goal[1], 'b*', markersize=15,
                       markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        
        # Skip conflicts visualization (user requested to ignore conflicts)
        # if result.conflicts:
        #     for uav1_id, uav2_id, conflict_time, conflict_location in result.conflicts:
        #         ax.plot(conflict_location[0], conflict_location[1], 'X',
        #                color='red', markersize=10, markeredgecolor='darkred',
        #                markeredgewidth=1.5, zorder=15)
        
        # Add info
        info_text = (
            f"Length: {result.total_cost:.1f}\n"
            f"Makespan: {result.makespan:.1f}s"
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
    parser = argparse.ArgumentParser(description='Visualize multi-UAV path planning results')
    parser.add_argument('--map', type=str, required=True,
                       help='Path to map JSON file')
    parser.add_argument('--algorithm', type=str,
                       choices=['MultiPathAStar', 'MultiPathThetaStar', 'MultiPathDijkstra',
                               'MultiPathAStarQIEA', 'MultiPathThetaStarQIEA', 'MultiPathDijkstraQIEA'],
                       help='Algorithm to visualize (if not provided, will test all)')
    parser.add_argument('--num-uavs', type=int, default=5,
                       help='Number of UAVs (default: 5)')
    parser.add_argument('--save', type=str,
                       help='Path to save visualization (PNG/PDF)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot (useful when saving)')
    
    args = parser.parse_args()
    
    # Load map
    map_data = load_map(args.map)
    
    # This is a visualization tool - in practice, you would load results from test
    # For now, we'll just show the structure
    print("Multi-UAV Path Visualization Tool")
    print("="*60)
    print(f"Map: {args.map}")
    print(f"Number of UAVs: {args.num_uavs}")
    print("\nNote: This tool visualizes MultiPathResult objects.")
    print("Run test_multi_path.py first to generate results, then load them here.")
    print("="*60)


if __name__ == "__main__":
    main()
