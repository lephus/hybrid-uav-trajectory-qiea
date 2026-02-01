"""
Visualize paths from different path planning algorithms on maps
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from algorithms import (
    AStar, ThetaStar, Dijkstra,
    AStarQIEA, ThetaStarQIEA, DijkstraQIEA
)


# Color scheme for different algorithms
ALGORITHM_COLORS = {
    'A*': '#FF6B6B',           # Red
    'Theta*': '#4ECDC4',        # Teal
    'Dijkstra': '#95E1D3',     # Light teal
    'A*+QIEA': '#F38181',      # Coral
    'Theta*+QIEA': '#AA96DA',   # Purple
    'Dijkstra+QIEA': '#FCBAD3', # Pink
}

ALGORITHM_LINESTYLES = {
    'A*': '-',
    'Theta*': '-',
    'Dijkstra': '--',
    'A*+QIEA': '-',
    'Theta*+QIEA': '-',
    'Dijkstra+QIEA': '--',
}

ALGORITHM_LINEWIDTHS = {
    'A*': 2.0,
    'Theta*': 2.5,
    'Dijkstra': 2.0,
    'A*+QIEA': 2.5,
    'Theta*+QIEA': 3.0,
    'Dijkstra+QIEA': 2.5,
}


def load_map(map_file: str):
    """Load map from JSON file"""
    with open(map_file, 'r') as f:
        data = json.load(f)
    
    return {
        'width': data['width'],
        'height': data['height'],
        'obstacles': [tuple(obs) for obs in data['obstacles']],
        'start': tuple(data['start']),
        'goal': tuple(data['goal'])
    }


def run_all_algorithms(map_data):
    """Run all algorithms and return results"""
    algorithms = {
        'A*': AStar,
        'Theta*': ThetaStar,
        'Dijkstra': Dijkstra,
        'A*+QIEA': AStarQIEA,
        'Theta*+QIEA': ThetaStarQIEA,
        'Dijkstra+QIEA': DijkstraQIEA,
    }
    
    results = {}
    
    for name, planner_class in algorithms.items():
        print(f"Running {name}...")
        planner = planner_class(
            map_data['obstacles'],
            map_data['width'],
            map_data['height']
        )
        result = planner.plan(map_data['start'], map_data['goal'])
        results[name] = result
    
    return results


def visualize_paths_overlay(map_data, results, save_path=None, show=True, figsize=(8.5, 8.5)):
    """
    Visualize all paths overlaid on the same map
    
    Args:
        map_data: Map data dictionary
        results: Dictionary of {algorithm_name: PathResult}
        save_path: Path to save figure
        show: Whether to display
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Draw map background
    ax.set_xlim(0, map_data['width'])
    ax.set_ylim(0, map_data['height'])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Draw grid
    ax.set_xticks(range(0, map_data['width'] + 1, max(1, map_data['width'] // 10)))
    ax.set_yticks(range(0, map_data['height'] + 1, max(1, map_data['height'] // 10)))
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Draw obstacles
    for cx, cy, radius in map_data['obstacles']:
        circle = patches.Circle(
            (cx, cy),
            radius,
            facecolor='#d32f2f',
            alpha=0.25,
            edgecolor='#b71c1c',
            linewidth=1.5,
            zorder=1
        )
        ax.add_patch(circle)
    
    # Draw start and goal
    start = map_data['start']
    goal = map_data['goal']
    
    ax.plot(start[0], start[1], 'go', markersize=18, 
            label='Start', markeredgecolor='#1b5e20', 
            markeredgewidth=2, zorder=10)
    ax.plot(goal[0], goal[1], 'b*', markersize=22,
            label='Goal', markeredgecolor='#0d47a1',
            markeredgewidth=2, zorder=10)
    
    # Draw paths
    for alg_name, result in results.items():
        if result.success and len(result.path) > 1:
            path = result.path
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            
            color = ALGORITHM_COLORS.get(alg_name, '#000000')
            linestyle = ALGORITHM_LINESTYLES.get(alg_name, '-')
            linewidth = ALGORITHM_LINEWIDTHS.get(alg_name, 2.0)
            
            label = f"{alg_name} (L={result.path_length:.2f})"
            ax.plot(x_coords, y_coords, color=color, linestyle=linestyle,
                   linewidth=linewidth, label=label, alpha=0.8, zorder=5)
            
            # Draw waypoints
            ax.scatter(x_coords, y_coords, color=color, s=30, 
                      alpha=0.6, zorder=6, edgecolors='white', linewidths=0.5)
    
    # Title and labels
    ax.set_xlabel('X (grid units)', fontsize=12)
    ax.set_ylabel('Y (grid units)', fontsize=12)
    ax.set_title('Path Planning Algorithms Comparison', fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
    
    # Add statistics text box
    stats_text = "Algorithm Statistics:\n"
    for alg_name, result in results.items():
        if result.success:
            stats_text += f"{alg_name}: {result.path_length:.2f} units, "
            stats_text += f"{result.computation_time:.4f}s, "
            stats_text += f"{len(result.path)} waypoints\n"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_paths_side_by_side(map_data, results, save_path=None, show=True, map_path: Path = None, save_individual: bool = False):
    """
    Visualize paths in separate subplots for each algorithm
    
    Args:
        map_data: Map data dictionary
        results: Dictionary of {algorithm_name: PathResult}
        save_path: Path to save combined figure (optional)
        show: Whether to display
        map_path: Optional Path object for naming individual outputs
        save_individual: If True, save one image per algorithm (850x850)
    """
    successful_results = {k: v for k, v in results.items() if v.success}
    
    if not successful_results:
        print("No successful paths to visualize!")
        return
    
    n_algorithms = len(successful_results)
    n_cols = 3
    n_rows = (n_algorithms + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_algorithms == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (alg_name, result) in enumerate(successful_results.items()):
        ax = axes[idx]
        
        # Draw map
        ax.set_xlim(0, map_data['width'])
        ax.set_ylim(0, map_data['height'])
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Draw obstacles
        for cx, cy, radius in map_data['obstacles']:
            circle = patches.Circle(
                (cx, cy), radius,
                facecolor='#d32f2f', alpha=0.25,
                edgecolor='#b71c1c', linewidth=1.5, zorder=1
            )
            ax.add_patch(circle)
        
        # Draw start and goal
        start = map_data['start']
        goal = map_data['goal']
        ax.plot(start[0], start[1], 'go', markersize=15,
                markeredgecolor='#1b5e20', markeredgewidth=2, zorder=10)
        ax.plot(goal[0], goal[1], 'b*', markersize=20,
                markeredgecolor='#0d47a1', markeredgewidth=2, zorder=10)
        
        # Draw path
        if len(result.path) > 1:
            path = result.path
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            
            color = ALGORITHM_COLORS.get(alg_name, '#000000')
            linestyle = ALGORITHM_LINESTYLES.get(alg_name, '-')
            linewidth = ALGORITHM_LINEWIDTHS.get(alg_name, 2.0)
            
            ax.plot(x_coords, y_coords, color=color, linestyle=linestyle,
                   linewidth=linewidth, alpha=0.8, zorder=5)
            ax.scatter(x_coords, y_coords, color=color, s=25,
                      alpha=0.6, zorder=6, edgecolors='white', linewidths=0.5)
        else:
            x_coords, y_coords = [], []
            color = ALGORITHM_COLORS.get(alg_name, '#000000')
            linestyle = ALGORITHM_LINESTYLES.get(alg_name, '-')
            linewidth = ALGORITHM_LINEWIDTHS.get(alg_name, 2.0)
        
        # Title with statistics
        title = f"{alg_name}\n"
        title += f"Length: {result.path_length:.2f} | "
        title += f"Time: {result.computation_time:.4f}s | "
        title += f"Waypoints: {len(result.path)}"
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        
        # Save individual figure if requested
        if save_individual:
            # High-quality per-algorithm export
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(12, 12))
            ax_ind.set_xlim(0, map_data['width'])
            ax_ind.set_ylim(0, map_data['height'])
            ax_ind.set_aspect('equal')
            ax_ind.invert_yaxis()
            ax_ind.grid(True, alpha=0.2, linestyle='--', linewidth=1.2)
            
            for cx, cy, radius in map_data['obstacles']:
                circle = patches.Circle((cx, cy), radius,
                                        facecolor='#d32f2f', alpha=0.25,
                                        edgecolor='#b71c1c', linewidth=2.0, zorder=1)
                ax_ind.add_patch(circle)
            ax_ind.plot(start[0], start[1], 'go', markersize=25,
                        markeredgecolor='#1b5e20', markeredgewidth=3, zorder=10)
            ax_ind.plot(goal[0], goal[1], 'b*', markersize=32,
                        markeredgecolor='#0d47a1', markeredgewidth=3, zorder=10)
            if len(result.path) > 1:
                ax_ind.plot(x_coords, y_coords, color=color, linestyle=linestyle,
                            linewidth=max(3.0, linewidth * 1.5), alpha=0.9, zorder=5)
                ax_ind.scatter(x_coords, y_coords, color=color, s=50,
                               alpha=0.7, zorder=6, edgecolors='white', linewidths=1.0)
            ax_ind.set_title(title, fontsize=16, fontweight='bold', pad=15)
            ax_ind.set_xlabel('X', fontsize=14)
            ax_ind.set_ylabel('Y', fontsize=14)
            ax_ind.tick_params(labelsize=12)
            plt.tight_layout()
            if map_path is not None:
                out_dir = map_path.parent / 'visualizations'
            else:
                out_dir = Path('visualizations')
            out_dir.mkdir(exist_ok=True)
            base = map_path.stem if map_path else 'map'
            out_png = out_dir / f"{base}_{alg_name.replace('*','star').replace('+','plus')}.png"
            out_pdf = out_dir / f"{base}_{alg_name.replace('*','star').replace('+','plus')}.pdf"
            plt.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
            plt.savefig(out_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
            print(f"✓ Saved individual PNG (600 DPI): {out_png}")
            print(f"✓ Saved individual PDF: {out_pdf}")
            plt.close(fig_ind)
    
    # Hide unused subplots
    for idx in range(n_algorithms, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Path Planning Algorithms - Individual Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize paths from path planning algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all paths overlaid on one map
  python3 visualize_paths.py --map maps/m1_20x20.json --mode overlay
  
  # Visualize each algorithm in separate subplots
  python3 visualize_paths.py --map maps/m1_20x20.json --mode side-by-side
  
  # Both modes and save
  python3 visualize_paths.py --map maps/m1_50x50.json --mode both --save comparison.png
        """
    )
    
    parser.add_argument(
        '--map',
        type=str,
        required=True,
        help='Path to map JSON file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['overlay', 'side-by-side', 'both'],
        default='overlay',
        help='Visualization mode: overlay (all paths together) or side-by-side (separate plots)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save figure (optional)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots'
    )
    
    args = parser.parse_args()
    
    # Load map
    map_path = Path(args.map)
    if not map_path.exists():
        print(f"Error: Map file {args.map} not found!")
        return
    
    print(f"Loading map: {map_path.name}")
    map_data = load_map(str(map_path))
    
    print(f"\nMap Info:")
    print(f"  Size: {map_data['width']}x{map_data['height']}")
    print(f"  Obstacles: {len(map_data['obstacles'])}")
    print(f"  Start: {map_data['start']}, Goal: {map_data['goal']}")
    
    # Run all algorithms
    print("\nRunning all algorithms...")
    results = run_all_algorithms(map_data)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for alg_name, result in results.items():
        status = "✓" if result.success else "✗"
        if result.success:
            print(f"{status} {alg_name:<20} Length: {result.path_length:.2f}, "
                  f"Time: {result.computation_time:.4f}s, "
                  f"Waypoints: {len(result.path)}")
        else:
            print(f"{status} {alg_name:<20} Failed: {result.message}")
    
    # Visualize
    if args.mode in ['overlay', 'both']:
        save_path = args.save
        if save_path and args.mode == 'both':
            save_path = str(Path(save_path).with_suffix('')) + '_overlay.png'
        visualize_paths_overlay(
            map_data, results,
            save_path=save_path,
            show=not args.no_show
        )
    
    if args.mode in ['side-by-side', 'both']:
        save_path = args.save
        if save_path and args.mode == 'both':
            save_path = str(Path(save_path).with_suffix('')) + '_side_by_side.png'
        visualize_paths_side_by_side(
            map_data, results,
            save_path=save_path,
            show=not args.no_show,
            map_path=map_path,
            save_individual=True
        )


if __name__ == "__main__":
    main()

