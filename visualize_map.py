"""
Map Visualizer for UAV Trajectory Planning Maps
Visualize individual maps or all maps at once
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from map_generator import MapGenerator


def load_map_from_json(filepath: str) -> MapGenerator:
    """Load a map from JSON file"""
    generator = MapGenerator(1, 1)  # Temporary size, will be overwritten
    generator.load_from_file(filepath)
    return generator


def visualize_map(generator: MapGenerator, title: str = None, 
                 save_path: str = None, show: bool = True,
                 figsize: tuple = (10, 10)):
    """
    Visualize a map with obstacles, start, and goal points
    
    Args:
        generator: MapGenerator instance
        title: Title for the plot
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Draw grid background
    ax.set_xlim(0, generator.width)
    ax.set_ylim(0, generator.height)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Match image coordinates (y increases downward)
    
    # Draw grid lines
    ax.set_xticks(range(0, generator.width + 1, max(1, generator.width // 10)))
    ax.set_yticks(range(0, generator.height + 1, max(1, generator.height // 10)))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Draw obstacles as circles
    for cx, cy, radius in generator.obstacles:
        circle = patches.Circle(
            (cx, cy), 
            radius, 
            facecolor='#d32f2f',  # Red color
            alpha=0.25,  # Reduced opacity to 25%
            edgecolor='#b71c1c',
            linewidth=1.5
        )
        ax.add_patch(circle)
    
    # Draw start point
    if generator.start:
        ax.plot(
            generator.start[0], 
            generator.start[1], 
            'go', 
            markersize=15, 
            label='Start', 
            markeredgecolor='#1b5e20',
            markeredgewidth=2,
            zorder=10
        )
        # Add text label
        ax.text(
            generator.start[0], 
            generator.start[1] - generator.height * 0.03,
            'START',
            fontsize=10,
            fontweight='bold',
            ha='center',
            color='#1b5e20',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    # Draw goal point
    if generator.goal:
        ax.plot(
            generator.goal[0], 
            generator.goal[1], 
            'b*', 
            markersize=20, 
            label='Goal', 
            markeredgecolor='#0d47a1',
            markeredgewidth=2,
            zorder=10
        )
        # Add text label
        ax.text(
            generator.goal[0], 
            generator.goal[1] - generator.height * 0.03,
            'GOAL',
            fontsize=10,
            fontweight='bold',
            ha='center',
            color='#0d47a1',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    # Set labels and title
    ax.set_xlabel('X (grid units)', fontsize=12)
    ax.set_ylabel('Y (grid units)', fontsize=12)
    
    if title is None:
        title = f'Map {generator.width}×{generator.height}'
    
    # Add statistics to title
    density = generator.get_obstacle_density()
    stats = f' | Obstacles: {len(generator.obstacles)} | Density: {density:.2f}%'
    ax.set_title(title + stats, fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_all_maps(maps_dir: str = "maps", show: bool = True):
    """
    Visualize all maps in the maps directory
    
    Args:
        maps_dir: Directory containing map JSON files
        show: Whether to display plots
    """
    maps_path = Path(maps_dir)
    
    if not maps_path.exists():
        print(f"Error: Directory '{maps_dir}' does not exist!")
        return
    
    # Find all JSON files (excluding visualizations folder)
    json_files = sorted(maps_path.glob("*.json"))
    
    if not json_files:
        print(f"No map files found in '{maps_dir}'")
        return
    
    print(f"Found {len(json_files)} map files. Visualizing...")
    print("=" * 60)
    
    for json_file in json_files:
        print(f"\nLoading: {json_file.name}")
        try:
            generator = load_map_from_json(str(json_file))
            
            # Extract map type and size from filename
            name_parts = json_file.stem.split('_')
            map_type = name_parts[0] if len(name_parts) > 0 else "unknown"
            map_size = name_parts[1] if len(name_parts) > 1 else "unknown"
            
            title = f"{map_type.upper()} - {map_size}"
            
            visualize_map(
                generator, 
                title=title,
                show=show,
                figsize=(10, 10)
            )
            
            print(f"  ✓ Obstacles: {len(generator.obstacles)}")
            print(f"  ✓ Density: {generator.get_obstacle_density():.2f}%")
            print(f"  ✓ Start: {generator.start}, Goal: {generator.goal}")
            
        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")


def visualize_single_map(map_file: str, show: bool = True, save: bool = False):
    """
    Visualize a single map file
    
    Args:
        map_file: Path to map JSON file
        show: Whether to display the plot
        save: Whether to save the visualization
    """
    map_path = Path(map_file)
    
    if not map_path.exists():
        print(f"Error: File '{map_file}' does not exist!")
        return
    
    print(f"Loading map: {map_path.name}")
    
    try:
        generator = load_map_from_json(str(map_path))
        
        # Extract info from filename
        name_parts = map_path.stem.split('_')
        map_type = name_parts[0] if len(name_parts) > 0 else "map"
        map_size = name_parts[1] if len(name_parts) > 1 else f"{generator.width}x{generator.height}"
        
        title = f"{map_type.upper()} - {map_size}"
        
        save_path = None
        if save:
            output_dir = Path("maps/visualizations")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{map_path.stem}_viz.png"
        
        visualize_map(
            generator,
            title=title,
            save_path=str(save_path) if save_path else None,
            show=show,
            figsize=(12, 12)
        )
        
        print(f"\nMap Statistics:")
        print(f"  Size: {generator.width} × {generator.height}")
        print(f"  Obstacles: {len(generator.obstacles)}")
        print(f"  Density: {generator.get_obstacle_density():.2f}%")
        print(f"  Start: {generator.start}")
        print(f"  Goal: {generator.goal}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize UAV trajectory planning maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all maps
  python visualize_map.py --all
  
  # Visualize a specific map
  python visualize_map.py --map maps/m1_50x50.json
  
  # Visualize and save without showing
  python visualize_map.py --map maps/m2_100x100.json --save --no-show
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Visualize all maps in the maps directory'
    )
    
    parser.add_argument(
        '--map',
        type=str,
        help='Path to a specific map JSON file to visualize'
    )
    
    parser.add_argument(
        '--maps-dir',
        type=str,
        default='maps',
        help='Directory containing map files (default: maps)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save visualization to file'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots (useful when saving)'
    )
    
    args = parser.parse_args()
    
    if args.all:
        visualize_all_maps(maps_dir=args.maps_dir, show=not args.no_show)
    elif args.map:
        visualize_single_map(
            map_file=args.map,
            show=not args.no_show,
            save=args.save
        )
    else:
        # Default: show all maps
        print("No specific map specified. Visualizing all maps...")
        print("(Use --map <file> for a specific map, or --all to explicitly show all)\n")
        visualize_all_maps(maps_dir=args.maps_dir, show=not args.no_show)


if __name__ == "__main__":
    main()

