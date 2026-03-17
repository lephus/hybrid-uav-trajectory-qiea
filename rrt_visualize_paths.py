"""
Visualize RRT* (Rapidly-exploring Random Tree Star) paths on UAV maps
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from algorithms import RRTStar


RRT_COLOR     = '#2EC4B6'   # Teal
RRT_LINESTYLE = '-'
RRT_LINEWIDTH = 2.5


def load_map(map_file: str):
    with open(map_file, 'r') as f:
        data = json.load(f)
    return {
        'width':     data['width'],
        'height':    data['height'],
        'obstacles': [tuple(obs) for obs in data['obstacles']],
        'start':     tuple(data['start']),
        'goal':      tuple(data['goal']),
    }


def run_rrt(map_data, max_iterations=3000, step_size=3.0,
            goal_sample_rate=0.10, goal_tolerance=1.5):
    """Run RRT* and return PathResult."""
    print(f"Running RRT*  (iter={max_iterations}, step={step_size}, "
          f"goal_rate={goal_sample_rate}) …")
    planner = RRTStar(
        map_data['obstacles'],
        map_data['width'],
        map_data['height'],
        max_iterations=max_iterations,
        step_size=step_size,
        goal_sample_rate=goal_sample_rate,
        goal_tolerance=goal_tolerance,
    )
    return planner.plan(map_data['start'], map_data['goal'])


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_map_base(ax, map_data):
    """Draw obstacles, grid, axis limits."""
    ax.set_xlim(0, map_data['width'])
    ax.set_ylim(0, map_data['height'])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(0, map_data['width']  + 1, max(1, map_data['width']  // 10)))
    ax.set_yticks(range(0, map_data['height'] + 1, max(1, map_data['height'] // 10)))
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    for cx, cy, radius in map_data['obstacles']:
        circle = patches.Circle(
            (cx, cy), radius,
            facecolor='#d32f2f', alpha=0.25,
            edgecolor='#b71c1c', linewidth=1.5, zorder=1,
        )
        ax.add_patch(circle)


def _draw_start_goal(ax, map_data, markersize_start=18, markersize_goal=22):
    start = map_data['start']
    goal  = map_data['goal']
    ax.plot(start[0], start[1], 'go', markersize=markersize_start,
            label='Start', markeredgecolor='#1b5e20', markeredgewidth=2, zorder=10)
    ax.plot(goal[0],  goal[1],  'b*', markersize=markersize_goal,
            label='Goal',  markeredgecolor='#0d47a1', markeredgewidth=2, zorder=10)


def _draw_rrt_path(ax, result, label_prefix="RRT*", scatter_size=30):
    if not result.success or len(result.path) < 2:
        return
    x = [p[0] for p in result.path]
    y = [p[1] for p in result.path]
    label = f"{label_prefix}  (L={result.path_length:.2f}, {len(result.path)} pts)"
    ax.plot(x, y, color=RRT_COLOR, linestyle=RRT_LINESTYLE,
            linewidth=RRT_LINEWIDTH, label=label, alpha=0.9, zorder=5)
    ax.scatter(x, y, color=RRT_COLOR, s=scatter_size,
               alpha=0.7, zorder=6, edgecolors='white', linewidths=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Overlay view (single axes)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_overlay(map_data, result, save_path=None, show=True, figsize=(8.5, 8.5)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    _draw_map_base(ax, map_data)
    _draw_start_goal(ax, map_data)
    _draw_rrt_path(ax, result)

    ax.set_xlabel('X (grid units)', fontsize=12)
    ax.set_ylabel('Y (grid units)', fontsize=12)
    ax.set_title('RRT* Path Planning', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Stats box
    status = "SUCCESS" if result.success else "FAILED (no path found)"
    stats  = (
        f"RRT* result\n"
        f"Status     : {status}\n"
        f"Length     : {result.path_length:.4f}\n"
        f"Waypoints  : {len(result.path)}\n"
        f"Time       : {result.computation_time:.4f} s\n"
        f"Tree nodes : {result.num_nodes_explored}"
    )
    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.85))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved overlay: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Individual high-quality export
# ─────────────────────────────────────────────────────────────────────────────

def visualize_individual(map_data, result, map_path: Path = None,
                         save_png=True, save_pdf=True, show=True):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    _draw_map_base(ax, map_data)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=1.2)
    _draw_start_goal(ax, map_data, markersize_start=25, markersize_goal=32)
    _draw_rrt_path(ax, result, scatter_size=55)

    status = "SUCCESS" if result.success else "FAILED (no path found)"
    title  = (
        f"RRT* – {status}\n"
        f"Length: {result.path_length:.2f}  |  "
        f"Waypoints: {len(result.path)}  |  "
        f"Time: {result.computation_time:.4f} s"
    )
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    plt.tight_layout()

    if save_png or save_pdf:
        out_dir = (map_path.parent / 'visualizations') if map_path else Path('visualizations')
        out_dir.mkdir(exist_ok=True)
        base = map_path.stem if map_path else 'map'

        if save_png:
            out_png = out_dir / f"{base}_rrt_star.png"
            plt.savefig(out_png, dpi=600, bbox_inches='tight',
                        facecolor='white', edgecolor='none', pad_inches=0.1)
            print(f"Saved PNG (600 dpi): {out_png}")

        if save_pdf:
            out_pdf = out_dir / f"{base}_rrt_star.pdf"
            plt.savefig(out_pdf, format='pdf', bbox_inches='tight',
                        facecolor='white', edgecolor='none', pad_inches=0.1)
            print(f"Saved PDF          : {out_pdf}")

    if show:
        plt.show()
    else:
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize RRT* path planning on a UAV map',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 rrt_visualize_paths.py --map maps/destinations/m1_50x50.json
  python3 rrt_visualize_paths.py --map maps/destinations/m1_50x50.json --mode both --save result.png
  python3 rrt_visualize_paths.py --map maps/destinations/m1_50x50.json --iterations 5000 --step 2.5
        """,
    )

    parser.add_argument('--map', type=str,
                        default='maps/destinations/m1_50x50.json',
                        help='Path to map JSON file')
    parser.add_argument('--mode', choices=['overlay', 'individual', 'both'],
                        default='both',
                        help='overlay = one axes; individual = high-res export; both = all')
    parser.add_argument('--save', type=str, default=None,
                        help='Base path to save overlay figure (optional)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not open plot windows')

    # RRT* hyper-parameters
    parser.add_argument('--iterations', type=int,   default=3000, help='Max tree iterations (default 3000)')
    parser.add_argument('--step',       type=float, default=3.0,  help='Steer step size (default 3.0)')
    parser.add_argument('--goal-rate',  type=float, default=0.10, help='Goal sample rate 0-1 (default 0.10)')
    parser.add_argument('--tolerance',  type=float, default=1.5,  help='Goal tolerance distance (default 1.5)')

    args = parser.parse_args()

    map_path = Path(args.map)
    if not map_path.exists():
        print(f"Error: map file '{args.map}' not found.")
        return

    print(f"Map : {map_path.name}")
    map_data = load_map(str(map_path))
    print(f"  Size      : {map_data['width']} × {map_data['height']}")
    print(f"  NFZ count : {len(map_data['obstacles'])}")
    print(f"  Start     : {map_data['start']}")
    print(f"  Goal      : {map_data['goal']}")
    print()

    result = run_rrt(
        map_data,
        max_iterations=args.iterations,
        step_size=args.step,
        goal_sample_rate=args.goal_rate,
        goal_tolerance=args.tolerance,
    )

    print()
    print("=" * 55)
    status = "SUCCESS" if result.success else "FAILED (no path found)"
    print(f"  Status     : {status}")
    if result.path:
        print(f"  Length     : {result.path_length:.4f}")
        print(f"  Waypoints  : {len(result.path)}")
        print(f"  Time (s)   : {result.computation_time:.4f}")
        print(f"  Tree nodes : {result.num_nodes_explored}")
    print("=" * 55)
    print()

    show = not args.no_show

    if args.mode in ('overlay', 'both'):
        save = args.save
        if save and args.mode == 'both':
            save = str(Path(save).with_suffix('')) + '_overlay.png'
        visualize_overlay(map_data, result, save_path=save, show=show)

    if args.mode in ('individual', 'both'):
        visualize_individual(map_data, result, map_path=map_path,
                             save_png=True, save_pdf=True, show=show)


if __name__ == "__main__":
    main()
