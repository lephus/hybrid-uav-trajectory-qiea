"""
Visualize PSO (Particle Swarm Optimization) paths on UAV maps
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from algorithms import PSO


PSO_COLOR     = '#F4A261'   # Orange
PSO_LINESTYLE = '-'
PSO_LINEWIDTH = 2.5


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


def run_pso(map_data, n_waypoints=5, n_particles=60, max_iter=300,
            w=0.6, c1=1.5, c2=2.0, collision_penalty=500.0):
    """Run PSO and return PathResult."""
    print(f"Running PSO  (particles={n_particles}, iter={max_iter}, waypoints={n_waypoints}) …")
    planner = PSO(
        map_data['obstacles'],
        map_data['width'],
        map_data['height'],
        n_waypoints=n_waypoints,
        n_particles=n_particles,
        max_iter=max_iter,
        w=w,
        c1=c1,
        c2=c2,
        collision_penalty=collision_penalty,
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


def _draw_pso_path(ax, result, label_prefix="PSO", scatter_size=30):
    if not result.success or len(result.path) < 2:
        return
    x = [p[0] for p in result.path]
    y = [p[1] for p in result.path]
    label = f"{label_prefix}  (L={result.path_length:.2f}, {len(result.path)} pts)"
    ax.plot(x, y, color=PSO_COLOR, linestyle=PSO_LINESTYLE,
            linewidth=PSO_LINEWIDTH, label=label, alpha=0.9, zorder=5)
    ax.scatter(x, y, color=PSO_COLOR, s=scatter_size,
               alpha=0.7, zorder=6, edgecolors='white', linewidths=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Overlay view (single axes)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_overlay(map_data, result, save_path=None, show=True, figsize=(8.5, 8.5)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    _draw_map_base(ax, map_data)
    _draw_start_goal(ax, map_data)
    _draw_pso_path(ax, result)

    ax.set_xlabel('X (grid units)', fontsize=12)
    ax.set_ylabel('Y (grid units)', fontsize=12)
    ax.set_title('PSO Path Planning', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Stats box
    status  = "SUCCESS" if result.success else "FAILED (path crosses NFZ)"
    stats   = (
        f"PSO result\n"
        f"Status    : {status}\n"
        f"Length    : {result.path_length:.4f}\n"
        f"Waypoints : {len(result.path)}\n"
        f"Time      : {result.computation_time:.4f} s\n"
        f"Fitness   : {result.cost:.4f}"
    )
    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

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
    _draw_pso_path(ax, result, scatter_size=55)

    status = "SUCCESS" if result.success else "FAILED (NFZ collision)"
    title  = (
        f"PSO – {status}\n"
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
            out_png = out_dir / f"{base}_pso.png"
            plt.savefig(out_png, dpi=600, bbox_inches='tight',
                        facecolor='white', edgecolor='none', pad_inches=0.1)
            print(f"Saved PNG (600 dpi): {out_png}")

        if save_pdf:
            out_pdf = out_dir / f"{base}_pso.pdf"
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
        description='Visualize PSO path planning on a UAV map',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pso_visualize_paths.py --map maps/destinations/m1_50x50.json
  python3 pso_visualize_paths.py --map maps/destinations/m1_50x50.json --mode both --save result.png
  python3 pso_visualize_paths.py --map maps/destinations/m1_50x50.json --particles 100 --iterations 500
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

    # PSO hyper-parameters
    parser.add_argument('--waypoints',  type=int,   default=5,     help='Intermediate waypoints (default 5)')
    parser.add_argument('--particles',  type=int,   default=60,    help='Swarm size (default 60)')
    parser.add_argument('--iterations', type=int,   default=300,   help='Max iterations (default 300)')
    parser.add_argument('--w',          type=float, default=0.6,   help='Inertia weight (default 0.6)')
    parser.add_argument('--c1',         type=float, default=1.5,   help='Cognitive coeff (default 1.5)')
    parser.add_argument('--c2',         type=float, default=2.0,   help='Social coeff (default 2.0)')
    parser.add_argument('--penalty',    type=float, default=500.0, help='NFZ collision penalty (default 500)')

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

    result = run_pso(
        map_data,
        n_waypoints=args.waypoints,
        n_particles=args.particles,
        max_iter=args.iterations,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        collision_penalty=args.penalty,
    )

    print()
    print("=" * 55)
    status = "SUCCESS" if result.success else "FAILED (NFZ collision)"
    print(f"  Status     : {status}")
    if result.path:
        print(f"  Length     : {result.path_length:.4f}")
        print(f"  Waypoints  : {len(result.path)}")
        print(f"  Time (s)   : {result.computation_time:.4f}")
        print(f"  Fitness    : {result.cost:.4f}")
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
