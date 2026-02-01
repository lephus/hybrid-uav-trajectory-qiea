# Path Planning Algorithms

This directory contains implementations of various path planning algorithms for UAV trajectory optimization.

## Algorithms

### Classical Algorithms

1. **A*** (`astar.py`)
   - A* search algorithm with 8-connected grid
   - Uses Euclidean distance as heuristic
   - Guaranteed to find optimal path if one exists

2. **Theta*** (`theta_star.py`)
   - Any-angle path planning algorithm
   - Allows paths to pass through grid cells
   - Produces smoother paths than A*

3. **Dijkstra** (`dijkstra.py`)
   - Classic Dijkstra's algorithm
   - Explores uniformly in all directions
   - No heuristic (slower but guaranteed optimal)

### Hybrid Algorithms (QIEA-Optimized)

4. **A* + QIEA** (`hybrid.py`)
   - Combines A* for initial path finding
   - Uses QIEA (Quantum-Inspired Evolutionary Algorithm) to optimize the path
   - QIEA uses quantum bits (Q-bits) and rotation gates to improve solutions

5. **Theta* + QIEA** (`hybrid.py`)
   - Combines Theta* for initial path finding
   - Uses QIEA to optimize the path

6. **Dijkstra + QIEA** (`hybrid.py`)
   - Combines Dijkstra for initial path finding
   - Uses QIEA to optimize the path

**Note:** QIEA is not used as a standalone algorithm. It is designed to optimize
paths found by classical algorithms, combining the reliability of classical methods
with the optimization power of quantum-inspired evolution.

## Usage

```python
from algorithms import AStar, ThetaStar, Dijkstra
from algorithms import AStarQIEA, ThetaStarQIEA, DijkstraQIEA

# Load map data
obstacles = [(cx1, cy1, r1), (cx2, cy2, r2), ...]
width, height = 50, 50

# Create planner
planner = AStar(obstacles, width, height)

# Plan path
start = (0, 0)
goal = (49, 49)
result = planner.plan(start, goal)

if result.success:
    print(f"Path found! Length: {result.path_length:.2f}")
    print(f"Computation time: {result.computation_time:.4f}s")
    print(f"Path: {result.path}")
else:
    print(f"Failed: {result.message}")
```

## Testing

Run the test script to test all algorithms:

```bash
python3 test_algorithms.py
```

## File Structure

```
algorithms/
├── __init__.py          # Package initialization
├── base.py              # Base classes (PathPlanner, PathResult)
├── utils.py             # Utility functions
├── astar.py             # A* algorithm
├── theta_star.py        # Theta* algorithm
├── dijkstra.py          # Dijkstra algorithm
├── qiea.py              # QIEA algorithm
└── hybrid.py            # Hybrid algorithms
```

## PathResult

All algorithms return a `PathResult` object with:

- `path`: List of waypoints `[(x1, y1), (x2, y2), ...]`
- `success`: Boolean indicating if path was found
- `computation_time`: Time taken in seconds
- `path_length`: Total path length
- `cost`: Multi-objective cost value
- `num_nodes_explored`: Number of nodes explored
- `message`: Status message

