# Hybrid UAV Trajectory Optimization via Search Initialization and Quantum-Inspired Evolutionary Algorithms

This repository contains the implementation of a hybrid path planning framework that combines classical path planning algorithms (A*, Theta*, Dijkstra) with Quantum-Inspired Evolutionary Algorithm (QIEA) for multi-objective UAV trajectory optimization. The framework addresses the challenge of finding optimal, safe, and smooth flight paths in complex environments with circular no-fly zones while minimizing path length, energy consumption, and maximizing safety margins.

## Key Contributions

### 1. **Hybrid Path Planning Framework**
- **Classical + Quantum-Inspired Optimization**: Combines the reliability of classical grid-based algorithms (A*, Theta*, Dijkstra) with the optimization power of QIEA for continuous space exploration
- **Seamless Integration**: Classical algorithms provide initial feasible paths, which are then optimized by QIEA in continuous space
- **Strict Line-of-Sight (LOS) Compliance**: Ensures paths maintain strict safety margins from obstacles (distance > radius, no tangency allowed)

### 2. **Advanced QIEA Enhancements**
- **Direction Vector Encoding**: Encodes path segments as direction vectors (angles) rather than absolute positions, enabling more effective quantum rotation gates and smoother path evolution
- **Adaptive Rotation Mechanism**: Dynamically adjusts rotation angles based on generation number (exploration → exploitation) and fitness gap (faster convergence for larger gaps)
- **Population Seeding**: Initializes QIEA population with classical algorithm solutions and their variations, significantly improving convergence speed and solution quality

### 3. **Multi-Objective Optimization**
- **Comprehensive Cost Function**: Optimizes three objectives simultaneously:
  - **Path Length** (weight: 1.0): Minimizes total travel distance
  - **Energy Consumption** (weight: 0.5): Reduces number of turns and path complexity
  - **Safety** (weight: 2.0): Maximizes distance from obstacles
- **Strict Collision Avoidance**: Heavy penalties for any path segments or waypoints violating obstacle constraints

### 4. **Performance Improvements**
- **Path Length Reduction**: ~30% reduction compared to classical algorithms
- **Waypoint Reduction**: ~75% reduction (from 35 waypoints to 6-9 waypoints)
- **Path Smoothness**: Significantly smoother paths with fewer unnecessary turns
- **Safety Guarantees**: All paths comply with strict LOS requirements

### 5. **Comprehensive Map Generation**
- **Three Map Types**: 
  - **m1 (Sparse)**: Maximum 4 alternative paths, moderate obstacle density
  - **m2 (Dense)**: Maximum 3 alternative paths, high obstacle density
  - **m3 (Trap)**: Maximum 2 alternative paths, very high obstacle density
- **Border Obstacles**: Prevents paths from going outside map boundaries
- **Controlled Complexity**: Ensures maps meet specific path count constraints

## Project Structure

```
FJCAI-2026_uav-qiea-hybrid-optimizer/
├── algorithms/                    # Core algorithm implementations
│   ├── __init__.py               # Package initialization
│   ├── base.py                   # Base classes (PathPlanner, PathResult)
│   ├── utils.py                  # Utility functions (collision detection, cost calculation)
│   ├── astar.py                  # A* algorithm implementation
│   ├── theta_star.py             # Theta* algorithm implementation
│   ├── dijkstra.py               # Dijkstra algorithm implementation
│   ├── qiea.py                   # QIEA algorithm with all enhancements
│   ├── hybrid.py                 # Hybrid algorithms (A*+QIEA, Theta*+QIEA, Dijkstra+QIEA)
│   └── README.md                 # Detailed algorithm documentation
│
├── maps/                         # Map data and visualizations
│   ├── m1_*.json                 # Sparse maps (20x20, 50x50, 100x100)
│   ├── m2_*.json                 # Dense maps (20x20, 50x50, 100x100)
│   ├── m3_*.json                 # Trap maps (20x20, 50x50, 100x100)
│   ├── visualizations/           # Generated path visualizations
│   └── result/                   # High-quality publication figures
│
├── map_generator.py              # Map generation with controlled complexity
├── test_algorithms.py            # Main testing script for all algorithms
├── validate_paths.py            # Path validation and strict LOS checking
├── visualize_paths.py           # Path visualization tool (overlay & side-by-side)
├── visualize_map.py              # Map visualization tool
│
├── ANALYSIS_QIEA_IMPROVEMENT.md  # Detailed analysis of QIEA improvements
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/lephus/FJCAI-2026_uav-qiea-hybrid-optimizer.git
cd FJCAI-2026_uav-qiea-hybrid-optimizer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Running All Algorithms on a Map

```bash
python3 test_algorithms.py
```

This will test all algorithms (A*, Theta*, Dijkstra, A*+QIEA, Theta*+QIEA, Dijkstra+QIEA) on available maps and display performance metrics.

#### Using Algorithms Programmatically

```python
from algorithms import AStar, ThetaStar, Dijkstra
from algorithms import AStarQIEA, ThetaStarQIEA, DijkstraQIEA
import json

# Load map data
with open('maps/m1_50x50.json', 'r') as f:
    map_data = json.load(f)

obstacles = [tuple(obs) for obs in map_data['obstacles']]
start = tuple(map_data['start'])
goal = tuple(map_data['goal'])
width = map_data['width']
height = map_data['height']

# Create planner (example: A* + QIEA)
planner = AStarQIEA(obstacles, width, height)

# Plan path
result = planner.plan(start, goal)

if result.success:
    print(f"Path found!")
    print(f"  Length: {result.path_length:.2f}")
    print(f"  Waypoints: {len(result.path)}")
    print(f"  Computation time: {result.computation_time:.4f}s")
    print(f"  Cost: {result.cost:.2f}")
else:
    print(f"Failed: {result.message}")
```

#### Visualizing Paths

**Overlay mode** (all paths on one map):
```bash
python3 visualize_paths.py --map maps/m1_50x50.json --mode overlay
```

**Side-by-side mode** (separate plots for each algorithm):
```bash
python3 visualize_paths.py --map maps/m1_50x50.json --mode side-by-side
```

This generates high-quality PNG (600 DPI) and PDF files for each algorithm in `maps/visualizations/`.

#### Generating Maps

```bash
python3 map_generator.py
```

This generates maps of different types (m1, m2, m3) and sizes (20x20, 50x50, 100x100) with controlled path complexity.

#### Validating Paths

```bash
python3 validate_paths.py --map maps/m1_50x50.json
```

Validates that all paths comply with strict LOS requirements (no touching or entering obstacles).

### Advanced Configuration

#### QIEA Parameters

You can customize QIEA parameters when creating hybrid planners:

```python
# Custom QIEA parameters
planner = AStarQIEA(
    obstacles, width, height,
    qiea_population_size=50,      # Population size (default: 30)
    qiea_max_generations=100      # Max generations (default: 50)
)
```

#### Multi-Objective Weights

Weights can be adjusted in `algorithms/qiea.py` in the `_evaluate_fitness()` function:

```python
cost = calculate_path_cost(
    path, self.obstacles,
    weight_length=1.0,    # Path length weight
    weight_energy=0.5,   # Energy (turns) weight
    weight_safety=2.0     # Safety weight
)
```

### Testing

Run comprehensive tests:

```bash
# Test all algorithms
python3 test_algorithms.py

# Test strict LOS compliance
python3 test_all_maps_strict_los.py

# Test QIEA improvements
python3 test_qiea_improvements.py
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{le2026multiobjective,
  title={Hybrid UAV Trajectory Optimization via Search Initialization and Quantum-Inspired Evolutionary Algorithms},
  author={Do Phuc Hao, Nguyen Nang Hung Van and Le Huu Phu},
  booktitle={Proceedings of the IEEE FJCAI 2026},
  year={2026},
  organization={IEEE}
}
```

**Note**: This code is provided for research purposes. If you use this implementation or build upon it, please cite the original paper and acknowledge this repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Classical path planning algorithms (A*, Theta*, Dijkstra) for initial path finding
- Quantum-Inspired Evolutionary Algorithm framework for continuous optimization
- Multi-objective optimization techniques for balancing path length, energy, and safety

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the authors.

---

**Keywords**: UAV Path Planning, Quantum-Inspired Evolutionary Algorithm, Multi-Objective Optimization, Trajectory Optimization, Path Planning, Obstacle Avoidance