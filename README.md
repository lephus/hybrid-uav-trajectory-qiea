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
- **Four Map Types**: 
  - **m1 (Sparse)**: Maximum 4 alternative paths, moderate obstacle density
  - **m2 (Dense)**: Maximum 3 alternative paths, high obstacle density
  - **m3 (Trap)**: Maximum 2 alternative paths, very high obstacle density
  - **m4 (QIEA Challenge)**: Complex maze-like structure with multiple alternative paths, narrow passages, and local optima (designed to showcase QIEA's advantages)
- **Border Obstacles**: Prevents paths from going outside map boundaries
- **Controlled Complexity**: Ensures maps meet specific path count constraints
- **Multi-UAV Support**: Supports both n-1 (n UAVs → 1 goal) and n-n (n UAVs → n destinations) scenarios

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
├── map_generator.py              # Map generation for single UAV (1-1)
├── map_generator_multi_uav.py   # Map generation for n-1 scenario (n UAVs → 1 goal)
├── map_generator_nn.py          # Map generation for n-n scenario (n UAVs → n destinations)
├── test_algorithms.py            # Main testing script for single UAV algorithms
├── test_multi_path.py            # Testing script for n-1 scenario
├── test_nn_paths.py              # Testing script for n-n scenario
├── validate_paths.py            # Path validation and strict LOS checking
├── visualize_paths.py           # Path visualization tool (overlay & side-by-side)
├── visualize_map.py              # Map visualization tool
├── visualize_multi_paths.py      # Visualization for n-1 scenario
├── visualize_nn_paths.py         # Visualization for n-n scenario
│
├── maps/
│   ├── multi_uav/               # Maps for n-1 scenario
│   └── nn_destinations/         # Maps for n-n scenario
│
├── ANALYSIS_QIEA_IMPROVEMENT.md  # Detailed analysis of QIEA improvements
├── QIEA_MAP_GUIDE.md            # Guide for creating maps that showcase QIEA advantages
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

### Multi-UAV Path Planning

The framework supports two multi-UAV scenarios:

#### 1. **n UAVs → 1 Goal (n-1 Scenario)**

Multiple UAVs start from different positions and all navigate to a single shared goal.

**Generate maps:**
```bash
# Generate a single map
python3 map_generator_multi_uav.py --type m1 --size 50 --uavs 5

# Generate all maps (all types, sizes, and UAV counts)
python3 map_generator_multi_uav.py --all
```

**Test algorithms:**
```bash
# Test with visualization
python3 test_multi_path.py --viz

# Test with specific map and save visualizations
python3 test_multi_path.py --map maps/multi_uav/m1_50x50_5uavs.json --viz --save-viz

# Test with specific number of UAVs
python3 test_multi_path.py --uavs 3 5 10 --viz
```

**Available algorithms for n-1:**
- `MultiPathAStar`: A* for each UAV independently
- `MultiPathThetaStar`: Theta* for each UAV independently
- `MultiPathDijkstra`: Dijkstra for each UAV independently
- `MultiPathAStarQIEA`: A* + QIEA optimization for each UAV
- `MultiPathThetaStarQIEA`: Theta* + QIEA optimization for each UAV
- `MultiPathDijkstraQIEA`: Dijkstra + QIEA optimization for each UAV

#### 2. **n UAVs → n Destinations (n-n Scenario)**

Each UAV is assigned to a specific destination (one-to-one assignment).

**Generate maps:**
```bash
# Generate a single n-n map
python3 map_generator_nn.py --type m1 --size 50 --uavs 5

# Generate m4 (QIEA Challenge) map - designed to showcase QIEA advantages
python3 map_generator_nn.py --type m4 --size 100 --uavs 5

# Generate all n-n maps
python3 map_generator_nn.py --all
```

**Test algorithms:**
```bash
# Test with visualization
python3 test_nn_paths.py --map maps/nn_destinations/m1_50x50_3uavs.json --viz --save-viz

# Test on QIEA Challenge map (m4)
python3 test_nn_paths.py --map maps/nn_destinations/m4_100x100_5nn.json --viz --save-viz
```

**Available algorithms for n-n:**
Each algorithm is tested independently on each UAV-destination pair:
- `AStar`: A* path planning
- `ThetaStar`: Theta* path planning
- `Dijkstra`: Dijkstra path planning
- `AStarQIEA`: A* + QIEA optimization
- `ThetaStarQIEA`: Theta* + QIEA optimization
- `DijkstraQIEA`: Dijkstra + QIEA optimization
- `QIEA`: Standalone QIEA path planning

**Map types for n-n:**
- **m1**: Sparse environment with low obstacle density
- **m2**: Dense environment with high obstacle density
- **m3**: Maximum difficulty with only 1-2 possible paths
- **m4**: QIEA Challenge - Complex maze-like structure with multiple alternative paths, narrow passages, and local optima (recommended for showcasing QIEA advantages)

**Example workflow for n-n:**
```bash
# 1. Generate a QIEA Challenge map
python3 map_generator_nn.py --type m4 --size 100 --uavs 5

# 2. Test all algorithms and visualize
python3 test_nn_paths.py --map maps/nn_destinations/m4_100x100_5nn.json --viz --save-viz

# 3. View results in maps/visualizations/
```

**Note**: For best results showcasing QIEA advantages, use map type **m4** with size >= 100. See `QIEA_MAP_GUIDE.md` for detailed guidance.

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
# Test single UAV algorithms
python3 test_algorithms.py

# Test multi-UAV n-1 scenario
python3 test_multi_path.py --viz

# Test multi-UAV n-n scenario
python3 test_nn_paths.py --map maps/nn_destinations/m1_50x50_3uavs.json --viz --save-viz

# Test strict LOS compliance
python3 test_all_maps_strict_los.py

# Test QIEA improvements
python3 test_qiea_improvements.py
```

## Understanding Output Metrics

When running `test_multi_path.py` or `test_nn_paths.py`, you'll see various metrics in both console logs and visualization images. This section explains what each metric means and how to interpret them.

### Console Log Output

#### Map Information
```
Map Info:
  Size: 50x50                    # Map dimensions (width × height in grid units)
  Obstacles: 82                  # Number of circular obstacles on the map
  Goal: (30, 10)                 # Shared goal position (all UAVs navigate here)
  Pre-defined starts: 3          # Number of pre-defined start positions
```

#### Algorithm Results
For each algorithm tested, you'll see:
```
Testing: MultiPathAStar with 3 UAVs
  Start positions: [(5, 5), (10, 8), (15, 12)]  # Starting positions of each UAV
  Goal: (30, 10)                                 # Shared destination
  
  Success: ✓                                     # Whether paths were found (✓ = success, ✗ = failed)
  Paths found: 3/3 UAVs                          # Number of successful paths / Total UAVs
  
  Total path length: 125.50                     # Sum of all path lengths (all UAVs combined)
  Makespan: 45.20 seconds                        # Time for the last UAV to reach the goal
  Computation time: 0.1234 seconds               # Algorithm execution time
  
  Message: Planned 3 UAVs to single goal        # Status message
  
  # Per-UAV details:
  UAV 0: 15 waypoints, length: 42.30             # UAV 0: 15 waypoints, path length 42.30
  UAV 1: 18 waypoints, length: 38.50            # UAV 1: 18 waypoints, path length 38.50
  UAV 2: 20 waypoints, length: 44.70            # UAV 2: 20 waypoints, path length 44.70
```

#### Summary Table
```
SUMMARY
Algorithm            Success    Total Length    Makespan      Time (s)
MultiPathAStar       ✓          125.50          45.20         0.1234
MultiPathThetaStar   ✓          118.30          42.10         0.2345
...
```

**Column meanings:**
- **Success**: ✓ (all paths found) or ✗ (some paths failed)
- **Total Length**: Sum of all path lengths (all UAVs combined)
- **Makespan**: Time for the last UAV to reach the goal (seconds)
- **Time (s)**: Algorithm computation time (seconds)

### Visualization Metrics

#### Individual Algorithm Visualization

In the information box (top-left corner):
```
Total Path Length: 125.50        # Sum of all path lengths
Makespan: 45.20s                 # Time for last UAV to reach goal
Computation Time: 0.1234s        # Algorithm execution time
```

**Visual elements on the map:**
- **Obstacles**: Red circles - no-fly zones
- **Start points**: Colored circles (○) - each UAV has a unique color
- **Goal**: Blue star (★) - shared destination
- **Paths**: Colored lines - each UAV's path in a unique color
- **Waypoints**: Small dots along paths - intermediate points

#### Comparison Visualization

In each subplot (top-left corner):
```
Length: 125.5                    # Total path length (abbreviated)
Makespan: 45.2s                  # Makespan time (abbreviated)
```

### Detailed Metric Explanations

#### 1. **Total Path Length**
- **Definition**: Sum of all path lengths for all UAVs
- **Formula**: `sum(path_length of UAV 0 + path_length of UAV 1 + ...)`
- **Units**: Grid units (or meters if scaled)
- **Interpretation**: Total distance traveled by all UAVs. **Lower is better.**

#### 2. **Makespan**
- **Definition**: Time for the last UAV to reach the goal
- **Formula**: `max(arrival_time of all UAVs)`
- **Units**: Seconds
- **Interpretation**: Mission completion time. Critical for multi-UAV coordination. **Lower is better.**

#### 3. **Computation Time**
- **Definition**: Time taken by the algorithm to find paths
- **Units**: Seconds
- **Interpretation**: Algorithm efficiency. **Lower is better**, but may trade off with path quality.

#### 4. **Waypoints**
- **Definition**: Number of intermediate points on a path (excluding start and goal)
- **Interpretation**: Path complexity. Fewer waypoints typically indicate smoother paths with fewer direction changes. **Lower is generally better.**

#### 5. **Path Length (per UAV)**
- **Definition**: Distance traveled by a specific UAV
- **Formula**: `sum(distances between consecutive waypoints)`
- **Units**: Grid units
- **Interpretation**: Individual UAV travel distance. **Lower is better.**

#### 6. **Success Rate**
- **Definition**: Percentage of UAVs that successfully found paths
- **Formula**: `(successful_paths / total_UAVs) × 100%`
- **Interpretation**: Algorithm reliability. **100% is ideal.**

### Example: Interpreting Results

```
SUMMARY
Algorithm            Success    Total Length    Makespan      Time (s)
MultiPathAStar       ✓          125.50          45.20         0.1234
MultiPathThetaStar   ✓          118.30          42.10         0.2345
MultiPathAStarQIEA   ✓          115.20          40.50         2.3456
```

**Analysis:**
- **MultiPathAStar**: Fastest (0.12s) but longest paths (125.50)
- **MultiPathThetaStar**: Slower (0.23s) but shorter paths (118.30)
- **MultiPathAStarQIEA**: Slowest (2.35s) but shortest paths (115.20) and best makespan (40.50s)

**Conclusion**: QIEA trades computation time for better path quality (shorter paths, better makespan).

### Important Notes

1. **Success**: Only shows ✓ when ALL UAVs successfully find paths
2. **Total Length**: Sum of all paths, NOT average
3. **Makespan**: Critical for multi-UAV scenarios - measures mission completion time
4. **Computation Time**: Does not include visualization time
5. **Waypoints**: Fewer waypoints usually indicate smoother, more efficient paths

### Tips for Analysis

- **Compare algorithms**: Lower Total Length and Makespan with reasonable Computation Time is ideal
- **QIEA variants**: Expect longer computation times but better path quality
- **Makespan importance**: In time-critical missions, makespan may be more important than total length
- **Waypoint count**: Fewer waypoints indicate smoother paths, which are easier for UAVs to follow

These metrics help evaluate and compare algorithm performance in multi-UAV path planning scenarios.

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