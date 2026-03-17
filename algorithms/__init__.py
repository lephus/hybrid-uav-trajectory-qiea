"""
Path Planning Algorithms for UAV Trajectory Optimization

This package provides classical path planning algorithms (A*, Theta*, Dijkstra),
their QIEA-optimized hybrid variants, and standalone continuous-space planners
(RRT*, PSO).

QIEA (Quantum-Inspired Evolutionary Algorithm) is used to optimize paths
found by classical algorithms, not as a standalone path planner.
"""

from .base import PathPlanner, PathResult, MultiPathResult
from .astar import AStar
from .theta_star import ThetaStar
from .dijkstra import Dijkstra
from .hybrid import AStarQIEA, ThetaStarQIEA, DijkstraQIEA
from .multi_path_base import MultiPathAStar, MultiPathThetaStar, MultiPathDijkstra
from .multi_path_hybrid import MultiPathAStarQIEA, MultiPathThetaStarQIEA, MultiPathDijkstraQIEA
from .rrt import RRTStar
from .pso import PSO

# QIEA is available internally for hybrid algorithms but not exported
# as it's designed to optimize classical algorithms, not work standalone
from .qiea import QIEA as _QIEA
from .multi_path_qiea import MultiPathQIEA as _MultiPathQIEA

__all__ = [
    'PathPlanner',
    'PathResult',
    'MultiPathResult',
    'AStar',
    'ThetaStar',
    'Dijkstra',
    'AStarQIEA',
    'ThetaStarQIEA',
    'DijkstraQIEA',
    'MultiPathAStar',
    'MultiPathThetaStar',
    'MultiPathDijkstra',
    'MultiPathAStarQIEA',
    'MultiPathThetaStarQIEA',
    'MultiPathDijkstraQIEA',
    'RRTStar',
    'PSO',
]

