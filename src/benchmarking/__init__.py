"""
Performance Benchmarking Framework

This module provides tools for benchmarking the performance of different
long-range dependence estimators in terms of accuracy, efficiency, and robustness.
"""

from .benchmark_runner import BenchmarkRunner
from .performance_metrics import PerformanceMetrics
from .leaderboard import PerformanceLeaderboard
from .synthetic_data import SyntheticDataGenerator

__all__ = [
    "BenchmarkRunner",
    "PerformanceMetrics", 
    "PerformanceLeaderboard",
    "SyntheticDataGenerator"
]
