"""
Performance Benchmarking Framework

This module provides tools for benchmarking the performance of different
long-range dependence estimators in terms of accuracy, efficiency, and robustness.
"""

# Try to import modules with graceful fallbacks
try:
    from .benchmark_runner import BenchmarkRunner
except ImportError as e:
    print(f"Warning: Could not import BenchmarkRunner: {e}")
    BenchmarkRunner = None

try:
    from .performance_metrics import PerformanceMetrics
except ImportError as e:
    print(f"Warning: Could not import PerformanceMetrics: {e}")
    PerformanceMetrics = None

try:
    from .leaderboard import PerformanceLeaderboard
except ImportError as e:
    print(f"Warning: Could not import PerformanceLeaderboard: {e}")
    PerformanceLeaderboard = None

try:
    from .synthetic_data import SyntheticDataGenerator
except ImportError as e:
    print(f"Warning: Could not import SyntheticDataGenerator: {e}")
    SyntheticDataGenerator = None

# Build __all__ list dynamically based on successful imports
__all__ = []
if BenchmarkRunner is not None:
    __all__.append("BenchmarkRunner")
if PerformanceMetrics is not None:
    __all__.append("PerformanceMetrics")
if PerformanceLeaderboard is not None:
    __all__.append("PerformanceLeaderboard")
if SyntheticDataGenerator is not None:
    __all__.append("SyntheticDataGenerator")

# Print status
print(f"✅ Benchmarking module initialized with {len(__all__)} available components")
