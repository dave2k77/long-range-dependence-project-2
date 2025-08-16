"""
Long-Range Dependence Estimators

This module contains implementations of various methods for estimating
long-range dependence in time series data.
"""

from .base import BaseEstimator
from .temporal import DFAEstimator, MFDFAEstimator, RSEstimator, HiguchiEstimator
from .spectral import WhittleMLEEstimator, PeriodogramEstimator, GPHEstimator
from .wavelet import WaveletLeadersEstimator, WaveletWhittleEstimator

# High-performance variants
try:
    from .high_performance import HighPerformanceMFDFAEstimator
    from .high_performance_dfa import HighPerformanceDFAEstimator
    from .high_performance_rs import HighPerformanceRSEstimator
    from .high_performance_higuchi import HighPerformanceHiguchiEstimator
    from .high_performance_whittle import HighPerformanceWhittleMLEEstimator
    HIGH_PERFORMANCE_AVAILABLE = True
except ImportError:
    HIGH_PERFORMANCE_AVAILABLE = False

__all__ = [
    "BaseEstimator",
    "DFAEstimator",
    "MFDFAEstimator", 
    "RSEstimator",
    "HiguchiEstimator",
    "WhittleMLEEstimator",
    "PeriodogramEstimator",
    "GPHEstimator",
    "WaveletLeadersEstimator",
    "WaveletWhittleEstimator",
]

# Add high-performance variants if available
if HIGH_PERFORMANCE_AVAILABLE:
    __all__.extend([
        "HighPerformanceDFAEstimator",
        "HighPerformanceMFDFAEstimator",
        "HighPerformanceRSEstimator",
        "HighPerformanceHiguchiEstimator",
        "HighPerformanceWhittleMLEEstimator"
    ])
