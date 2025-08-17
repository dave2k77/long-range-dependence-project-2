"""
Long-Range Dependence Estimators

This module contains implementations of various methods for estimating
long-range dependence in time series data.
"""

from .base import BaseEstimator
from .temporal import DFAEstimator, MFDFAEstimator, RSEstimator, HiguchiEstimator, DMAEstimator
from .spectral import WhittleMLEEstimator, PeriodogramEstimator, GPHEstimator
from .wavelet import WaveletLeadersEstimator, WaveletWhittleEstimator, WaveletLogVarianceEstimator, WaveletVarianceEstimator

# High-performance variants
try:
    from .high_performance import HighPerformanceMFDFAEstimator
    from .high_performance_dfa import HighPerformanceDFAEstimator
    from .high_performance_rs import HighPerformanceRSEstimator
    from .high_performance_higuchi import HighPerformanceHiguchiEstimator
    from .high_performance_whittle import HighPerformanceWhittleMLEEstimator
    from .high_performance_periodogram import HighPerformancePeriodogramEstimator
    from .high_performance_gph import HighPerformanceGPHEstimator
    from .high_performance_wavelet_leaders import HighPerformanceWaveletLeadersEstimator
    from .high_performance_wavelet_whittle import HighPerformanceWaveletWhittleEstimator
    from .high_performance_wavelet_log_variance import HighPerformanceWaveletLogVarianceEstimator
    from .high_performance_dma import HighPerformanceDMAEstimator
    from .high_performance_wavelet_variance import HighPerformanceWaveletVarianceEstimator
    HIGH_PERFORMANCE_AVAILABLE = True
except ImportError:
    HIGH_PERFORMANCE_AVAILABLE = False

__all__ = [
    "BaseEstimator",
    "DFAEstimator",
    "MFDFAEstimator", 
    "RSEstimator",
    "HiguchiEstimator",
    "DMAEstimator",
    "WhittleMLEEstimator",
    "PeriodogramEstimator",
    "GPHEstimator",
    "WaveletLeadersEstimator",
    "WaveletWhittleEstimator",
    "WaveletLogVarianceEstimator",
    "WaveletVarianceEstimator",
]

# Add high-performance variants if available
if HIGH_PERFORMANCE_AVAILABLE:
    __all__.extend([
        "HighPerformanceDFAEstimator",
        "HighPerformanceMFDFAEstimator",
        "HighPerformanceRSEstimator",
        "HighPerformanceHiguchiEstimator",
        "HighPerformanceWhittleMLEEstimator",
        "HighPerformancePeriodogramEstimator",
        "HighPerformanceGPHEstimator",
        "HighPerformanceWaveletLeadersEstimator",
        "HighPerformanceWaveletWhittleEstimator",
        "HighPerformanceWaveletLogVarianceEstimator",
        "HighPerformanceDMAEstimator",
        "HighPerformanceWaveletVarianceEstimator"
    ])
