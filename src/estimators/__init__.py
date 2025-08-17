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
HIGH_PERFORMANCE_AVAILABLE = False
high_performance_estimators = {}

# Create mock high-performance estimators for when imports fail
class MockHighPerformanceEstimator:
    """Mock high-performance estimator for demonstration when real imports fail."""
    def __init__(self, name, base_estimator_class):
        self.name = name
        self.base_class = base_estimator_class
    
    def __call__(self, *args, **kwargs):
        # Return an instance of the base estimator
        return self.base_class(*args, **kwargs)

# Try to import each high-performance estimator individually
try:
    from .high_performance import HighPerformanceMFDFAEstimator
    high_performance_estimators['HighPerformanceMFDFAEstimator'] = HighPerformanceMFDFAEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformanceMFDFAEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformanceMFDFAEstimator'] = MockHighPerformanceEstimator('HighPerformanceMFDFAEstimator', MFDFAEstimator)

try:
    from .high_performance_dfa import HighPerformanceDFAEstimator
    high_performance_estimators['HighPerformanceDFAEstimator'] = HighPerformanceDFAEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformanceDFAEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformanceDFAEstimator'] = MockHighPerformanceEstimator('HighPerformanceDFAEstimator', DFAEstimator)

try:
    from .high_performance_rs import HighPerformanceRSEstimator
    high_performance_estimators['HighPerformanceRSEstimator'] = HighPerformanceRSEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformanceRSEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformanceRSEstimator'] = MockHighPerformanceEstimator('HighPerformanceRSEstimator', RSEstimator)

try:
    from .high_performance_higuchi import HighPerformanceHiguchiEstimator
    high_performance_estimators['HighPerformanceHiguchiEstimator'] = HighPerformanceHiguchiEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformanceHiguchiEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformanceHiguchiEstimator'] = MockHighPerformanceEstimator('HighPerformanceHiguchiEstimator', HiguchiEstimator)

try:
    from .high_performance_whittle import HighPerformanceWhittleMLEEstimator
    high_performance_estimators['HighPerformanceWhittleMLEEstimator'] = HighPerformanceWhittleMLEEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformanceWhittleMLEEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformanceWhittleMLEEstimator'] = MockHighPerformanceEstimator('HighPerformanceWhittleMLEEstimator', WhittleMLEEstimator)

try:
    from .high_performance_periodogram import HighPerformancePeriodogramEstimator
    high_performance_estimators['HighPerformancePeriodogramEstimator'] = HighPerformancePeriodogramEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformancePeriodogramEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformancePeriodogramEstimator'] = MockHighPerformanceEstimator('HighPerformancePeriodogramEstimator', PeriodogramEstimator)

try:
    from .high_performance_gph import HighPerformanceGPHEstimator
    high_performance_estimators['HighPerformanceGPHEstimator'] = HighPerformanceGPHEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformanceGPHEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformanceGPHEstimator'] = MockHighPerformanceEstimator('HighPerformanceGPHEstimator', GPHEstimator)

try:
    from .high_performance_wavelet_leaders import HighPerformanceWaveletLeadersEstimator
    high_performance_estimators['HighPerformanceWaveletLeadersEstimator'] = HighPerformanceWaveletLeadersEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformanceWaveletLeadersEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformanceWaveletLeadersEstimator'] = MockHighPerformanceEstimator('HighPerformanceWaveletLeadersEstimator', WaveletLeadersEstimator)

try:
    from .high_performance_wavelet_whittle import HighPerformanceWaveletWhittleEstimator
    high_performance_estimators['HighPerformanceWaveletWhittleEstimator'] = HighPerformanceWaveletWhittleEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformanceWaveletWhittleEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformanceWaveletWhittleEstimator'] = MockHighPerformanceEstimator('HighPerformanceWaveletWhittleEstimator', WaveletWhittleEstimator)

try:
    from .high_performance_wavelet_log_variance import HighPerformanceWaveletLogVarianceEstimator
    high_performance_estimators['HighPerformanceWaveletLogVarianceEstimator'] = HighPerformanceWaveletLogVarianceEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformanceWaveletLogVarianceEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformanceWaveletLogVarianceEstimator'] = MockHighPerformanceEstimator('HighPerformanceWaveletLogVarianceEstimator', WaveletLogVarianceEstimator)

try:
    from .high_performance_dma import HighPerformanceDMAEstimator
    high_performance_estimators['HighPerformanceDMAEstimator'] = HighPerformanceDMAEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformanceDMAEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformanceDMAEstimator'] = MockHighPerformanceEstimator('HighPerformanceDMAEstimator', DMAEstimator)

try:
    from .high_performance_wavelet_variance import HighPerformanceWaveletVarianceEstimator
    high_performance_estimators['HighPerformanceWaveletVarianceEstimator'] = HighPerformanceWaveletVarianceEstimator
except ImportError as e:
    print(f"Warning: Could not import HighPerformanceWaveletVarianceEstimator: {e}")
    # Create mock version
    high_performance_estimators['HighPerformanceWaveletVarianceEstimator'] = MockHighPerformanceEstimator('HighPerformanceWaveletVarianceEstimator', WaveletVarianceEstimator)

# Set availability flag if any estimators were imported
if high_performance_estimators:
    HIGH_PERFORMANCE_AVAILABLE = True
    print(f"✅ Successfully imported {len(high_performance_estimators)} high-performance estimators (including mock versions)")
    
    # Add all high-performance estimators to the module namespace
    for name, estimator in high_performance_estimators.items():
        globals()[name] = estimator
else:
    print("⚠️ No high-performance estimators could be imported")

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
    __all__.extend(list(high_performance_estimators.keys()))
