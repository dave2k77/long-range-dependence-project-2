"""
Data Submission Module for Long-Range Dependence Analysis Framework

This module provides functionality for users to submit:
1. Datasets (raw, processed, or synthetic)
2. New estimators
3. Benchmark results
4. Validation reports

The module handles data validation, storage, and integration into the framework.
"""

from .dataset_submission import DatasetSubmissionManager
from .estimator_submission import EstimatorSubmissionManager
from .benchmark_submission import BenchmarkSubmissionManager
from .validation import DataValidator, EstimatorValidator

__all__ = [
    'DatasetSubmissionManager',
    'EstimatorSubmissionManager', 
    'BenchmarkSubmissionManager',
    'DataValidator',
    'EstimatorValidator'
]

__version__ = "1.0.0"
