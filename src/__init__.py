"""
Long-Range Dependence Benchmarking Framework

A comprehensive framework for detecting and characterising long-range dependence
in time series data with statistical validation and performance benchmarking.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import estimators
from . import validation
from . import benchmarking
from . import utils

__all__ = ["estimators", "validation", "benchmarking", "utils"]
