"""
Base class for Long-Range Dependence estimators.

This module provides the abstract base class that all LRD estimators
must implement, ensuring consistent interface and functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class BaseEstimator(ABC):
    """
    Abstract base class for Long-Range Dependence estimators.
    
    All LRD estimators must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, name: str = None, **kwargs):
        """
        Initialize the base estimator.
        
        Parameters
        ----------
        name : str, optional
            Name identifier for the estimator
        **kwargs
            Additional keyword arguments
        """
        self.name = name or self.__class__.__name__
        self.parameters = kwargs
        self.results = {}
        self.execution_time = None
        self.memory_usage = None
        
    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs) -> 'BaseEstimator':
        """
        Fit the estimator to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        self : BaseEstimator
            Fitted estimator instance
        """
        pass
    
    @abstractmethod
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Estimate long-range dependence parameters.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Input time series data (if not provided, uses fitted data)
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing estimation results
        """
        pass
    
    def fit_estimate(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Fit the estimator and estimate parameters in one step.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional parameters for fitting and estimation
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing estimation results
        """
        start_time = time.time()
        
        try:
            # Fit the estimator
            self.fit(data, **kwargs)
            
            # Estimate parameters
            results = self.estimate(**kwargs)
            
            # Record execution time
            self.execution_time = time.time() - start_time
            
            # Store results
            self.results.update(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fit_estimate for {self.name}: {str(e)}")
            raise
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the stored results from the last estimation.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the last estimation results
        """
        return self.results.copy()
    
    def get_execution_time(self) -> Optional[float]:
        """
        Get the execution time of the last estimation.
        
        Returns
        -------
        float or None
            Execution time in seconds, or None if not yet executed
        """
        return self.execution_time
    
    def get_memory_usage(self) -> Optional[float]:
        """
        Get the memory usage of the last estimation.
        
        Returns
        -------
        float or None
            Memory usage in bytes, or None if not yet executed
        """
        return self.memory_usage
    
    def reset(self):
        """Reset the estimator to initial state."""
        self.results = {}
        self.execution_time = None
        self.memory_usage = None
        self.data = None
    
    def __repr__(self) -> str:
        """String representation of the estimator."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __str__(self) -> str:
        """String representation of the estimator."""
        return self.__repr__()
