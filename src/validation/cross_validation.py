"""
Cross Validation for Long-Range Dependence

This module provides cross validation methods for
long-range dependence estimates.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Cross validation framework for long-range dependence estimates.
    
    This class provides methods for validating estimates using
    various cross validation strategies.
    """
    
    def __init__(self, n_folds: int = 5, random_state: Optional[int] = None):
        """
        Initialize the cross validator.
        
        Parameters
        ----------
        n_folds : int
            Number of folds for cross validation (default: 5)
        random_state : Optional[int]
            Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            
    def k_fold_cross_validation(self, estimator_func, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Perform k-fold cross validation.
        
        Parameters
        ----------
        estimator_func : callable
            Function that estimates the parameter of interest
        data : np.ndarray
            Input data for estimation
        **kwargs : dict
            Additional arguments for the estimator function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing cross validation results
        """
        n_samples = len(data)
        fold_size = n_samples // self.n_folds
        
        # Create fold indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        fold_estimates = []
        fold_errors = []
        
        for fold in range(self.n_folds):
            try:
                # Define test set for this fold
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < self.n_folds - 1 else n_samples
                test_indices = indices[start_idx:end_idx]
                train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
                
                # Split data
                train_data = data[train_indices]
                test_data = data[test_indices]
                
                # Fit estimator on training data
                train_result = estimator_func(train_data, **kwargs)
                if isinstance(train_result, dict) and 'hurst_exponent' in train_result:
                    train_estimate = train_result['hurst_exponent']
                elif isinstance(train_result, (int, float)):
                    train_estimate = train_result
                else:
                    train_estimate = np.nan
                    
                # Test on test data
                test_result = estimator_func(test_data, **kwargs)
                if isinstance(test_result, dict) and 'hurst_exponent' in test_result:
                    test_estimate = test_result['hurst_exponent']
                elif isinstance(test_result, (int, float)):
                    test_estimate = test_result
                else:
                    test_estimate = np.nan
                    
                if not np.isnan(train_estimate) and not np.isnan(test_estimate):
                    fold_estimates.append(test_estimate)
                    fold_errors.append(abs(test_estimate - train_estimate))
                    
            except Exception as e:
                logger.warning(f"Fold {fold} failed: {e}")
                continue
                
        if not fold_estimates:
            return {
                'mean_estimate': np.nan,
                'std_estimate': np.nan,
                'mean_error': np.nan,
                'fold_estimates': [],
                'fold_errors': [],
                'n_successful_folds': 0
            }
            
        results = {
            'mean_estimate': np.mean(fold_estimates),
            'std_estimate': np.std(fold_estimates, ddof=1),
            'mean_error': np.mean(fold_errors),
            'fold_estimates': fold_estimates,
            'fold_errors': fold_errors,
            'n_successful_folds': len(fold_estimates),
            'n_folds': self.n_folds
        }
        
        return results
        
    def time_series_cross_validation(self, estimator_func, data: np.ndarray, 
                                   min_train_size: int = None, **kwargs) -> Dict[str, Any]:
        """
        Perform time series cross validation (expanding window).
        
        Parameters
        ----------
        estimator_func : callable
            Function that estimates the parameter of interest
        data : np.ndarray
            Input data for estimation
        min_train_size : int
            Minimum training set size (default: 100)
        **kwargs : dict
            Additional arguments for the estimator function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing time series cross validation results
        """
        if min_train_size is None:
            min_train_size = min(100, len(data) // 4)
            
        n_samples = len(data)
        estimates = []
        errors = []
        
        for i in range(min_train_size, n_samples):
            try:
                # Training set: data from start to i-1
                train_data = data[:i]
                
                # Test set: single point at i
                test_data = data[i:i+1]
                
                # Fit estimator on training data
                train_result = estimator_func(train_data, **kwargs)
                if isinstance(train_result, dict) and 'hurst_exponent' in train_result:
                    train_estimate = train_result['hurst_exponent']
                elif isinstance(train_result, (int, float)):
                    train_estimate = train_result
                else:
                    train_estimate = np.nan
                    
                # For time series, we can't really test on a single point
                # So we'll use the training estimate as the "test" estimate
                # and calculate error based on stability
                if not np.isnan(train_estimate):
                    estimates.append(train_estimate)
                    
                    # Calculate error based on change from previous estimate
                    if len(estimates) > 1:
                        error = abs(estimates[-1] - estimates[-2])
                        errors.append(error)
                        
            except Exception as e:
                logger.warning(f"Time series CV iteration {i} failed: {e}")
                continue
                
        if not estimates:
            return {
                'mean_estimate': np.nan,
                'std_estimate': np.nan,
                'mean_error': np.nan,
                'estimates': [],
                'errors': [],
                'n_iterations': 0
            }
            
        results = {
            'mean_estimate': np.mean(estimates),
            'std_estimate': np.std(estimates, ddof=1),
            'mean_error': np.mean(errors) if errors else 0.0,
            'estimates': estimates,
            'errors': errors,
            'n_iterations': len(estimates),
            'min_train_size': min_train_size
        }
        
        return results
        
    def leave_one_out_cross_validation(self, estimator_func, data: np.ndarray, 
                                     max_samples: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Perform leave-one-out cross validation (limited for computational efficiency).
        
        Parameters
        ----------
        estimator_func : callable
            Function that estimates the parameter of interest
        data : np.ndarray
            Input data for estimation
        max_samples : int
            Maximum number of samples to test (default: 100)
        **kwargs : dict
            Additional arguments for the estimator function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing leave-one-out cross validation results
        """
        n_samples = min(len(data), max_samples)
        estimates = []
        
        for i in range(n_samples):
            try:
                # Leave out sample i
                test_indices = [i]
                train_indices = list(range(n_samples))
                train_indices.remove(i)
                
                # Split data
                train_data = data[train_indices]
                test_data = data[test_indices]
                
                # Fit estimator on training data
                train_result = estimator_func(train_data, **kwargs)
                if isinstance(train_result, dict) and 'hurst_exponent' in train_result:
                    train_estimate = train_result['hurst_exponent']
                elif isinstance(train_result, (int, float)):
                    train_estimate = train_result
                else:
                    train_estimate = np.nan
                    
                if not np.isnan(train_estimate):
                    estimates.append(train_estimate)
                    
            except Exception as e:
                logger.warning(f"Leave-one-out CV iteration {i} failed: {e}")
                continue
                
        if not estimates:
            return {
                'mean_estimate': np.nan,
                'std_estimate': np.nan,
                'estimates': [],
                'n_iterations': 0
            }
            
        results = {
            'mean_estimate': np.mean(estimates),
            'std_estimate': np.std(estimates, ddof=1),
            'estimates': estimates,
            'n_iterations': len(estimates),
            'max_samples': max_samples
        }
        
        return results
