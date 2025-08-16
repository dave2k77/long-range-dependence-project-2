"""
Performance Metrics for Long-Range Dependence Estimators

This module provides tools for measuring and comparing the performance
of different LRD estimators.
"""

import numpy as np
import time
import psutil
import os
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Performance measurement framework for long-range dependence estimators.
    
    This class provides methods for measuring accuracy, efficiency,
    memory usage, and other performance aspects.
    """
    
    def __init__(self):
        """Initialize the performance metrics collector."""
        self.process = psutil.Process(os.getpid())
        
    def measure_execution_time(self, func, *args, **kwargs) -> Dict[str, Any]:
        """
        Measure execution time of a function.
        
        Parameters
        ----------
        func : callable
            Function to measure
        *args : tuple
            Positional arguments for the function
        **kwargs : dict
            Keyword arguments for the function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing timing information
        """
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logger.error(f"Function execution failed: {e}")
            
        end_time = time.perf_counter()
        end_cpu_time = time.process_time()
        
        wall_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time
        
        return {
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'success': success,
            'result': result
        }
        
    def measure_memory_usage(self, func, *args, **kwargs) -> Dict[str, Any]:
        """
        Measure memory usage of a function.
        
        Parameters
        ----------
        func : callable
            Function to measure
        *args : tuple
            Positional arguments for the function
        **kwargs : dict
            Keyword arguments for the function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing memory usage information
        """
        # Get initial memory usage
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logger.error(f"Function execution failed: {e}")
            
        # Get final memory usage
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        memory_delta = final_memory - initial_memory
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_delta_mb': memory_delta,
            'success': success,
            'result': result
        }
        
    def measure_accuracy(self, estimated_values: List[float], 
                        true_values: List[float]) -> Dict[str, float]:
        """
        Measure accuracy of estimates against true values.
        
        Parameters
        ----------
        estimated_values : List[float]
            List of estimated values
        true_values : List[float]
            List of true values
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing accuracy metrics
        """
        if len(estimated_values) != len(true_values):
            raise ValueError("Estimated and true values must have the same length")
            
        # Remove NaN values
        valid_mask = ~(np.isnan(estimated_values) | np.isnan(true_values))
        est_valid = np.array(estimated_values)[valid_mask]
        true_valid = np.array(true_values)[valid_mask]
        
        if len(est_valid) == 0:
            return {
                'mse': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'r_squared': np.nan,
                'bias': np.nan
            }
            
        # Calculate error metrics
        errors = est_valid - true_valid
        squared_errors = errors ** 2
        absolute_errors = np.abs(errors)
        
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)
        mae = np.mean(absolute_errors)
        
        # Mean absolute percentage error
        mape = np.mean(np.abs(errors / true_valid)) * 100
        
        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Bias
        bias = np.mean(errors)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r_squared': r_squared,
            'bias': bias
        }
        
    def measure_precision(self, estimated_values: List[float], 
                         true_values: List[float]) -> Dict[str, float]:
        """
        Measure precision (reproducibility) of estimates.
        
        Parameters
        ----------
        estimated_values : List[float]
            List of estimated values
        true_values : List[float]
            List of true values
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing precision metrics
        """
        if len(estimated_values) != len(true_values):
            raise ValueError("Estimated and true values must have the same length")
            
        # Remove NaN values
        valid_mask = ~(np.isnan(estimated_values) | np.isnan(true_values))
        est_valid = np.array(estimated_values)[valid_mask]
        true_valid = np.array(true_values)[valid_mask]
        
        if len(est_valid) == 0:
            return {
                'std_error': np.nan,
                'coefficient_of_variation': np.nan,
                'confidence_interval_95': (np.nan, np.nan)
            }
            
        # Standard error
        std_error = np.std(est_valid, ddof=1)
        
        # Coefficient of variation
        mean_estimate = np.mean(est_valid)
        coefficient_of_variation = std_error / abs(mean_estimate) if mean_estimate != 0 else np.nan
        
        # 95% confidence interval
        n = len(est_valid)
        t_value = 1.96  # Approximate for large n
        margin_of_error = t_value * std_error / np.sqrt(n)
        ci_lower = mean_estimate - margin_of_error
        ci_upper = mean_estimate + margin_of_error
        
        return {
            'std_error': std_error,
            'coefficient_of_variation': coefficient_of_variation,
            'confidence_interval_95': (ci_lower, ci_upper)
        }
        
    def measure_robustness(self, estimator_func, data: np.ndarray,
                          noise_levels: List[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Measure robustness of an estimator to noise.
        
        Parameters
        ----------
        estimator_func : callable
            Function that estimates the parameter of interest
        data : np.ndarray
            Input data for estimation
        noise_levels : List[float]
            List of noise levels to test
        **kwargs : dict
            Additional arguments for the estimator function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing robustness metrics
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
            
        data_std = np.std(data)
        estimates = []
        
        for noise_level in noise_levels:
            try:
                # Add noise to data
                noise = np.random.normal(0, noise_level * data_std, size=data.shape)
                noisy_data = data + noise
                
                # Estimate with noisy data
                result = estimator_func(noisy_data, **kwargs)
                if isinstance(result, dict) and 'hurst_exponent' in result:
                    estimate = result['hurst_exponent']
                elif isinstance(result, (int, float)):
                    estimate = result
                else:
                    estimate = np.nan
                    
                estimates.append(estimate)
                
            except Exception as e:
                logger.warning(f"Robustness test failed for noise level {noise_level}: {e}")
                estimates.append(np.nan)
                
        # Calculate robustness metrics
        valid_estimates = [e for e in estimates if not np.isnan(e)]
        
        if len(valid_estimates) < 2:
            return {
                'noise_levels': noise_levels,
                'estimates': estimates,
                'robustness_score': 0.0,
                'stability': np.nan
            }
            
        # Robustness score based on coefficient of variation
        mean_estimate = np.mean(valid_estimates)
        std_estimate = np.std(valid_estimates, ddof=1)
        cv = std_estimate / abs(mean_estimate) if mean_estimate != 0 else np.inf
        
        # Robustness score (lower CV = higher robustness)
        robustness_score = 1.0 / (1.0 + cv)
        
        # Stability (inverse of variance)
        stability = 1.0 / (1.0 + np.var(valid_estimates))
        
        return {
            'noise_levels': noise_levels,
            'estimates': estimates,
            'robustness_score': robustness_score,
            'stability': stability,
            'coefficient_of_variation': cv
        }
        
    def comprehensive_benchmark(self, estimator_func, data: np.ndarray,
                              true_value: float = None, **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive performance benchmarking.
        
        Parameters
        ----------
        estimator_func : callable
            Function that estimates the parameter of interest
        data : np.ndarray
            Input data for estimation
        true_value : float
            True value for accuracy measurement (if known)
        **kwargs : dict
            Additional arguments for the estimator function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing comprehensive benchmark results
        """
        benchmark_results = {}
        
        # Measure execution time
        timing_results = self.measure_execution_time(estimator_func, data, **kwargs)
        benchmark_results.update(timing_results)
        
        # Measure memory usage
        memory_results = self.measure_memory_usage(estimator_func, data, **kwargs)
        benchmark_results.update(memory_results)
        
        # Measure robustness
        robustness_results = self.measure_robustness(estimator_func, data, **kwargs)
        benchmark_results.update(robustness_results)
        
        # If true value is provided, measure accuracy
        if true_value is not None and timing_results['success']:
            try:
                result = timing_results['result']
                if isinstance(result, dict) and 'hurst_exponent' in result:
                    estimated_value = result['hurst_exponent']
                elif isinstance(result, (int, float)):
                    estimated_value = result
                else:
                    estimated_value = np.nan
                    
                if not np.isnan(estimated_value):
                    accuracy_results = self.measure_accuracy([estimated_value], [true_value])
                    benchmark_results.update(accuracy_results)
                    
            except Exception as e:
                logger.warning(f"Accuracy measurement failed: {e}")
                
        return benchmark_results
