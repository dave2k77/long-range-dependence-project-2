"""
Robustness Testing for Long-Range Dependence

This module provides robustness testing methods for
long-range dependence estimates.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class RobustnessTester:
    """
    Robustness testing framework for long-range dependence estimates.
    
    This class provides methods for testing the robustness of estimates
    under various perturbations and conditions.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the robustness tester.
        
        Parameters
        ----------
        random_state : Optional[int]
            Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            
    def test_noise_robustness(self, estimator_func, data: np.ndarray, 
                            noise_levels: List[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Test robustness to additive noise.
        
        Parameters
        ----------
        estimator_func : callable
            Function that estimates the parameter of interest
        data : np.ndarray
            Input data for estimation
        noise_levels : List[float]
            List of noise levels to test (as fraction of data std)
        **kwargs : dict
            Additional arguments for the estimator function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing robustness test results
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
            
        data_std = np.std(data)
        results = {
            'noise_levels': noise_levels,
            'estimates': [],
            'data_std': data_std,
            'robustness_scores': []
        }
        
        # Get baseline estimate
        try:
            baseline_result = estimator_func(data, **kwargs)
            if isinstance(baseline_result, dict) and 'hurst_exponent' in baseline_result:
                baseline_estimate = baseline_result['hurst_exponent']
            elif isinstance(baseline_result, (int, float)):
                baseline_estimate = baseline_result
            else:
                baseline_estimate = np.nan
        except Exception as e:
            logger.warning(f"Baseline estimation failed: {e}")
            baseline_estimate = np.nan
            
        results['baseline_estimate'] = baseline_estimate
        
        for noise_level in noise_levels:
            try:
                # Add noise to data
                noise = np.random.normal(0, noise_level * data_std, size=data.shape)
                noisy_data = data + noise
                
                # Estimate with noisy data
                noisy_result = estimator_func(noisy_data, **kwargs)
                if isinstance(noisy_result, dict) and 'hurst_exponent' in noisy_result:
                    noisy_estimate = noisy_result['hurst_exponent']
                elif isinstance(noisy_result, (int, float)):
                    noisy_estimate = noisy_result
                else:
                    noisy_estimate = np.nan
                    
                results['estimates'].append(noisy_estimate)
                
                # Calculate robustness score (relative change)
                if not np.isnan(baseline_estimate) and not np.isnan(noisy_estimate):
                    relative_change = abs(noisy_estimate - baseline_estimate) / abs(baseline_estimate)
                    robustness_score = 1.0 / (1.0 + relative_change)  # Higher is better
                else:
                    robustness_score = 0.0
                    
                results['robustness_scores'].append(robustness_score)
                
            except Exception as e:
                logger.warning(f"Noise robustness test failed for level {noise_level}: {e}")
                results['estimates'].append(np.nan)
                results['robustness_scores'].append(0.0)
                
        # Calculate overall robustness
        valid_scores = [s for s in results['robustness_scores'] if s > 0]
        if valid_scores:
            results['overall_robustness'] = np.mean(valid_scores)
        else:
            results['overall_robustness'] = 0.0
            
        return results
        
    def test_outlier_robustness(self, estimator_func, data: np.ndarray,
                              outlier_fractions: List[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Test robustness to outliers.
        
        Parameters
        ----------
        estimator_func : callable
            Function that estimates the parameter of interest
        data : np.ndarray
            Input data for estimation
        outlier_fractions : List[float]
            List of outlier fractions to test
        **kwargs : dict
            Additional arguments for the estimator function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing outlier robustness test results
        """
        if outlier_fractions is None:
            outlier_fractions = [0.0, 0.01, 0.05, 0.1, 0.2]
            
        results = {
            'outlier_fractions': outlier_fractions,
            'estimates': [],
            'robustness_scores': []
        }
        
        # Get baseline estimate
        try:
            baseline_result = estimator_func(data, **kwargs)
            if isinstance(baseline_result, dict) and 'hurst_exponent' in baseline_result:
                baseline_estimate = baseline_result['hurst_exponent']
            elif isinstance(baseline_result, (int, float)):
                baseline_estimate = baseline_result
            else:
                baseline_estimate = np.nan
        except Exception as e:
            logger.warning(f"Baseline estimation failed: {e}")
            baseline_estimate = np.nan
            
        results['baseline_estimate'] = baseline_estimate
        
        for outlier_fraction in outlier_fractions:
            try:
                # Create data with outliers
                outlier_data = data.copy()
                n_outliers = int(len(data) * outlier_fraction)
                
                if n_outliers > 0:
                    # Replace random points with extreme values
                    outlier_indices = np.random.choice(len(data), size=n_outliers, replace=False)
                    outlier_values = np.random.choice([-10, 10], size=n_outliers) * np.std(data)
                    outlier_data[outlier_indices] = outlier_values
                
                # Estimate with outlier data
                outlier_result = estimator_func(outlier_data, **kwargs)
                if isinstance(outlier_result, dict) and 'hurst_exponent' in outlier_result:
                    outlier_estimate = outlier_result['hurst_exponent']
                elif isinstance(outlier_result, (int, float)):
                    outlier_estimate = outlier_result
                else:
                    outlier_estimate = np.nan
                    
                results['estimates'].append(outlier_estimate)
                
                # Calculate robustness score
                if not np.isnan(baseline_estimate) and not np.isnan(outlier_estimate):
                    relative_change = abs(outlier_estimate - baseline_estimate) / abs(baseline_estimate)
                    robustness_score = 1.0 / (1.0 + relative_change)
                else:
                    robustness_score = 0.0
                    
                results['robustness_scores'].append(robustness_score)
                
            except Exception as e:
                logger.warning(f"Outlier robustness test failed for fraction {outlier_fraction}: {e}")
                results['estimates'].append(np.nan)
                results['robustness_scores'].append(0.0)
                
        # Calculate overall robustness
        valid_scores = [s for s in results['robustness_scores'] if s > 0]
        if valid_scores:
            results['overall_robustness'] = np.mean(valid_scores)
        else:
            results['overall_robustness'] = 0.0
            
        return results
        
    def test_parameter_robustness(self, estimator_func, data: np.ndarray,
                                parameter_name: str, parameter_values: List[Any], **kwargs) -> Dict[str, Any]:
        """
        Test robustness to parameter changes.
        
        Parameters
        ----------
        estimator_func : callable
            Function that estimates the parameter of interest
        data : np.ndarray
            Input data for estimation
        parameter_name : str
            Name of the parameter to vary
        parameter_values : List[Any]
            List of parameter values to test
        **kwargs : dict
            Additional arguments for the estimator function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameter robustness test results
        """
        results = {
            'parameter_name': parameter_name,
            'parameter_values': parameter_values,
            'estimates': [],
            'robustness_scores': []
        }
        
        # Get baseline estimate with default parameters
        try:
            baseline_result = estimator_func(data, **kwargs)
            if isinstance(baseline_result, dict) and 'hurst_exponent' in baseline_result:
                baseline_estimate = baseline_result['hurst_exponent']
            elif isinstance(baseline_result, (int, float)):
                baseline_estimate = baseline_result
            else:
                baseline_estimate = np.nan
        except Exception as e:
            logger.warning(f"Baseline estimation failed: {e}")
            baseline_estimate = np.nan
            
        results['baseline_estimate'] = baseline_estimate
        
        for param_value in parameter_values:
            try:
                # Create parameter dict with the varying parameter
                test_params = kwargs.copy()
                test_params[parameter_name] = param_value
                
                # Estimate with modified parameter
                param_result = estimator_func(data, **test_params)
                if isinstance(param_result, dict) and 'hurst_exponent' in param_result:
                    param_estimate = param_result['hurst_exponent']
                elif isinstance(param_result, (int, float)):
                    param_estimate = param_result
                else:
                    param_estimate = np.nan
                    
                results['estimates'].append(param_estimate)
                
                # Calculate robustness score
                if not np.isnan(baseline_estimate) and not np.isnan(param_estimate):
                    relative_change = abs(param_estimate - baseline_estimate) / abs(baseline_estimate)
                    robustness_score = 1.0 / (1.0 + relative_change)
                else:
                    robustness_score = 0.0
                    
                results['robustness_scores'].append(robustness_score)
                
            except Exception as e:
                logger.warning(f"Parameter robustness test failed for {parameter_name}={param_value}: {e}")
                results['estimates'].append(np.nan)
                results['robustness_scores'].append(0.0)
                
        # Calculate overall robustness
        valid_scores = [s for s in results['robustness_scores'] if s > 0]
        if valid_scores:
            results['overall_robustness'] = np.mean(valid_scores)
        else:
            results['overall_robustness'] = 0.0
            
        return results
