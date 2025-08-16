"""
Bootstrap Validation for Long-Range Dependence

This module provides bootstrap-based validation methods for
long-range dependence estimates.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class BootstrapValidator:
    """
    Bootstrap validation framework for long-range dependence estimates.
    
    This class provides methods for estimating confidence intervals
    and standard errors using bootstrap resampling.
    """
    
    def __init__(self, n_bootstrap: int = 1000, random_state: Optional[int] = None):
        """
        Initialize the bootstrap validator.
        
        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap samples (default: 1000)
        random_state : Optional[int]
            Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            
    def bootstrap_confidence_interval(self, estimator_func, data: np.ndarray, 
                                   confidence_level: float = 0.95, **kwargs) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence interval for an estimator.
        
        Parameters
        ----------
        estimator_func : callable
            Function that estimates the parameter of interest
        data : np.ndarray
            Input data for estimation
        confidence_level : float
            Confidence level for the interval (default: 0.95)
        **kwargs : dict
            Additional arguments for the estimator function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing bootstrap results
        """
        bootstrap_estimates = []
        
        for _ in range(self.n_bootstrap):
            # Generate bootstrap sample
            bootstrap_indices = np.random.choice(len(data), size=len(data), replace=True)
            bootstrap_data = data[bootstrap_indices]
            
            try:
                # Apply estimator to bootstrap sample
                result = estimator_func(bootstrap_data, **kwargs)
                if isinstance(result, dict) and 'hurst_exponent' in result:
                    bootstrap_estimates.append(result['hurst_exponent'])
                elif isinstance(result, (int, float)):
                    bootstrap_estimates.append(result)
            except Exception as e:
                logger.warning(f"Bootstrap iteration failed: {e}")
                continue
                
        if not bootstrap_estimates:
            return {
                'confidence_interval': (np.nan, np.nan),
                'standard_error': np.nan,
                'bootstrap_estimates': [],
                'success_rate': 0.0
            }
            
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
        ci_upper = np.percentile(bootstrap_estimates, upper_percentile)
        
        # Calculate standard error
        standard_error = np.std(bootstrap_estimates, ddof=1)
        
        # Calculate success rate
        success_rate = len(bootstrap_estimates) / self.n_bootstrap
        
        return {
            'confidence_interval': (ci_lower, ci_upper),
            'standard_error': standard_error,
            'bootstrap_estimates': bootstrap_estimates.tolist(),
            'success_rate': success_rate,
            'confidence_level': confidence_level,
            'n_bootstrap': self.n_bootstrap
        }
        
    def bootstrap_hypothesis_test(self, estimator_func, data: np.ndarray,
                                null_hypothesis: float = 0.5, **kwargs) -> Dict[str, Any]:
        """
        Perform bootstrap hypothesis test.
        
        Parameters
        ----------
        estimator_func : callable
            Function that estimates the parameter of interest
        data : np.ndarray
            Input data for estimation
        null_hypothesis : float
            Value under the null hypothesis (default: 0.5 for no LRD)
        **kwargs : dict
            Additional arguments for the estimator function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing test results
        """
        # Get original estimate
        try:
            original_result = estimator_func(data, **kwargs)
            if isinstance(original_result, dict) and 'hurst_exponent' in original_result:
                original_estimate = original_result['hurst_exponent']
            elif isinstance(original_result, (int, float)):
                original_estimate = original_result
            else:
                raise ValueError("Estimator must return a numeric value or dict with 'hurst_exponent'")
        except Exception as e:
            logger.error(f"Failed to get original estimate: {e}")
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'reject_null': False,
                'original_estimate': np.nan,
                'null_hypothesis': null_hypothesis
            }
            
        # Perform bootstrap test
        bootstrap_results = self.bootstrap_confidence_interval(estimator_func, data, **kwargs)
        
        if bootstrap_results['success_rate'] == 0:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'reject_null': False,
                'original_estimate': original_estimate,
                'null_hypothesis': null_hypothesis
            }
            
        # Calculate p-value based on bootstrap distribution
        bootstrap_estimates = np.array(bootstrap_results['bootstrap_estimates'])
        
        # Two-tailed test: count how many bootstrap estimates are as extreme as the original
        # under the null hypothesis
        test_statistic = abs(original_estimate - null_hypothesis)
        extreme_count = np.sum(np.abs(bootstrap_estimates - null_hypothesis) >= test_statistic)
        p_value = extreme_count / len(bootstrap_estimates)
        
        # Decision (using alpha = 0.05)
        alpha = 0.05
        reject_null = p_value < alpha
        
        return {
            'test_statistic': test_statistic,
            'p_value': p_value,
            'reject_null': reject_null,
            'original_estimate': original_estimate,
            'null_hypothesis': null_hypothesis,
            'alpha': alpha,
            'bootstrap_estimates': bootstrap_results['bootstrap_estimates'],
            'confidence_interval': bootstrap_results['confidence_interval'],
            'standard_error': bootstrap_results['standard_error']
        }
