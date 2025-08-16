"""
Hypothesis Testing for Long-Range Dependence

This module provides tools for testing hypotheses about long-range
dependence parameters and validating estimation results.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class HypothesisTester:
    """
    Hypothesis testing framework for long-range dependence estimates.
    
    This class provides methods for testing various hypotheses about
    long-range dependence parameters and their statistical significance.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the hypothesis tester.
        
        Parameters
        ----------
        alpha : float
            Significance level for hypothesis tests (default: 0.05)
        """
        self.alpha = alpha
        self.test_results = {}
    
    def test_no_lrd(self, hurst_estimate: float, std_error: float, 
                    sample_size: int) -> Dict[str, Any]:
        """
        Test the null hypothesis of no long-range dependence (H = 0.5).
        
        Parameters
        ----------
        hurst_estimate : float
            Estimated Hurst exponent
        std_error : float
            Standard error of the estimate
        sample_size : int
            Sample size used for estimation
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing test results
        """
        # Null hypothesis: H = 0.5 (no long-range dependence)
        # Alternative hypothesis: H ≠ 0.5 (long-range dependence exists)
        
        null_hypothesis = 0.5
        test_statistic = (hurst_estimate - null_hypothesis) / std_error
        
        # Two-tailed test
        p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
        
        # Calculate confidence interval
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = hurst_estimate - z_critical * std_error
        ci_upper = hurst_estimate + z_critical * std_error
        
        # Decision
        reject_null = p_value < self.alpha
        
        results = {
            'test_type': 'No LRD Test (H = 0.5)',
            'null_hypothesis': f'H = {null_hypothesis}',
            'alternative_hypothesis': 'H ≠ 0.5',
            'test_statistic': test_statistic,
            'p_value': p_value,
            'significance_level': self.alpha,
            'reject_null': reject_null,
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': 1 - self.alpha,
            'sample_size': sample_size,
            'estimated_hurst': hurst_estimate,
            'standard_error': std_error
        }
        
        self.test_results['no_lrd_test'] = results
        return results
    
    def test_lrd_strength(self, hurst_estimate: float, std_error: float,
                          threshold: float = 0.6) -> Dict[str, Any]:
        """
        Test whether long-range dependence is strong (H > threshold).
        
        Parameters
        ----------
        hurst_estimate : float
            Estimated Hurst exponent
        std_error : float
            Standard error of the estimate
        threshold : float
            Threshold for strong LRD (default: 0.6)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing test results
        """
        # Null hypothesis: H ≤ threshold (weak or no LRD)
        # Alternative hypothesis: H > threshold (strong LRD)
        
        test_statistic = (hurst_estimate - threshold) / std_error
        
        # One-tailed test (right-tailed)
        p_value = 1 - stats.norm.cdf(test_statistic)
        
        # Calculate confidence interval
        z_critical = stats.norm.ppf(1 - self.alpha)
        ci_lower = hurst_estimate - z_critical * std_error
        ci_upper = hurst_estimate + z_critical * std_error
        
        # Decision
        reject_null = p_value < self.alpha
        
        results = {
            'test_type': f'Strong LRD Test (H > {threshold})',
            'null_hypothesis': f'H ≤ {threshold}',
            'alternative_hypothesis': f'H > {threshold}',
            'test_statistic': test_statistic,
            'p_value': p_value,
            'significance_level': self.alpha,
            'reject_null': reject_null,
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': 1 - self.alpha,
            'threshold': threshold,
            'estimated_hurst': hurst_estimate,
            'standard_error': std_error
        }
        
        self.test_results['strong_lrd_test'] = results
        return results
    
    def test_parameter_equality(self, estimate1: float, std_error1: float,
                               estimate2: float, std_error2: float,
                               sample_size1: int, sample_size2: int) -> Dict[str, Any]:
        """
        Test whether two long-range dependence estimates are equal.
        
        Parameters
        ----------
        estimate1, estimate2 : float
            Two parameter estimates to compare
        std_error1, std_error2 : float
            Standard errors of the estimates
        sample_size1, sample_size2 : int
            Sample sizes used for estimation
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing test results
        """
        # Null hypothesis: θ1 = θ2
        # Alternative hypothesis: θ1 ≠ θ2
        
        # Pooled standard error
        pooled_std_error = np.sqrt(std_error1**2 + std_error2**2)
        
        test_statistic = (estimate1 - estimate2) / pooled_std_error
        
        # Two-tailed test
        p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
        
        # Decision
        reject_null = p_value < self.alpha
        
        results = {
            'test_type': 'Parameter Equality Test',
            'null_hypothesis': 'θ1 = θ2',
            'alternative_hypothesis': 'θ1 ≠ θ2',
            'test_statistic': test_statistic,
            'p_value': p_value,
            'significance_level': self.alpha,
            'reject_null': reject_null,
            'estimate1': estimate1,
            'estimate2': estimate2,
            'std_error1': std_error1,
            'std_error2': std_error2,
            'sample_size1': sample_size1,
            'sample_size2': sample_size2,
            'difference': estimate1 - estimate2,
            'pooled_std_error': pooled_std_error
        }
        
        self.test_results['equality_test'] = results
        return results
    
    def test_trend_significance(self, trend_estimate: float, std_error: float,
                               sample_size: int) -> Dict[str, Any]:
        """
        Test the significance of a trend in long-range dependence.
        
        Parameters
        ----------
        trend_estimate : float
            Estimated trend coefficient
        std_error : float
            Standard error of the trend estimate
        sample_size : int
            Sample size used for estimation
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing test results
        """
        # Null hypothesis: trend = 0 (no trend)
        # Alternative hypothesis: trend ≠ 0 (trend exists)
        
        test_statistic = trend_estimate / std_error
        
        # Two-tailed test
        p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
        
        # Decision
        reject_null = p_value < self.alpha
        
        results = {
            'test_type': 'Trend Significance Test',
            'null_hypothesis': 'trend = 0',
            'alternative_hypothesis': 'trend ≠ 0',
            'test_statistic': test_statistic,
            'p_value': p_value,
            'significance_level': self.alpha,
            'reject_null': reject_null,
            'trend_estimate': trend_estimate,
            'standard_error': std_error,
            'sample_size': sample_size
        }
        
        self.test_results['trend_test'] = results
        return results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all performed tests.
        
        Returns
        -------
        Dict[str, Any]
            Summary of test results
        """
        if not self.test_results:
            return {'message': 'No tests performed yet'}
        
        summary = {
            'total_tests': len(self.test_results),
            'tests_performed': list(self.test_results.keys()),
            'rejected_tests': [],
            'accepted_tests': [],
            'overall_results': {}
        }
        
        for test_name, results in self.test_results.items():
            if results.get('reject_null', False):
                summary['rejected_tests'].append(test_name)
            else:
                summary['accepted_tests'].append(test_name)
            
            summary['overall_results'][test_name] = {
                'p_value': results.get('p_value'),
                'reject_null': results.get('reject_null'),
                'test_statistic': results.get('test_statistic')
            }
        
        return summary
    
    def reset(self):
        """Reset all test results."""
        self.test_results = {}
