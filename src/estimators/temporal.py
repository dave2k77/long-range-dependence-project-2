"""
Temporal Long-Range Dependence Estimators

This module contains implementations of temporal methods for estimating
long-range dependence, including DFA, MFDFA, R/S, and Higuchi methods.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.optimize import curve_fit
import logging
import time

from .base import BaseEstimator

logger = logging.getLogger(__name__)


class DFAEstimator(BaseEstimator):
    """
    Detrended Fluctuation Analysis (DFA) estimator.
    
    DFA is a method for determining the statistical self-affinity of a signal.
    It is useful for analyzing time series that appear to be long-memory processes.
    """
    
    def __init__(self, name: str = "DFA", **kwargs):
        """
        Initialize DFA estimator.
        
        Parameters
        ----------
        name : str
            Name identifier for the estimator
        **kwargs
            Additional parameters including:
            - min_scale: Minimum scale for analysis (default: 4)
            - max_scale: Maximum scale for analysis (default: len(data)//4)
            - num_scales: Number of scales to analyze (default: 20)
            - polynomial_order: Order of polynomial for detrending (default: 1)
        """
        super().__init__(name=name, **kwargs)
        self.min_scale = kwargs.get('min_scale', 4)
        self.max_scale = kwargs.get('max_scale', None)
        self.num_scales = kwargs.get('num_scales', 20)
        self.polynomial_order = kwargs.get('polynomial_order', 1)
        
        # Validate parameters
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive")
        if self.polynomial_order <= 0:
            raise ValueError("polynomial_order must be positive")
        if self.num_scales <= 0:
            raise ValueError("num_scales must be positive")
        
        self.data = None
        self.scales = None
        self.fluctuations = None
        
    def fit(self, data: np.ndarray, **kwargs) -> 'DFAEstimator':
        """
        Fit the DFA estimator to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        self : DFAEstimator
            Fitted estimator instance
        """
        self.data = np.asarray(data, dtype=float)
        self._validate_data()
        self._generate_scales()
        return self
    
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Estimate long-range dependence using DFA.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Input time series data (if not provided, uses fitted data)
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing DFA estimation results
        """
        start_time = time.time()
        
        if data is not None:
            self.fit(data, **kwargs)
        
        if self.data is None:
            raise ValueError("No data provided. Call fit() first.")
        
        # Generate scales
        self._generate_scales()
        
        # Calculate fluctuations for each scale
        self.fluctuations = self._calculate_fluctuations()
        
        # Fit power law to get Hurst exponent
        hurst_exponent, intercept, r_squared = self._fit_power_law()
        
        # Calculate alpha (long-range dependence parameter)
        alpha = 2 * hurst_exponent - 1
        
        results = {
            'hurst_exponent': hurst_exponent,
            'alpha': alpha,
            'r_squared': r_squared,
            'intercept': intercept,
            'scales': self.scales,
            'fluctuations': self.fluctuations,
            'method': 'DFA',
            'execution_time': self.execution_time
        }
        
        # Record execution time
        self.execution_time = time.time() - start_time
        
        # Update results with execution time
        results['execution_time'] = self.execution_time
        
        # Store results
        self.results.update(results)
        
        return results
    
    def _validate_data(self):
        """Validate input data."""
        if len(self.data) == 0:
            raise ValueError("Data cannot be empty")
        
        if len(self.data) < 100:
            raise ValueError("Data must have at least 100 points for reliable DFA estimation")
        
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        
        # Check if data is all zeros (which would cause issues)
        if np.all(self.data == 0):
            raise ValueError("Data cannot be all zeros")
    
    def _generate_scales(self):
        """Generate scales for analysis."""
        if self.max_scale is None:
            self.max_scale = len(self.data) // 4
        
        # Generate scales logarithmically spaced
        self.scales = np.logspace(
            np.log10(self.min_scale),
            np.log10(self.max_scale),
            self.num_scales,
            dtype=int
        )
        
        # Ensure unique scales
        self.scales = np.unique(self.scales)
    
    def _calculate_fluctuations(self) -> np.ndarray:
        """Calculate fluctuations for each scale."""
        fluctuations = []
        
        for scale in self.scales:
            # Divide data into segments
            num_segments = len(self.data) // scale
            if num_segments == 0:
                continue
                
            segment_fluctuations = []
            
            for i in range(num_segments):
                start_idx = i * scale
                end_idx = start_idx + scale
                segment = self.data[start_idx:end_idx]
                
                # Detrend segment
                x = np.arange(scale)
                coeffs = np.polyfit(x, segment, self.polynomial_order)
                trend = np.polyval(coeffs, x)
                detrended = segment - trend
                
                # Calculate RMS fluctuation
                rms = np.sqrt(np.mean(detrended**2))
                segment_fluctuations.append(rms)
            
            if segment_fluctuations:
                fluctuations.append(np.mean(segment_fluctuations))
        
        result = np.array(fluctuations)
        self.fluctuations = result
        return result
    
    def _fit_power_law(self) -> Tuple[float, float, float]:
        """Fit power law to fluctuations vs scales."""
        if len(self.scales) != len(self.fluctuations):
            # Filter out scales where fluctuations couldn't be calculated
            valid_indices = np.arange(len(self.scales))[:len(self.fluctuations)]
            scales = self.scales[valid_indices]
        else:
            scales = self.scales
        
        # Log-log relationship: log(F) = H * log(s) + C
        log_scales = np.log(scales)
        log_fluctuations = np.log(self.fluctuations)
        
        # Linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_scales, log_fluctuations
        )
        
        hurst_exponent = slope
        r_squared = r_value**2
        
        return hurst_exponent, intercept, r_squared


class MFDFAEstimator(BaseEstimator):
    """
    Multifractal Detrended Fluctuation Analysis (MFDFA) estimator.
    
    MFDFA extends DFA to analyze multifractal properties by computing
    fluctuation functions for different moments q.
    """
    
    def __init__(self, name: str = "MFDFA", **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_scale = kwargs.get('min_scale', 4)
        self.max_scale = kwargs.get('max_scale', None)
        self.num_scales = kwargs.get('num_scales', 20)
        self.polynomial_order = kwargs.get('polynomial_order', 1)
        self.q_values = kwargs.get('q_values', np.arange(-5, 6, 0.5))
        self.data = None
        self.scales = None
        self.fluctuations = None
        self.hurst_exponents = None
        self.multifractal_spectrum = None
        
    def fit(self, data: np.ndarray, **kwargs) -> 'MFDFAEstimator':
        """Fit the MFDFA estimator to the data."""
        self.data = np.asarray(data, dtype=float)
        self._validate_data()
        self._generate_scales()
        return self
        
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """Estimate multifractal properties using MFDFA."""
        import time
        start_time = time.time()
        
        if data is not None:
            self.fit(data, **kwargs)
            
        if self.data is None:
            raise ValueError("No data available. Call fit() first.")
            
        # Calculate fluctuation functions for all q values
        self._calculate_fluctuations()
        
        # Fit scaling laws for each q value
        self._fit_scaling_laws()
        
        # Calculate multifractal spectrum
        self._calculate_multifractal_spectrum()
        
        # Record execution time
        self.execution_time = time.time() - start_time
        
        return self.get_results()
        
    def _validate_data(self):
        """Validate input data."""
        if self.data is None:
            raise ValueError("No data provided")
        if len(self.data) < 100:
            raise ValueError("Data must have at least 100 points for reliable MFDFA analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        # Check if data is constant (which would cause issues)
        if np.std(self.data) == 0:
            raise ValueError("Data is constant, cannot estimate LRD")
            
    def _generate_scales(self):
        """Generate scale values for analysis."""
        if self.max_scale is None:
            self.max_scale = len(self.data) // 4
            
        self.scales = np.logspace(
            np.log10(self.min_scale), 
            np.log10(self.max_scale), 
            self.num_scales, 
            dtype=int
        )
        # Ensure unique scales
        self.scales = np.unique(self.scales)
        
    def _calculate_fluctuations(self):
        """Calculate fluctuation functions for all q values and scales."""
        n_scales = len(self.scales)
        n_q = len(self.q_values)
        
        self.fluctuations = np.zeros((n_scales, n_q))
        
        for i, scale in enumerate(self.scales):
            for j, q in enumerate(self.q_values):
                self.fluctuations[i, j] = self._calculate_fluctuation_at_scale(scale, q)
                
    def _calculate_fluctuation_at_scale(self, scale: int, q: float) -> float:
        """Calculate fluctuation function for a specific scale and q value."""
        n_segments = len(self.data) // scale
        if n_segments == 0:
            return np.nan
            
        fluctuations = []
        
        for i in range(n_segments):
            start_idx = i * scale
            end_idx = start_idx + scale
            segment = self.data[start_idx:end_idx]
            
            # Detrend the segment
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, self.polynomial_order)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend
            
            # Calculate variance
            variance = np.var(detrended)
            if variance > 0:
                fluctuations.append(variance)
                
        if not fluctuations:
            return np.nan
            
        # Calculate q-th order fluctuation
        if abs(q) < 1e-10:  # q ≈ 0
            fq = np.exp(np.mean(np.log(fluctuations)))
        else:
            fq = np.mean(np.array(fluctuations) ** (q/2))
            
        return fq ** (1/q) if q != 0 else fq
        
    def _fit_scaling_laws(self):
        """Fit scaling laws for each q value to extract Hurst exponents."""
        self.hurst_exponents = np.zeros(len(self.q_values))
        self.scaling_errors = np.zeros(len(self.q_values))
        
        for i, q in enumerate(self.q_values):
            # Get valid fluctuations for this q
            valid_mask = ~np.isnan(self.fluctuations[:, i])
            if np.sum(valid_mask) < 3:
                self.hurst_exponents[i] = np.nan
                self.scaling_errors[i] = np.nan
                continue
                
            scales_valid = self.scales[valid_mask]
            fluct_valid = self.fluctuations[valid_mask, i]
            
            # Log-log regression
            log_scales = np.log(scales_valid)
            log_fluct = np.log(fluct_valid)
            
            try:
                coeffs = np.polyfit(log_scales, log_fluct, 1)
                self.hurst_exponents[i] = coeffs[0]
                
                # Calculate R-squared
                y_pred = np.polyval(coeffs, log_scales)
                ss_res = np.sum((log_fluct - y_pred) ** 2)
                ss_tot = np.sum((log_fluct - np.mean(log_fluct)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                self.scaling_errors[i] = 1 - r_squared
                
            except (np.linalg.LinAlgError, ValueError):
                self.hurst_exponents[i] = np.nan
                self.scaling_errors[i] = np.nan
                
    def _calculate_multifractal_spectrum(self):
        """Calculate the multifractal spectrum f(alpha)."""
        # Find valid Hurst exponents
        valid_mask = ~np.isnan(self.hurst_exponents)
        if np.sum(valid_mask) < 3:
            self.multifractal_spectrum = None
            return
            
        q_valid = self.q_values[valid_mask]
        h_valid = self.hurst_exponents[valid_mask]
        
        # Calculate alpha (singularity strength) and f(alpha)
        alpha = h_valid + q_valid * np.gradient(h_valid, q_valid)
        f_alpha = q_valid * alpha - h_valid
        
        self.multifractal_spectrum = {
            'alpha': alpha,
            'f_alpha': f_alpha,
            'q_values': q_valid,
            'hurst_exponents': h_valid
        }
        
    def get_results(self) -> Dict[str, Any]:
        """Get estimation results."""
        results = {
            'hurst_exponents': self.hurst_exponents,
            'q_values': self.q_values,
            'scales': self.scales,
            'fluctuations': self.fluctuations,
            'scaling_errors': getattr(self, 'scaling_errors', None),
            'multifractal_spectrum': self.multifractal_spectrum,
            'parameters': {
                'min_scale': self.min_scale,
                'max_scale': self.max_scale,
                'num_scales': self.num_scales,
                'polynomial_order': self.polynomial_order
            }
        }
        
        # Add summary statistics
        if self.hurst_exponents is not None:
            valid_h = self.hurst_exponents[~np.isnan(self.hurst_exponents)]
            if len(valid_h) > 0:
                results['summary'] = {
                    'mean_hurst': np.mean(valid_h),
                    'std_hurst': np.std(valid_h),
                    'min_hurst': np.min(valid_h),
                    'max_hurst': np.max(valid_h),
                    'is_multifractal': np.std(valid_h) > 0.05  # Threshold for multifractality
                }
                
        return results


class RSEstimator(BaseEstimator):
    """
    Rescaled Range (R/S) Analysis estimator.
    
    R/S analysis estimates the Hurst exponent by analyzing the scaling
    behavior of the rescaled range statistic across different time scales.
    """
    
    def __init__(self, name: str = "R/S", **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_scale = kwargs.get('min_scale', 4)
        self.max_scale = kwargs.get('max_scale', None)
        self.num_scales = kwargs.get('num_scales', 20)
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.n_bootstrap = kwargs.get('n_bootstrap', 1000)
        self.data = None
        self.scales = None
        self.rs_values = None
        self.hurst_exponent = None
        self.confidence_interval = None
        
    def fit(self, data: np.ndarray, **kwargs) -> 'RSEstimator':
        """Fit the R/S estimator to the data."""
        self.data = np.asarray(data, dtype=float)
        self._validate_data()
        self._generate_scales()
        return self
        
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """Estimate Hurst exponent using R/S analysis."""
        import time
        start_time = time.time()
        
        if data is not None:
            self.fit(data, **kwargs)
            
        if self.data is None:
            raise ValueError("No data available. Call fit() first.")
            
        # Calculate R/S values for all scales
        self._calculate_rs_values()
        
        # Fit scaling law to extract Hurst exponent
        self._fit_scaling_law()
        
        # Calculate confidence interval
        self._calculate_confidence_interval()
        
        # Record execution time
        self.execution_time = time.time() - start_time
        
        return self.get_results()
        
    def _validate_data(self):
        """Validate input data."""
        if self.data is None:
            raise ValueError("No data provided")
        if len(self.data) < 100:
            raise ValueError("Data must have at least 100 points for reliable R/S analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        # Check if data is constant (which would cause issues)
        if np.std(self.data) == 0:
            raise ValueError("Data is constant, cannot estimate LRD")
            
    def _generate_scales(self):
        """Generate scale values for analysis."""
        if self.max_scale is None:
            self.max_scale = len(self.data) // 4
            
        self.scales = np.logspace(
            np.log10(self.min_scale), 
            np.log10(self.max_scale), 
            self.num_scales, 
            dtype=int
        )
        # Ensure unique scales
        self.scales = np.unique(self.scales)
        
    def _calculate_rs_values(self):
        """Calculate R/S values for all scales."""
        self.rs_values = np.zeros(len(self.scales))
        self.rs_std = np.zeros(len(self.scales))
        
        for i, scale in enumerate(self.scales):
            rs_list = []
            
            # Calculate R/S for all possible segments of this scale
            n_segments = len(self.data) // scale
            if n_segments == 0:
                self.rs_values[i] = np.nan
                self.rs_std[i] = np.nan
                continue
                
            for j in range(n_segments):
                start_idx = j * scale
                end_idx = start_idx + scale
                segment = self.data[start_idx:end_idx]
                
                rs = self._calculate_rs_for_segment(segment)
                if rs is not None:
                    rs_list.append(rs)
                    
            if rs_list:
                self.rs_values[i] = np.mean(rs_list)
                self.rs_std[i] = np.std(rs_list)
            else:
                self.rs_values[i] = np.nan
                self.rs_std[i] = np.nan
                
    def _calculate_rs_for_segment(self, segment: np.ndarray) -> Optional[float]:
        """Calculate R/S value for a single segment."""
        if len(segment) < 2:
            return None
            
        # Calculate cumulative sum
        cumsum = np.cumsum(segment - np.mean(segment))
        
        # Calculate range R
        R = np.max(cumsum) - np.min(cumsum)
        
        # Calculate standard deviation S
        S = np.std(segment)
        
        # Avoid division by zero
        if S < 1e-10:
            return None
            
        return R / S
        
    def _fit_scaling_law(self):
        """Fit scaling law to extract Hurst exponent."""
        # Get valid R/S values
        valid_mask = ~np.isnan(self.rs_values)
        if np.sum(valid_mask) < 3:
            self.hurst_exponent = np.nan
            self.scaling_error = np.nan
            return
            
        scales_valid = self.scales[valid_mask]
        rs_valid = self.rs_values[valid_mask]
        
        # Log-log regression: log(R/S) = H * log(scale) + constant
        log_scales = np.log(scales_valid)
        log_rs = np.log(rs_valid)
        
        try:
            coeffs = np.polyfit(log_scales, log_rs, 1)
            self.hurst_exponent = coeffs[0]
            
            # Calculate R-squared
            y_pred = np.polyval(coeffs, log_scales)
            ss_res = np.sum((log_rs - y_pred) ** 2)
            ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.scaling_error = 1 - r_squared
            
        except (np.linalg.LinAlgError, ValueError):
            self.hurst_exponent = np.nan
            self.scaling_error = np.nan
            
    def _calculate_confidence_interval(self):
        """Calculate confidence interval for the Hurst exponent."""
        if np.isnan(self.hurst_exponent):
            self.confidence_interval = (np.nan, np.nan)
            return
            
        # Get valid R/S values for error estimation
        valid_mask = ~np.isnan(self.rs_values)
        if np.sum(valid_mask) < 3:
            self.confidence_interval = (np.nan, np.nan)
            return
            
        scales_valid = self.scales[valid_mask]
        rs_valid = self.rs_values[valid_mask]
        rs_std_valid = self.rs_std[valid_mask]
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        hurst_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Add noise to R/S values based on their standard deviations
            rs_noisy = rs_valid + np.random.normal(0, rs_std_valid)
            rs_noisy = np.maximum(rs_noisy, 0.1)  # Ensure positive values
            
            try:
                log_scales = np.log(scales_valid)
                log_rs = np.log(rs_noisy)
                coeffs = np.polyfit(log_scales, log_rs, 1)
                hurst_bootstrap.append(coeffs[0])
            except:
                continue
                
        if hurst_bootstrap:
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            self.confidence_interval = (
                np.percentile(hurst_bootstrap, lower_percentile),
                np.percentile(hurst_bootstrap, upper_percentile)
            )
        else:
            self.confidence_interval = (np.nan, np.nan)
            
    def get_results(self) -> Dict[str, Any]:
        """Get estimation results."""
        results = {
            'hurst_exponent': self.hurst_exponent,
            'scales': self.scales,
            'rs_values': self.rs_values,
            'rs_std': getattr(self, 'rs_std', None),
            'scaling_error': getattr(self, 'scaling_error', None),
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'parameters': {
                'min_scale': self.min_scale,
                'max_scale': self.max_scale,
                'num_scales': self.num_scales
            }
        }
        
        # Add interpretation
        if not np.isnan(self.hurst_exponent):
            if self.hurst_exponent < 0.5:
                lrd_type = "Anti-persistent (short-range dependent)"
            elif self.hurst_exponent > 0.5:
                lrd_type = "Persistent (long-range dependent)"
            else:
                lrd_type = "Random walk (no long-range dependence)"
                
            results['interpretation'] = {
                'lrd_type': lrd_type,
                'strength': abs(self.hurst_exponent - 0.5),
                'reliability': 1 - getattr(self, 'scaling_error', 1.0)
            }
            
        return results


class HiguchiEstimator(BaseEstimator):
    """
    Higuchi method estimator for fractal dimension.
    
    The Higuchi method estimates the fractal dimension of time series
    by analyzing the length of curves at different scales.
    """
    
    def __init__(self, name: str = "Higuchi", **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_k = kwargs.get('min_k', 2)
        self.max_k = kwargs.get('max_k', None)
        self.num_k = kwargs.get('num_k', 20)
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.n_bootstrap = kwargs.get('n_bootstrap', 1000)
        self.data = None
        self.k_values = None
        self.lengths = None
        self.fractal_dimension = None
        self.confidence_interval = None
        
    def fit(self, data: np.ndarray, **kwargs) -> 'HiguchiEstimator':
        """Fit the Higuchi estimator to the data."""
        self.data = np.asarray(data, dtype=float)
        self._validate_data()
        self._generate_k_values()
        return self
        
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """Estimate fractal dimension using Higuchi method."""
        import time
        start_time = time.time()
        
        if data is not None:
            self.fit(data, **kwargs)
            
        if self.data is None:
            raise ValueError("No data available. Call fit() first.")
            
        # Calculate curve lengths for all k values
        self._calculate_lengths()
        
        # Fit scaling law to extract fractal dimension
        self._fit_scaling_law()
        
        # Calculate confidence interval
        self._calculate_confidence_interval()
        
        # Record execution time
        self.execution_time = time.time() - start_time
        
        return self.get_results()
        
    def _validate_data(self):
        """Validate input data."""
        if self.data is None:
            raise ValueError("No data provided")
        if len(self.data) < 50:
            raise ValueError("Data must have at least 50 points for reliable Higuchi analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        # Check if data is constant (which would cause issues)
        if np.std(self.data) == 0:
            raise ValueError("Data is constant, cannot estimate LRD")
            
    def _generate_k_values(self):
        """Generate k values for analysis."""
        if self.max_k is None:
            self.max_k = len(self.data) // 3
            
        self.k_values = np.logspace(
            np.log10(self.min_k), 
            np.log10(self.max_k), 
            self.num_k, 
            dtype=int
        )
        # Ensure unique k values
        self.k_values = np.unique(self.k_values)
        
    def _calculate_lengths(self):
        """Calculate curve lengths for all k values."""
        self.lengths = np.zeros(len(self.k_values))
        self.length_std = np.zeros(len(self.k_values))
        
        for i, k in enumerate(self.k_values):
            length_list = []
            
            # Calculate length for different starting points
            n_starting_points = min(10, len(self.data) // k)
            if n_starting_points == 0:
                self.lengths[i] = np.nan
                self.length_std[i] = np.nan
                continue
                
            for start_idx in range(0, len(self.data) - k, max(1, (len(self.data) - k) // n_starting_points)):
                length = self._calculate_length_for_k(k, start_idx)
                if length is not None:
                    length_list.append(length)
                    
            if length_list:
                self.lengths[i] = np.mean(length_list)
                self.length_std[i] = np.std(length_list)
            else:
                self.lengths[i] = np.nan
                self.length_std[i] = np.nan
                
    def _calculate_length_for_k(self, k: int, start_idx: int) -> Optional[float]:
        """Calculate curve length for a specific k value and starting point."""
        if start_idx + k >= len(self.data):
            return None
            
        # Extract segment
        segment = self.data[start_idx:start_idx + k]
        
        # Calculate length using Higuchi's method
        length = 0.0
        
        for i in range(1, k):
            # Calculate difference between points separated by i
            diff_sum = 0.0
            count = 0
            
            for j in range(0, k - i, i):
                if j + i < len(segment):
                    diff_sum += abs(segment[j + i] - segment[j])
                    count += 1
                    
            if count > 0:
                length += (diff_sum / count) * (k - 1) / (i * i)
                
        return length
        
    def _fit_scaling_law(self):
        """Fit scaling law to extract fractal dimension."""
        # Get valid length values
        valid_mask = ~np.isnan(self.lengths)
        if np.sum(valid_mask) < 3:
            self.fractal_dimension = np.nan
            self.scaling_error = np.nan
            return
            
        k_valid = self.k_values[valid_mask]
        length_valid = self.lengths[valid_mask]
        
        # Log-log regression: log(L) = -D * log(k) + constant
        log_k = np.log(k_valid)
        log_length = np.log(length_valid)
        
        try:
            coeffs = np.polyfit(log_k, log_length, 1)
            # For Higuchi method: log(L) = -D * log(k) + constant
            # The slope should be negative for increasing k, so D = -slope
            # But let's check the actual relationship and adjust accordingly
            slope = coeffs[0]
            
            # If slope is positive, the relationship might be inverted
            # Try both conventions and pick the one that gives reasonable results
            if slope > 0:
                raw_dimension = slope  # Try positive slope
            else:
                raw_dimension = -slope  # Try negative slope
                
            # Clamp fractal dimension to valid range [1, 2]
            self.fractal_dimension = np.clip(raw_dimension, 1.0, 2.0)
            
            # Calculate R-squared
            y_pred = np.polyval(coeffs, log_k)
            ss_res = np.sum((log_length - y_pred) ** 2)
            ss_tot = np.sum((log_length - np.mean(log_length)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.scaling_error = 1 - r_squared
            
        except (np.linalg.LinAlgError, ValueError):
            self.fractal_dimension = np.nan
            self.scaling_error = np.nan
            
    def _calculate_confidence_interval(self):
        """Calculate confidence interval for the fractal dimension."""
        if np.isnan(self.fractal_dimension):
            self.confidence_interval = (np.nan, np.nan)
            return
            
        # Get valid length values for error estimation
        valid_mask = ~np.isnan(self.lengths)
        if np.sum(valid_mask) < 3:
            self.confidence_interval = (np.nan, np.nan)
            return
            
        k_valid = self.k_values[valid_mask]
        length_valid = self.lengths[valid_mask]
        length_std_valid = self.length_std[valid_mask]
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        dim_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Add noise to length values based on their standard deviations
            length_noisy = length_valid + np.random.normal(0, length_std_valid)
            length_noisy = np.maximum(length_noisy, 0.1)  # Ensure positive values
            
            try:
                log_k = np.log(k_valid)
                log_length = np.log(length_noisy)
                coeffs = np.polyfit(log_k, log_length, 1)
                dim_bootstrap.append(-coeffs[0])
            except:
                continue
                
        if dim_bootstrap:
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            self.confidence_interval = (
                np.percentile(dim_bootstrap, lower_percentile),
                np.percentile(dim_bootstrap, upper_percentile)
            )
        else:
            self.confidence_interval = (np.nan, np.nan)
            
    def get_results(self) -> Dict[str, Any]:
        """Get estimation results."""
        results = {
            'fractal_dimension': self.fractal_dimension,
            'k_values': self.k_values,
            'lengths': self.lengths,
            'length_std': getattr(self, 'length_std', None),
            'scaling_error': getattr(self, 'scaling_error', None),
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'parameters': {
                'min_k': self.min_k,
                'max_k': self.max_k,
                'num_k': self.num_k
            }
        }
        
        # Add interpretation
        if not np.isnan(self.fractal_dimension):
            if self.fractal_dimension < 1.5:
                complexity = "Low complexity"
            elif self.fractal_dimension < 1.8:
                complexity = "Medium complexity"
            else:
                complexity = "High complexity"
                
            results['interpretation'] = {
                'complexity': complexity,
                'reliability': 1 - getattr(self, 'scaling_error', 1.0)
            }
            
        return results


class DMAEstimator(BaseEstimator):
    """
    Detrended Moving Average (DMA) estimator for Hurst exponent.
    
    The DMA method estimates the Hurst exponent by analyzing the variance
    of detrended moving averages at different scales. It's particularly
    effective for non-stationary time series with trends.
    
    References:
    - Vandewalle, N., & Ausloos, M. (1998). Coherent and random sequences 
      in financial fluctuations. Physica A: Statistical Mechanics and its 
      Applications, 246(3-4), 454-459.
    """
    
    def __init__(self, name: str = "DMA", **kwargs):
        super().__init__(name=name, **kwargs)
        self.window_sizes = kwargs.get('window_sizes', None)
        self.min_window = kwargs.get('min_window', 10)
        self.max_window = kwargs.get('max_window', None)
        self.num_windows = kwargs.get('num_windows', 20)
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.n_bootstrap = kwargs.get('n_bootstrap', 1000)
        self.data = None
        self.window_sizes_used = None
        self.fluctuations = None
        self.hurst_exponent = None
        self.confidence_interval = None
        self.scaling_error = None
        
    def fit(self, data: np.ndarray, **kwargs) -> 'DMAEstimator':
        """Fit the DMA estimator to the data."""
        self.data = np.asarray(data, dtype=float)
        self._validate_data()
        self._generate_window_sizes()
        return self
        
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """Estimate Hurst exponent using DMA method."""
        import time
        start_time = time.time()
        
        if data is not None:
            self.fit(data, **kwargs)
            
        if self.data is None:
            raise ValueError("No data available. Call fit() first.")
            
        # Calculate fluctuations for all window sizes
        self._calculate_fluctuations()
        
        # Fit scaling law to extract Hurst exponent
        self._fit_scaling_law()
        
        # Calculate confidence interval
        self._calculate_confidence_interval()
        
        # Record execution time
        self.execution_time = time.time() - start_time
        
        return self.get_results()
        
    def _validate_data(self):
        """Validate input data."""
        if self.data is None:
            raise ValueError("No data provided")
        if len(self.data) < 100:
            raise ValueError("Data must have at least 100 points for reliable DMA analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        # Check if data is constant (which would cause issues)
        if np.std(self.data) == 0:
            raise ValueError("Data is constant, cannot estimate LRD")
            
    def _generate_window_sizes(self):
        """Generate window sizes for analysis."""
        if self.max_window is None:
            self.max_window = len(self.data) // 4
            
        # Generate window sizes logarithmically spaced
        self.window_sizes_used = np.logspace(
            np.log10(self.min_window), 
            np.log10(self.max_window), 
            self.num_windows, 
            dtype=int
        )
        # Ensure unique window sizes and minimum size
        self.window_sizes_used = np.unique(np.maximum(self.window_sizes_used, self.min_window))
        
    def _calculate_fluctuations(self):
        """Calculate fluctuations for all window sizes."""
        self.fluctuations = np.zeros(len(self.window_sizes_used))
        self.fluctuation_std = np.zeros(len(self.window_sizes_used))
        
        for i, window_size in enumerate(self.window_sizes_used):
            fluctuation_list = []
            
            # Calculate fluctuations for different starting points
            n_starting_points = min(20, len(self.data) // window_size)
            if n_starting_points == 0:
                self.fluctuations[i] = np.nan
                self.fluctuation_std[i] = np.nan
                continue
                
            for start_idx in range(0, len(self.data) - window_size, 
                                 max(1, (len(self.data) - window_size) // n_starting_points)):
                fluctuation = self._calculate_fluctuation_for_window(window_size, start_idx)
                if not np.isnan(fluctuation):
                    fluctuation_list.append(fluctuation)
                    
            if fluctuation_list:
                self.fluctuations[i] = np.mean(fluctuation_list)
                self.fluctuation_std[i] = np.std(fluctuation_list)
            else:
                self.fluctuations[i] = np.nan
                self.fluctuation_std[i] = np.nan
                
    def _calculate_fluctuation_for_window(self, window_size: int, start_idx: int) -> float:
        """Calculate fluctuation for a specific window and starting point."""
        try:
            # Extract window data
            window_data = self.data[start_idx:start_idx + window_size]
            
            # Calculate moving average
            moving_avg = self._calculate_moving_average(window_data, window_size // 4)
            
            # Detrend the data
            detrended = window_data - moving_avg
            
            # Calculate root mean square fluctuation
            fluctuation = np.sqrt(np.mean(detrended ** 2))
            
            return fluctuation
            
        except Exception:
            return np.nan
            
    def _calculate_moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average with given window size."""
        if window <= 1:
            return np.zeros_like(data)
            
        # Use convolution for efficient moving average
        kernel = np.ones(window) / window
        padded_data = np.pad(data, (window//2, window//2), mode='edge')
        moving_avg = np.convolve(padded_data, kernel, mode='valid')
        
        # Ensure same length as original data
        if len(moving_avg) > len(data):
            moving_avg = moving_avg[:len(data)]
        elif len(moving_avg) < len(data):
            # Pad with edge values if needed
            moving_avg = np.pad(moving_avg, (0, len(data) - len(moving_avg)), mode='edge')
            
        return moving_avg
        
    def _fit_scaling_law(self):
        """Fit scaling law to extract Hurst exponent."""
        # Get valid fluctuation values
        valid_mask = ~np.isnan(self.fluctuations)
        if np.sum(valid_mask) < 3:
            self.hurst_exponent = np.nan
            self.scaling_error = np.nan
            return
            
        window_valid = self.window_sizes_used[valid_mask]
        fluctuation_valid = self.fluctuations[valid_mask]
        
        try:
            # Log-log linear fit: log(F) = H * log(w) + C
            log_window = np.log(window_valid)
            log_fluctuation = np.log(fluctuation_valid)
            
            # Fit linear relationship
            coeffs = np.polyfit(log_window, log_fluctuation, 1)
            slope = coeffs[0]
            
            # Hurst exponent is the slope
            self.hurst_exponent = slope
            
            # Calculate R-squared
            y_pred = np.polyval(coeffs, log_window)
            ss_res = np.sum((log_fluctuation - y_pred) ** 2)
            ss_tot = np.sum((log_fluctuation - np.mean(log_fluctuation)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.scaling_error = 1 - r_squared
            
        except (np.linalg.LinAlgError, ValueError):
            self.hurst_exponent = np.nan
            self.scaling_error = np.nan
            
    def _calculate_confidence_interval(self):
        """Calculate confidence interval for the Hurst exponent."""
        if np.isnan(self.hurst_exponent):
            self.confidence_interval = (np.nan, np.nan)
            return
            
        # Get valid fluctuation values for error estimation
        valid_mask = ~np.isnan(self.fluctuations)
        if np.sum(valid_mask) < 3:
            self.confidence_interval = (np.nan, np.nan)
            return
            
        window_valid = self.window_sizes_used[valid_mask]
        fluctuation_valid = self.fluctuations[valid_mask]
        fluctuation_std_valid = self.fluctuation_std[valid_mask]
        
        # Bootstrap confidence interval
        n_bootstrap = self.n_bootstrap
        h_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Add noise to fluctuation values based on their standard deviations
            fluctuation_noisy = fluctuation_valid + np.random.normal(0, fluctuation_std_valid)
            fluctuation_noisy = np.maximum(fluctuation_noisy, 1e-10)  # Ensure positive values
            
            try:
                log_window = np.log(window_valid)
                log_fluctuation = np.log(fluctuation_noisy)
                coeffs = np.polyfit(log_window, log_fluctuation, 1)
                h_bootstrap.append(coeffs[0])
            except:
                continue
                
        if h_bootstrap:
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            self.confidence_interval = (
                np.percentile(h_bootstrap, lower_percentile),
                np.percentile(h_bootstrap, upper_percentile)
            )
        else:
            self.confidence_interval = (np.nan, np.nan)
            
    def get_results(self) -> Dict[str, Any]:
        """Get estimation results."""
        results = {
            'hurst_exponent': self.hurst_exponent,
            'window_sizes': self.window_sizes_used,
            'fluctuations': self.fluctuations,
            'fluctuation_std': getattr(self, 'fluctuation_std', None),
            'scaling_error': self.scaling_error,
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'parameters': {
                'min_window': self.min_window,
                'max_window': self.max_window,
                'num_windows': self.num_windows
            }
        }
        
        # Add interpretation
        if not np.isnan(self.hurst_exponent):
            if self.hurst_exponent < 0.5:
                dependence = "Short-range dependent (anti-persistent)"
            elif self.hurst_exponent < 0.6:
                dependence = "Short-range dependent (weakly anti-persistent)"
            elif self.hurst_exponent < 0.9:
                dependence = "Long-range dependent (persistent)"
            else:
                dependence = "Strongly long-range dependent (highly persistent)"
                
            results['interpretation'] = {
                'dependence_type': dependence,
                'reliability': 1 - getattr(self, 'scaling_error', 1.0)
            }
            
        return results
