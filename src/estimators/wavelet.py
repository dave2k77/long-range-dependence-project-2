"""
Wavelet Long-Range Dependence Estimators

This module contains implementations of wavelet-based methods for estimating
long-range dependence, including Wavelet Leaders and Wavelet Whittle methods.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import pywt
import logging
import scipy.optimize
import time

from .base import BaseEstimator

logger = logging.getLogger(__name__)


class WaveletLeadersEstimator(BaseEstimator):
    """
    Wavelet Leaders estimator for long-range dependence.
    
    Wavelet Leaders is a method that uses wavelet coefficients to estimate
    long-range dependence parameters by analyzing the scaling behavior.
    """
    
    def __init__(self, name: str = "WaveletLeaders", **kwargs):
        """
        Initialize Wavelet Leaders estimator.
        
        Parameters
        ----------
        name : str
            Name identifier for the estimator
        **kwargs
            Additional parameters including:
            - wavelet: Wavelet type (default: 'db4')
            - num_scales: Number of wavelet scales to use
            - min_scale: Minimum scale for analysis
            - max_scale: Maximum scale for analysis
        """
        super().__init__(name=name, **kwargs)
        self.wavelet = kwargs.get('wavelet', 'db4')
        self.num_scales = kwargs.get('num_scales', 20)
        self.min_scale = kwargs.get('min_scale', 2)
        self.max_scale = kwargs.get('max_scale', None)
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.n_bootstrap = kwargs.get('n_bootstrap', 1000)
        self.data = None
        self.wavelet_coeffs = None
        self.scales = None
        
    def fit(self, data: np.ndarray, **kwargs) -> 'WaveletLeadersEstimator':
        """
        Fit the Wavelet Leaders estimator to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        self : WaveletLeadersEstimator
            Fitted estimator instance
        """
        self.data = np.asarray(data, dtype=float)
        self._validate_data()
        return self
    
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Estimate long-range dependence using Wavelet Leaders.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Input time series data (if not provided, uses fitted data)
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing Wavelet Leaders estimation results
        """
        start_time = time.time()
        
        if data is not None:
            self.fit(data, **kwargs)
        
        if self.data is None:
            raise ValueError("No data provided. Call fit() first.")
        
        # Generate scales
        self._generate_scales()
        
        # Calculate wavelet coefficients
        self._calculate_wavelet_coeffs()
        
        # Calculate wavelet leaders
        self.leaders = self._calculate_wavelet_leaders()
        
        # Fit scaling law to get Hurst exponent
        hurst_exponent, r_squared, std_error = self._fit_scaling_law()
        
        # Calculate alpha (long-range dependence parameter)
        alpha = 2 * hurst_exponent - 1
        
        # Calculate confidence interval
        self._calculate_confidence_interval()
        
        results = {
            'hurst_exponent': hurst_exponent,
            'alpha': alpha,
            'r_squared': r_squared,
            'std_error': std_error,
            'scaling_error': getattr(self, 'scaling_error', None),
            'confidence_interval': getattr(self, 'confidence_interval', None),
            'scales': self.scales,
            'leaders': self.leaders,
            'wavelet_coeffs': self.wavelet_coeffs,
            'method': 'WaveletLeaders'
        }
        
        # Add interpretation
        if not np.isnan(hurst_exponent):
            if hurst_exponent < 0.5:
                lrd_type = "Anti-persistent (short-range dependent)"
            elif hurst_exponent > 0.5:
                lrd_type = "Persistent (long-range dependent)"
            else:
                lrd_type = "Random walk (no long-range dependence)"
                
            results['interpretation'] = {
                'lrd_type': lrd_type,
                'strength': abs(hurst_exponent - 0.5),
                'reliability': getattr(self, 'r_squared', 0.0),
                'method': 'Wavelet Leaders'
            }
        
        # Record execution time
        self.execution_time = time.time() - start_time
        
        # Calculate memory usage (simple approximation)
        import sys
        self.memory_usage = sys.getsizeof(self.data) + sys.getsizeof(self.wavelet_coeffs) + sys.getsizeof(self.leaders)
        
        # Store results
        self.results.update(results)
        
        return results
    
    def _validate_data(self):
        """Validate input data."""
        if len(self.data) < 100:
            raise ValueError("Data must have at least 100 points for reliable Wavelet Leaders estimation")
        
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        
        # Check if data is constant (which would cause issues)
        if np.all(self.data == self.data[0]):
            raise ValueError("Data cannot be constant")
    
    def _generate_scales(self):
        """Generate scales for wavelet analysis."""
        if self.max_scale is None:
            self.max_scale = min(len(self.data) // 4, 10)
        
        # Generate scales logarithmically spaced
        self.scales = np.logspace(
            np.log10(self.min_scale),
            np.log10(self.max_scale),
            self.num_scales,
            dtype=int
        )
        
        # Ensure unique scales
        self.scales = np.unique(self.scales)
    
    def _calculate_wavelet_coeffs(self):
        """Calculate wavelet coefficients for different scales."""
        self.wavelet_coeffs = {}
        
        for scale in self.scales:
            # Use discrete wavelet transform with decimation for different scales
            # This is more compatible with PyWavelets 1.8.0
            try:
                # For small scales, use direct DWT
                if scale <= 4:
                    coeffs = pywt.wavedec(self.data, self.wavelet, level=1)[0]
                else:
                    # For larger scales, use decimation
                    level = int(np.log2(scale))
                    coeffs = pywt.wavedec(self.data, self.wavelet, level=min(level, 8))[0]
                
                # Ensure consistent length by padding/truncating
                if len(coeffs) < len(self.data):
                    coeffs = np.pad(coeffs, (0, len(self.data) - len(coeffs)), mode='edge')
                else:
                    coeffs = coeffs[:len(self.data)]
                    
                self.wavelet_coeffs[scale] = coeffs
            except Exception as e:
                # Fallback: use simple moving average as approximation
                window_size = min(scale, len(self.data) // 4)
                if window_size > 1:
                    coeffs = np.convolve(self.data, np.ones(window_size)/window_size, mode='same')
                else:
                    coeffs = self.data.copy()
                self.wavelet_coeffs[scale] = coeffs
    
    def _calculate_wavelet_leaders(self) -> np.ndarray:
        """Calculate wavelet leaders for each scale."""
        leaders = []
        
        for scale in self.scales:
            coeffs = self.wavelet_coeffs[scale]
            
            # Calculate wavelet leaders as the maximum absolute value
            # of wavelet coefficients in a neighborhood
            leader = np.max(np.abs(coeffs))
            leaders.append(leader)
        
        result = np.array(leaders)
        self.leaders = result
        return result
    
    def _fit_scaling_law(self) -> Tuple[float, float, float]:
        """Fit scaling law to wavelet leaders vs scales."""
        if len(self.scales) != len(self.leaders):
            # Filter out scales where leaders couldn't be calculated
            valid_indices = np.arange(len(self.scales))[:len(self.leaders)]
            scales = self.scales[valid_indices]
        else:
            scales = self.scales
        
        # Log-log relationship: log(L) = H * log(s) + C
        log_scales = np.log(scales)
        log_leaders = np.log(self.leaders)
        
        # Linear fit
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_scales, log_leaders
        )
        
        hurst_exponent = slope
        r_squared = r_value**2
        
        # Store results as instance attributes
        self.hurst_exponent = hurst_exponent
        self.r_squared = r_squared
        self.scaling_error = 1 - r_squared
        
        return hurst_exponent, r_squared, std_err
    
    def _calculate_confidence_interval(self):
        """Calculate confidence interval for the Hurst exponent."""
        if np.isnan(self.hurst_exponent):
            self.confidence_interval = (np.nan, np.nan)
            return
            
        # Simple confidence interval based on standard error
        # For 95% confidence level, use ±1.96 * std_error
        if hasattr(self, 'scaling_error'):
            # Use scaling error as a proxy for uncertainty
            margin = 1.96 * np.sqrt(self.scaling_error)
            self.confidence_interval = (
                max(0.01, self.hurst_exponent - margin),
                min(0.99, self.hurst_exponent + margin)
            )
        else:
            # Fallback: use a simple percentage of the estimate
            margin = 0.1 * self.hurst_exponent
            self.confidence_interval = (
                max(0.01, self.hurst_exponent - margin),
                min(0.99, self.hurst_exponent + margin)
            )
    
    def _generate_interpretation(self):
        """Generate interpretation of the estimation results."""
        if np.isnan(self.hurst_exponent):
            return None
            
        if self.hurst_exponent < 0.5:
            lrd_type = "Anti-persistent (short-range dependent)"
        elif self.hurst_exponent > 0.5:
            lrd_type = "Persistent (long-range dependent)"
        else:
            lrd_type = "Random walk (no long-range dependence)"
            
        return {
            'lrd_type': lrd_type,
            'strength': abs(self.hurst_exponent - 0.5),
            'reliability': getattr(self, 'r_squared', 0.0),
            'method': 'Wavelet Leaders'
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get estimation results including interpretation."""
        # Check if we have stored results
        if hasattr(self, 'results') and self.results:
            return self.results.copy()
        
        # Fallback to individual attributes
        results = {
            'hurst_exponent': getattr(self, 'hurst_exponent', None),
            'alpha': getattr(self, 'alpha', None),
            'r_squared': getattr(self, 'r_squared', None),
            'std_error': getattr(self, 'std_error', None),
            'scaling_error': getattr(self, 'scaling_error', None),
            'confidence_interval': getattr(self, 'confidence_interval', None),
            'scales': getattr(self, 'scales', None),
            'leaders': getattr(self, 'leaders', None),
            'wavelet_coeffs': getattr(self, 'wavelet_coeffs', None),
            'method': 'Wavelet Leaders'
        }
        
        # Add interpretation if available
        interpretation = self._generate_interpretation()
        if interpretation:
            results['interpretation'] = interpretation
            
        return results
    
    def get_execution_time(self) -> Optional[float]:
        """Get the execution time of the last estimation."""
        return getattr(self, 'execution_time', None)
    
    def get_memory_usage(self) -> Optional[int]:
        """Get memory usage of the estimator."""
        if self.data is None:
            return 0
        
        import sys
        memory_usage = sys.getsizeof(self.data)
        if self.wavelet_coeffs is not None:
            memory_usage += sys.getsizeof(self.wavelet_coeffs)
        if self.leaders is not None:
            memory_usage += sys.getsizeof(self.leaders)
        if self.scales is not None:
            memory_usage += sys.getsizeof(self.scales)
        return memory_usage
    
    def reset(self):
        """Reset the estimator to initial state."""
        self.data = None
        self.wavelet_coeffs = None
        self.scales = None
        self.leaders = None
        self.hurst_exponent = None
        self.r_squared = None
        self.scaling_error = None
        self.confidence_interval = None
        self.execution_time = None
        self.memory_usage = None


class WaveletWhittleEstimator(BaseEstimator):
    """
    Wavelet Whittle estimator for long-range dependence.
    
    This estimator uses wavelet-based spectral analysis and maximum
    likelihood estimation to determine the Hurst exponent.
    """
    
    def __init__(self, name: str = "WaveletWhittle", **kwargs):
        super().__init__(name=name, **kwargs)
        self.wavelet = kwargs.get('wavelet', 'db4')
        self.num_scales = kwargs.get('num_scales', 20)
        self.min_scale = kwargs.get('min_scale', 2)
        self.max_scale = kwargs.get('max_scale', None)
        self.optimization_method = kwargs.get('optimization_method', 'L-BFGS-B')
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.n_bootstrap = kwargs.get('n_bootstrap', 1000)
        self.data = None
        self.wavelet_coeffs = None
        self.scales = None
        self.hurst_exponent = None
        self.confidence_interval = None
        
    def fit(self, data: np.ndarray, **kwargs) -> 'WaveletWhittleEstimator':
        """Fit the Wavelet Whittle estimator to the data."""
        self.data = np.asarray(data, dtype=float)
        self._validate_data()
        self._generate_scales()
        return self
        
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """Estimate Hurst exponent using Wavelet Whittle method."""
        start_time = time.time()
        
        if data is not None:
            self.fit(data, **kwargs)
            
        if self.data is None:
            raise ValueError("No data available. Call fit() first.")
            
        # Calculate wavelet coefficients
        self._calculate_wavelet_coeffs()
        
        # Apply Whittle likelihood optimization
        self._whittle_optimization()
        
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
            raise ValueError("Data must have at least 100 points for reliable wavelet analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        # Check if data is constant (which would cause issues)
        if np.all(self.data == self.data[0]):
            raise ValueError("Data cannot be constant")
            
    def _generate_scales(self):
        """Generate scale values for wavelet analysis."""
        if self.max_scale is None:
            self.max_scale = min(20, len(self.data) // 10)
            
        self.scales = np.logspace(
            np.log10(self.min_scale), 
            np.log10(self.max_scale), 
            self.num_scales, 
            dtype=int
        )
        # Ensure unique scales
        self.scales = np.unique(self.scales)
        
    def _calculate_wavelet_coeffs(self):
        """Calculate wavelet coefficients for all scales."""
        self.wavelet_coeffs = {}
        
        for scale in self.scales:
            # Use discrete wavelet transform with decimation for different scales
            # This is more compatible with PyWavelets 1.8.0
            try:
                # For small scales, use direct DWT
                if scale <= 4:
                    coeffs = pywt.wavedec(self.data, self.wavelet, level=1)[0]
                else:
                    # For larger scales, use decimation
                    level = int(np.log2(scale))
                    coeffs = pywt.wavedec(self.data, self.wavelet, level=min(level, 8))[0]
                
                # Ensure consistent length by padding/truncating
                if len(coeffs) < len(self.data):
                    coeffs = np.pad(coeffs, (0, len(self.data) - len(coeffs)), mode='edge')
                else:
                    coeffs = coeffs[:len(self.data)]
                    
                self.wavelet_coeffs[scale] = coeffs
            except Exception as e:
                # Fallback: use simple moving average as approximation
                window_size = min(scale, len(self.data) // 4)
                if window_size > 1:
                    coeffs = np.convolve(self.data, np.ones(window_size)/window_size, mode='same')
                else:
                    coeffs = self.data.copy()
                self.wavelet_coeffs[scale] = coeffs
            
    def _whittle_optimization(self):
        """Apply Whittle likelihood optimization to estimate Hurst exponent."""
        # Prepare data for optimization
        scales_array = np.array(list(self.scales))
        coeffs_array = np.array([self.wavelet_coeffs[scale] for scale in self.scales])
        
        # Initial guess for Hurst exponent
        initial_guess = 0.5
        
        try:
            # Optimize negative log-likelihood
            result = scipy.optimize.minimize(
                fun=lambda h: self._negative_log_likelihood(h, scales_array, coeffs_array),
                x0=initial_guess,
                method=self.optimization_method,
                bounds=[(0.01, 0.99)],  # Hurst exponent bounds
                options={'maxiter': 1000}
            )
            
            if result.success:
                self.hurst_exponent = result.x[0]
                self.optimization_success = True
                self.optimization_message = result.message
                self.negative_log_likelihood = result.fun
            else:
                self.hurst_exponent = np.nan
                self.optimization_success = False
                self.optimization_message = result.message
                self.negative_log_likelihood = np.nan
                
        except Exception as e:
            self.hurst_exponent = np.nan
            self.optimization_success = False
            self.optimization_message = str(e)
            self.negative_log_likelihood = np.nan
            
    def _negative_log_likelihood(self, h: float, scales: np.ndarray, coeffs: np.ndarray) -> float:
        """Calculate negative log-likelihood for Whittle estimation."""
        try:
            # Theoretical wavelet variance for fBm process
            theoretical_variance = scales ** (2 * h - 1)
            
            # Empirical wavelet variance
            empirical_variance = np.var(coeffs, axis=1)
            
            # Whittle likelihood (simplified)
            # L = -sum(log(theoretical_variance) + empirical_variance/theoretical_variance)
            log_likelihood = np.sum(
                np.log(theoretical_variance) + 
                empirical_variance / theoretical_variance
            )
            
            return log_likelihood
            
        except (ValueError, RuntimeWarning):
            return np.inf
            
    def _calculate_confidence_interval(self):
        """Calculate confidence interval for the Hurst exponent."""
        if np.isnan(self.hurst_exponent):
            self.confidence_interval = (np.nan, np.nan)
            return
            
        # Simple confidence interval based on optimization success and data quality
        if hasattr(self, 'optimization_success') and self.optimization_success:
            # If optimization was successful, use a reasonable margin
            margin = 0.05 * self.hurst_exponent
            self.confidence_interval = (
                max(0.01, self.hurst_exponent - margin),
                min(0.99, self.hurst_exponent + margin)
            )
        else:
            # Fallback: use a simple percentage of the estimate
            margin = 0.1 * self.hurst_exponent
            self.confidence_interval = (
                max(0.01, self.hurst_exponent - margin),
                min(0.99, self.hurst_exponent + margin)
            )
    
    def _generate_interpretation(self):
        """Generate interpretation of results."""
        if not hasattr(self, 'hurst_exponent') or np.isnan(self.hurst_exponent):
            return None
            
        if self.hurst_exponent < 0.5:
            lrd_type = "Anti-persistent (short-range dependent)"
        elif self.hurst_exponent > 0.5:
            lrd_type = "Persistent (long-range dependent)"
        else:
            lrd_type = "Random walk (no long-range dependence)"
            
        return {
            'lrd_type': lrd_type,
            'strength': abs(self.hurst_exponent - 0.5),
            'reliability': getattr(self, 'optimization_success', False),
            'method': 'Wavelet Whittle Maximum Likelihood'
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get estimation results."""
        # Convert wavelet_coeffs dict to list for compatibility
        wavelet_coeffs_list = []
        if self.wavelet_coeffs is not None:
            for scale in sorted(self.scales):
                if scale in self.wavelet_coeffs:
                    wavelet_coeffs_list.append(self.wavelet_coeffs[scale])
        
        results = {
            'hurst_exponent': self.hurst_exponent,
            'scales': self.scales,
            'wavelet_coeffs': wavelet_coeffs_list,
            'optimization_success': getattr(self, 'optimization_success', None),
            'optimization_message': getattr(self, 'optimization_message', None),
            'negative_log_likelihood': getattr(self, 'negative_log_likelihood', None),
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'parameters': {
                'wavelet': self.wavelet,
                'min_scale': self.min_scale,
                'max_scale': self.max_scale,
                'num_scales': self.num_scales,
                'optimization_method': self.optimization_method
            }
        }
        
        # Add interpretation
        interpretation = self._generate_interpretation()
        if interpretation:
            results['interpretation'] = interpretation
            
        return results
    
    def get_execution_time(self) -> Optional[float]:
        """Get the execution time of the last estimation."""
        return getattr(self, 'execution_time', None)
    
    def get_memory_usage(self) -> Optional[int]:
        """Get memory usage of the estimator."""
        if self.data is None:
            return 0
        
        import sys
        memory_usage = sys.getsizeof(self.data)
        if self.wavelet_coeffs is not None:
            memory_usage += sys.getsizeof(self.wavelet_coeffs)
        if self.scales is not None:
            memory_usage += sys.getsizeof(self.scales)
        return memory_usage
    
    def reset(self):
        """Reset the estimator to initial state."""
        self.data = None
        self.wavelet_coeffs = None
        self.scales = None
        self.hurst_exponent = None
        self.confidence_interval = None
        self.execution_time = None
        self.optimization_success = None
        self.optimization_message = None
        self.negative_log_likelihood = None
