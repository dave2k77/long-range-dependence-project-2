"""
High-Performance Geweke-Porter-Hudak (GPH) Estimator

This module provides an optimized implementation of the GPH method
for estimating long-range dependence using JAX acceleration and NumPy fallbacks.
"""

import numpy as np
import jax.numpy as jnp
import jax
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import psutil
from functools import lru_cache
from scipy import signal

try:
    from .base import BaseEstimator
    from ..utils.jax_utils import JAXOptimizer
except ImportError:
    from src.estimators.base import BaseEstimator
    from src.utils.jax_utils import JAXOptimizer

logger = logging.getLogger(__name__)

# Configure JAX for optimal performance
jax.config.update('jax_enable_x64', True)


class HighPerformanceGPHEstimator(BaseEstimator):
    """
    High-performance Geweke-Porter-Hudak (GPH) estimator for long-range dependence.
    
    This estimator uses JAX acceleration for core computations with robust
    NumPy fallbacks for reliability. It includes performance optimizations
    such as vectorized operations, caching, and memory management.
    
    Parameters
    ----------
    name : str, optional
        Name identifier for the estimator
    num_frequencies : int, optional
        Number of frequencies to use in GPH regression (default: 50)
    frequency_threshold : float, optional
        Frequency threshold for low-frequency analysis (default: 0.1)
    min_freq : float, optional
        Minimum frequency for analysis (default: 0.01)
    num_freq : int, optional
        Number of frequency points (default: 50)
    confidence_level : float, optional
        Confidence level for bootstrap intervals (default: 0.95)
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 1000)
    use_jax : bool, optional
        Whether to use JAX acceleration (default: True)
    enable_caching : bool, optional
        Whether to enable result caching (default: True)
    vectorized : bool, optional
        Whether to use vectorized operations (default: True)
    """
    
    def __init__(self, name: str = "HighPerformanceGPH", **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Core parameters
        self.num_frequencies = kwargs.get('num_frequencies', 50)
        self.frequency_threshold = kwargs.get('frequency_threshold', 0.1)
        self.min_freq = kwargs.get('min_freq', 0.01)
        self.num_freq = kwargs.get('num_freq', 50)
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.n_bootstrap = kwargs.get('n_bootstrap', 1000)
        
        # Performance options
        self.use_jax = kwargs.get('use_jax', True)
        self.enable_caching = kwargs.get('enable_caching', True)
        self.vectorized = kwargs.get('vectorized', True)
        
        # Data storage
        self.data = None
        self.frequencies = None
        self.periodogram = None
        self.fractional_d = None
        self.hurst_exponent = None
        self.regression_error = None
        self.confidence_interval = None
        
        # Performance monitoring
        self.execution_time = None
        self.memory_usage = None
        self.jax_usage = False
        self.fallback_usage = False
        
        # Caching
        if self.enable_caching:
            self._periodogram_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate initialization parameters."""
        if self.num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")
        if not 0 < self.frequency_threshold <= 0.5:
            raise ValueError("frequency_threshold must be between 0 and 0.5")
        if not 0 < self.min_freq < self.frequency_threshold:
            raise ValueError("min_freq must be between 0 and frequency_threshold")
        if self.num_freq <= 0:
            raise ValueError("num_freq must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
            
    def fit(self, data: np.ndarray, **kwargs) -> 'HighPerformanceGPHEstimator':
        """
        Fit the GPH estimator to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        self : HighPerformanceGPHEstimator
            Fitted estimator instance
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            self.data = np.asarray(data, dtype=float)
            self._validate_data()
            
            self.execution_time = time.time() - start_time
            self.memory_usage = psutil.Process().memory_info().rss - start_memory
            
            return self
            
        except Exception as e:
            logger.error(f"Error in fit for {self.name}: {str(e)}")
            raise
            
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Estimate long-range dependence using GPH method.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Input time series data (if not provided, uses fitted data)
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing GPH estimation results
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            if data is not None:
                self.fit(data, **kwargs)
                
            if self.data is None:
                raise ValueError("No data available. Call fit() first.")
                
            # Calculate periodogram
            self._calculate_periodogram()
            
            # Apply GPH regression
            if self.use_jax:
                try:
                    self._gph_regression_jax()
                    self.jax_usage = True
                except Exception as e:
                    logger.warning(f"JAX GPH regression failed: {e}. Using NumPy fallback.")
                    self._gph_regression_numpy()
                    self.fallback_usage = True
            else:
                self._gph_regression_numpy()
                
            # Calculate confidence interval
            self._calculate_confidence_interval()
            
            # Record execution time and memory usage
            self.execution_time = time.time() - start_time
            self.memory_usage = psutil.Process().memory_info().rss - start_memory
            
            return self.get_results()
            
        except Exception as e:
            logger.error(f"Error in estimate for {self.name}: {str(e)}")
            raise
            
    def _validate_data(self):
        """Validate input data."""
        if self.data is None:
            raise ValueError("No data provided")
        if len(self.data) < 200:
            raise ValueError("Data must have at least 200 points for reliable GPH analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        if np.std(self.data) == 0:
            raise ValueError("Data is constant, cannot estimate LRD")
            
    def _calculate_periodogram(self):
        """Calculate the periodogram of the data with caching."""
        # Check cache first
        cache_key = (len(self.data), hash(self.data.tobytes()))
        if self.enable_caching and cache_key in self._periodogram_cache:
            self.frequencies, self.periodogram = self._periodogram_cache[cache_key]
            self._cache_hits += 1
            return
            
        try:
            # Use Welch's method for better spectral estimation
            self.frequencies, self.periodogram = signal.welch(
                self.data,
                fs=1.0,
                window='hann',
                nperseg=min(256, len(self.data) // 4),
                noverlap=0
            )
            
            # Cache the result
            if self.enable_caching:
                self._periodogram_cache[cache_key] = (self.frequencies, self.periodogram)
                self._cache_misses += 1
                
        except Exception as e:
            logger.error(f"Periodogram calculation failed: {e}")
            raise
            
    def _gph_regression_jax(self):
        """Apply GPH regression using JAX."""
        try:
            # Convert to JAX arrays
            frequencies_jax = jnp.array(self.frequencies)
            periodogram_jax = jnp.array(self.periodogram)
            
            # Filter low frequencies
            low_freq_mask = frequencies_jax <= self.frequency_threshold
            
            if jnp.sum(low_freq_mask) < self.num_frequencies:
                # Use all available low frequencies
                freq_filtered = frequencies_jax[low_freq_mask]
                periodogram_filtered = periodogram_jax[low_freq_mask]
            else:
                # Use specified number of lowest frequencies
                freq_filtered = frequencies_jax[:self.num_frequencies]
                periodogram_filtered = periodogram_jax[:self.num_frequencies]
                
            # Remove zero and negative values
            positive_mask = periodogram_filtered > 0
            if jnp.sum(positive_mask) < 5:
                self.fractional_d = np.nan
                self.hurst_exponent = np.nan
                self.regression_error = np.nan
                return
                
            freq_positive = freq_filtered[positive_mask]
            periodogram_positive = periodogram_filtered[positive_mask]
            
            # GPH regression: log(I(f)) = c - 2d * log(4 * sin²(πf)) + ε
            # where d is the fractional differencing parameter
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            x_gph = jnp.log(4 * jnp.sin(jnp.pi * freq_positive) ** 2 + epsilon)
            y_gph = jnp.log(periodogram_positive)
            
            try:
                # Use JAX linear regression
                jax_optimizer = JAXOptimizer()
                slope, intercept, r_value, p_value, std_err = jax_optimizer.fast_linregress(
                    x_gph, y_gph
                )
                
                self.fractional_d = -float(slope) / 2  # Extract d from slope
                
                # Convert d to Hurst exponent: H = d + 0.5
                self.hurst_exponent = self.fractional_d + 0.5
                self.regression_error = 1 - float(r_value)**2
                
                # Store regression details
                self.gph_x = np.array(x_gph)
                self.gph_y = np.array(y_gph)
                self.gph_y_pred = np.array(intercept + slope * x_gph)
                self.intercept = float(intercept)
                
            except Exception as e:
                logger.debug(f"JAX linear regression failed: {e}. Using NumPy fallback.")
                self._gph_regression_numpy()
                
        except Exception as e:
            logger.error(f"JAX GPH regression failed: {e}")
            raise
            
    def _gph_regression_numpy(self):
        """Apply GPH regression using NumPy."""
        # Filter low frequencies
        low_freq_mask = self.frequencies <= self.frequency_threshold
        
        if np.sum(low_freq_mask) < self.num_frequencies:
            # Use all available low frequencies
            freq_filtered = self.frequencies[low_freq_mask]
            periodogram_filtered = self.periodogram[low_freq_mask]
        else:
            # Use specified number of lowest frequencies
            freq_filtered = self.frequencies[:self.num_frequencies]
            periodogram_filtered = self.periodogram[:self.num_frequencies]
            
        # Remove zero and negative values
        positive_mask = periodogram_filtered > 0
        if np.sum(positive_mask) < 5:
            self.fractional_d = np.nan
            self.hurst_exponent = np.nan
            self.regression_error = np.nan
            return
            
        freq_positive = freq_filtered[positive_mask]
        periodogram_positive = periodogram_filtered[positive_mask]
        
        # GPH regression: log(I(f)) = c - 2d * log(4 * sin²(πf)) + ε
        # where d is the fractional differencing parameter
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        x_gph = np.log(4 * np.sin(np.pi * freq_positive) ** 2 + epsilon)
        y_gph = np.log(periodogram_positive)
        
        try:
            # Linear regression with intercept
            coeffs = np.polyfit(x_gph, y_gph, 1)
            self.fractional_d = -coeffs[0] / 2  # Extract d from slope
            
            # Convert d to Hurst exponent: H = d + 0.5
            self.hurst_exponent = self.fractional_d + 0.5
            
            # Calculate R-squared
            y_pred = np.polyval(coeffs, x_gph)
            ss_res = np.sum((y_gph - y_pred) ** 2)
            ss_tot = np.sum((y_gph - np.mean(y_gph)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.regression_error = 1 - r_squared
            
            # Store regression details
            self.gph_x = x_gph
            self.gph_y = y_gph
            self.gph_y_pred = y_pred
            self.intercept = coeffs[1]
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"GPH regression failed: {e}")
            self.fractional_d = np.nan
            self.hurst_exponent = np.nan
            self.regression_error = np.nan
            
    def _calculate_confidence_interval(self):
        """Calculate confidence interval for the fractional differencing parameter."""
        if np.isnan(self.fractional_d):
            self.confidence_interval = (np.nan, np.nan)
            return
            
        # Get GPH regression data for error estimation
        if not hasattr(self, 'gph_x'):
            self.confidence_interval = (np.nan, np.nan)
            return
            
        x_valid = self.gph_x
        y_valid = self.gph_y
        
        # Bootstrap confidence interval
        d_bootstrap = []
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(x_valid), size=len(x_valid), replace=True)
            x_boot = x_valid[indices]
            y_boot = y_valid[indices]
            
            try:
                coeffs = np.polyfit(x_boot, y_boot, 1)
                d_bootstrap.append(-coeffs[0] / 2)
            except:
                continue
                
        if d_bootstrap:
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            self.confidence_interval = (
                np.percentile(d_bootstrap, lower_percentile),
                np.percentile(d_bootstrap, upper_percentile)
            )
        else:
            self.confidence_interval = (np.nan, np.nan)
            
    def get_results(self) -> Dict[str, Any]:
        """Get estimation results."""
        results = {
            'fractional_d': self.fractional_d,
            'hurst_exponent': self.hurst_exponent,
            'frequencies': self.frequencies,
            'periodogram': self.periodogram,
            'regression_error': self.regression_error,
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'gph_x': getattr(self, 'gph_x', None),
            'gph_y': getattr(self, 'gph_y', None),
            'gph_y_pred': getattr(self, 'gph_y_pred', None),
            'intercept': getattr(self, 'intercept', None),
            'parameters': {
                'num_frequencies': self.num_frequencies,
                'frequency_threshold': self.frequency_threshold
            },
            'performance': {
                'execution_time': self.execution_time,
                'memory_usage': self.memory_usage,
                'jax_usage': self.jax_usage,
                'fallback_usage': self.fallback_usage
            }
        }
        
        # Add interpretation
        if not np.isnan(self.fractional_d):
            if self.fractional_d < 0:
                lrd_type = "Anti-persistent (short-range dependent)"
            elif self.fractional_d > 0:
                lrd_type = "Persistent (long-range dependent)"
            else:
                lrd_type = "Random walk (no long-range dependence)"
                
            results['interpretation'] = {
                'lrd_type': lrd_type,
                'strength': abs(self.fractional_d),
                'reliability': 1 - getattr(self, 'regression_error', 1.0),
                'fractional_d_interpretation': f"d = {self.fractional_d:.3f}"
            }
            
        return results
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        if not self.enable_caching:
            return {'caching_enabled': False}
            
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'caching_enabled': True,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._periodogram_cache)
        }
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cache_stats = self.get_cache_stats()
        
        return {
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'jax_usage': self.jax_usage,
            'fallback_usage': self.fallback_usage,
            'cache_performance': cache_stats,
            'optimization_features': {
                'vectorized': self.vectorized,
                'caching': self.enable_caching,
                'jax_acceleration': self.use_jax
            }
        }
        
    def reset(self):
        """Reset the estimator to initial state."""
        super().reset()
        self.frequencies = None
        self.periodogram = None
        self.fractional_d = None
        self.hurst_exponent = None
        self.regression_error = None
        self.confidence_interval = None
        self.jax_usage = False
        self.fallback_usage = False
        
        if self.enable_caching:
            self._periodogram_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
