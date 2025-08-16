"""
High-Performance Periodogram-based Long-Range Dependence Estimator

This module provides an optimized implementation of the periodogram method
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


class HighPerformancePeriodogramEstimator(BaseEstimator):
    """
    High-performance periodogram-based long-range dependence estimator.
    
    This estimator uses JAX acceleration for core computations with robust
    NumPy fallbacks for reliability. It includes performance optimizations
    such as vectorized operations, caching, and memory management.
    
    Parameters
    ----------
    name : str, optional
        Name identifier for the estimator
    window : str, optional
        Window function for periodogram calculation (default: 'hann')
    nperseg : int, optional
        Number of points per segment (default: None, auto-determined)
    nfft : int, optional
        Number of FFT points (default: None, auto-determined)
    frequency_range : List[float], optional
        Range of frequencies to use (default: [0.01, 0.5])
    min_freq : float, optional
        Minimum frequency for analysis (default: 0.01)
    num_freq : int, optional
        Number of frequency points (default: 256)
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
    
    def __init__(self, name: str = "HighPerformancePeriodogram", **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Core parameters
        self.window = kwargs.get('window', 'hann')
        self.nperseg = kwargs.get('nperseg', None)
        self.nfft = kwargs.get('nfft', None)
        self.frequency_range = kwargs.get('frequency_range', [0.01, 0.5])
        self.min_freq = kwargs.get('min_freq', 0.01)
        self.num_freq = kwargs.get('num_freq', 256)
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
        self.hurst_exponent = None
        self.beta = None
        self.scaling_error = None
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
        if len(self.frequency_range) != 2:
            raise ValueError("frequency_range must have exactly 2 elements")
        if self.frequency_range[0] >= self.frequency_range[1]:
            raise ValueError("frequency_range must be [min_freq, max_freq]")
        if not all(0 < f <= 0.5 for f in self.frequency_range):
            raise ValueError("frequencies must be between 0 and 0.5 (inclusive)")
        if self.num_freq <= 0:
            raise ValueError("num_freq must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
            
    def fit(self, data: np.ndarray, **kwargs) -> 'HighPerformancePeriodogramEstimator':
        """
        Fit the periodogram estimator to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        self : HighPerformancePeriodogramEstimator
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
        Estimate long-range dependence using periodogram analysis.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Input time series data (if not provided, uses fitted data)
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing periodogram estimation results
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
            
            # Filter frequencies and fit scaling law
            if self.use_jax:
                try:
                    self._filter_and_fit_jax()
                    self.jax_usage = True
                except Exception as e:
                    logger.warning(f"JAX filtering and fitting failed: {e}. Using NumPy fallback.")
                    self._filter_and_fit_numpy()
                    self.fallback_usage = True
            else:
                self._filter_and_fit_numpy()
                
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
        if len(self.data) < 100:
            raise ValueError("Data must have at least 100 points for reliable periodogram analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        if np.std(self.data) == 0:
            raise ValueError("Data is constant, cannot estimate LRD")
            
    def _calculate_periodogram(self):
        """Calculate the periodogram of the data with caching."""
        # Check cache first
        cache_key = (len(self.data), self.window, self.nperseg, self.nfft, hash(self.data.tobytes()))
        if self.enable_caching and cache_key in self._periodogram_cache:
            self.frequencies, self.periodogram = self._periodogram_cache[cache_key]
            self._cache_hits += 1
            return
            
        try:
            # Set default parameters if not provided
            if self.nperseg is None:
                self.nperseg = min(256, len(self.data) // 4)
            if self.nfft is None:
                self.nfft = max(512, 2 * self.nperseg)
                
            # Calculate periodogram
            self.frequencies, self.periodogram = signal.periodogram(
                self.data,
                fs=1.0,
                window=self.window,
                nfft=self.nfft,
                scaling='density'
            )
            
            # Cache the result
            if self.enable_caching:
                self._periodogram_cache[cache_key] = (self.frequencies, self.periodogram)
                self._cache_misses += 1
                
        except Exception as e:
            logger.error(f"Periodogram calculation failed: {e}")
            raise
            
    def _filter_and_fit_jax(self):
        """Filter frequencies and fit scaling law using JAX."""
        try:
            # Convert to JAX arrays
            frequencies_jax = jnp.array(self.frequencies)
            periodogram_jax = jnp.array(self.periodogram)
            
            # Filter frequencies within the specified range
            freq_mask = (frequencies_jax >= self.frequency_range[0]) & \
                       (frequencies_jax <= self.frequency_range[1])
            
            if jnp.sum(freq_mask) < 10:
                self.hurst_exponent = np.nan
                self.scaling_error = np.nan
                self.beta = np.nan
                return
                
            freq_filtered = frequencies_jax[freq_mask]
            periodogram_filtered = periodogram_jax[freq_mask]
            
            # Remove zero and negative values
            positive_mask = periodogram_filtered > 0
            if jnp.sum(positive_mask) < 5:
                self.hurst_exponent = np.nan
                self.scaling_error = np.nan
                self.beta = np.nan
                return
                
            freq_positive = freq_filtered[positive_mask]
            periodogram_positive = periodogram_filtered[positive_mask]
            
            # Log-log regression: log(P(f)) = -β * log(f) + constant
            # For long-range dependent processes: β = 2H - 1
            log_freq = jnp.log(freq_positive)
            log_periodogram = jnp.log(periodogram_positive)
            
            try:
                # Use JAX linear regression
                jax_optimizer = JAXOptimizer()
                slope, intercept, r_value, p_value, std_err = jax_optimizer.fast_linregress(
                    log_freq, log_periodogram
                )
                
                beta = -float(slope)  # Negative slope gives β
                
                # Convert β to Hurst exponent: H = (β + 1) / 2
                self.hurst_exponent = (beta + 1) / 2
                self.scaling_error = 1 - float(r_value)**2
                self.beta = beta
                
                # Store additional information
                self.fitted_frequencies = np.array(freq_positive)
                self.fitted_periodogram = np.array(periodogram_positive)
                
            except Exception as e:
                logger.debug(f"JAX linear regression failed: {e}. Using NumPy fallback.")
                self._filter_and_fit_numpy()
                
        except Exception as e:
            logger.error(f"JAX filtering and fitting failed: {e}")
            raise
            
    def _filter_and_fit_numpy(self):
        """Filter frequencies and fit scaling law using NumPy."""
        # Filter frequencies within the specified range
        freq_mask = (self.frequencies >= self.frequency_range[0]) & \
                   (self.frequencies <= self.frequency_range[1])
        
        if np.sum(freq_mask) < 10:
            self.hurst_exponent = np.nan
            self.scaling_error = np.nan
            self.beta = np.nan
            return
            
        freq_filtered = self.frequencies[freq_mask]
        periodogram_filtered = self.periodogram[freq_mask]
        
        # Remove zero and negative values
        positive_mask = periodogram_filtered > 0
        if np.sum(positive_mask) < 5:
            self.hurst_exponent = np.nan
            self.scaling_error = np.nan
            self.beta = np.nan
            return
            
        freq_positive = freq_filtered[positive_mask]
        periodogram_positive = periodogram_filtered[positive_mask]
        
        # Log-log regression: log(P(f)) = -β * log(f) + constant
        # For long-range dependent processes: β = 2H - 1
        log_freq = np.log(freq_positive)
        log_periodogram = np.log(periodogram_positive)
        
        try:
            coeffs = np.polyfit(log_freq, log_periodogram, 1)
            beta = -coeffs[0]  # Negative slope gives β
            
            # Convert β to Hurst exponent: H = (β + 1) / 2
            self.hurst_exponent = (beta + 1) / 2
            
            # Calculate R-squared
            y_pred = np.polyval(coeffs, log_freq)
            ss_res = np.sum((log_periodogram - y_pred) ** 2)
            ss_tot = np.sum((log_periodogram - np.mean(log_periodogram)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.scaling_error = 1 - r_squared
            
            # Store additional information
            self.beta = beta
            self.fitted_frequencies = freq_positive
            self.fitted_periodogram = periodogram_positive
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Scaling law fitting failed: {e}")
            self.hurst_exponent = np.nan
            self.scaling_error = np.nan
            self.beta = np.nan
            
    def _calculate_confidence_interval(self):
        """Calculate confidence interval for the Hurst exponent."""
        if np.isnan(self.hurst_exponent):
            self.confidence_interval = (np.nan, np.nan)
            return
            
        # Get filtered data for error estimation
        if not hasattr(self, 'fitted_frequencies'):
            self.confidence_interval = (np.nan, np.nan)
            return
            
        freq_valid = self.fitted_frequencies
        periodogram_valid = self.fitted_periodogram
        
        # Bootstrap confidence interval
        hurst_bootstrap = []
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(freq_valid), size=len(freq_valid), replace=True)
            freq_boot = freq_valid[indices]
            periodogram_boot = periodogram_valid[indices]
            
            try:
                log_freq = np.log(freq_boot)
                log_periodogram = np.log(periodogram_boot)
                coeffs = np.polyfit(log_freq, log_periodogram, 1)
                beta = -coeffs[0]
                hurst_bootstrap.append((beta + 1) / 2)
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
            'frequencies': self.frequencies,
            'periodogram': self.periodogram,
            'beta': self.beta,
            'scaling_error': self.scaling_error,
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'fitted_frequencies': getattr(self, 'fitted_frequencies', None),
            'fitted_periodogram': getattr(self, 'fitted_periodogram', None),
            'parameters': {
                'window': self.window,
                'nperseg': self.nperseg,
                'nfft': self.nfft,
                'frequency_range': self.frequency_range
            },
            'performance': {
                'execution_time': self.execution_time,
                'memory_usage': self.memory_usage,
                'jax_usage': self.jax_usage,
                'fallback_usage': self.fallback_usage
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
                'reliability': 1 - getattr(self, 'scaling_error', 1.0),
                'spectral_exponent': self.beta
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
        self.hurst_exponent = None
        self.beta = None
        self.scaling_error = None
        self.confidence_interval = None
        self.jax_usage = False
        self.fallback_usage = False
        
        if self.enable_caching:
            self._periodogram_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
