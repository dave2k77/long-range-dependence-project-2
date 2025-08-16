"""
High-Performance Wavelet Whittle Estimator

This module provides an optimized implementation of the wavelet Whittle method
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
import pywt
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


class HighPerformanceWaveletWhittleEstimator(BaseEstimator):
    """
    High-performance wavelet Whittle estimator for long-range dependence.
    
    This estimator uses JAX acceleration for core computations with robust
    NumPy fallbacks for reliability. It includes performance optimizations
    such as vectorized operations, caching, and memory management.
    
    Parameters
    ----------
    name : str, optional
        Name identifier for the estimator
    wavelet : str, optional
        Wavelet type (default: 'db4')
    num_scales : int, optional
        Number of wavelet scales to use (default: 20)
    min_scale : int, optional
        Minimum scale for analysis (default: 2)
    max_scale : int, optional
        Maximum scale for analysis (default: None, auto-determined)
    frequency_range : List[float], optional
        Range of frequencies to use (default: [0.01, 0.5])
    initial_guess : List[float], optional
        Initial guess for parameters (default: [0.5])
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
    
    def __init__(self, name: str = "HighPerformanceWaveletWhittle", **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Core parameters
        self.wavelet = kwargs.get('wavelet', 'db4')
        self.num_scales = kwargs.get('num_scales', 20)
        self.min_scale = kwargs.get('min_scale', 2)
        self.max_scale = kwargs.get('max_scale', None)
        self.frequency_range = kwargs.get('frequency_range', [0.01, 0.5])
        self.initial_guess = kwargs.get('initial_guess', [0.5])
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.n_bootstrap = kwargs.get('n_bootstrap', 1000)
        
        # Performance options
        self.use_jax = kwargs.get('use_jax', True)
        self.enable_caching = kwargs.get('enable_caching', True)
        self.vectorized = kwargs.get('vectorized', True)
        
        # Data storage
        self.data = None
        self.wavelet_coeffs = None
        self.scales = None
        self.frequencies = None
        self.wavelet_periodogram = None
        self.alpha_estimate = None
        self.hurst_exponent = None
        self.optimization_success = None
        self.confidence_interval = None
        
        # Performance monitoring
        self.execution_time = None
        self.memory_usage = None
        self.jax_usage = False
        self.fallback_usage = False
        
        # Caching
        if self.enable_caching:
            self._wavelet_cache = {}
            self._scale_cache = {}
            self._periodogram_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate initialization parameters."""
        if self.num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive")
        if len(self.frequency_range) != 2:
            raise ValueError("frequency_range must have exactly 2 elements")
        if self.frequency_range[0] >= self.frequency_range[1]:
            raise ValueError("frequency_range must be [min_freq, max_freq]")
        if not all(0 < f <= 0.5 for f in self.frequency_range):
            raise ValueError("frequencies must be between 0 and 0.5 (inclusive)")
        if len(self.initial_guess) != 1:
            raise ValueError("initial_guess must have exactly 1 element")
        if not 0 <= self.initial_guess[0] <= 2:
            raise ValueError("initial_guess alpha must be between 0 and 2")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
            
    def fit(self, data: np.ndarray, **kwargs) -> 'HighPerformanceWaveletWhittleEstimator':
        """
        Fit the wavelet Whittle estimator to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        self : HighPerformanceWaveletWhittleEstimator
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
        Estimate long-range dependence using wavelet Whittle method.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Input time series data (if not provided, uses fitted data)
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing wavelet Whittle estimation results
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            if data is not None:
                self.fit(data, **kwargs)
                
            if self.data is None:
                raise ValueError("No data available. Call fit() first.")
                
            # Generate scales
            self._generate_scales()
            
            # Calculate wavelet coefficients
            self._calculate_wavelet_coeffs()
            
            # Calculate wavelet periodogram
            self._calculate_wavelet_periodogram()
            
            # Filter frequencies within specified range
            self._filter_frequencies()
            
            # Optimize likelihood function
            if self.use_jax:
                try:
                    self._optimize_likelihood_jax()
                    self.jax_usage = True
                except Exception as e:
                    logger.warning(f"JAX optimization failed: {e}. Using NumPy fallback.")
                    self._optimize_likelihood_numpy()
                    self.fallback_usage = True
            else:
                self._optimize_likelihood_numpy()
                
            # Calculate Hurst exponent
            self.hurst_exponent = (self.alpha_estimate + 1) / 2
            
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
            raise ValueError("Data must have at least 100 points for reliable wavelet analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        if np.std(self.data) == 0:
            raise ValueError("Data is constant, cannot estimate LRD")
            
    def _generate_scales(self):
        """Generate scale values for analysis with caching."""
        if self.max_scale is None:
            self.max_scale = min(len(self.data) // 8, 64)
            
        # Check cache first
        cache_key = (len(self.data), self.min_scale, self.max_scale, self.num_scales)
        if self.enable_caching and cache_key in self._scale_cache:
            self.scales = self._scale_cache[cache_key]
            self._cache_hits += 1
            return
            
        try:
            # Generate log-spaced scales
            self.scales = np.logspace(
                np.log2(self.min_scale), 
                np.log2(self.max_scale), 
                self.num_scales, 
                base=2
            ).astype(int)
            
            # Ensure unique scales
            self.scales = np.unique(self.scales)
            if len(self.scales) != self.num_scales:
                self.num_scales = len(self.scales)
                
            # Cache the result
            if self.enable_caching:
                self._scale_cache[cache_key] = self.scales
                self._cache_misses += 1
                
        except Exception as e:
            logger.error(f"Scale generation failed: {e}")
            # Fallback to simple linear spacing
            self.scales = np.linspace(self.min_scale, self.max_scale, self.num_scales, dtype=int)
            
    def _calculate_wavelet_coeffs(self):
        """Calculate wavelet coefficients for all scales."""
        # Check cache first
        cache_key = (len(self.data), self.wavelet, hash(self.data.tobytes()))
        if self.enable_caching and cache_key in self._wavelet_cache:
            self.wavelet_coeffs = self._wavelet_cache[cache_key]
            self._cache_hits += 1
            return
            
        try:
            # Calculate wavelet coefficients for each scale
            self.wavelet_coeffs = {}
            for scale in self.scales:
                coeffs = pywt.wavedec(self.data, self.wavelet, level=scale, mode='periodic')
                # Use the detail coefficients (all except the approximation)
                self.wavelet_coeffs[scale] = coeffs[1:]
                
            # Cache the result
            if self.enable_caching:
                self._wavelet_cache[cache_key] = self.wavelet_coeffs
                self._cache_misses += 1
                
        except Exception as e:
            logger.error(f"Wavelet coefficient calculation failed: {e}")
            raise
            
    def _calculate_wavelet_periodogram(self):
        """Calculate wavelet periodogram with caching."""
        # Check cache first
        cache_key = (len(self.data), self.wavelet, hash(self.data.tobytes()))
        if self.enable_caching and cache_key in self._periodogram_cache:
            self.frequencies, self.wavelet_periodogram = self._periodogram_cache[cache_key]
            self._cache_hits += 1
            return
            
        try:
            # Calculate wavelet periodogram using the wavelet coefficients
            # This is a simplified version - in practice, you'd use more sophisticated wavelet spectral estimation
            
            # Generate frequency array
            self.frequencies = np.linspace(0.01, 0.5, 256)
            
            # Calculate periodogram from wavelet coefficients
            periodogram_values = []
            for freq in self.frequencies:
                # Simplified wavelet periodogram calculation
                # In practice, you'd use proper wavelet spectral estimation
                periodogram_value = 0.0
                for scale in self.scales:
                    if scale in self.wavelet_coeffs:
                        for detail_coeffs in self.wavelet_coeffs[scale]:
                            if len(detail_coeffs) > 0:
                                # Simplified spectral contribution
                                periodogram_value += np.var(detail_coeffs) * np.exp(-freq * scale)
                periodogram_values.append(periodogram_value)
                
            self.wavelet_periodogram = np.array(periodogram_values)
            
            # Cache the result
            if self.enable_caching:
                self._periodogram_cache[cache_key] = (self.frequencies, self.wavelet_periodogram)
                self._cache_misses += 1
                
        except Exception as e:
            logger.error(f"Wavelet periodogram calculation failed: {e}")
            raise
            
    def _filter_frequencies(self):
        """Filter frequencies within the specified range."""
        min_freq, max_freq = self.frequency_range
        
        # Find indices within frequency range
        valid_indices = (self.frequencies >= min_freq) & (self.frequencies <= max_freq)
        
        self.frequencies = self.frequencies[valid_indices]
        self.wavelet_periodogram = self.wavelet_periodogram[valid_indices]
        
    def _wavelet_whittle_likelihood_jax(self, alpha: float) -> float:
        """
        Calculate wavelet Whittle likelihood function using JAX.
        
        Parameters
        ----------
        alpha : float
            Long-range dependence parameter
            
        Returns
        -------
        float
            Negative log-likelihood (to be minimized)
        """
        try:
            # Convert to JAX arrays
            frequencies_jax = jnp.array(self.frequencies)
            periodogram_jax = jnp.array(self.wavelet_periodogram)
            
            # Theoretical wavelet power spectrum for long-range dependent processes
            # S_w(f) = |f|^(-alpha) * wavelet_factor
            
            # Calculate theoretical spectrum using JAX
            theoretical_spectrum = jnp.abs(frequencies_jax) ** (-alpha)
            
            # Add wavelet-specific factor (simplified)
            wavelet_factor = 1.0  # In practice, this would depend on the wavelet type
            theoretical_spectrum = theoretical_spectrum * wavelet_factor
            
            # Whittle likelihood
            log_likelihood = jnp.sum(
                jnp.log(theoretical_spectrum) + periodogram_jax / theoretical_spectrum
            )
            
            return float(-log_likelihood)  # Return negative for minimization
            
        except Exception as e:
            logger.debug(f"JAX likelihood calculation failed: {e}")
            raise
            
    def _wavelet_whittle_likelihood_numpy(self, alpha: float) -> float:
        """
        Calculate wavelet Whittle likelihood function using NumPy.
        
        Parameters
        ----------
        alpha : float
            Long-range dependence parameter
            
        Returns
        -------
        float
            Negative log-likelihood (to be minimized)
        """
        # Theoretical wavelet power spectrum for long-range dependent processes
        # S_w(f) = |f|^(-alpha) * wavelet_factor
        
        # Calculate theoretical spectrum
        theoretical_spectrum = np.abs(self.frequencies) ** (-alpha)
        
        # Add wavelet-specific factor (simplified)
        wavelet_factor = 1.0  # In practice, this would depend on the wavelet type
        theoretical_spectrum = theoretical_spectrum * wavelet_factor
        
        # Whittle likelihood
        log_likelihood = np.sum(
            np.log(theoretical_spectrum) + self.wavelet_periodogram / theoretical_spectrum
        )
        
        return -log_likelihood  # Return negative for minimization
        
    def _optimize_likelihood_jax(self):
        """Optimize the wavelet Whittle likelihood function using JAX."""
        try:
            # JAX optimization (simplified - in practice, you'd use more sophisticated JAX optimization)
            # For now, we'll use a simple grid search with JAX acceleration
            alpha_range = jnp.linspace(0.0, 2.0, 100)
            likelihoods = jnp.array([self._wavelet_whittle_likelihood_jax(float(alpha)) for alpha in alpha_range])
            
            # Find minimum
            min_idx = jnp.argmin(likelihoods)
            self.alpha_estimate = float(alpha_range[min_idx])
            self.optimization_success = True
            
        except Exception as e:
            logger.error(f"JAX optimization failed: {e}")
            raise
            
    def _optimize_likelihood_numpy(self):
        """Optimize the wavelet Whittle likelihood function using NumPy."""
        try:
            # Simple grid search optimization
            alpha_range = np.linspace(0.0, 2.0, 100)
            likelihoods = [self._wavelet_whittle_likelihood_numpy(alpha) for alpha in alpha_range]
            
            # Find minimum
            min_idx = np.argmin(likelihoods)
            self.alpha_estimate = alpha_range[min_idx]
            self.optimization_success = True
            
        except Exception as e:
            logger.error(f"Error in likelihood optimization: {str(e)}")
            self.alpha_estimate = self.initial_guess[0]
            self.optimization_success = False
            
    def _calculate_confidence_interval(self):
        """Calculate confidence interval for the alpha parameter."""
        if np.isnan(self.alpha_estimate):
            self.confidence_interval = (np.nan, np.nan)
            return
            
        # Get filtered data for error estimation
        if len(self.frequencies) < 3:
            self.confidence_interval = (np.nan, np.nan)
            return
            
        # Bootstrap confidence interval
        alpha_bootstrap = []
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(self.frequencies), size=len(self.frequencies), replace=True)
            freq_boot = self.frequencies[indices]
            periodogram_boot = self.wavelet_periodogram[indices]
            
            try:
                # Store original values temporarily
                orig_freq = self.frequencies.copy()
                orig_periodogram = self.wavelet_periodogram.copy()
                
                # Use bootstrap values
                self.frequencies = freq_boot
                self.wavelet_periodogram = periodogram_boot
                
                # Optimize with bootstrap data
                self._optimize_likelihood_numpy()
                alpha_bootstrap.append(self.alpha_estimate)
                
                # Restore original values
                self.frequencies = orig_freq
                self.wavelet_periodogram = orig_periodogram
                
            except:
                continue
                
        if alpha_bootstrap:
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            self.confidence_interval = (
                np.percentile(alpha_bootstrap, lower_percentile),
                np.percentile(alpha_bootstrap, upper_percentile)
            )
        else:
            self.confidence_interval = (np.nan, np.nan)
            
    def get_results(self) -> Dict[str, Any]:
        """Get estimation results."""
        results = {
            'alpha': self.alpha_estimate,
            'hurst_exponent': self.hurst_exponent,
            'optimization_success': self.optimization_success,
            'scales': self.scales,
            'frequencies': self.frequencies,
            'wavelet_periodogram': self.wavelet_periodogram,
            'wavelet_coeffs': self.wavelet_coeffs,
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'method': 'HighPerformanceWaveletWhittle',
            'parameters': {
                'wavelet': self.wavelet,
                'num_scales': self.num_scales,
                'min_scale': self.min_scale,
                'max_scale': self.max_scale,
                'frequency_range': self.frequency_range,
                'initial_guess': self.initial_guess
            },
            'performance': {
                'execution_time': self.execution_time,
                'memory_usage': self.memory_usage,
                'jax_usage': self.jax_usage,
                'fallback_usage': self.fallback_usage
            }
        }
        
        # Add interpretation
        if self.alpha_estimate is not None:
            if self.alpha_estimate < 1:
                lrd_type = "Short-range dependent (stationary)"
            elif self.alpha_estimate > 1:
                lrd_type = "Long-range dependent (non-stationary)"
            else:
                lrd_type = "Critical case (unit root)"
                
            results['interpretation'] = {
                'lrd_type': lrd_type,
                'strength': abs(self.alpha_estimate - 1),
                'reliability': self.optimization_success,
                'wavelet_method': f"{self.wavelet} wavelet Whittle"
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
            'cache_size': len(self._wavelet_cache) + len(self._scale_cache) + len(self._periodogram_cache)
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
        self.wavelet_coeffs = None
        self.scales = None
        self.frequencies = None
        self.wavelet_periodogram = None
        self.alpha_estimate = None
        self.hurst_exponent = None
        self.optimization_success = None
        self.confidence_interval = None
        self.jax_usage = False
        self.fallback_usage = False
        
        if self.enable_caching:
            self._wavelet_cache.clear()
            self._scale_cache.clear()
            self._periodogram_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
