"""
High-Performance Wavelet Log-Variance Estimator

This module provides an optimized implementation of the wavelet log-variance method
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

try:
    from .base import BaseEstimator
    from ..utils.jax_utils import JAXOptimizer
except ImportError:
    from src.estimators.base import BaseEstimator
    from src.utils.jax_utils import JAXOptimizer

logger = logging.getLogger(__name__)

# Configure JAX for optimal performance
jax.config.update('jax_enable_x64', True)


class HighPerformanceWaveletLogVarianceEstimator(BaseEstimator):
    """
    High-performance wavelet log-variance estimator for long-range dependence.
    
    This estimator uses JAX acceleration for core computations with robust
    NumPy fallbacks for reliability. It includes performance optimizations
    such as vectorized operations, caching, and memory management.
    
    The wavelet log-variance method estimates long-range dependence by analyzing
    the scaling behavior of wavelet coefficient variances across different scales.
    
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
    
    def __init__(self, name: str = "HighPerformanceWaveletLogVariance", **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Core parameters
        self.wavelet = kwargs.get('wavelet', 'db4')
        self.num_scales = kwargs.get('num_scales', 20)
        self.min_scale = kwargs.get('min_scale', 2)
        self.max_scale = kwargs.get('max_scale', None)
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
        self.wavelet_variances = None
        self.hurst_exponent = None
        self.alpha = None
        self.scaling_error = None
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
            self._variance_cache = {}
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
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
            
    def fit(self, data: np.ndarray, **kwargs) -> 'HighPerformanceWaveletLogVarianceEstimator':
        """
        Fit the wavelet log-variance estimator to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        self : HighPerformanceWaveletLogVarianceEstimator
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
        Estimate long-range dependence using wavelet log-variance method.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Input time series data (if not provided, uses fitted data)
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing wavelet log-variance estimation results
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
            
            # Calculate wavelet variances
            if self.use_jax:
                try:
                    self.wavelet_variances = self._calculate_wavelet_variances_jax()
                    self.jax_usage = True
                except Exception as e:
                    logger.warning(f"JAX wavelet variance calculation failed: {e}. Using NumPy fallback.")
                    self.wavelet_variances = self._calculate_wavelet_variances_numpy()
                    self.fallback_usage = True
            else:
                self.wavelet_variances = self._calculate_wavelet_variances_numpy()
                
            # Fit scaling law to get Hurst exponent
            if self.use_jax:
                try:
                    hurst_exponent, r_squared, std_error = self._fit_scaling_law_jax()
                    self.jax_usage = True
                except Exception as e:
                    logger.warning(f"JAX scaling law fitting failed: {e}. Using NumPy fallback.")
                    hurst_exponent, r_squared, std_error = self._fit_scaling_law_numpy()
                    self.fallback_usage = True
            else:
                hurst_exponent, r_squared, std_error = self._fit_scaling_law_numpy()
                
            self.hurst_exponent = hurst_exponent
            self.scaling_error = 1 - r_squared
            
            # Calculate alpha (long-range dependence parameter)
            self.alpha = 2 * hurst_exponent - 1
            
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
            
    def _calculate_wavelet_variances_jax(self) -> Dict[int, float]:
        """Calculate wavelet variances using JAX."""
        try:
            variances = {}
            
            for scale in self.scales:
                if scale not in self.wavelet_coeffs:
                    continue
                    
                # Calculate variance for this scale
                scale_variances = []
                for detail_coeffs in self.wavelet_coeffs[scale]:
                    if len(detail_coeffs) > 0:
                        # Convert to JAX array for calculation
                        coeffs_jax = jnp.array(detail_coeffs)
                        
                        # Calculate variance using JAX
                        variance_value = float(jnp.var(coeffs_jax))
                        scale_variances.append(variance_value)
                        
                if scale_variances:
                    # Store the mean of variances for this scale
                    variances[scale] = float(jnp.mean(jnp.array(scale_variances)))
                    
            return variances
            
        except Exception as e:
            logger.error(f"JAX wavelet variance calculation failed: {e}")
            raise
            
    def _calculate_wavelet_variances_numpy(self) -> Dict[int, float]:
        """Calculate wavelet variances using NumPy."""
        variances = {}
        
        for scale in self.scales:
            if scale not in self.wavelet_coeffs:
                continue
                
            # Calculate variance for this scale
            scale_variances = []
            for detail_coeffs in self.wavelet_coeffs[scale]:
                if len(detail_coeffs) > 0:
                    # Calculate variance using NumPy
                    variance_value = np.var(detail_coeffs)
                    scale_variances.append(variance_value)
                    
            if scale_variances:
                # Store the mean of variances for this scale
                variances[scale] = np.mean(scale_variances)
                
        return variances
        
    def _fit_scaling_law_jax(self) -> Tuple[float, float, float]:
        """Fit scaling law using JAX to extract Hurst exponent."""
        try:
            # Get valid scales and variances
            valid_scales = []
            valid_variances = []
            
            for scale in self.scales:
                if scale in self.wavelet_variances and not np.isnan(self.wavelet_variances[scale]):
                    valid_scales.append(scale)
                    valid_variances.append(self.wavelet_variances[scale])
                    
            if len(valid_scales) < 3:
                return np.nan, 0.0, np.nan
                
            # Convert to JAX arrays
            scales_jax = jnp.array(valid_scales)
            variances_jax = jnp.array(valid_variances)
            
            # Log-log regression: log(V(s)) = (2H - 1) * log(s) + constant
            # where V(s) is the wavelet variance at scale s
            log_scales = jnp.log(scales_jax)
            log_variances = jnp.log(variances_jax)
            
            # Use JAX linear regression
            jax_optimizer = JAXOptimizer()
            slope, intercept, r_value, p_value, std_err = jax_optimizer.fast_linregress(
                log_scales, log_variances
            )
            
            # Extract Hurst exponent: H = (slope + 1) / 2
            hurst_exponent = (float(slope) + 1) / 2
            r_squared = float(r_value)**2
            std_error = float(std_err)
            
            return hurst_exponent, r_squared, std_error
            
        except Exception as e:
            logger.error(f"JAX scaling law fitting failed: {e}")
            raise
            
    def _fit_scaling_law_numpy(self) -> Tuple[float, float, float]:
        """Fit scaling law using NumPy to extract Hurst exponent."""
        # Get valid scales and variances
        valid_scales = []
        valid_variances = []
        
        for scale in self.scales:
            if scale in self.wavelet_variances and not np.isnan(self.wavelet_variances[scale]):
                valid_scales.append(scale)
                valid_variances.append(self.wavelet_variances[scale])
                
        if len(valid_scales) < 3:
            return np.nan, 0.0, np.nan
            
        # Log-log regression: log(V(s)) = (2H - 1) * log(s) + constant
        # where V(s) is the wavelet variance at scale s
        log_scales = np.log(valid_scales)
        log_variances = np.log(valid_variances)
        
        try:
            coeffs = np.polyfit(log_scales, log_variances, 1)
            slope = coeffs[0]
            
            # Extract Hurst exponent: H = (slope + 1) / 2
            hurst_exponent = (slope + 1) / 2
            
            # Calculate R-squared
            y_pred = np.polyval(coeffs, log_scales)
            ss_res = np.sum((log_variances - y_pred) ** 2)
            ss_tot = np.sum((log_variances - np.mean(log_variances)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate standard error
            std_error = np.sqrt(ss_res / (len(valid_scales) - 2))
            
            return hurst_exponent, r_squared, std_error
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Scaling law fitting failed: {e}")
            return np.nan, 0.0, np.nan
            
    def _calculate_confidence_interval(self):
        """Calculate confidence interval for the Hurst exponent."""
        if np.isnan(self.hurst_exponent):
            self.confidence_interval = (np.nan, np.nan)
            return
            
        # Get valid scales and variances for error estimation
        valid_scales = []
        valid_variances = []
        
        for scale in self.scales:
            if scale in self.wavelet_variances and not np.isnan(self.wavelet_variances[scale]):
                valid_scales.append(scale)
                valid_variances.append(self.wavelet_variances[scale])
                
        if len(valid_scales) < 3:
            self.confidence_interval = (np.nan, np.nan)
            return
            
        # Bootstrap confidence interval
        hurst_bootstrap = []
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(valid_scales), size=len(valid_scales), replace=True)
            scales_boot = [valid_scales[i] for i in indices]
            variances_boot = [valid_variances[i] for i in indices]
            
            try:
                log_scales = np.log(scales_boot)
                log_variances = np.log(variances_boot)
                coeffs = np.polyfit(log_scales, log_variances, 1)
                slope = coeffs[0]
                hurst_bootstrap.append((slope + 1) / 2)
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
            'alpha': self.alpha,
            'scales': self.scales,
            'wavelet_coeffs': self.wavelet_coeffs,
            'wavelet_variances': self.wavelet_variances,
            'scaling_error': self.scaling_error,
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'parameters': {
                'wavelet': self.wavelet,
                'num_scales': self.num_scales,
                'min_scale': self.min_scale,
                'max_scale': self.max_scale
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
                'wavelet_method': f"{self.wavelet} wavelet log-variance",
                'method_description': "Wavelet variance scaling analysis for LRD estimation"
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
            'cache_size': len(self._wavelet_cache) + len(self._scale_cache) + len(self._variance_cache)
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
        self.wavelet_variances = None
        self.hurst_exponent = None
        self.alpha = None
        self.scaling_error = None
        self.confidence_interval = None
        self.jax_usage = False
        self.fallback_usage = False
        
        if self.enable_caching:
            self._wavelet_cache.clear()
            self._scale_cache.clear()
            self._variance_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
