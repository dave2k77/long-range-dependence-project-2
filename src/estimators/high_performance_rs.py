"""
High-Performance Rescaled Range (R/S) Analysis Estimator

This module provides an optimized implementation of the R/S analysis method
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

try:
    from .base import BaseEstimator
    from ..utils.jax_utils import JAXOptimizer
except ImportError:
    from src.estimators.base import BaseEstimator
    from src.utils.jax_utils import JAXOptimizer

logger = logging.getLogger(__name__)

# Configure JAX for optimal performance
jax.config.update('jax_enable_x64', True)


class HighPerformanceRSEstimator(BaseEstimator):
    """
    High-performance Rescaled Range (R/S) Analysis estimator.
    
    This estimator uses JAX acceleration for core computations with robust
    NumPy fallbacks for reliability. It includes performance optimizations
    such as vectorized operations, caching, and memory management.
    
    Parameters
    ----------
    name : str, optional
        Name identifier for the estimator
    min_scale : int, optional
        Minimum scale for analysis (default: 4)
    max_scale : int, optional
        Maximum scale for analysis (default: len(data)//4)
    num_scales : int, optional
        Number of scales to analyze (default: 20)
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
    
    def __init__(self, name: str = "HighPerformanceR/S", **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Core parameters
        self.min_scale = kwargs.get('min_scale', 4)
        self.max_scale = kwargs.get('max_scale', None)
        self.num_scales = kwargs.get('num_scales', 20)
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.n_bootstrap = kwargs.get('n_bootstrap', 1000)
        
        # Performance options
        self.use_jax = kwargs.get('use_jax', True)
        self.enable_caching = kwargs.get('enable_caching', True)
        self.vectorized = kwargs.get('vectorized', True)
        
        # Data storage
        self.data = None
        self.scales = None
        self.rs_values = None
        self.rs_std = None
        self.hurst_exponent = None
        self.scaling_error = None
        self.confidence_interval = None
        
        # Performance monitoring
        self.execution_time = None
        self.memory_usage = None
        self.jax_usage = False
        self.fallback_usage = False
        
        # Caching
        if self.enable_caching:
            self._scale_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate initialization parameters."""
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive")
        if self.num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
            
    def fit(self, data: np.ndarray, **kwargs) -> 'HighPerformanceRSEstimator':
        """
        Fit the R/S estimator to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        self : HighPerformanceRSEstimator
            Fitted estimator instance
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            self.data = np.asarray(data, dtype=float)
            self._validate_data()
            self._generate_scales()
            
            self.execution_time = time.time() - start_time
            self.memory_usage = psutil.Process().memory_info().rss - start_memory
            
            return self
            
        except Exception as e:
            logger.error(f"Error in fit for {self.name}: {str(e)}")
            raise
            
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Estimate long-range dependence using R/S analysis.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Input time series data (if not provided, uses fitted data)
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing R/S estimation results
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            if data is not None:
                self.fit(data, **kwargs)
                
            if self.data is None:
                raise ValueError("No data available. Call fit() first.")
                
            # Calculate R/S values for all scales
            if self.use_jax and self.vectorized:
                try:
                    self._calculate_rs_values_jax_vectorized()
                    self.jax_usage = True
                except Exception as e:
                    logger.warning(f"JAX vectorized R/S calculation failed: {e}. Using NumPy fallback.")
                    self._calculate_rs_values_numpy_vectorized()
                    self.fallback_usage = True
            elif self.use_jax:
                try:
                    self._calculate_rs_values_jax()
                    self.jax_usage = True
                except Exception as e:
                    logger.warning(f"JAX R/S calculation failed: {e}. Using NumPy fallback.")
                    self._calculate_rs_values_numpy()
                    self.fallback_usage = True
            else:
                self._calculate_rs_values_numpy()
                
            # Fit scaling law to extract Hurst exponent
            self._fit_scaling_law()
            
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
            raise ValueError("Data must have at least 100 points for reliable R/S analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        if np.std(self.data) == 0:
            raise ValueError("Data is constant, cannot estimate LRD")
            
    def _generate_scales(self):
        """Generate scale values for analysis with caching."""
        if self.max_scale is None:
            self.max_scale = len(self.data) // 4
            
        # Check cache first
        cache_key = (len(self.data), self.min_scale, self.max_scale, self.num_scales)
        if self.enable_caching and cache_key in self._scale_cache:
            self.scales = self._scale_cache[cache_key]
            self._cache_hits += 1
            return
            
        try:
            # Try JAX logspace first
            if self.use_jax:
                try:
                    jax_optimizer = JAXOptimizer()
                    self.scales = jax_optimizer.fast_logspace(
                        np.log10(self.min_scale), 
                        np.log10(self.max_scale), 
                        self.num_scales
                    )
                    # Ensure unique scales
                    self.scales = np.unique(self.scales)
                    # Adjust num_scales if needed
                    if len(self.scales) != self.num_scales:
                        self.num_scales = len(self.scales)
                    return
                except Exception as e:
                    logger.debug(f"JAX scale generation failed: {e}. Using NumPy fallback.")
                    
            # NumPy fallback
            self.scales = np.logspace(
                np.log10(self.min_scale), 
                np.log10(self.max_scale), 
                self.num_scales, 
                dtype=int
            )
            self.scales = np.unique(self.scales)
            if len(self.scales) != self.num_scales:
                self.num_scales = len(self.scales)
                
        except Exception as e:
            logger.error(f"Scale generation failed: {e}")
            # Fallback to simple linear spacing
            self.scales = np.linspace(self.min_scale, self.max_scale, self.num_scales, dtype=int)
            
        # Cache the result
        if self.enable_caching:
            self._scale_cache[cache_key] = self.scales
            self._cache_misses += 1
            
    def _calculate_rs_values_jax_vectorized(self):
        """Calculate R/S values using JAX vectorized operations."""
        try:
            # Convert data to JAX array
            data_jax = jnp.array(self.data)
            scales_jax = jnp.array(self.scales)
            
            # Vectorized R/S calculation
            rs_values, rs_std = self._calculate_rs_jax_vectorized(data_jax, scales_jax)
            
            self.rs_values = np.array(rs_values)
            self.rs_std = np.array(rs_std)
            
        except Exception as e:
            logger.error(f"JAX vectorized R/S calculation failed: {e}")
            raise
            
    def _calculate_rs_values_jax(self):
        """Calculate R/S values using JAX (non-vectorized)."""
        try:
            # Convert data to JAX array
            data_jax = jnp.array(self.data)
            scales_jax = jnp.array(self.scales)
            
            # Calculate R/S for each scale
            rs_values = []
            rs_std_values = []
            
            for scale in scales_jax:
                rs_list = self._calculate_rs_for_scale_jax(data_jax, scale)
                if len(rs_list) > 0:
                    rs_values.append(jnp.mean(jnp.array(rs_list)))
                    rs_std_values.append(jnp.std(jnp.array(rs_list)))
                else:
                    rs_values.append(jnp.nan)
                    rs_std_values.append(jnp.nan)
                    
            self.rs_values = np.array(rs_values)
            self.rs_std = np.array(rs_std_values)
            
        except Exception as e:
            logger.error(f"JAX R/S calculation failed: {e}")
            raise
            
    def _calculate_rs_values_numpy_vectorized(self):
        """Calculate R/S values using vectorized NumPy operations."""
        self.rs_values = np.zeros(len(self.scales))
        self.rs_std = np.zeros(len(self.scales))
        
        # Vectorized processing
        for i, scale in enumerate(self.scales):
            rs_list = self._process_scale_numpy_vectorized(scale)
            if len(rs_list) > 0:
                self.rs_values[i] = np.mean(rs_list)
                self.rs_std[i] = np.std(rs_list)
            else:
                self.rs_values[i] = np.nan
                self.rs_std[i] = np.nan
                
    def _calculate_rs_values_numpy(self):
        """Calculate R/S values using standard NumPy operations."""
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
                    
            if len(rs_list) > 0:
                self.rs_values[i] = np.mean(rs_list)
                self.rs_std[i] = np.std(rs_list)
            else:
                self.rs_values[i] = np.nan
                self.rs_std[i] = np.nan
                
    def _process_scale_numpy_vectorized(self, scale: int) -> List[float]:
        """Process a single scale using vectorized NumPy operations."""
        n_segments = len(self.data) // scale
        if n_segments == 0:
            return []
            
        # Create all segments at once
        segments = np.array([
            self.data[i*scale:(i+1)*scale] 
            for i in range(n_segments)
        ])
        
        # Vectorized R/S calculation
        rs_values = []
        for segment in segments:
            rs = self._calculate_rs_for_segment(segment)
            if rs is not None:
                rs_values.append(rs)
                
        return rs_values
        
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
        
    def _calculate_rs_for_scale_jax(self, data: jnp.ndarray, scale: int) -> List[float]:
        """Calculate R/S values for a specific scale using JAX."""
        try:
            n_segments = len(data) // scale
            if n_segments == 0:
                return []
                
            rs_list = []
            for i in range(n_segments):
                start_idx = i * scale
                end_idx = start_idx + scale
                segment = data[start_idx:end_idx]
                
                # Calculate R/S using JAX operations
                cumsum = jnp.cumsum(segment - jnp.mean(segment))
                R = jnp.max(cumsum) - jnp.min(cumsum)
                S = jnp.std(segment)
                
                if S > 1e-10:
                    rs_list.append(float(R / S))
                    
            return rs_list
            
        except Exception as e:
            logger.debug(f"JAX R/S calculation for scale {scale} failed: {e}")
            return []
            
    def _calculate_rs_jax_vectorized(self, data: jnp.ndarray, scales: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Vectorized R/S calculation using JAX."""
        try:
            # This is a simplified vectorized version
            # In practice, we'd need more complex JAX operations for full vectorization
            rs_values = []
            rs_std_values = []
            
            for scale in scales:
                rs_list = self._calculate_rs_for_scale_jax(data, int(scale))
                if len(rs_list) > 0:
                    rs_values.append(jnp.mean(jnp.array(rs_list)))
                    rs_std_values.append(jnp.std(jnp.array(rs_list)))
                else:
                    rs_values.append(jnp.nan)
                    rs_std_values.append(jnp.nan)
                    
            return jnp.array(rs_values), jnp.array(rs_std_values)
            
        except Exception as e:
            logger.error(f"Vectorized JAX R/S calculation failed: {e}")
            raise
            
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
        
        try:
            # Use fast linear regression if available
            try:
                jax_optimizer = JAXOptimizer()
                slope, intercept, r_value, p_value, std_err = jax_optimizer.fast_linregress(
                    jnp.array(np.log(scales_valid)), jnp.array(np.log(rs_valid))
                )
                self.hurst_exponent = float(slope)
                self.scaling_error = 1 - float(r_value)**2
            except Exception as e:
                logger.debug(f"JAX linear regression failed: {e}. Using NumPy fallback.")
                # Standard polyfit
                coeffs = np.polyfit(np.log(scales_valid), np.log(rs_valid), 1)
                self.hurst_exponent = coeffs[0]
                
                # Calculate R-squared
                y_pred = np.polyval(coeffs, np.log(scales_valid))
                ss_res = np.sum((np.log(rs_valid) - y_pred) ** 2)
                ss_tot = np.sum((np.log(rs_valid) - np.mean(np.log(rs_valid))) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                self.scaling_error = 1 - r_squared
                
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Scaling law fitting failed: {e}")
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
        hurst_bootstrap = []
        
        for _ in range(self.n_bootstrap):
            # Add noise to R/S values based on their standard deviations
            rs_noisy = rs_valid + np.random.normal(0, rs_std_valid)
            rs_noisy = np.maximum(rs_noisy, 0.1)  # Ensure positive values
            
            try:
                try:
                    jax_optimizer = JAXOptimizer()
                    slope, _, _, _, _ = jax_optimizer.fast_linregress(
                        jnp.array(np.log(scales_valid)), jnp.array(np.log(rs_noisy))
                    )
                    hurst_bootstrap.append(float(slope))
                except Exception as e:
                    logger.debug(f"JAX bootstrap regression failed: {e}. Using NumPy fallback.")
                    coeffs = np.polyfit(np.log(scales_valid), np.log(rs_noisy), 1)
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
            'rs_std': self.rs_std,
            'scaling_error': self.scaling_error,
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'parameters': {
                'min_scale': self.min_scale,
                'max_scale': self.max_scale,
                'num_scales': self.num_scales,
                'n_bootstrap': self.n_bootstrap
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
                'reliability': 1 - getattr(self, 'scaling_error', 1.0)
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
            'cache_size': len(self._scale_cache)
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
        self.scales = None
        self.rs_values = None
        self.rs_std = None
        self.hurst_exponent = None
        self.scaling_error = None
        self.confidence_interval = None
        self.jax_usage = False
        self.fallback_usage = False
        
        if self.enable_caching:
            self._scale_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
