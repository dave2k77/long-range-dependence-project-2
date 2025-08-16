"""
High-Performance Long-Range Dependence Estimators

This module provides NUMBA and JAX optimized versions of the classical
LRD estimators for improved performance on CPU and GPU.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from .base import BaseEstimator

logger = logging.getLogger(__name__)

try:
    import numba
    from numba import jit as numba_jit, prange, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("NUMBA not available. High-performance variants will not work.")

try:
    import jax
    import jax.numpy as jnp
    from jax import jit as jax_jit, vmap, grad
    from jax.scipy.optimize import minimize
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    logger.warning("JAX not available. High-performance variants will not work.")


class HighPerformanceDFAEstimator(BaseEstimator):
    """
    High-performance DFA estimator using NUMBA JIT compilation.
    
    This estimator provides significant speedup over the standard DFA
    implementation through just-in-time compilation and parallelization.
    """
    
    def __init__(self, name: str = "HighPerformanceDFA", **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_scale = kwargs.get('min_scale', 4)
        self.max_scale = kwargs.get('max_scale', None)
        self.num_scales = kwargs.get('num_scales', 20)
        self.polynomial_order = kwargs.get('polynomial_order', 1)
        self.use_parallel = kwargs.get('use_parallel', True)
        self.data = None
        self.scales = None
        self.fluctuations = None
        
        if not NUMBA_AVAILABLE:
            raise ImportError("NUMBA is required for HighPerformanceDFAEstimator")
            
    def fit(self, data: np.ndarray, **kwargs) -> 'HighPerformanceDFAEstimator':
        """Fit the high-performance DFA estimator to the data."""
        self.data = np.asarray(data, dtype=np.float64)
        self._validate_data()
        self._generate_scales()
        return self
        
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """Estimate Hurst exponent using high-performance DFA."""
        if data is not None:
            self.fit(data, **kwargs)
            
        if self.data is None:
            raise ValueError("No data available. Call fit() first.")
            
        # Calculate fluctuations using NUMBA-optimized function
        self._calculate_fluctuations_optimized()
        
        # Fit power law to extract Hurst exponent
        self._fit_power_law_optimized()
        
        return self.get_results()
        
    def _validate_data(self):
        """Validate input data."""
        if len(self.data) < 100:
            raise ValueError("Data must have at least 100 points for reliable DFA analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
            
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
        self.scales = np.unique(self.scales)
        
    def _calculate_fluctuations_optimized(self):
        """Calculate fluctuations using NUMBA-optimized function."""
        if self.use_parallel:
            self.fluctuations = _calculate_fluctuations_numba_parallel(
                self.data, 
                self.scales, 
                self.polynomial_order
            )
        else:
            self.fluctuations = _calculate_fluctuations_numba_sequential(
                self.data, 
                self.scales, 
                self.polynomial_order
            )
        
    def _fit_power_law_optimized(self):
        """Fit power law using NUMBA-optimized function."""
        self.hurst_exponent, self.intercept, self.r_squared = _fit_power_law_numba(
            self.scales, 
            self.fluctuations
        )
        
    def get_results(self) -> Dict[str, Any]:
        """Get estimation results."""
        results = {
            'hurst_exponent': self.hurst_exponent,
            'scales': self.scales,
            'fluctuations': self.fluctuations,
            'intercept': getattr(self, 'intercept', None),
            'r_squared': getattr(self, 'r_squared', None),
            'parameters': {
                'min_scale': self.min_scale,
                'max_scale': self.max_scale,
                'num_scales': self.num_scales,
                'polynomial_order': self.polynomial_order,
                'use_parallel': self.use_parallel
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
                'reliability': getattr(self, 'r_squared', 0.0),
                'method': 'High-Performance DFA (NUMBA)'
            }
            
        return results


class HighPerformanceMFDFAEstimator(BaseEstimator):
    """
    High-performance MFDFA estimator using JAX for GPU acceleration.
    
    This estimator leverages JAX's automatic differentiation and GPU
    capabilities for efficient multifractal analysis.
    """
    
    def __init__(self, name: str = "HighPerformanceMFDFA", **kwargs):
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
        
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for HighPerformanceMFDFAEstimator")
            
    def fit(self, data: np.ndarray, **kwargs) -> 'HighPerformanceMFDFAEstimator':
        """Fit the high-performance MFDFA estimator to the data."""
        self.data = jnp.asarray(data, dtype=jnp.float32)
        self._validate_data()
        self._generate_scales()
        return self
        
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """Estimate multifractal properties using high-performance MFDFA."""
        if data is not None:
            self.fit(data, **kwargs)
            
        if self.data is None:
            raise ValueError("No data available. Call fit() first.")
            
        # Calculate fluctuation functions using JAX-optimized functions
        self._calculate_fluctuations_jax()
        
        # Fit scaling laws using JAX optimization
        self._fit_scaling_laws_jax()
        
        # Calculate multifractal spectrum
        self._calculate_multifractal_spectrum_jax()
        
        return self.get_results()
        
    def _validate_data(self):
        """Validate input data."""
        if len(self.data) < 100:
            raise ValueError("Data must have at least 100 points for reliable MFDFA analysis")
        if jnp.any(jnp.isnan(self.data)) or jnp.any(jnp.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
            
    def _generate_scales(self):
        """Generate scale values for analysis."""
        if self.max_scale is None:
            self.max_scale = len(self.data) // 4
            
        self.scales = jnp.logspace(
            jnp.log10(self.min_scale), 
            jnp.log10(self.max_scale), 
            self.num_scales, 
            dtype=jnp.int32
        )
        self.scales = jnp.unique(self.scales)
        
    def _calculate_fluctuations_jax(self):
        """Calculate fluctuation functions using JAX vectorization."""
        # Vectorized calculation across scales and q values
        self.fluctuations = _calculate_fluctuations_jax_vectorized(
            self.data, 
            self.scales, 
            self.q_values, 
            self.polynomial_order
        )
        
    def _fit_scaling_laws_jax(self):
        """Fit scaling laws using JAX optimization."""
        self.hurst_exponents = _fit_scaling_laws_jax_vectorized(
            self.scales, 
            self.fluctuations, 
            self.q_values
        )
        
    def _calculate_multifractal_spectrum_jax(self):
        """Calculate multifractal spectrum using JAX."""
        if self.hurst_exponents is not None:
            self.multifractal_spectrum = _calculate_multifractal_spectrum_jax(
                self.q_values, 
                self.hurst_exponents
            )
        
    def get_results(self) -> Dict[str, Any]:
        """Get estimation results."""
        # Convert JAX arrays to numpy for compatibility
        results = {
            'hurst_exponents': np.array(self.hurst_exponents) if self.hurst_exponents is not None else None,
            'q_values': np.array(self.q_values),
            'scales': np.array(self.scales),
            'fluctuations': np.array(self.fluctuations) if self.fluctuations is not None else None,
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
            valid_h = self.hurst_exponents[~jnp.isnan(self.hurst_exponents)]
            if len(valid_h) > 0:
                results['summary'] = {
                    'mean_hurst': float(jnp.mean(valid_h)),
                    'std_hurst': float(jnp.std(valid_h)),
                    'min_hurst': float(jnp.min(valid_h)),
                    'max_hurst': float(jnp.max(valid_h)),
                    'is_multifractal': float(jnp.std(valid_h)) > 0.05
                }
                
        return results


# NUMBA-optimized functions
if NUMBA_AVAILABLE:
    @numba_jit
    def _calculate_fluctuations_numba_parallel(data: np.ndarray, scales: np.ndarray,
                                               polynomial_order: int) -> np.ndarray:
        """NUMBA-optimized parallel fluctuation calculation."""
        n_scales = len(scales)
        fluctuations = np.zeros(n_scales)
        
        for i in prange(n_scales):
            fluctuations[i] = _calculate_fluctuation_at_scale_numba(
                data, scales[i], polynomial_order
            )
                
        return fluctuations
        
    @numba_jit
    def _calculate_fluctuations_numba_sequential(data: np.ndarray, scales: np.ndarray,
                                                 polynomial_order: int) -> np.ndarray:
        """NUMBA-optimized sequential fluctuation calculation."""
        n_scales = len(scales)
        fluctuations = np.zeros(n_scales)
        
        for i in range(n_scales):
            fluctuations[i] = _calculate_fluctuation_at_scale_numba(
                data, scales[i], polynomial_order
            )
                
        return fluctuations
        
    @numba_jit
    def _calculate_fluctuation_at_scale_numba(data: np.ndarray, scale: int,
                                               polynomial_order: int) -> float:
        """NUMBA-optimized single scale fluctuation calculation."""
        n_segments = len(data) // scale
        if n_segments == 0:
            return np.nan
            
        # Pre-allocate array for fluctuations
        max_fluctuations = n_segments
        fluctuations = np.zeros(max_fluctuations)
        n_valid = 0
        
        for i in range(n_segments):
            start_idx = i * scale
            end_idx = start_idx + scale
            segment = data[start_idx:end_idx]
            
            # Detrend using simple linear fit (polynomial_order=1) for NUMBA compatibility
            if polynomial_order == 1:
                x = np.arange(scale, dtype=np.float64)
                x_mean = np.mean(x)
                y_mean = np.mean(segment)
                
                # Simple linear regression
                numerator = np.sum((x - x_mean) * (segment - y_mean))
                denominator = np.sum((x - x_mean) ** 2)
                
                if denominator == 0:
                    slope = 0.0
                else:
                    slope = numerator / denominator
                    
                intercept = y_mean - slope * x_mean
                trend = slope * x + intercept
            else:
                # For higher orders, use simple mean subtraction (fallback)
                trend = np.full(scale, np.mean(segment))
                
            detrended = segment - trend
            
            # Calculate variance
            variance = np.var(detrended)
            if variance > 0:
                fluctuations[n_valid] = variance
                n_valid += 1
                
        if n_valid == 0:
            return np.nan
            
        # Return mean of valid fluctuations
        return np.sum(fluctuations[:n_valid]) / n_valid
        
    @numba_jit
    def _fit_power_law_numba(scales: np.ndarray, fluctuations: np.ndarray) -> Tuple[float, float, float]:
        """NUMBA-optimized power law fitting."""
        # Remove NaN values
        valid_mask = ~np.isnan(fluctuations)
        if np.sum(valid_mask) < 3:
            return np.nan, np.nan, np.nan
            
        scales_valid = scales[valid_mask]
        fluct_valid = fluctuations[valid_mask]
        
        # Log-log regression
        log_scales = np.log(scales_valid.astype(np.float64))
        log_fluct = np.log(fluct_valid)
        
        # Simple linear regression for NUMBA compatibility
        n = len(log_scales)
        if n < 2:
            return np.nan, np.nan, np.nan
            
        x_mean = np.mean(log_scales)
        y_mean = np.mean(log_fluct)
        
        # Calculate slope and intercept
        numerator = np.sum((log_scales - x_mean) * (log_fluct - y_mean))
        denominator = np.sum((log_scales - x_mean) ** 2)
        
        if denominator == 0:
            return np.nan, np.nan, np.nan
            
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_pred = slope * log_scales + intercept
        ss_res = np.sum((log_fluct - y_pred) ** 2)
        ss_tot = np.sum((log_fluct - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return slope, intercept, r_squared


# JAX-optimized functions
if JAX_AVAILABLE:
    @jax_jit
    def _calculate_fluctuations_jax_vectorized(data: jnp.ndarray, scales: jnp.ndarray, 
                                              q_values: jnp.ndarray, polynomial_order: int) -> jnp.ndarray:
        """JAX-vectorized fluctuation calculation."""
        # Vectorize over scales and q values
        def calculate_for_scale_and_q(scale, q):
            return _calculate_fluctuation_jax(data, scale, q, polynomial_order)
            
        return vmap(vmap(calculate_for_scale_and_q, in_axes=(0, None)), in_axes=(None, 0))(
            scales, q_values
        )
        
    @jax_jit
    def _calculate_fluctuation_jax(data: jnp.ndarray, scale: int, q: float, 
                                   polynomial_order: int) -> float:
        """JAX-optimized single scale fluctuation calculation."""
        n_segments = len(data) // scale
        if n_segments == 0:
            return jnp.nan
            
        def process_segment(segment):
            x = jnp.arange(scale, dtype=jnp.float32)
            coeffs = jnp.polyfit(x, segment, polynomial_order)
            trend = jnp.polyval(coeffs, x)
            detrended = segment - trend
            return jnp.var(detrended)
            
        # Vectorize over segments
        segments = jnp.array([data[i*scale:(i+1)*scale] for i in range(n_segments)])
        variances = vmap(process_segment)(segments)
        
        # Filter positive variances
        positive_variances = variances[variances > 0]
        
        if len(positive_variances) == 0:
            return jnp.nan
            
        # Calculate q-th order fluctuation
        if jnp.abs(q) < 1e-10:  # q ≈ 0
            fq = jnp.exp(jnp.mean(jnp.log(positive_variances)))
        else:
            fq = jnp.mean(positive_variances ** (q/2))
            
        return fq ** (1/q) if q != 0 else fq
        
    @jax_jit
    def _fit_scaling_laws_jax_vectorized(scales: jnp.ndarray, fluctuations: jnp.ndarray, 
                                         q_values: jnp.ndarray) -> jnp.ndarray:
        """JAX-vectorized scaling law fitting."""
        def fit_for_q(fluct_col):
            return _fit_scaling_law_jax(scales, fluct_col)
            
        return vmap(fit_for_q)(fluctuations.T)
        
    @jax_jit
    def _fit_scaling_law_jax(scales: jnp.ndarray, fluctuations: jnp.ndarray) -> float:
        """JAX-optimized single scaling law fitting."""
        # Remove NaN values
        valid_mask = ~jnp.isnan(fluctuations)
        if jnp.sum(valid_mask) < 3:
            return jnp.nan
            
        scales_valid = scales[valid_mask]
        fluct_valid = fluctuations[valid_mask]
        
        # Log-log regression
        log_scales = jnp.log(scales_valid.astype(jnp.float32))
        log_fluct = jnp.log(fluct_valid)
        
        try:
            coeffs = jnp.polyfit(log_scales, log_fluct, 1)
            return coeffs[0]
        except:
            return jnp.nan
            
    @jax_jit
    def _calculate_multifractal_spectrum_jax(q_values: jnp.ndarray, 
                                            hurst_exponents: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """JAX-optimized multifractal spectrum calculation."""
        # Find valid Hurst exponents
        valid_mask = ~jnp.isnan(hurst_exponents)
        if jnp.sum(valid_mask) < 3:
            return None
            
        q_valid = q_values[valid_mask]
        h_valid = hurst_exponents[valid_mask]
        
        # Calculate alpha (singularity strength) and f(alpha)
        alpha = h_valid + q_valid * jnp.gradient(h_valid, q_valid)
        f_alpha = q_valid * alpha - h_valid
        
        return {
            'alpha': alpha,
            'f_alpha': f_alpha,
            'q_values': q_valid,
            'hurst_exponents': h_valid
        }
