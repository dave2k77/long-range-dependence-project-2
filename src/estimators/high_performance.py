"""
High-Performance Long-Range Dependence Estimators - Utility Functions

This module provides NUMBA and JAX optimized utility functions for
LRD estimators. The actual estimator classes are in separate files.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

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
    def _calculate_fluctuation_jax_simple(data: jnp.ndarray, scale: int, q: float, 
                                          polynomial_order: int) -> float:
        """Simple JAX-compatible fluctuation calculation."""
        # Use a simple approach that's guaranteed to work with JAX
        data_len = len(data)
        n_segments = data_len // scale
        
        if n_segments == 0:
            return jnp.nan
        
        # Create fixed-size arrays with known dimensions
        max_scale = 1000  # Maximum expected scale
        max_segments = max_scale // 4  # Minimum scale is 4
        
        # Create fixed-size segment array
        segment_data = jnp.zeros((max_segments, max_scale), dtype=data.dtype)
        
        # Fill segments up to n_segments
        def fill_segment(i):
            start_idx = i * scale
            end_idx = start_idx + scale
            segment = data[start_idx:end_idx]
            # Pad or truncate to max_scale
            if len(segment) < max_scale:
                padded = jnp.pad(segment, (0, max_scale - len(segment)), mode='constant')
            else:
                padded = segment[:max_scale]
            return padded
        
        # Use vmap to fill segments
        segment_data = vmap(fill_segment)(jnp.arange(max_segments))
        
        # Create fixed-size x array
        x = jnp.linspace(0, 1, max_scale, dtype=jnp.float32)
        
        # Process segments using vmap
        def process_segment(segment):
            # Use only the first 'scale' elements
            segment_actual = segment[:scale]
            x_actual = x[:scale]
            
            coeffs = jnp.polyfit(x_actual, segment_actual, polynomial_order)
            trend = jnp.polyval(coeffs, x_actual)
            detrended = segment_actual - trend
            return jnp.var(detrended)
        
        # Apply to all segments
        variances = vmap(process_segment)(segment_data)
        
        # Calculate q-th order fluctuation
        q_abs = jnp.abs(q)
        q_small = q_abs < 1e-10
        
        # Use jnp.where for conditional logic
        fq = jnp.where(q_small,
                      jnp.exp(jnp.mean(jnp.log(variances))),
                      jnp.mean(variances ** (q/2)))
        
        # Apply power safely
        result = jnp.where(q_small, fq, fq ** (1/q))
        return result
    
    @jax_jit
    def _calculate_fluctuation_jax_pure(data: jnp.ndarray, scale: int, q: float, 
                                        polynomial_order: int) -> float:
        """Pure JAX fluctuation calculation without any Python control flow."""
        # Use pure JAX operations - no Python control flow at all
        data_len = len(data)
        n_segments = data_len // scale
        
        # Use JAX-safe conditional logic
        def process_data():
            # Create segment indices using JAX operations
            segment_indices = jnp.arange(n_segments)
            
            # Create fixed-size x array
            x = jnp.linspace(0, 1, scale, dtype=jnp.float32)
            
            # Process segments using vmap
            def process_segment(segment_idx):
                start_idx = segment_idx * scale
                end_idx = start_idx + scale
                segment = data[start_idx:end_idx]
                
                # Polynomial fitting
                coeffs = jnp.polyfit(x, segment, polynomial_order)
                trend = jnp.polyval(coeffs, x)
                detrended = segment - trend
                return jnp.var(detrended)
            
            # Apply to all segments
            variances = vmap(process_segment)(segment_indices)
            
            # Calculate q-th order fluctuation
            q_abs = jnp.abs(q)
            q_small = q_abs < 1e-10
            
            # Use jnp.where for conditional logic
            fq = jnp.where(q_small,
                          jnp.exp(jnp.mean(jnp.log(variances))),
                          jnp.mean(variances ** (q/2)))
            
            # Apply power safely
            result = jnp.where(q_small, fq, fq ** (1/q))
            return result
        
        # Return result only if we have segments
        return jnp.where(n_segments > 0, process_data(), jnp.nan)
    
    @jax_jit
    def _calculate_fluctuations_jax_vectorized(data: jnp.ndarray, scales: jnp.ndarray, 
                                               q_values: jnp.ndarray, polynomial_order: int) -> jnp.ndarray:
        """JAX-vectorized fluctuation calculation for multiple scales and q values."""
        # Use a more JAX-compatible approach that avoids dynamic shapes
        def calculate_for_scale_and_q(scale, q):
            return _calculate_fluctuation_jax_pure(data, scale, q, polynomial_order)
            
        return vmap(vmap(calculate_for_scale_and_q, in_axes=(0, None)), in_axes=(None, 0))(
            scales, q_values
        )
    
    @jax_jit
    def _fit_scaling_laws_jax_vectorized(scales: jnp.ndarray, fluctuations: jnp.ndarray, 
                                         q_values: jnp.ndarray) -> jnp.ndarray:
        """JAX-vectorized scaling law fitting."""
        def fit_for_q(fluct_col):
            return _fit_scaling_law_jax(scales, fluct_col)
            
        # Ensure compatible shapes by truncating to minimum length
        min_len = min(fluctuations.shape[1], len(scales))
        fluctuations_truncated = fluctuations[:, :min_len]
        scales_truncated = scales[:min_len]
            
        return vmap(fit_for_q)(fluctuations_truncated.T)
        
    @jax_jit
    def _fit_scaling_law_jax(scales: jnp.ndarray, fluctuations: jnp.ndarray) -> float:
        """JAX-optimized single scaling law fitting."""
        # Remove NaN values using JAX-safe operations
        valid_mask = ~jnp.isnan(fluctuations)
        n_valid = jnp.sum(valid_mask)
        
        # Use JAX-safe conditional logic
        def fit_regression():
            # Use jnp.where to safely extract valid values
            scales_valid = jnp.where(valid_mask, scales, 0)
            fluct_valid = jnp.where(valid_mask, fluctuations, 0)
            
            # Log-log regression
            log_scales = jnp.log(scales_valid.astype(jnp.float32))
            log_fluct = jnp.log(fluct_valid)
            
            # Use JAX-safe polyfit
            coeffs = jnp.polyfit(log_scales, log_fluct, 1)
            return coeffs[0]
        
        # Return result only if we have enough valid points
        return jnp.where(n_valid >= 3, fit_regression(), jnp.nan)
            
    @jax_jit
    def _calculate_multifractal_spectrum_jax(q_values: jnp.ndarray, 
                                            hurst_exponents: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """JAX-optimized multifractal spectrum calculation."""
        # Find valid Hurst exponents using JAX-safe operations
        valid_mask = ~jnp.isnan(hurst_exponents)
        n_valid = jnp.sum(valid_mask)
        
        # Use JAX-safe conditional logic
        def calculate_spectrum():
            # Use jnp.where to safely extract valid values
            q_valid = jnp.where(valid_mask, q_values, 0)
            h_valid = jnp.where(valid_mask, hurst_exponents, 0)
            
            # Calculate alpha (singularity strength) and f(alpha)
            alpha = h_valid + q_valid * jnp.gradient(h_valid, q_valid)
            f_alpha = q_valid * alpha - h_valid
            
            return {
                'alpha': alpha,
                'f_alpha': f_alpha,
                'q_values': q_valid,
                'hurst_exponents': h_valid
            }
        
        # Return result only if we have enough valid points
        # Use a default result with same shape for JAX compatibility
        def get_default_result():
            # Return arrays with same shape as input but filled with NaN
            return {
                'alpha': jnp.full_like(q_values, jnp.nan),
                'f_alpha': jnp.full_like(q_values, jnp.nan),
                'q_values': q_values,
                'hurst_exponents': hurst_exponents
            }
        
        # Use conditional return to avoid type issues
        return jax.lax.cond(
            n_valid >= 3,
            lambda _: calculate_spectrum(),
            lambda _: get_default_result(),
            operand=None
        )
