"""
JAX-optimized utility functions for high-performance computing.

This module provides JAX-based functions for GPU acceleration,
automatic differentiation, and vectorized operations.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, hessian
from jax.scipy import stats, optimize
# from jax.scipy.signal import periodogram  # Not available in current JAX version
from jax.lax import scan, fori_loop
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)

# Set JAX to use float64 for better numerical stability
jax.config.update("jax_enable_x64", True)


class JAXOptimizer:
    """
    JAX optimization utilities for long-range dependence estimators.
    
    This class provides JAX-compiled functions for common operations
    used in LRD estimation, including GPU acceleration and automatic differentiation.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize JAX optimizer.
        
        Parameters
        ----------
        device : str
            Device to use ('cpu', 'gpu', 'tpu', or 'auto')
        """
        self.device = device
        self.available_devices = jax.devices()
        
        if device == "auto":
            # Prefer GPU if available
            if any("gpu" in str(d).lower() for d in self.available_devices):
                self.device = "gpu"
                logger.info("GPU acceleration enabled with JAX")
            else:
                self.device = "cpu"
                logger.info("Using CPU with JAX")
        
        logger.info(f"Available devices: {self.available_devices}")
    
    @staticmethod
    @jit
    def fast_polyfit(x: jnp.ndarray, y: jnp.ndarray, degree: int) -> jnp.ndarray:
        """
        Fast polynomial fitting using JAX.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input x values
        y : jnp.ndarray
            Input y values
        degree : int
            Polynomial degree
            
        Returns
        -------
        jnp.ndarray
            Polynomial coefficients
        """
        n = len(x)
        if n <= degree:
            return jnp.zeros(degree + 1)
        
        # Vandermonde matrix
        A = jnp.vander(x, degree + 1, increasing=True)
        
        # Solve using least squares
        coeffs, residuals, rank, s = jnp.linalg.lstsq(A, y, rcond=None)
        
        return coeffs
    
    @staticmethod
    @jit
    def fast_polyval(coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """
        Fast polynomial evaluation using JAX.
        
        Parameters
        ----------
        coeffs : jnp.ndarray
            Polynomial coefficients
        x : jnp.ndarray
            Input x values
            
        Returns
        -------
        jnp.ndarray
            Polynomial values
        """
        return jnp.polyval(coeffs, x)
    
    @staticmethod
    @jit
    def fast_detrend(data: jnp.ndarray, degree: int = 1) -> jnp.ndarray:
        """
        Fast detrending using JAX.
        
        Parameters
        ----------
        data : jnp.ndarray
            Input data
        degree : int
            Polynomial degree for detrending
            
        Returns
        -------
        jnp.ndarray
            Detrended data
        """
        n = len(data)
        x = jnp.arange(n, dtype=jnp.float64)
        
        # Fit polynomial
        coeffs = JAXOptimizer.fast_polyfit(x, data, degree)
        
        # Evaluate and subtract
        trend = JAXOptimizer.fast_polyval(coeffs, x)
        return data - trend
    
    @staticmethod
    @jit
    def fast_rms(data: jnp.ndarray) -> jnp.ndarray:
        """
        Fast RMS calculation using JAX.
        
        Parameters
        ----------
        data : jnp.ndarray
            Input data
            
        Returns
        -------
        jnp.ndarray
            RMS value
        """
        return jnp.sqrt(jnp.mean(data ** 2))
    
    @staticmethod
    @jit
    def fast_logspace(start: float, stop: float, num: int) -> jnp.ndarray:
        """
        Fast logspace generation using JAX.
        
        Parameters
        ----------
        start : float
            Start value (log10)
        stop : float
            Stop value (log10)
        num : int
            Number of points
            
        Returns
        -------
        jnp.ndarray
            Log-spaced array
        """
        try:
            return jnp.logspace(start, stop, num)
        except Exception as e:
            # JAX failed - this is expected for dynamic shapes
            raise RuntimeError(f"JAX logspace failed (expected for dynamic shapes): {e}")
    
    @staticmethod
    def fast_logspace_numpy(start: float, stop: float, num: int) -> np.ndarray:
        """
        Numpy fallback for logspace generation.
        
        Parameters
        ----------
        start : float
            Start value (log10)
        stop : float
            Stop value (log10)
        num : int
            Number of points
            
        Returns
        -------
        np.ndarray
            Log-spaced array
        """
        return np.logspace(start, stop, num)
    
    @staticmethod
    @jit
    def fast_linregress(x: jnp.ndarray, y: jnp.ndarray) -> tuple:
        """
        Fast linear regression using JAX.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input x values
        y : jnp.ndarray
            Input y values
            
        Returns
        -------
        tuple
            (slope, intercept, r_value, p_value, std_err)
        """
        n = len(x)
        if n < 2:
            return 0.0, 0.0, 0.0, 1.0, 0.0
        
        # Calculate means
        x_mean = jnp.mean(x)
        y_mean = jnp.mean(y)
        
        # Calculate sums
        dx = x - x_mean
        dy = y - y_mean
        sum_xy = jnp.sum(dx * dy)
        sum_xx = jnp.sum(dx * dx)
        sum_yy = jnp.sum(dy * dy)
        
        # Calculate slope and intercept
        if sum_xx == 0.0:
            return 0.0, y_mean, 0.0, 1.0, 0.0
        
        slope = sum_xy / sum_xx
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        r_squared = (sum_xy * sum_xy) / (sum_xx * sum_yy) if sum_yy > 0.0 else 0.0
        r_value = jnp.sqrt(r_squared) if r_squared >= 0.0 else 0.0
        
        # Calculate standard error
        if n > 2:
            mse = (sum_yy - slope * sum_xy) / (n - 2)
            std_err = jnp.sqrt(mse / sum_xx) if sum_xx > 0.0 else 0.0
        else:
            std_err = 0.0
        
        # Simple p-value approximation
        if n > 10:
            t_stat = slope / std_err if std_err > 0.0 else 0.0
            p_value = 2.0 * (1.0 - stats.norm.cdf(jnp.abs(t_stat)))
        else:
            p_value = 0.5
        
        return slope, intercept, r_value, p_value, std_err
    
    @staticmethod
    @jit
    def fast_periodogram(data: jnp.ndarray) -> tuple:
        """
        Fast periodogram calculation using JAX.
        
        Parameters
        ----------
        data : jnp.ndarray
            Input time series data
            
        Returns
        -------
        tuple
            (frequencies, power_spectrum)
        """
        # Custom periodogram implementation since jax.scipy.signal.periodogram is not available
        n = len(data)
        freqs = jnp.fft.fftfreq(n)
        
        # Compute FFT
        fft_vals = jnp.fft.fft(data)
        
        # Compute periodogram (power spectral density)
        periodogram_values = jnp.abs(fft_vals) ** 2 / n
        
        # Return only positive frequencies (one-sided)
        positive_freqs = freqs[:n//2 + 1]
        positive_periodogram = periodogram_values[:n//2 + 1]
        
        return positive_freqs, positive_periodogram
    
    @staticmethod
    @jit
    def fast_wavelet_transform(data: jnp.ndarray, scales: jnp.ndarray) -> jnp.ndarray:
        """
        Fast wavelet transform approximation using JAX.
        
        Parameters
        ----------
        data : jnp.ndarray
            Input time series data
        scales : jnp.ndarray
            Wavelet scales
            
        Returns
        -------
        jnp.ndarray
            Wavelet coefficients
        """
        n = len(data)
        n_scales = len(scales)
        
        def wavelet_scale(scale):
            """Compute wavelet coefficients for a single scale."""
            scale_int = int(scale)
            if scale_int < 2:
                return jnp.zeros(n)
            
            # Simple Haar wavelet approximation
            coeffs = jnp.zeros(n)
            
            for j in range(0, n - scale_int + 1, scale_int):
                if j + scale_int <= n:
                    segment = data[j:j + scale_int]
                    mean_val = jnp.mean(segment)
                    coeffs = coeffs.at[j:j + scale_int].set(data[j:j + scale_int] - mean_val)
            
            return coeffs
        
        # Vectorize over scales
        return vmap(wavelet_scale)(scales)
    
    @staticmethod
    @jit
    def fast_whittle_likelihood(alpha: jnp.ndarray, frequencies: jnp.ndarray, 
                               periodogram_values: jnp.ndarray) -> jnp.ndarray:
        """
        Fast Whittle likelihood calculation using JAX.
        
        Parameters
        ----------
        alpha : jnp.ndarray
            Long-range dependence parameter
        frequencies : jnp.ndarray
            Frequency values
        periodogram_values : jnp.ndarray
            Periodogram values
            
        Returns
        -------
        jnp.ndarray
            Negative log-likelihood
        """
        # Theoretical power spectrum for ARFIMA(0,d,0) process
        # S(f) = |1 - exp(-2πif)|^(-2d) where d = (α-1)/2
        
        d = (alpha - 1) / 2
        
        # Calculate theoretical spectrum
        theoretical_spectrum = jnp.abs(1 - jnp.exp(-2j * jnp.pi * frequencies)) ** (-2 * d)
        
        # Whittle likelihood
        log_likelihood = jnp.sum(
            jnp.log(theoretical_spectrum) + periodogram_values / theoretical_spectrum
        )
        
        return -log_likelihood  # Return negative for minimization
    
    @staticmethod
    def optimize_whittle_likelihood(frequencies: jnp.ndarray, 
                                  periodogram_values: jnp.ndarray,
                                  initial_guess: float = 0.5,
                                  bounds: tuple = (0.0, 2.0)) -> tuple:
        """
        Optimize Whittle likelihood using JAX.
        
        Parameters
        ----------
        frequencies : jnp.ndarray
            Frequency values
        periodogram_values : jnp.ndarray
            Periodogram values
        initial_guess : float
            Initial guess for alpha
        bounds : tuple
            Bounds for alpha (min, max)
            
        Returns
        -------
        tuple
            (optimal_alpha, optimization_success)
        """
        # Define objective function
        def objective(alpha):
            return JAXOptimizer.fast_whittle_likelihood(
                alpha, frequencies, periodogram_values
            )
        
        # Get gradient and Hessian
        grad_obj = grad(objective)
        hess_obj = hessian(objective)
        
        try:
            # Use JAX optimization
            result = optimize.minimize(
                objective,
                x0=jnp.array([initial_guess]),
                method='BFGS',
                jac=grad_obj
            )
            
            if result.success:
                return result.x[0], True
            else:
                logger.warning(f"Optimization failed: {result.message}")
                return result.x[0], False
                
        except Exception as e:
            logger.error(f"Error in JAX optimization: {str(e)}")
            return initial_guess, False
    
    @staticmethod
    @jit
    def fast_fractional_brownian_motion(size: int, hurst: float, 
                                      noise_level: float = 0.0) -> jnp.ndarray:
        """
        Fast fractional Brownian motion generation using JAX.
        
        Parameters
        ----------
        size : int
            Size of the time series
        hurst : float
            Hurst exponent
        noise_level : float
            Noise level to add
            
        Returns
        -------
        jnp.ndarray
            Fractional Brownian motion time series
        """
        # Generate frequencies
        freqs = jnp.fft.fftfreq(size)
        
        # Power spectrum for fBm
        power_spectrum = jnp.where(
            freqs != 0,
            jnp.abs(freqs) ** (-2 * hurst - 1),
            0.0
        )
        
        # Generate complex Gaussian noise
        key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        noise = jax.random.normal(key, (size,)) + 1j * jax.random.normal(key, (size,))
        
        # Scale by power spectrum
        scaled_noise = noise * jnp.sqrt(power_spectrum)
        
        # Inverse FFT
        fbm = jnp.real(jnp.fft.ifft(scaled_noise))
        
        # Add additional noise if specified
        if noise_level > 0.0:
            key2 = jax.random.PRNGKey(123)
            additional_noise = noise_level * jax.random.normal(key2, (size,))
            fbm = fbm + additional_noise
        
        return fbm
    
    @staticmethod
    @jit
    def vectorized_estimation(data_batch: jnp.ndarray, 
                            estimator_func: Callable) -> jnp.ndarray:
        """
        Vectorized estimation over multiple datasets using JAX.
        
        Parameters
        ----------
        data_batch : jnp.ndarray
            Batch of datasets (batch_size, data_length)
        estimator_func : Callable
            Function to apply to each dataset
            
        Returns
        -------
        jnp.ndarray
            Batch of estimation results
        """
        return vmap(estimator_func)(data_batch)
    
    @staticmethod
    def parallel_estimation(data_list: List[jnp.ndarray], 
                          estimator_func: Callable,
                          batch_size: int = 32) -> List[Any]:
        """
        Parallel estimation using JAX batching.
        
        Parameters
        ----------
        data_list : List[jnp.ndarray]
            List of datasets
        estimator_func : Callable
            Function to apply to each dataset
        batch_size : int
            Batch size for processing
            
        Returns
        -------
        List[Any]
            List of estimation results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            batch_array = jnp.array(batch)
            
            # Vectorized estimation
            batch_results = JAXOptimizer.vectorized_estimation(batch_array, estimator_func)
            results.extend(batch_results)
        
        return results
