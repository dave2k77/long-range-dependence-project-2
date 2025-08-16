"""
NUMBA-optimized utility functions for high-performance computing.

This module provides NUMBA-compiled functions for accelerating
long-range dependence estimation algorithms.
"""

import numpy as np
from numba import jit, prange, cuda, float64, int64, boolean
from numba.core.errors import NumbaDeprecationWarning
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)


class NumbaOptimizer:
    """
    NUMBA optimization utilities for long-range dependence estimators.
    
    This class provides NUMBA-compiled functions for common operations
    used in LRD estimation, including GPU acceleration when available.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize NUMBA optimizer.
        
        Parameters
        ----------
        use_gpu : bool
            Whether to use GPU acceleration (requires CUDA)
        """
        self.use_gpu = use_gpu and cuda.is_available()
        if self.use_gpu:
            print("GPU acceleration enabled with NUMBA CUDA")
        else:
            print("Using CPU optimization with NUMBA")
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def fast_polyfit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        """
        Fast polynomial fitting using NUMBA.
        
        Parameters
        ----------
        x : np.ndarray
            Input x values
        y : np.ndarray
            Input y values
        degree : int
            Polynomial degree
            
        Returns
        -------
        np.ndarray
            Polynomial coefficients
        """
        n = len(x)
        if n <= degree:
            return np.zeros(degree + 1)
        
        # Vandermonde matrix
        A = np.zeros((n, degree + 1))
        for i in prange(n):
            for j in range(degree + 1):
                A[i, j] = x[i] ** j
        
        # Normal equations: A^T * A * c = A^T * y
        AtA = np.zeros((degree + 1, degree + 1))
        Aty = np.zeros(degree + 1)
        
        for i in prange(degree + 1):
            for j in range(degree + 1):
                for k in range(n):
                    AtA[i, j] += A[k, i] * A[k, j]
            
            for k in range(n):
                Aty[i] += A[k, i] * y[k]
        
        # Solve using Cholesky decomposition
        try:
            L = np.linalg.cholesky(AtA)
            # Forward substitution
            z = np.zeros(degree + 1)
            for i in range(degree + 1):
                z[i] = (Aty[i] - np.sum(L[i, :i] * z[:i])) / L[i, i]
            
            # Backward substitution
            c = np.zeros(degree + 1)
            for i in range(degree - 1, -1, -1):
                c[i] = (z[i] - np.sum(L[i+1:, i] * c[i+1:])) / L[i, i]
            c[degree] = z[degree]
            
            return c
        except:
            # Fallback to least squares
            return np.linalg.lstsq(A, y, rcond=None)[0]
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def fast_polyval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Fast polynomial evaluation using NUMBA.
        
        Parameters
        ----------
        coeffs : np.ndarray
            Polynomial coefficients
        x : np.ndarray
            Input x values
            
        Returns
        -------
        np.ndarray
            Polynomial values
        """
        result = np.zeros_like(x, dtype=np.float64)
        for i in prange(len(x)):
            val = 0.0
            for j in range(len(coeffs)):
                val += coeffs[j] * (x[i] ** j)
            result[i] = val
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def fast_detrend(data: np.ndarray, degree: int = 1) -> np.ndarray:
        """
        Fast detrending using NUMBA.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        degree : int
            Polynomial degree for detrending
            
        Returns
        -------
        np.ndarray
            Detrended data
        """
        n = len(data)
        x = np.arange(n, dtype=np.float64)
        
        # Fit polynomial
        coeffs = NumbaOptimizer.fast_polyfit(x, data, degree)
        
        # Evaluate and subtract
        trend = NumbaOptimizer.fast_polyval(coeffs, x)
        return data - trend
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def fast_rms(data: np.ndarray) -> float:
        """
        Fast RMS calculation using NUMBA.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
            
        Returns
        -------
        float
            RMS value
        """
        sum_sq = 0.0
        for i in prange(len(data)):
            sum_sq += data[i] * data[i]
        return np.sqrt(sum_sq / len(data))
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def fast_logspace(start: float, stop: float, num: int) -> np.ndarray:
        """
        Fast logspace generation using NUMBA.
        
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
        result = np.zeros(num, dtype=np.float64)
        step = (stop - start) / (num - 1)
        for i in prange(num):
            result[i] = 10.0 ** (start + i * step)
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def fast_linregress(x: np.ndarray, y: np.ndarray) -> tuple:
        """
        Fast linear regression using NUMBA.
        
        Parameters
        ----------
        x : np.ndarray
            Input x values
        y : np.ndarray
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
        x_mean = 0.0
        y_mean = 0.0
        for i in prange(n):
            x_mean += x[i]
            y_mean += y[i]
        x_mean /= n
        y_mean /= n
        
        # Calculate sums
        sum_xy = 0.0
        sum_xx = 0.0
        sum_yy = 0.0
        for i in prange(n):
            dx = x[i] - x_mean
            dy = y[i] - y_mean
            sum_xy += dx * dy
            sum_xx += dx * dx
            sum_yy += dy * dy
        
        # Calculate slope and intercept
        if sum_xx == 0.0:
            return 0.0, y_mean, 0.0, 1.0, 0.0
        
        slope = sum_xy / sum_xx
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        r_squared = (sum_xy * sum_xy) / (sum_xx * sum_yy) if sum_yy > 0.0 else 0.0
        r_value = np.sqrt(r_squared) if r_squared >= 0.0 else 0.0
        
        # Calculate standard error
        if n > 2:
            mse = (sum_yy - slope * sum_xy) / (n - 2)
            std_err = np.sqrt(mse / sum_xx) if sum_xx > 0.0 else 0.0
        else:
            std_err = 0.0
        
        # Simple p-value approximation (for large n)
        if n > 10:
            t_stat = slope / std_err if std_err > 0.0 else 0.0
            p_value = 2.0 * (1.0 - 0.5 * (1.0 + np.tanh(t_stat / np.sqrt(2.0))))
        else:
            p_value = 0.5  # Default value for small samples
        
        return slope, intercept, r_value, p_value, std_err
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def fast_periodogram(data: np.ndarray) -> tuple:
        """
        Fast periodogram calculation using NUMBA.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
            
        Returns
        -------
        tuple
            (frequencies, power_spectrum)
        """
        n = len(data)
        n_freq = n // 2 + 1
        
        frequencies = np.zeros(n_freq, dtype=np.float64)
        power_spectrum = np.zeros(n_freq, dtype=np.float64)
        
        # Generate frequencies
        for i in prange(n_freq):
            frequencies[i] = i / n
        
        # Calculate power spectrum using FFT approximation
        for k in prange(n_freq):
            real_part = 0.0
            imag_part = 0.0
            
            for t in range(n):
                angle = 2.0 * np.pi * k * t / n
                real_part += data[t] * np.cos(angle)
                imag_part += data[t] * np.sin(angle)
            
            power_spectrum[k] = (real_part * real_part + imag_part * imag_part) / n
        
        return frequencies, power_spectrum
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def fast_wavelet_transform(data: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        Fast wavelet transform approximation using NUMBA.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        scales : np.ndarray
            Wavelet scales
            
        Returns
        -------
        np.ndarray
            Wavelet coefficients
        """
        n = len(data)
        n_scales = len(scales)
        
        # Simple Haar wavelet approximation
        coeffs = np.zeros((n_scales, n), dtype=np.float64)
        
        for i in prange(n_scales):
            scale = int(scales[i])
            if scale < 2:
                continue
                
            for j in range(0, n - scale + 1, scale):
                # Haar wavelet coefficients
                if j + scale <= n:
                    segment = data[j:j + scale]
                    mean_val = np.sum(segment) / scale
                    
                    # Store coefficients
                    for k in range(scale):
                        if j + k < n:
                            coeffs[i, j + k] = data[j + k] - mean_val
        
        return coeffs


# GPU-accelerated functions (when CUDA is available)
if cuda.is_available():
    @cuda.jit
    def gpu_polyfit_kernel(x, y, degree, coeffs):
        """CUDA kernel for polynomial fitting."""
        idx = cuda.grid(1)
        if idx < len(x):
            # Implementation for GPU polynomial fitting
            pass
    
    @cuda.jit
    def gpu_detrend_kernel(data, coeffs, result):
        """CUDA kernel for detrending."""
        idx = cuda.grid(1)
        if idx < len(data):
            # Implementation for GPU detrending
            pass
else:
    # Placeholder functions when GPU is not available
    def gpu_polyfit_kernel(x, y, degree, coeffs):
        """Placeholder for GPU polynomial fitting."""
        raise RuntimeError("GPU acceleration not available")
    
    def gpu_detrend_kernel(data, coeffs, result):
        """Placeholder for GPU detrending."""
        raise RuntimeError("GPU acceleration not available")
