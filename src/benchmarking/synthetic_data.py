"""
Synthetic Data Generation for Long-Range Dependence

This module provides tools for generating synthetic time series data
with known long-range dependence properties for benchmarking.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generator for synthetic time series data with known long-range dependence.
    
    This class provides methods for creating various types of synthetic
    data useful for benchmarking LRD estimators.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the synthetic data generator.
        
        Parameters
        ----------
        random_state : Optional[int]
            Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            
    def fractional_brownian_motion(self, n_points: int, hurst_exponent: float = None, 
                                 sigma: float = 1.0, noise_level: float = None, 
                                 hurst: float = None) -> np.ndarray:
        """
        Generate fractional Brownian motion (fBm) with given Hurst exponent.
        
        Parameters
        ----------
        n_points : int
            Number of points to generate
        hurst_exponent : float, optional
            Hurst exponent (0 < H < 1)
        sigma : float
            Standard deviation of the process
        noise_level : float, optional
            Noise level (alias for sigma, overrides sigma if provided)
        hurst : float, optional
            Hurst exponent (alias for hurst_exponent)
            
        Returns
        -------
        np.ndarray
            Fractional Brownian motion time series
        """
        # Handle hurst parameter (alias for hurst_exponent)
        if hurst is not None:
            hurst_exponent = hurst
        elif hurst_exponent is None:
            raise ValueError("Either hurst_exponent or hurst must be provided")
            
        if not 0 < hurst_exponent < 1:
            raise ValueError("Hurst exponent must be between 0 and 1")
        
        # Handle noise_level parameter
        if noise_level is not None:
            sigma = noise_level
            
        # Use the Davies-Harte method for efficient fBm generation
        n = n_points
        H = hurst_exponent
        
        # Generate frequencies
        freqs = np.fft.fftfreq(n)
        
        # Power spectrum
        power_spectrum = np.zeros_like(freqs)
        power_spectrum[1:] = np.abs(freqs[1:]) ** (-2*H - 1)
        power_spectrum[0] = 0  # DC component
        
        # Generate complex Gaussian random variables
        real_part = np.random.normal(0, 1, n)
        imag_part = np.random.normal(0, 1, n)
        complex_noise = (real_part + 1j * imag_part) / np.sqrt(2)
        
        # Apply power spectrum
        filtered_noise = complex_noise * np.sqrt(power_spectrum)
        
        # Inverse FFT to get time series
        fbm = np.real(np.fft.ifft(filtered_noise))
        
        # Normalize and scale
        fbm = (fbm - np.mean(fbm)) / np.std(fbm) * sigma
        
        return fbm
        
    def fractional_gaussian_noise(self, n_points: int, hurst_exponent: float,
                                sigma: float = 1.0) -> np.ndarray:
        """
        Generate fractional Gaussian noise (fGn) with given Hurst exponent.
        
        Parameters
        ----------
        n_points : int
            Number of points to generate
        hurst_exponent : float
            Hurst exponent (0 < H < 1)
        sigma : float
            Standard deviation of the process
            
        Returns
        -------
        np.ndarray
            Fractional Gaussian noise time series
        """
        # Generate fBm and take differences
        fbm = self.fractional_brownian_motion(n_points + 1, hurst_exponent, sigma)
        fgn = np.diff(fbm)
        
        return fgn
        
    def arfima_process(self, n_points: int, d: float, ar_params: List[float] = None,
                      ma_params: List[float] = None, sigma: float = 1.0) -> np.ndarray:
        """
        Generate ARFIMA(p,d,q) process.
        
        Parameters
        ----------
        n_points : int
            Number of points to generate
        d : float
            Fractional differencing parameter (-0.5 < d < 0.5)
        ar_params : List[float]
            AR parameters (default: [])
        ma_params : List[float]
            MA parameters (default: [])
        sigma : float
            Standard deviation of innovations
            
        Returns
        -------
        np.ndarray
            ARFIMA time series
        """
        if not -0.5 < d < 0.5:
            raise ValueError("Fractional differencing parameter must be between -0.5 and 0.5")
            
        if ar_params is None:
            ar_params = []
        if ma_params is None:
            ma_params = []
            
        # Generate white noise
        white_noise = np.random.normal(0, sigma, n_points)
        
        # Apply fractional differencing
        if abs(d) > 1e-10:
            # Use binomial expansion for fractional differencing
            n = n_points
            weights = np.zeros(n)
            weights[0] = 1
            
            for k in range(1, n):
                weights[k] = weights[k-1] * (d + k - 1) / k
                
            # Apply weights
            frac_diff = np.convolve(white_noise, weights, mode='same')
        else:
            frac_diff = white_noise
            
        # Apply AR and MA filters if specified
        if ar_params:
            # AR filter
            for i, ar_param in enumerate(ar_params):
                if i + 1 < len(frac_diff):
                    frac_diff[i+1:] += ar_param * frac_diff[:-i-1]
                    
        if ma_params:
            # MA filter
            ma_filtered = frac_diff.copy()
            for i, ma_param in enumerate(ma_params):
                if i + 1 < len(frac_diff):
                    ma_filtered[i+1:] += ma_param * white_noise[:-i-1]
            frac_diff = ma_filtered
            
        return frac_diff
        
    def multifractal_cascade(self, n_points: int, hurst_exponent: float,
                           multifractality: float = 0.1, sigma: float = 1.0) -> np.ndarray:
        """
        Generate multifractal cascade process.
        
        Parameters
        ----------
        n_points : int
            Number of points to generate
        hurst_exponent : float
            Base Hurst exponent
        multifractality : float
            Degree of multifractality (0 = monofractal, >0 = multifractal)
        sigma : float
            Standard deviation of the process
            
        Returns
        -------
        np.ndarray
            Multifractal time series
        """
        # Start with fBm
        base_process = self.fractional_brownian_motion(n_points, hurst_exponent, sigma)
        
        if multifractality > 0:
            # Add multifractal modulation
            modulation = np.ones(n_points)
            
            # Create cascade structure
            n_levels = int(np.log2(n_points))
            for level in range(n_levels):
                scale = 2 ** level
                if scale < n_points:
                    # Random modulation at this scale
                    level_mod = np.random.lognormal(0, multifractality, n_points // scale)
                    # Upsample to full length
                    level_mod_full = np.repeat(level_mod, scale)[:n_points]
                    modulation *= level_mod_full
                    
            # Normalize modulation
            modulation = (modulation - np.mean(modulation)) / np.std(modulation)
            
            # Apply modulation
            multifractal_process = base_process * (1 + 0.5 * modulation)
        else:
            multifractal_process = base_process
            
        return multifractal_process
        
    def generate_benchmark_dataset(self, n_points: int = 1000, 
                                 hurst_exponents: List[float] = None) -> Dict[str, np.ndarray]:
        """
        Generate a comprehensive benchmark dataset.
        
        Parameters
        ----------
        n_points : int
            Number of points for each time series
        hurst_exponents : List[float]
            List of Hurst exponents to test
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing various synthetic datasets
        """
        if hurst_exponents is None:
            hurst_exponents = [0.3, 0.5, 0.7, 0.9]
            
        datasets = {}
        
        # Generate fBm processes
        for H in hurst_exponents:
            datasets[f'fbm_H{H}'] = self.fractional_brownian_motion(n_points, H)
            datasets[f'fgn_H{H}'] = self.fractional_gaussian_noise(n_points, H)
            
        # Generate ARFIMA processes
        for d in [0.1, 0.2, 0.3]:
            H = 0.5 + d
            datasets[f'arfima_d{d}'] = self.arfima_process(n_points, d)
            
        # Generate multifractal processes
        for H in [0.5, 0.7]:
            datasets[f'multifractal_H{H}'] = self.multifractal_cascade(n_points, H, 0.2)
            
        # Add white noise (H = 0.5)
        datasets['white_noise'] = np.random.normal(0, 1, n_points)
        
        # Add random walk (H = 1.0)
        datasets['random_walk'] = np.cumsum(np.random.normal(0, 1, n_points))
        
        return datasets
