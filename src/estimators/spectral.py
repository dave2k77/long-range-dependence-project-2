"""
Spectral Long-Range Dependence Estimators

This module contains implementations of spectral methods for estimating
long-range dependence, including Whittle MLE, Periodogram, and GPH methods.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import periodogram
import logging
import scipy.signal

from .base import BaseEstimator

logger = logging.getLogger(__name__)


class WhittleMLEEstimator(BaseEstimator):
    """
    Whittle Maximum Likelihood Estimation (MLE) estimator.
    
    Whittle MLE is a spectral method that estimates long-range dependence
    parameters by maximizing the likelihood function in the frequency domain.
    """
    
    def __init__(self, name: str = "WhittleMLE", **kwargs):
        """
        Initialize Whittle MLE estimator.
        
        Parameters
        ----------
        name : str
            Name identifier for the estimator
        **kwargs
            Additional parameters including:
            - frequency_range: Range of frequencies to use (default: [0.01, 0.5])
            - initial_guess: Initial guess for parameters (default: [0.5])
            - optimization_method: Optimization method (default: 'L-BFGS-B')
        """
        super().__init__(name=name, **kwargs)
        self.frequency_range = kwargs.get('frequency_range', [0.01, 0.5])
        self.initial_guess = kwargs.get('initial_guess', [0.5])
        self.optimization_method = kwargs.get('optimization_method', 'L-BFGS-B')
        self.data = None
        self.frequencies = None
        self.periodogram_values = None
        
    def fit(self, data: np.ndarray, **kwargs) -> 'WhittleMLEEstimator':
        """
        Fit the Whittle MLE estimator to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        self : WhittleMLEEstimator
            Fitted estimator instance
        """
        self.data = np.asarray(data, dtype=float)
        self._validate_data()
        return self
    
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Estimate long-range dependence using Whittle MLE.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Input time series data (if not provided, uses fitted data)
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing Whittle MLE estimation results
        """
        if data is not None:
            self.fit(data, **kwargs)
        
        if self.data is None:
            raise ValueError("No data provided. Call fit() first.")
        
        # Calculate periodogram
        self._calculate_periodogram()
        
        # Filter frequencies within specified range
        self._filter_frequencies()
        
        # Optimize likelihood function
        alpha_estimate, optimization_success = self._optimize_likelihood()
        
        # Calculate Hurst exponent
        hurst_exponent = (alpha_estimate + 1) / 2
        
        results = {
            'alpha': alpha_estimate,
            'hurst_exponent': hurst_exponent,
            'optimization_success': optimization_success,
            'frequencies': self.frequencies,
            'periodogram_values': self.periodogram_values,
            'method': 'WhittleMLE'
        }
        
        return results
    
    def _validate_data(self):
        """Validate input data."""
        if len(self.data) < 100:
            logger.warning("Data length is small for reliable Whittle MLE estimation")
        
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
    
    def _calculate_periodogram(self):
        """Calculate periodogram of the data."""
        # Remove mean
        centered_data = self.data - np.mean(self.data)
        
        # Calculate periodogram
        self.frequencies, self.periodogram_values = periodogram(
            centered_data, 
            fs=1.0, 
            return_onesided=True
        )
        
        # Convert to one-sided frequencies
        self.frequencies = self.frequencies[1:]  # Remove zero frequency
        self.periodogram_values = self.periodogram_values[1:]
    
    def _filter_frequencies(self):
        """Filter frequencies within the specified range."""
        min_freq, max_freq = self.frequency_range
        
        # Find indices within frequency range
        valid_indices = (self.frequencies >= min_freq) & (self.frequencies <= max_freq)
        
        self.frequencies = self.frequencies[valid_indices]
        self.periodogram_values = self.periodogram_values[valid_indices]
    
    def _whittle_likelihood(self, alpha: float) -> float:
        """
        Calculate Whittle likelihood function.
        
        Parameters
        ----------
        alpha : float
            Long-range dependence parameter
            
        Returns
        -------
        float
            Negative log-likelihood (to be minimized)
        """
        # Theoretical power spectrum for ARFIMA(0,d,0) process
        # S(f) = |1 - exp(-2πif)|^(-2d) where d = (α-1)/2
        
        d = (alpha - 1) / 2
        
        # Calculate theoretical spectrum
        theoretical_spectrum = np.abs(1 - np.exp(-2j * np.pi * self.frequencies)) ** (-2 * d)
        
        # Whittle likelihood
        log_likelihood = np.sum(
            np.log(theoretical_spectrum) + self.periodogram_values / theoretical_spectrum
        )
        
        return -log_likelihood  # Return negative for minimization
    
    def _optimize_likelihood(self) -> Tuple[float, bool]:
        """
        Optimize the Whittle likelihood function.
        
        Returns
        -------
        Tuple[float, bool]
            Estimated alpha parameter and optimization success status
        """
        try:
            # Bounds for alpha (typically between 0 and 2)
            bounds = [(0.0, 2.0)]
            
            # Optimize
            result = minimize(
                self._whittle_likelihood,
                self.initial_guess,
                method=self.optimization_method,
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x[0], True
            else:
                logger.warning(f"Optimization failed: {result.message}")
                return result.x[0], False
                
        except Exception as e:
            logger.error(f"Error in likelihood optimization: {str(e)}")
            return self.initial_guess[0], False


class PeriodogramEstimator(BaseEstimator):
    """
    Periodogram-based Long-Range Dependence estimator.
    
    This estimator uses the power spectral density (periodogram) to estimate
    the Hurst exponent by analyzing the scaling behavior of the spectrum.
    """
    
    def __init__(self, name: str = "Periodogram", **kwargs):
        super().__init__(name=name, **kwargs)
        self.window = kwargs.get('window', 'hann')
        self.nperseg = kwargs.get('nperseg', None)
        self.nfft = kwargs.get('nfft', None)
        self.frequency_range = kwargs.get('frequency_range', [0.01, 0.5])
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.data = None
        self.frequencies = None
        self.periodogram = None
        self.hurst_exponent = None
        self.confidence_interval = None
        
    def fit(self, data: np.ndarray, **kwargs) -> 'PeriodogramEstimator':
        """Fit the Periodogram estimator to the data."""
        self.data = np.asarray(data, dtype=float)
        self._validate_data()
        return self
        
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """Estimate Hurst exponent using periodogram analysis."""
        if data is not None:
            self.fit(data, **kwargs)
            
        if self.data is None:
            raise ValueError("No data available. Call fit() first.")
            
        # Calculate periodogram
        self._calculate_periodogram()
        
        # Filter frequencies and fit scaling law
        self._filter_and_fit()
        
        # Calculate confidence interval
        self._calculate_confidence_interval()
        
        return self.get_results()
        
    def _validate_data(self):
        """Validate input data."""
        if len(self.data) < 100:
            raise ValueError("Data must have at least 100 points for reliable periodogram analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
            
    def _calculate_periodogram(self):
        """Calculate the periodogram of the data."""
        # Set default parameters if not provided
        if self.nperseg is None:
            self.nperseg = min(256, len(self.data) // 4)
        if self.nfft is None:
            self.nfft = max(512, 2 * self.nperseg)
            
        # Calculate periodogram
        frequencies, periodogram = scipy.signal.periodogram(
            self.data,
            fs=1.0,
            window=self.window,
            nfft=self.nfft,
            scaling='density'
        )
        
        self.frequencies = frequencies
        self.periodogram = periodogram
        
    def _filter_and_fit(self):
        """Filter frequencies and fit scaling law to extract Hurst exponent."""
        # Filter frequencies within the specified range
        freq_mask = (self.frequencies >= self.frequency_range[0]) & \
                   (self.frequencies <= self.frequency_range[1])
        
        if np.sum(freq_mask) < 10:
            self.hurst_exponent = np.nan
            self.scaling_error = np.nan
            return
            
        freq_filtered = self.frequencies[freq_mask]
        periodogram_filtered = self.periodogram[freq_mask]
        
        # Remove zero and negative values
        positive_mask = periodogram_filtered > 0
        if np.sum(positive_mask) < 5:
            self.hurst_exponent = np.nan
            self.scaling_error = np.nan
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
            
        except (np.linalg.LinAlgError, ValueError):
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
        n_bootstrap = 1000
        hurst_bootstrap = []
        
        for _ in range(n_bootstrap):
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
            'beta': getattr(self, 'beta', None),
            'scaling_error': getattr(self, 'scaling_error', None),
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'fitted_frequencies': getattr(self, 'fitted_frequencies', None),
            'fitted_periodogram': getattr(self, 'fitted_periodogram', None),
            'parameters': {
                'window': self.window,
                'nperseg': self.nperseg,
                'nfft': self.nfft,
                'frequency_range': self.frequency_range
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
                'spectral_exponent': getattr(self, 'beta', None)
            }
            
        return results


class GPHEstimator(BaseEstimator):
    """
    Geweke-Porter-Hudak (GPH) estimator for long-range dependence.
    
    The GPH method estimates the fractional differencing parameter d
    by analyzing the periodogram in the low-frequency region.
    """
    
    def __init__(self, name: str = "GPH", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_frequencies = kwargs.get('num_frequencies', 50)
        self.frequency_threshold = kwargs.get('frequency_threshold', 0.1)
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.data = None
        self.frequencies = None
        self.periodogram = None
        self.fractional_d = None
        self.hurst_exponent = None
        self.confidence_interval = None
        
    def fit(self, data: np.ndarray, **kwargs) -> 'GPHEstimator':
        """Fit the GPH estimator to the data."""
        self.data = np.asarray(data, dtype=float)
        self._validate_data()
        return self
        
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """Estimate fractional differencing parameter using GPH method."""
        if data is not None:
            self.fit(data, **kwargs)
            
        if self.data is None:
            raise ValueError("No data available. Call fit() first.")
            
        # Calculate periodogram
        self._calculate_periodogram()
        
        # Apply GPH regression
        self._gph_regression()
        
        # Calculate confidence interval
        self._calculate_confidence_interval()
        
        return self.get_results()
        
    def _validate_data(self):
        """Validate input data."""
        if len(self.data) < 200:
            raise ValueError("Data must have at least 200 points for reliable GPH analysis")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
            
    def _calculate_periodogram(self):
        """Calculate the periodogram of the data."""
        # Use Welch's method for better spectral estimation
        frequencies, periodogram = scipy.signal.welch(
            self.data,
            fs=1.0,
            window='hann',
            nperseg=min(256, len(self.data) // 4),
            noverlap=0
        )
        
        self.frequencies = frequencies
        self.periodogram = periodogram
        
    def _gph_regression(self):
        """Apply GPH regression to estimate fractional differencing parameter."""
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
            
        except (np.linalg.LinAlgError, ValueError):
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
        n_bootstrap = 1000
        d_bootstrap = []
        
        for _ in range(n_bootstrap):
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
            'regression_error': getattr(self, 'regression_error', None),
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'gph_x': getattr(self, 'gph_x', None),
            'gph_y': getattr(self, 'gph_y', None),
            'gph_y_pred': getattr(self, 'gph_y_pred', None),
            'intercept': getattr(self, 'intercept', None),
            'parameters': {
                'num_frequencies': self.num_frequencies,
                'frequency_threshold': self.frequency_threshold
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
