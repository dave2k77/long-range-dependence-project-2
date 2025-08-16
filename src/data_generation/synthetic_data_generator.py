"""
Synthetic Data Generator for Long-Range Dependence Analysis

This module provides comprehensive synthetic data generation capabilities for:
- Base models: ARFIMA, fBm, fGn
- Realistic confounds: Non-stationarity, heavy-tails, artifacts
- Domain-specific patterns: EEG, Hydrology/Climate, Financial

Designed for benchmarking physics-based fractional machine learning models.
"""

import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
import warnings
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Suppress warnings for numerical stability
warnings.filterwarnings('ignore', category=RuntimeWarning)


class DataType(Enum):
    """Types of synthetic data."""
    ARFIMA = "arfima"
    FRACTIONAL_BROWNIAN_MOTION = "fbm"
    FRACTIONAL_GAUSSIAN_NOISE = "fgn"
    MIXED = "mixed"


class ConfoundType(Enum):
    """Types of realistic confounds."""
    NON_STATIONARITY = "non_stationarity"
    HEAVY_TAILS = "heavy_tails"
    BASELINE_DRIFT = "baseline_drift"
    ARTIFACTS = "artifacts"
    SEASONALITY = "seasonality"
    TREND_CHANGES = "trend_changes"
    VOLATILITY_CLUSTERING = "volatility_clustering"
    REGIME_CHANGES = "regime_changes"
    JUMPS = "jumps"
    NOISE = "noise"


class DomainType(Enum):
    """Domain-specific data types."""
    EEG = "eeg"
    HYDROLOGY = "hydrology"
    CLIMATE = "climate"
    FINANCIAL = "financial"
    GENERAL = "general"


@dataclass
class DataSpecification:
    """Specification for synthetic data generation."""
    n_points: int = 1000
    hurst_exponent: float = 0.7
    alpha_stable_alpha: float = 2.0  # For heavy tails (2.0 = Gaussian)
    alpha_stable_beta: float = 0.0   # Skewness
    alpha_stable_gamma: float = 1.0  # Scale
    alpha_stable_delta: float = 0.0  # Location
    
    # ARFIMA parameters
    ar_coeffs: Optional[List[float]] = None
    ma_coeffs: Optional[List[float]] = None
    d_parameter: Optional[float] = None
    
    # Confound parameters
    confound_strength: float = 0.1
    noise_level: float = 0.05
    artifact_probability: float = 0.01
    
    # Domain-specific parameters
    domain_type: DomainType = DomainType.GENERAL
    seasonal_period: Optional[int] = None
    trend_change_points: Optional[List[int]] = None
    volatility_regimes: Optional[List[float]] = None
    
    # Validation
    def validate(self):
        """Validate the data specification."""
        if self.n_points <= 0:
            raise ValueError("n_points must be positive")
        if not 0 < self.hurst_exponent < 1:
            raise ValueError("hurst_exponent must be between 0 and 1")
        if not 0 < self.alpha_stable_alpha <= 2:
            raise ValueError("alpha_stable_alpha must be between 0 and 2")
        if self.confound_strength < 0 or self.confound_strength > 1:
            raise ValueError("confound_strength must be between 0 and 1")
        if self.noise_level < 0:
            raise ValueError("noise_level must be non-negative")


class SyntheticDataGenerator:
    """
    Comprehensive synthetic data generator for LRD analysis.
    
    This generator creates realistic time series with controlled properties
    for benchmarking physics-based fractional machine learning models.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the synthetic data generator.
        
        Parameters:
        -----------
        random_seed : int, optional
            Random seed for reproducible results
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            logger.info(f"Set random seed to {random_seed}")
        
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Validate required dependencies."""
        try:
            import scipy
            logger.info(f"SciPy version: {scipy.__version__}")
        except ImportError:
            logger.warning("SciPy not available. Some features may be limited.")
    
    def generate_data(self, spec: DataSpecification, 
                     confounds: Optional[List[ConfoundType]] = None) -> Dict[str, Any]:
        """
        Generate synthetic data according to specification.
        
        Parameters:
        -----------
        spec : DataSpecification
            Data generation specification
        confounds : List[ConfoundType], optional
            List of confounds to apply
            
        Returns:
        --------
        Dict[str, Any]
            Generated data with metadata
        """
        spec.validate()
        
        logger.info(f"Generating {spec.n_points} points with H={spec.hurst_exponent:.3f}")
        logger.info(f"Domain: {spec.domain_type.value}")
        
        # Generate base data
        if spec.domain_type == DomainType.EEG:
            base_data = self._generate_eeg_base(spec)
        elif spec.domain_type in [DomainType.HYDROLOGY, DomainType.CLIMATE]:
            base_data = self._generate_hydrology_base(spec)
        elif spec.domain_type == DomainType.FINANCIAL:
            base_data = self._generate_financial_base(spec)
        else:
            base_data = self._generate_general_base(spec)
        
        # Apply confounds
        if confounds:
            contaminated_data = self._apply_confounds(base_data, spec, confounds)
        else:
            contaminated_data = base_data
        
        # Add noise
        if spec.noise_level > 0:
            contaminated_data = self._add_noise(contaminated_data, spec.noise_level)
        
        # Validate final data
        self._validate_generated_data(contaminated_data)
        
        # Prepare results
        results = {
            'data': contaminated_data,
            'specification': spec,
            'confounds_applied': confounds or [],
            'metadata': self._generate_metadata(spec, base_data, contaminated_data)
        }
        
        logger.info("Data generation completed successfully")
        return results
    
    def _generate_eeg_base(self, spec: DataSpecification) -> np.ndarray:
        """Generate EEG-like base data."""
        logger.info("Generating EEG-like base data")
        
        # Start with fGn as base
        base_data = self._generate_fractional_gaussian_noise(
            spec.n_points, spec.hurst_exponent
        )
        
        # Add EEG-specific characteristics
        # 1. Multiple frequency components
        frequencies = [0.5, 1.0, 2.0, 4.0, 8.0, 13.0, 20.0]  # Hz
        amplitudes = [0.3, 0.5, 0.8, 1.0, 0.6, 0.4, 0.2]
        
        t = np.linspace(0, spec.n_points / 100, spec.n_points)  # 100 Hz sampling
        
        for freq, amp in zip(frequencies, amplitudes):
            base_data += amp * np.sin(2 * np.pi * freq * t)
        
        # 2. Baseline drift (slow trend)
        baseline_drift = 0.1 * np.cumsum(np.random.randn(spec.n_points) * 0.01)
        base_data += baseline_drift
        
        return base_data
    
    def _generate_hydrology_base(self, spec: DataSpecification) -> np.ndarray:
        """Generate hydrology/climate-like base data."""
        logger.info("Generating hydrology/climate-like base data")
        
        # Start with fBm as base
        base_data = self._generate_fractional_brownian_motion(
            spec.n_points, spec.hurst_exponent
        )
        
        # Add seasonal patterns
        if spec.seasonal_period:
            seasonal_period = spec.seasonal_period
        else:
            seasonal_period = spec.n_points // 4  # Default seasonal pattern
        
        seasonal_pattern = 0.5 * np.sin(2 * np.pi * np.arange(spec.n_points) / seasonal_period)
        base_data += seasonal_pattern
        
        # Add trend changes
        if spec.trend_change_points:
            for change_point in spec.trend_change_points:
                if change_point < spec.n_points:
                    trend_slope = np.random.uniform(-0.1, 0.1)
                    base_data[change_point:] += trend_slope * np.arange(len(base_data[change_point:]))
        
        return base_data
    
    def _generate_financial_base(self, spec: DataSpecification) -> np.ndarray:
        """Generate financial-like base data."""
        logger.info("Generating financial-like base data")
        
        # Start with fGn as base for returns
        returns = self._generate_fractional_gaussian_noise(
            spec.n_points, spec.hurst_exponent
        )
        
        # Scale returns to reasonable levels (avoid extreme values)
        returns *= 0.01  # 1% daily returns
        
        # Add volatility clustering
        if spec.volatility_regimes:
            volatility_levels = spec.volatility_regimes
        else:
            volatility_levels = [0.5, 1.0, 1.5, 0.8]  # Default volatility regimes
        
        # Create volatility process
        volatility = np.ones(spec.n_points)
        regime_length = spec.n_points // len(volatility_levels)
        
        for i, vol_level in enumerate(volatility_levels):
            start_idx = i * regime_length
            end_idx = min((i + 1) * regime_length, spec.n_points)
            volatility[start_idx:end_idx] = vol_level
        
        # Apply volatility to returns
        returns *= volatility
        
        # Convert to price series (start from 100 to avoid extreme values)
        price_series = 100 * np.exp(np.cumsum(returns))
        
        # Normalize to reasonable range
        price_series = (price_series - np.mean(price_series)) / np.std(price_series)
        
        return price_series
    
    def _generate_general_base(self, spec: DataSpecification) -> np.ndarray:
        """Generate general-purpose base data."""
        logger.info("Generating general-purpose base data")
        
        if spec.ar_coeffs or spec.ma_coeffs or spec.d_parameter is not None:
            # Generate ARFIMA data
            return self._generate_arfima_data(spec)
        else:
            # Generate fGn as default
            return self._generate_fractional_gaussian_noise(
                spec.n_points, spec.hurst_exponent
            )
    
    def _generate_fractional_gaussian_noise(self, n_points: int, hurst: float) -> np.ndarray:
        """Generate fractional Gaussian noise using FFT method."""
        # Use FFT method for efficient generation
        freqs = np.fft.fftfreq(n_points)
        
        # Power spectrum: S(f) ~ |f|^(-2H-1)
        power_spectrum = np.abs(freqs) ** (-2 * hurst - 1)
        power_spectrum[0] = 0  # Remove DC component
        
        # Generate complex random phases
        phases = np.random.uniform(0, 2 * np.pi, n_points)
        complex_spectrum = np.sqrt(power_spectrum) * np.exp(1j * phases)
        
        # Inverse FFT to get time series
        fgn = np.real(np.fft.ifft(complex_spectrum))
        
        # Normalize
        fgn = (fgn - np.mean(fgn)) / np.std(fgn)
        
        return fgn
    
    def _generate_fractional_brownian_motion(self, n_points: int, hurst: float) -> np.ndarray:
        """Generate fractional Brownian motion by integrating fGn."""
        fgn = self._generate_fractional_gaussian_noise(n_points, hurst)
        fbm = np.cumsum(fgn)
        
        # Normalize
        fbm = (fbm - np.mean(fbm)) / np.std(fbm)
        
        return fbm
    
    def _generate_arfima_data(self, spec: DataSpecification) -> np.ndarray:
        """Generate ARFIMA data."""
        logger.info("Generating ARFIMA data")
        
        n_points = spec.n_points
        d = spec.d_parameter if spec.d_parameter is not None else 0.3
        
        # Generate white noise
        white_noise = np.random.randn(n_points)
        
        # Apply fractional differencing
        if d != 0:
            # Use binomial expansion for fractional differencing
            coeffs = self._fractional_differencing_coefficients(d, n_points)
            fgn = np.convolve(white_noise, coeffs, mode='same')
        else:
            fgn = white_noise
        
        # Apply AR coefficients if specified
        if spec.ar_coeffs:
            ar_data = self._apply_ar_coefficients(fgn, spec.ar_coeffs)
        else:
            ar_data = fgn
        
        # Apply MA coefficients if specified
        if spec.ma_coeffs:
            ma_data = self._apply_ma_coefficients(ar_data, spec.ma_coeffs)
        else:
            ma_data = ar_data
        
        # Normalize
        ma_data = (ma_data - np.mean(ma_data)) / np.std(ma_data)
        
        return ma_data
    
    def _fractional_differencing_coefficients(self, d: float, max_lag: int) -> np.ndarray:
        """Generate fractional differencing coefficients."""
        coeffs = np.zeros(max_lag)
        coeffs[0] = 1.0
        
        for k in range(1, max_lag):
            coeffs[k] = coeffs[k-1] * (d - k + 1) / k
        
        return coeffs
    
    def _apply_ar_coefficients(self, data: np.ndarray, ar_coeffs: List[float]) -> np.ndarray:
        """Apply AR coefficients to data."""
        result = np.copy(data)
        p = len(ar_coeffs)
        
        for t in range(p, len(data)):
            for i, coeff in enumerate(ar_coeffs):
                result[t] += coeff * result[t - i - 1]
        
        return result
    
    def _apply_ma_coefficients(self, data: np.ndarray, ma_coeffs: List[float]) -> np.ndarray:
        """Apply MA coefficients to data."""
        result = np.copy(data)
        q = len(ma_coeffs)
        
        for t in range(q, len(data)):
            for i, coeff in enumerate(ma_coeffs):
                result[t] += coeff * data[t - i - 1]
        
        return result
    
    def _apply_confounds(self, base_data: np.ndarray, spec: DataSpecification, 
                        confounds: List[ConfoundType]) -> np.ndarray:
        """Apply specified confounds to the base data."""
        contaminated_data = np.copy(base_data)
        
        for confound in confounds:
            logger.info(f"Applying confound: {confound.value}")
            
            if confound == ConfoundType.NON_STATIONARITY:
                contaminated_data = self._add_non_stationarity(contaminated_data, spec)
            elif confound == ConfoundType.HEAVY_TAILS:
                contaminated_data = self._add_heavy_tails(contaminated_data, spec)
            elif confound == ConfoundType.BASELINE_DRIFT:
                contaminated_data = self._add_baseline_drift(contaminated_data, spec)
            elif confound == ConfoundType.ARTIFACTS:
                contaminated_data = self._add_artifacts(contaminated_data, spec)
            elif confound == ConfoundType.SEASONALITY:
                contaminated_data = self._add_seasonality(contaminated_data, spec)
            elif confound == ConfoundType.TREND_CHANGES:
                contaminated_data = self._add_trend_changes(contaminated_data, spec)
            elif confound == ConfoundType.VOLATILITY_CLUSTERING:
                contaminated_data = self._add_volatility_clustering(contaminated_data, spec)
            elif confound == ConfoundType.REGIME_CHANGES:
                contaminated_data = self._add_regime_changes(contaminated_data, spec)
            elif confound == ConfoundType.JUMPS:
                contaminated_data = self._add_jumps(contaminated_data, spec)
        
        return contaminated_data
    
    def _add_non_stationarity(self, data: np.ndarray, spec: DataSpecification) -> np.ndarray:
        """Add non-stationarity to the data."""
        strength = spec.confound_strength
        
        # Add time-varying mean
        time_trend = np.linspace(0, strength, len(data))
        data += time_trend
        
        # Add time-varying variance
        variance_trend = 1 + strength * np.sin(2 * np.pi * np.arange(len(data)) / (len(data) / 4))
        data *= np.sqrt(variance_trend)
        
        return data
    
    def _add_heavy_tails(self, data: np.ndarray, spec: DataSpecification) -> np.ndarray:
        """Add heavy-tailed characteristics using alpha-stable distribution."""
        strength = spec.confound_strength
        
        # Generate alpha-stable noise with corrected parameter names
        try:
            # Try newer SciPy parameter names
            alpha_stable_noise = stats.levy_stable.rvs(
                alpha=spec.alpha_stable_alpha,
                beta=spec.alpha_stable_beta,
                loc=spec.alpha_stable_delta,
                scale=spec.alpha_stable_gamma * strength,
                size=len(data)
            )
        except TypeError:
            # Fallback to older parameter names
            alpha_stable_noise = stats.levy_stable.rvs(
                alpha=spec.alpha_stable_alpha,
                beta=spec.alpha_stable_beta,
                gamma=spec.alpha_stable_gamma * strength,
                delta=spec.alpha_stable_delta,
                size=len(data)
            )
        
        # Mix with original data
        data = (1 - strength) * data + strength * alpha_stable_noise
        
        return data
    
    def _add_baseline_drift(self, data: np.ndarray, spec: DataSpecification) -> np.ndarray:
        """Add baseline drift to the data."""
        strength = spec.confound_strength
        
        # Generate slow baseline drift
        drift = np.cumsum(np.random.randn(len(data)) * strength * 0.01)
        data += drift
        
        return data
    
    def _add_artifacts(self, data: np.ndarray, spec: DataSpecification) -> np.ndarray:
        """Add artifacts typical in real data."""
        strength = spec.confound_strength
        prob = spec.artifact_probability
        
        # Add random spikes
        spike_indices = np.random.choice(len(data), size=int(len(data) * prob), replace=False)
        for idx in spike_indices:
            spike_amplitude = np.random.uniform(-strength, strength)
            data[idx] += spike_amplitude
        
        # Add step changes
        step_indices = np.random.choice(len(data), size=int(len(data) * prob * 0.1), replace=False)
        for idx in step_indices:
            step_amplitude = np.random.uniform(-strength * 2, strength * 2)
            data[idx:] += step_amplitude
        
        return data
    
    def _add_seasonality(self, data: np.ndarray, spec: DataSpecification) -> np.ndarray:
        """Add seasonal patterns to the data."""
        strength = spec.confound_strength
        
        if spec.seasonal_period:
            period = spec.seasonal_period
        else:
            period = len(data) // 4
        
        # Add multiple seasonal components
        seasonal_pattern = np.zeros(len(data))
        for freq in [1, 2, 4]:  # Multiple seasonal frequencies
            seasonal_pattern += (strength / freq) * np.sin(2 * np.pi * np.arange(len(data)) / (period / freq))
        
        data += seasonal_pattern
        return data
    
    def _add_trend_changes(self, data: np.ndarray, spec: DataSpecification) -> np.ndarray:
        """Add trend changes to the data."""
        strength = spec.confound_strength
        
        if spec.trend_change_points:
            change_points = spec.trend_change_points
        else:
            # Generate random change points
            n_changes = max(1, len(data) // 500)
            change_points = np.sort(np.random.choice(len(data), size=n_changes, replace=False))
        
        # Apply trend changes
        for i, change_point in enumerate(change_points):
            if change_point < len(data):
                trend_slope = np.random.uniform(-strength, strength)
                data[change_point:] += trend_slope * np.arange(len(data[change_point:]))
        
        return data
    
    def _add_volatility_clustering(self, data: np.ndarray, spec: DataSpecification) -> np.ndarray:
        """Add volatility clustering to the data."""
        strength = spec.confound_strength
        
        # Generate GARCH-like volatility process
        volatility = np.ones(len(data))
        for t in range(1, len(data)):
            volatility[t] = np.sqrt(0.9 * volatility[t-1]**2 + 0.1 * data[t-1]**2)
        
        # Normalize and scale
        volatility = volatility / np.mean(volatility)
        volatility = 1 + strength * (volatility - 1)
        
        # Apply volatility
        data *= volatility
        
        return data
    
    def _add_regime_changes(self, data: np.ndarray, spec: DataSpecification) -> np.ndarray:
        """Add regime changes to the data."""
        strength = spec.confound_strength
        
        # Generate regime switching
        n_regimes = 3
        regime_length = len(data) // n_regimes
        
        for i in range(n_regimes):
            start_idx = i * regime_length
            end_idx = min((i + 1) * regime_length, len(data))
            
            # Different regime characteristics
            if i == 0:
                # High volatility regime
                data[start_idx:end_idx] *= (1 + strength)
            elif i == 1:
                # Low volatility regime
                data[start_idx:end_idx] *= (1 - strength * 0.5)
            else:
                # Trend regime
                trend = np.linspace(0, strength, end_idx - start_idx)
                data[start_idx:end_idx] += trend
        
        return data
    
    def _add_jumps(self, data: np.ndarray, spec: DataSpecification) -> np.ndarray:
        """Add jumps to the data."""
        strength = spec.confound_strength
        prob = spec.artifact_probability * 0.1  # Lower probability for jumps
        
        # Generate random jumps
        jump_indices = np.random.choice(len(data), size=int(len(data) * prob), replace=False)
        for idx in jump_indices:
            jump_amplitude = np.random.uniform(-strength * 3, strength * 3)
            data[idx:] += jump_amplitude
        
        return data
    
    def _add_noise(self, data: np.ndarray, noise_level: float) -> np.ndarray:
        """Add measurement noise to the data."""
        noise = np.random.normal(0, noise_level, len(data))
        data += noise
        return data
    
    def _validate_generated_data(self, data: np.ndarray):
        """Validate the generated data."""
        if np.any(np.isnan(data)):
            raise ValueError("Generated data contains NaN values")
        if np.any(np.isinf(data)):
            raise ValueError("Generated data contains infinite values")
        if len(data) == 0:
            raise ValueError("Generated data is empty")
        
        logger.info(f"Data validation passed: {len(data)} points, "
                   f"mean={np.mean(data):.4f}, std={np.std(data):.4f}")
    
    def _generate_metadata(self, spec: DataSpecification, base_data: np.ndarray, 
                          final_data: np.ndarray) -> Dict[str, Any]:
        """Generate metadata about the generated data."""
        metadata = {
            'generation_timestamp': np.datetime64('now'),
            'base_data_properties': {
                'mean': float(np.mean(base_data)),
                'std': float(np.std(base_data)),
                'min': float(np.min(base_data)),
                'max': float(np.max(base_data)),
                'skewness': float(stats.skew(base_data)),
                'kurtosis': float(stats.kurtosis(base_data))
            },
            'final_data_properties': {
                'mean': float(np.mean(final_data)),
                'std': float(np.std(final_data)),
                'min': float(np.min(final_data)),
                'max': float(np.max(final_data)),
                'skewness': float(stats.skew(final_data)),
                'kurtosis': float(stats.kurtosis(final_data))
            },
            'theoretical_properties': {
                'hurst_exponent': spec.hurst_exponent,
                'alpha_stable_alpha': spec.alpha_stable_alpha,
                'expected_lrd': spec.hurst_exponent > 0.5
            }
        }
        
        return metadata


def create_standard_dataset_specifications() -> Dict[str, DataSpecification]:
    """Create standard dataset specifications for common use cases."""
    specs = {}
    
    # EEG-like data
    specs['eeg_resting'] = DataSpecification(
        n_points=5000,
        hurst_exponent=0.6,
        domain_type=DomainType.EEG,
        confound_strength=0.15,
        noise_level=0.02,
        artifact_probability=0.02
    )
    
    specs['eeg_active'] = DataSpecification(
        n_points=5000,
        hurst_exponent=0.4,
        domain_type=DomainType.EEG,
        confound_strength=0.25,
        noise_level=0.05,
        artifact_probability=0.05
    )
    
    # Hydrology/Climate data
    specs['hydrology_daily'] = DataSpecification(
        n_points=3650,  # 10 years
        hurst_exponent=0.8,
        domain_type=DomainType.HYDROLOGY,
        seasonal_period=365,
        confound_strength=0.2,
        noise_level=0.1
    )
    
    specs['climate_monthly'] = DataSpecification(
        n_points=1200,  # 100 years
        hurst_exponent=0.7,
        domain_type=DomainType.CLIMATE,
        seasonal_period=12,
        confound_strength=0.15,
        noise_level=0.08
    )
    
    # Financial data
    specs['financial_daily'] = DataSpecification(
        n_points=2520,  # 10 years
        hurst_exponent=0.55,
        domain_type=DomainType.FINANCIAL,
        volatility_regimes=[0.8, 1.2, 0.6, 1.5],
        confound_strength=0.3,
        noise_level=0.05
    )
    
    specs['financial_high_freq'] = DataSpecification(
        n_points=100000,  # High-frequency data
        hurst_exponent=0.52,
        domain_type=DomainType.FINANCIAL,
        volatility_regimes=[0.5, 1.0, 2.0, 0.8],
        confound_strength=0.4,
        noise_level=0.02
    )
    
    # ARFIMA models
    specs['arfima_stationary'] = DataSpecification(
        n_points=2000,
        hurst_exponent=0.6,
        d_parameter=0.1,
        ar_coeffs=[0.5, -0.3],
        ma_coeffs=[0.2],
        confound_strength=0.1,
        noise_level=0.05
    )
    
    specs['arfima_nonstationary'] = DataSpecification(
        n_points=2000,
        hurst_exponent=0.8,
        d_parameter=0.4,
        ar_coeffs=[0.7, -0.5],
        ma_coeffs=[0.3, -0.1],
        confound_strength=0.2,
        noise_level=0.08
    )
    
    return specs


def generate_benchmark_dataset(generator: SyntheticDataGenerator, 
                             spec_name: str = 'eeg_resting',
                             confounds: Optional[List[ConfoundType]] = None) -> Dict[str, Any]:
    """
    Generate a benchmark dataset using standard specifications.
    
    Parameters:
    -----------
    generator : SyntheticDataGenerator
        Initialized data generator
    spec_name : str
        Name of the standard specification to use
    confounds : List[ConfoundType], optional
        List of confounds to apply
        
    Returns:
    --------
    Dict[str, Any]
        Generated benchmark dataset
    """
    specs = create_standard_dataset_specifications()
    
    if spec_name not in specs:
        raise ValueError(f"Unknown specification: {spec_name}. "
                       f"Available: {list(specs.keys())}")
    
    spec = specs[spec_name]
    
    if confounds is None:
        # Use domain-appropriate default confounds
        if spec.domain_type == DomainType.EEG:
            confounds = [ConfoundType.BASELINE_DRIFT, ConfoundType.ARTIFACTS, ConfoundType.NOISE]
        elif spec.domain_type in [DomainType.HYDROLOGY, DomainType.CLIMATE]:
            confounds = [ConfoundType.SEASONALITY, ConfoundType.TREND_CHANGES, ConfoundType.HEAVY_TAILS]
        elif spec.domain_type == DomainType.FINANCIAL:
            confounds = [ConfoundType.VOLATILITY_CLUSTERING, ConfoundType.REGIME_CHANGES, ConfoundType.JUMPS]
        else:
            confounds = [ConfoundType.NON_STATIONARITY, ConfoundType.HEAVY_TAILS]
    
    return generator.generate_data(spec, confounds)


if __name__ == "__main__":
    # Example usage
    generator = SyntheticDataGenerator(random_seed=42)
    
    # Generate EEG-like data
    eeg_data = generate_benchmark_dataset(generator, 'eeg_resting')
    print(f"Generated EEG data: {len(eeg_data['data'])} points")
    
    # Generate financial data
    financial_data = generate_benchmark_dataset(generator, 'financial_daily')
    print(f"Generated financial data: {len(financial_data['data'])} points")
