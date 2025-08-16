"""
Data Preprocessing Utilities for Long-Range Dependence Analysis

This module provides comprehensive data preprocessing capabilities including:
- Data normalization and scaling
- Missing value handling
- Outlier detection and treatment
- Domain-specific preprocessing
- Quality checks and validation

All preprocessing methods are designed to preserve LRD properties while ensuring
fair comparison across datasets with different scales and characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Available normalization methods."""
    ZSCORE = "zscore"           # Standardization (mean=0, std=1)
    MINMAX = "minmax"           # Min-max scaling to [0,1]
    ROBUST = "robust"           # Robust scaling using median and IQR
    DECIMAL = "decimal"         # Decimal scaling
    LOG = "log"                 # Log transformation
    BOXCOX = "boxcox"          # Box-Cox transformation
    QUANTILE = "quantile"      # Quantile normalization
    NONE = "none"              # No normalization


class PreprocessingStep(Enum):
    """Available preprocessing steps."""
    NORMALIZATION = "normalization"
    MISSING_VALUE_HANDLING = "missing_value_handling"
    OUTLIER_DETECTION = "outlier_detection"
    OUTLIER_TREATMENT = "outlier_treatment"
    TREND_REMOVAL = "trend_removal"
    SEASONAL_REMOVAL = "seasonal_removal"
    NOISE_REDUCTION = "noise_reduction"
    QUALITY_CHECK = "quality_check"


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    normalization_method: NormalizationMethod = NormalizationMethod.ZSCORE
    handle_missing_values: bool = True
    detect_outliers: bool = True
    treat_outliers: bool = False  # Default to detection only
    remove_trend: bool = False
    remove_seasonality: bool = False
    noise_reduction: bool = False
    quality_check: bool = True
    
    # Normalization parameters
    normalization_params: Dict[str, Any] = None
    
    # Outlier detection parameters
    outlier_method: str = "iqr"  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 1.5
    
    # Quality check parameters
    min_data_points: int = 50
    max_missing_ratio: float = 0.1
    max_outlier_ratio: float = 0.05
    
    def __post_init__(self):
        if self.normalization_params is None:
            self.normalization_params = {}


@dataclass
class PreprocessingResult:
    """Result of data preprocessing."""
    original_data: np.ndarray
    processed_data: np.ndarray
    preprocessing_info: Dict[str, Any]
    quality_report: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.preprocessing_info is None:
            self.preprocessing_info = {}
        if self.quality_report is None:
            self.quality_report = {}
        if self.metadata is None:
            self.metadata = {}


class DataPreprocessor:
    """
    Comprehensive data preprocessor for LRD analysis.
    
    Handles multiple preprocessing steps while preserving long-range dependence
    properties and ensuring data quality.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the data preprocessor.
        
        Parameters:
        -----------
        config : PreprocessingConfig, optional
            Configuration for preprocessing steps
        """
        self.config = config or PreprocessingConfig()
        self.preprocessing_history = []
        
        logger.info(f"DataPreprocessor initialized with {self.config.normalization_method.value} normalization")
    
    def preprocess_dataset(self, 
                          data: np.ndarray,
                          metadata: Optional[Dict[str, Any]] = None,
                          domain: Optional[str] = None) -> PreprocessingResult:
        """
        Preprocess a complete dataset with all configured steps.
        
        Parameters:
        -----------
        data : np.ndarray
            Input time series data
        metadata : Dict[str, Any], optional
            Dataset metadata for domain-specific processing
        domain : str, optional
            Data domain (hydrology, financial, biomedical, climate, etc.)
            
        Returns:
        --------
        PreprocessingResult
            Complete preprocessing result with original and processed data
        """
        logger.info(f"Starting preprocessing of dataset with {len(data)} points")
        
        # Store original data
        original_data = data.copy()
        processed_data = data.copy()
        preprocessing_info = {}
        quality_report = {}
        
        # Quality check first
        if self.config.quality_check:
            quality_report = self._quality_check(processed_data)
            if not quality_report['passed']:
                logger.warning(f"Quality check failed: {quality_report['issues']}")
        
        # Handle missing values
        if self.config.handle_missing_values:
            processed_data, missing_info = self._handle_missing_values(processed_data)
            preprocessing_info['missing_values'] = missing_info
        
        # Detect outliers
        if self.config.detect_outliers:
            outlier_info = self._detect_outliers(processed_data)
            preprocessing_info['outlier_detection'] = outlier_info
            
            # Treat outliers if configured
            if self.config.treat_outliers:
                processed_data, treatment_info = self._treat_outliers(processed_data, outlier_info)
                preprocessing_info['outlier_treatment'] = treatment_info
        
        # Remove trend if configured
        if self.config.remove_trend:
            processed_data, trend_info = self._remove_trend(processed_data)
            preprocessing_info['trend_removal'] = trend_info
        
        # Remove seasonality if configured
        if self.config.remove_seasonality:
            processed_data, seasonal_info = self._remove_seasonality(processed_data)
            preprocessing_info['seasonal_removal'] = seasonal_info
        
        # Apply normalization
        if self.config.normalization_method != NormalizationMethod.NONE:
            processed_data, norm_info = self._normalize_data(processed_data, domain)
            preprocessing_info['normalization'] = norm_info
        
        # Noise reduction if configured
        if self.config.noise_reduction:
            processed_data, noise_info = self._reduce_noise(processed_data)
            preprocessing_info['noise_reduction'] = noise_info
        
        # Final quality check
        if self.config.quality_check:
            final_quality = self._quality_check(processed_data)
            quality_report['final'] = final_quality
        
        # Create result
        result = PreprocessingResult(
            original_data=original_data,
            processed_data=processed_data,
            preprocessing_info=preprocessing_info,
            quality_report=quality_report,
            metadata=metadata or {}
        )
        
        # Store in history
        self.preprocessing_history.append(result)
        
        logger.info(f"Preprocessing completed. Data shape: {processed_data.shape}")
        return result
    
    def _normalize_data(self, data: np.ndarray, domain: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalize data using the configured method.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
        domain : str, optional
            Data domain for domain-specific normalization
            
        Returns:
        --------
        Tuple[np.ndarray, Dict[str, Any]]
            Normalized data and normalization information
        """
        method = self.config.normalization_method
        params = self.config.normalization_params or {}
        
        logger.info(f"Applying {method.value} normalization")
        
        if method == NormalizationMethod.ZSCORE:
            return self._zscore_normalization(data, params)
        elif method == NormalizationMethod.MINMAX:
            return self._minmax_normalization(data, params)
        elif method == NormalizationMethod.ROBUST:
            return self._robust_normalization(data, params)
        elif method == NormalizationMethod.DECIMAL:
            return self._decimal_normalization(data, params)
        elif method == NormalizationMethod.LOG:
            return self._log_normalization(data, params)
        elif method == NormalizationMethod.BOXCOX:
            return self._boxcox_normalization(data, params)
        elif method == NormalizationMethod.QUANTILE:
            return self._quantile_normalization(data, params)
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return data, {"method": "none", "applied": False}
    
    def _zscore_normalization(self, data: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Standardize data to mean=0, std=1."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            logger.warning("Standard deviation is 0, cannot apply z-score normalization")
            return data, {"method": "zscore", "applied": False, "error": "std=0"}
        
        normalized = (data - mean_val) / std_val
        
        info = {
            "method": "zscore",
            "applied": True,
            "original_mean": float(mean_val),
            "original_std": float(std_val),
            "new_mean": float(np.mean(normalized)),
            "new_std": float(np.std(normalized))
        }
        
        return normalized, info
    
    def _minmax_normalization(self, data: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Scale data to [0,1] range."""
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val == min_val:
            logger.warning("Min and max are equal, cannot apply min-max normalization")
            return data, {"method": "minmax", "applied": False, "error": "min=max"}
        
        normalized = (data - min_val) / (max_val - min_val)
        
        info = {
            "method": "minmax",
            "applied": True,
            "original_min": float(min_val),
            "original_max": float(max_val),
            "new_min": float(np.min(normalized)),
            "new_max": float(np.max(normalized))
        }
        
        return normalized, info
    
    def _robust_normalization(self, data: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Robust scaling using median and IQR."""
        median_val = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        
        if iqr == 0:
            logger.warning("IQR is 0, cannot apply robust normalization")
            return data, {"method": "robust", "applied": False, "error": "iqr=0"}
        
        normalized = (data - median_val) / iqr
        
        info = {
            "method": "robust",
            "applied": True,
            "original_median": float(median_val),
            "original_iqr": float(iqr),
            "new_median": float(np.median(normalized)),
            "new_iqr": float(np.percentile(normalized, 75) - np.percentile(normalized, 25))
        }
        
        return normalized, info
    
    def _decimal_normalization(self, data: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decimal scaling to [-1,1] range."""
        max_abs = np.max(np.abs(data))
        
        if max_abs == 0:
            logger.warning("Maximum absolute value is 0, cannot apply decimal normalization")
            return data, {"method": "decimal", "applied": False, "error": "max_abs=0"}
        
        # Find the smallest power of 10 that makes max_abs < 1
        scale_factor = 10 ** np.ceil(np.log10(max_abs))
        normalized = data / scale_factor
        
        info = {
            "method": "decimal",
            "applied": True,
            "scale_factor": float(scale_factor),
            "original_max_abs": float(max_abs),
            "new_max_abs": float(np.max(np.abs(normalized)))
        }
        
        return normalized, info
    
    def _log_normalization(self, data: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Log transformation with offset for negative values."""
        # Add offset to ensure all values are positive
        offset = np.abs(np.min(data)) + 1e-10
        data_positive = data + offset
        
        # Apply log transformation
        normalized = np.log(data_positive)
        
        info = {
            "method": "log",
            "applied": True,
            "offset": float(offset),
            "original_min": float(np.min(data)),
            "new_min": float(np.min(normalized)),
            "new_max": float(np.max(normalized))
        }
        
        return normalized, info
    
    def _boxcox_normalization(self, data: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Box-Cox transformation for positive data."""
        try:
            from scipy.stats import boxcox
            
            # Ensure positive data
            if np.any(data <= 0):
                offset = np.abs(np.min(data)) + 1e-10
                data_positive = data + offset
            else:
                offset = 0
                data_positive = data
            
            # Apply Box-Cox transformation
            normalized, lambda_param = boxcox(data_positive)
            
            info = {
                "method": "boxcox",
                "applied": True,
                "lambda": float(lambda_param),
                "offset": float(offset),
                "original_min": float(np.min(data)),
                "new_min": float(np.min(normalized)),
                "new_max": float(np.max(normalized))
            }
            
            return normalized, info
            
        except ImportError:
            logger.warning("SciPy not available, falling back to log normalization")
            return self._log_normalization(data, params)
    
    def _quantile_normalization(self, data: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantile normalization to standard normal distribution."""
        from scipy.stats import norm
        
        # Calculate empirical quantiles
        sorted_indices = np.argsort(data)
        ranks = np.argsort(sorted_indices)
        
        # Convert ranks to quantiles
        quantiles = (ranks + 1) / (len(data) + 1)
        
        # Transform to standard normal
        normalized = norm.ppf(quantiles)
        
        info = {
            "method": "quantile",
            "applied": True,
            "original_min": float(np.min(data)),
            "original_max": float(np.max(data)),
            "new_min": float(np.min(normalized)),
            "new_max": float(np.max(normalized)),
            "new_mean": float(np.mean(normalized)),
            "new_std": float(np.std(normalized))
        }
        
        return normalized, info
    
    def _handle_missing_values(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Handle missing values in the data."""
        missing_mask = np.isnan(data) | np.isinf(data)
        n_missing = np.sum(missing_mask)
        
        if n_missing == 0:
            return data, {"missing_values": 0, "handled": False}
        
        logger.info(f"Found {n_missing} missing/infinite values")
        
        # For now, use forward fill then backward fill
        # In practice, you might want more sophisticated methods
        data_filled = data.copy()
        
        # Forward fill
        for i in range(1, len(data_filled)):
            if missing_mask[i]:
                data_filled[i] = data_filled[i-1]
        
        # Backward fill for any remaining missing values at the beginning
        for i in range(len(data_filled)-2, -1, -1):
            if missing_mask[i]:
                data_filled[i] = data_filled[i+1]
        
        info = {
            "missing_values": int(n_missing),
            "handled": True,
            "method": "forward_backward_fill",
            "missing_ratio": float(n_missing / len(data))
        }
        
        return data_filled, info
    
    def _detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using the configured method."""
        method = self.config.outlier_method
        threshold = self.config.outlier_threshold
        
        if method == "iqr":
            return self._detect_outliers_iqr(data, threshold)
        elif method == "zscore":
            return self._detect_outliers_zscore(data, threshold)
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            return {"method": method, "outliers": [], "count": 0}
    
    def _detect_outliers_iqr(self, data: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = np.where(outlier_mask)[0]
        
        return {
            "method": "iqr",
            "outliers": outlier_indices.tolist(),
            "count": len(outlier_indices),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "iqr": float(iqr),
            "threshold": threshold
        }
    
    def _detect_outliers_zscore(self, data: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return {"method": "zscore", "outliers": [], "count": 0, "error": "std=0"}
        
        z_scores = np.abs((data - mean_val) / std_val)
        outlier_mask = z_scores > threshold
        outlier_indices = np.where(outlier_mask)[0]
        
        return {
            "method": "zscore",
            "outliers": outlier_indices.tolist(),
            "count": len(outlier_indices),
            "threshold": threshold,
            "max_zscore": float(np.max(z_scores))
        }
    
    def _treat_outliers(self, data: np.ndarray, outlier_info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Treat detected outliers."""
        if outlier_info['count'] == 0:
            return data, {"treated": False, "method": "none"}
        
        # Use winsorization (cap at percentiles)
        data_treated = data.copy()
        outlier_indices = outlier_info['outliers']
        
        # Calculate percentiles excluding outliers
        clean_data = np.delete(data, outlier_indices)
        p01, p99 = np.percentile(clean_data, [1, 99])
        
        # Cap outliers
        data_treated[outlier_indices] = np.clip(data[outlier_indices], p01, p99)
        
        info = {
            "treated": True,
            "method": "winsorization",
            "outliers_treated": outlier_info['count'],
            "lower_percentile": float(p01),
            "upper_percentile": float(p99)
        }
        
        return data_treated, info
    
    def _remove_trend(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Remove linear trend from data."""
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        
        detrended = data - trend
        
        info = {
            "trend_removed": True,
            "trend_coefficients": coeffs.tolist(),
            "trend_slope": float(coeffs[0]),
            "trend_intercept": float(coeffs[1])
        }
        
        return detrended, info
    
    def _remove_seasonality(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Remove seasonal patterns from data."""
        # Simple seasonal removal using moving average
        # In practice, you might want more sophisticated methods like STL decomposition
        
        # Estimate seasonality period (simplified)
        # For daily data, assume annual seasonality
        if len(data) >= 365:
            period = 365
        elif len(data) >= 52:
            period = 52  # Weekly
        else:
            period = min(12, len(data) // 4)  # Default to 12 or 1/4 of data length
        
        # Calculate seasonal component using moving average
        seasonal = np.convolve(data, np.ones(period)/period, mode='same')
        
        # Remove seasonality
        deseasonalized = data - seasonal
        
        info = {
            "seasonality_removed": True,
            "estimated_period": period,
            "seasonal_component_std": float(np.std(seasonal))
        }
        
        return deseasonalized, info
    
    def _reduce_noise(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reduce noise using simple smoothing."""
        # Simple moving average smoothing
        window_size = min(5, len(data) // 10)
        if window_size < 3:
            window_size = 3
        
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        
        info = {
            "noise_reduced": True,
            "smoothing_window": window_size,
            "original_std": float(np.std(data)),
            "smoothed_std": float(np.std(smoothed))
        }
        
        return smoothed, info
    
    def _quality_check(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform quality checks on the data."""
        n_points = len(data)
        
        # Check data length
        length_ok = n_points >= self.config.min_data_points
        
        # Check for missing values
        missing_mask = np.isnan(data) | np.isinf(data)
        missing_ratio = np.sum(missing_mask) / n_points
        missing_ok = missing_ratio <= self.config.max_missing_ratio
        
        # Check for outliers
        outlier_info = self._detect_outliers(data)
        outlier_ratio = outlier_info['count'] / n_points
        outlier_ok = outlier_ratio <= self.config.max_outlier_ratio
        
        # Check data range
        data_range = np.max(data) - np.min(data)
        range_ok = data_range > 0
        
        # Overall quality
        passed = all([length_ok, missing_ok, outlier_ok, range_ok])
        
        issues = []
        if not length_ok:
            issues.append(f"Insufficient data points: {n_points} < {self.config.min_data_points}")
        if not missing_ok:
            issues.append(f"Too many missing values: {missing_ratio:.2%} > {self.config.max_missing_ratio:.2%}")
        if not outlier_ok:
            issues.append(f"Too many outliers: {outlier_ratio:.2%} > {self.config.max_outlier_ratio:.2%}")
        if not range_ok:
            issues.append("Data has no range (all values are equal)")
        
        return {
            "passed": passed,
            "issues": issues,
            "n_points": n_points,
            "missing_ratio": float(missing_ratio),
            "outlier_ratio": float(outlier_ratio),
            "data_range": float(data_range),
            "checks": {
                "length": length_ok,
                "missing_values": missing_ok,
                "outliers": outlier_ok,
                "range": range_ok
            }
        }
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get a summary of all preprocessing operations."""
        if not self.preprocessing_history:
            return {"message": "No preprocessing operations performed"}
        
        summary = {
            "total_datasets": len(self.preprocessing_history),
            "normalization_methods_used": [],
            "quality_check_results": [],
            "common_issues": []
        }
        
        # Collect information from all preprocessing operations
        for result in self.preprocessing_history:
            # Normalization methods
            if 'normalization' in result.preprocessing_info:
                norm_method = result.preprocessing_info['normalization']['method']
                if norm_method not in summary["normalization_methods_used"]:
                    summary["normalization_methods_used"].append(norm_method)
            
            # Quality check results
            if result.quality_report:
                summary["quality_check_results"].append({
                    "passed": result.quality_report.get('passed', False),
                    "issues": result.quality_report.get('issues', [])
                })
        
        # Analyze common issues
        all_issues = []
        for result in self.preprocessing_history:
            if result.quality_report and 'issues' in result.quality_report:
                all_issues.extend(result.quality_report['issues'])
        
        from collections import Counter
        issue_counts = Counter(all_issues)
        summary["common_issues"] = [{"issue": issue, "count": count} 
                                   for issue, count in issue_counts.most_common()]
        
        return summary
    
    def save_preprocessing_config(self, filepath: Union[str, Path]):
        """Save the current preprocessing configuration to a file."""
        config_dict = {
            "normalization_method": self.config.normalization_method.value,
            "handle_missing_values": self.config.handle_missing_values,
            "detect_outliers": self.config.detect_outliers,
            "treat_outliers": self.config.treat_outliers,
            "remove_trend": self.config.remove_trend,
            "remove_seasonality": self.config.remove_seasonality,
            "noise_reduction": self.config.noise_reduction,
            "quality_check": self.config.quality_check,
            "normalization_params": self.config.normalization_params,
            "outlier_method": self.config.outlier_method,
            "outlier_threshold": self.config.outlier_threshold,
            "min_data_points": self.config.min_data_points,
            "max_missing_ratio": self.config.max_missing_ratio,
            "max_outlier_ratio": self.config.max_outlier_ratio
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Preprocessing configuration saved to {filepath}")
    
    @classmethod
    def load_preprocessing_config(cls, filepath: Union[str, Path]) -> 'DataPreprocessor':
        """Load a preprocessing configuration from a file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert string back to enum
        if 'normalization_method' in config_dict:
            config_dict['normalization_method'] = NormalizationMethod(config_dict['normalization_method'])
        
        config = PreprocessingConfig(**config_dict)
        return cls(config)


def create_domain_specific_preprocessor(domain: str) -> DataPreprocessor:
    """
    Create a preprocessor configured for a specific domain.
    
    Parameters:
    -----------
    domain : str
        Data domain (hydrology, financial, biomedical, climate, etc.)
        
    Returns:
    --------
    DataPreprocessor
        Configured preprocessor for the specified domain
    """
    domain = domain.lower()
    
    if domain == "hydrology":
        config = PreprocessingConfig(
            normalization_method=NormalizationMethod.ZSCORE,
            handle_missing_values=True,
            detect_outliers=True,
            treat_outliers=False,  # Hydrology data often has natural outliers
            remove_trend=True,     # Remove long-term trends
            remove_seasonality=True, # Remove seasonal patterns
            noise_reduction=False,
            quality_check=True
        )
    
    elif domain == "financial":
        config = PreprocessingConfig(
            normalization_method=NormalizationMethod.ROBUST,  # Robust to outliers
            handle_missing_values=True,
            detect_outliers=True,
            treat_outliers=True,   # Financial outliers often indicate errors
            remove_trend=False,    # Keep trends in financial data
            remove_seasonality=False, # Keep seasonality
            noise_reduction=False,
            quality_check=True
        )
    
    elif domain == "biomedical":
        config = PreprocessingConfig(
            normalization_method=NormalizationMethod.ZSCORE,
            handle_missing_values=True,
            detect_outliers=True,
            treat_outliers=True,   # Biomedical artifacts should be removed
            remove_trend=True,     # Remove baseline drift
            remove_seasonality=False, # Keep physiological rhythms
            noise_reduction=True,  # Reduce measurement noise
            quality_check=True
        )
    
    elif domain == "climate":
        config = PreprocessingConfig(
            normalization_method=NormalizationMethod.ZSCORE,
            handle_missing_values=True,
            detect_outliers=True,
            treat_outliers=False,   # Climate extremes are important
            remove_trend=True,      # Remove climate change trends
            remove_seasonality=True, # Remove seasonal patterns
            noise_reduction=False,
            quality_check=True
        )
    
    else:
        # Default configuration
        config = PreprocessingConfig(
            normalization_method=NormalizationMethod.ZSCORE,
            handle_missing_values=True,
            detect_outliers=True,
            treat_outliers=False,
            remove_trend=False,
            remove_seasonality=False,
            noise_reduction=False,
            quality_check=True
        )
    
    return DataPreprocessor(config)
