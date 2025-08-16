"""
Synthetic Data Quality Evaluation

This module provides comprehensive evaluation measures for assessing the quality
of synthetic time series data against realistic parameters. Inspired by the TSGBench
framework, it includes measures for:

1. Statistical Distribution Similarity
2. Temporal Structure Preservation
3. Long-Range Dependence Characteristics
4. Domain-Specific Quality Metrics
5. Overall Quality Scoring

The goal is to ensure synthetic data closely matches realistic data characteristics
while maintaining the intended LRD properties.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class QualityMetricType(Enum):
    """Types of quality metrics."""
    STATISTICAL = "statistical"           # Distribution properties
    TEMPORAL = "temporal"                 # Time series structure
    LRD = "lrd"                          # Long-range dependence
    DOMAIN = "domain"                     # Domain-specific features
    COMPOSITE = "composite"               # Overall quality score


@dataclass
class QualityMetricResult:
    """Result of a quality metric evaluation."""
    metric_name: str
    metric_type: QualityMetricType
    value: float
    score: float  # Normalized score [0, 1] where 1 is best
    weight: float  # Weight for composite scoring
    description: str
    details: Dict[str, Any]
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class QualityEvaluationResult:
    """Complete quality evaluation result."""
    synthetic_data: np.ndarray
    reference_data: np.ndarray
    reference_metadata: Dict[str, Any]
    metrics: List[QualityMetricResult]
    overall_score: float
    quality_level: str
    recommendations: List[str]
    evaluation_date: str
    normalization_info: Dict[str, Any]
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = []
        if self.recommendations is None:
            self.recommendations = []
        if self.normalization_info is None:
            self.normalization_info = {}


class SyntheticDataQualityEvaluator:
    """
    Comprehensive evaluator for synthetic data quality.
    
    Implements TSGBench-inspired measures to assess how well synthetic data
    matches realistic parameters and characteristics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the quality evaluator.
        
        Parameters:
        -----------
        config : Dict[str, Any], optional
            Configuration for evaluation parameters
        """
        self.config = config or self._default_config()
        self.evaluation_history = []
        
        logger.info("SyntheticDataQualityEvaluator initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for quality evaluation."""
        return {
            "statistical_metrics": {
                "distribution_similarity": True,
                "moment_preservation": True,
                "tail_behavior": True,
                "quantile_matching": True
            },
            "temporal_metrics": {
                "autocorrelation": True,
                "seasonality": True,
                "trend_preservation": True,
                "volatility_clustering": True
            },
            "lrd_metrics": {
                "hurst_preservation": True,
                "spectral_properties": True,
                "scaling_behavior": True
            },
            "domain_metrics": {
                "hydrology": True,
                "financial": True,
                "biomedical": True,
                "climate": True
            },
            "weights": {
                "statistical": 0.25,
                "temporal": 0.25,
                "lrd": 0.35,
                "domain": 0.15
            },
            "thresholds": {
                "excellent": 0.9,
                "good": 0.7,
                "acceptable": 0.5,
                "poor": 0.3
            }
        }
    
    def evaluate_quality(self, 
                        synthetic_data: np.ndarray,
                        reference_data: np.ndarray,
                        reference_metadata: Dict[str, Any],
                        domain: Optional[str] = None,
                        normalize_for_comparison: bool = True,
                        normalization_method: str = "zscore") -> QualityEvaluationResult:
        """
        Evaluate the quality of synthetic data against reference data.
        
        Parameters:
        -----------
        synthetic_data : np.ndarray
            Synthetic time series data to evaluate
        reference_data : np.ndarray
            Reference/realistic data for comparison
        reference_metadata : Dict[str, Any]
            Metadata about the reference data
        domain : str, optional
            Data domain for domain-specific evaluation
        normalize_for_comparison : bool, optional
            Whether to normalize data for fair comparison (default: True)
        normalization_method : str, optional
            Normalization method to use ('zscore', 'minmax', 'robust')
            
        Returns:
        --------
        QualityEvaluationResult
            Comprehensive quality evaluation results
        """
        logger.info(f"Starting quality evaluation for {len(synthetic_data)} synthetic points")
        
        # Normalize data for fair comparison if requested
        if normalize_for_comparison:
            norm_synthetic, norm_reference, norm_info = self._normalize_for_comparison(
                synthetic_data, reference_data, normalization_method
            )
            logger.info(f"Data normalized using {norm_info['method']} method for fair comparison")
        else:
            norm_synthetic, norm_reference = synthetic_data, reference_data
            norm_info = {"method": "none", "applied": False}
        
        # Initialize results
        metrics = []
        
        # 1. Statistical quality metrics
        if self.config["statistical_metrics"]["distribution_similarity"]:
            metrics.extend(self._evaluate_statistical_quality(norm_synthetic, norm_reference))
        
        # 2. Temporal structure metrics
        if self.config["temporal_metrics"]["autocorrelation"]:
            metrics.extend(self._evaluate_temporal_quality(norm_synthetic, norm_reference))
        
        # 3. LRD-specific metrics
        if self.config["lrd_metrics"]["hurst_preservation"]:
            metrics.extend(self._evaluate_lrd_quality(norm_synthetic, norm_reference))
        
        # 4. Domain-specific metrics
        if domain and self.config["domain_metrics"].get(domain, False):
            metrics.extend(self._evaluate_domain_quality(norm_synthetic, norm_reference, domain))
        
        # Calculate overall score
        overall_score = self._calculate_composite_score(metrics)
        quality_level = self._determine_quality_level(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, overall_score)
        
        # Create result
        result = QualityEvaluationResult(
            synthetic_data=synthetic_data,
            reference_data=reference_data,
            reference_metadata=reference_metadata,
            metrics=metrics,
            overall_score=overall_score,
            quality_level=quality_level,
            recommendations=recommendations,
            evaluation_date=pd.Timestamp.now().isoformat(),
            normalization_info=norm_info
        )
        
        # Store in history
        self.evaluation_history.append(result)
        
        logger.info(f"Quality evaluation completed. Overall score: {overall_score:.3f} ({quality_level})")
        return result
    
    def _normalize_for_comparison(self, 
                                 synthetic_data: np.ndarray,
                                 reference_data: np.ndarray,
                                 method: str = "zscore") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Normalize both synthetic and reference data to the same scale for fair comparison.
        
        Parameters:
        -----------
        synthetic_data : np.ndarray
            Synthetic data to normalize
        reference_data : np.ndarray
            Reference data to normalize
        method : str
            Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
            Normalized synthetic data, normalized reference data, and normalization info
        """
        logger.info(f"Normalizing data for comparison using {method} method")
        
        if method == "zscore":
            # Z-score normalization to mean=0, std=1
            ref_mean, ref_std = np.mean(reference_data), np.std(reference_data)
            synth_mean, synth_std = np.mean(synthetic_data), np.std(synthetic_data)
            
            if ref_std == 0 or synth_std == 0:
                logger.warning("Standard deviation is 0, using min-max normalization instead")
                return self._normalize_for_comparison(synthetic_data, reference_data, "minmax")
            
            norm_synthetic = (synthetic_data - synth_mean) / synth_std
            norm_reference = (reference_data - ref_mean) / ref_std
            
            info = {
                "method": "zscore",
                "synthetic_original_mean": float(synth_mean),
                "synthetic_original_std": float(synth_std),
                "reference_original_mean": float(ref_mean),
                "reference_original_std": float(ref_std),
                "normalized_mean": 0.0,
                "normalized_std": 1.0
            }
            
        elif method == "minmax":
            # Min-max normalization to [0,1]
            ref_min, ref_max = np.min(reference_data), np.max(reference_data)
            synth_min, synth_max = np.min(synthetic_data), np.max(synthetic_data)
            
            if ref_max == ref_min or synth_max == synth_min:
                logger.warning("Min and max are equal, using z-score normalization instead")
                return self._normalize_for_comparison(synthetic_data, reference_data, "zscore")
            
            norm_synthetic = (synthetic_data - synth_min) / (synth_max - synth_min)
            norm_reference = (reference_data - ref_min) / (ref_max - ref_min)
            
            info = {
                "method": "minmax",
                "synthetic_original_min": float(synth_min),
                "synthetic_original_max": float(synth_max),
                "reference_original_min": float(ref_min),
                "reference_original_max": float(ref_max),
                "normalized_min": 0.0,
                "normalized_max": 1.0
            }
            
        elif method == "robust":
            # Robust normalization using median and IQR
            ref_median, ref_q75, ref_q25 = np.median(reference_data), np.percentile(reference_data, 75), np.percentile(reference_data, 25)
            synth_median, synth_q75, synth_q25 = np.median(synthetic_data), np.percentile(synthetic_data, 75), np.percentile(synthetic_data, 25)
            
            ref_iqr = ref_q75 - ref_q25
            synth_iqr = synth_q75 - synth_q25
            
            if ref_iqr == 0 or synth_iqr == 0:
                logger.warning("IQR is 0, using z-score normalization instead")
                return self._normalize_for_comparison(synthetic_data, reference_data, "zscore")
            
            norm_synthetic = (synthetic_data - synth_median) / synth_iqr
            norm_reference = (reference_data - ref_median) / ref_iqr
            
            info = {
                "method": "robust",
                "synthetic_original_median": float(synth_median),
                "synthetic_original_iqr": float(synth_iqr),
                "reference_original_median": float(ref_median),
                "reference_original_iqr": float(ref_iqr),
                "normalized_median": 0.0,
                "normalized_iqr": 1.0
            }
            
        else:
            logger.warning(f"Unknown normalization method: {method}, using z-score")
            return self._normalize_for_comparison(synthetic_data, reference_data, "zscore")
        
        logger.info(f"Data normalized successfully. Synthetic shape: {norm_synthetic.shape}, Reference shape: {norm_reference.shape}")
        return norm_synthetic, norm_reference, info

    def _evaluate_statistical_quality(self, synthetic: np.ndarray, reference: np.ndarray) -> List[QualityMetricResult]:
        """Evaluate statistical distribution quality."""
        metrics = []
        
        # 1. Distribution similarity (Jensen-Shannon divergence)
        try:
            # Create histograms for comparison
            bins = np.linspace(min(np.min(synthetic), np.min(reference)),
                             max(np.max(synthetic), np.max(reference)), 50)
            
            hist_synthetic, _ = np.histogram(synthetic, bins=bins, density=True)
            hist_reference, _ = np.histogram(reference, bins=bins, density=True)
            
            # Add small epsilon to avoid zero probabilities
            hist_synthetic = hist_synthetic + 1e-10
            hist_reference = hist_reference + 1e-10
            
            # Normalize
            hist_synthetic = hist_synthetic / np.sum(hist_synthetic)
            hist_reference = hist_reference / np.sum(hist_reference)
            
            js_divergence = jensenshannon(hist_synthetic, hist_reference)
            # Convert to similarity score (0 = identical, 1 = completely different)
            similarity_score = 1 / (1 + js_divergence)
            
            metrics.append(QualityMetricResult(
                metric_name="distribution_similarity",
                metric_type=QualityMetricType.STATISTICAL,
                value=float(js_divergence),
                score=float(similarity_score),
                weight=self.config["weights"]["statistical"] * 0.4,
                description="Similarity of probability distributions",
                details={
                    "js_divergence": float(js_divergence),
                    "similarity_score": float(similarity_score),
                    "bins": len(bins)
                }
            ))
        except Exception as e:
            logger.warning(f"Distribution similarity evaluation failed: {e}")
        
        # 2. Moment preservation
        try:
            # Compare first four moments
            synthetic_mean = np.mean(synthetic)
            synthetic_std = np.std(synthetic)
            synthetic_skew = stats.skew(synthetic)
            synthetic_kurt = stats.kurtosis(synthetic)
            
            reference_mean = np.mean(reference)
            reference_std = np.std(reference)
            reference_skew = stats.skew(reference)
            reference_kurt = stats.kurtosis(reference)
            
            # Calculate relative errors
            mean_error = abs(synthetic_mean - reference_mean) / (abs(reference_mean) + 1e-10)
            std_error = abs(synthetic_std - reference_std) / (abs(reference_std) + 1e-10)
            skew_error = abs(synthetic_skew - reference_skew) / (abs(reference_skew) + 1e-10)
            kurt_error = abs(synthetic_kurt - reference_kurt) / (abs(reference_kurt) + 1e-10)
            
            # Average error across moments
            avg_moment_error = np.mean([mean_error, std_error, skew_error, kurt_error])
            moment_score = 1 / (1 + avg_moment_error)
            
            metrics.append(QualityMetricResult(
                metric_name="moment_preservation",
                metric_type=QualityMetricType.STATISTICAL,
                value=float(avg_moment_error),
                score=float(moment_score),
                weight=self.config["weights"]["statistical"] * 0.3,
                description="Preservation of statistical moments",
                details={
                    "mean_error": float(mean_error),
                    "std_error": float(std_error),
                    "skew_error": float(skew_error),
                    "kurt_error": float(kurt_error),
                    "synthetic_moments": {
                        "mean": float(synthetic_mean),
                        "std": float(synthetic_std),
                        "skew": float(synthetic_skew),
                        "kurt": float(synthetic_kurt)
                    },
                    "reference_moments": {
                        "mean": float(reference_mean),
                        "std": float(reference_std),
                        "skew": float(reference_skew),
                        "kurt": float(reference_kurt)
                    }
                }
            ))
        except Exception as e:
            logger.warning(f"Moment preservation evaluation failed: {e}")
        
        # 3. Quantile matching
        try:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            synthetic_quantiles = np.percentile(synthetic, [q * 100 for q in quantiles])
            reference_quantiles = np.percentile(reference, [q * 100 for q in quantiles])
            
            # Calculate relative errors
            quantile_errors = []
            for i, q in enumerate(quantiles):
                if abs(reference_quantiles[i]) > 1e-10:
                    error = abs(synthetic_quantiles[i] - reference_quantiles[i]) / abs(reference_quantiles[i])
                    quantile_errors.append(error)
            
            avg_quantile_error = np.mean(quantile_errors) if quantile_errors else 1.0
            quantile_score = 1 / (1 + avg_quantile_error)
            
            metrics.append(QualityMetricResult(
                metric_name="quantile_matching",
                metric_type=QualityMetricType.STATISTICAL,
                value=float(avg_quantile_error),
                score=float(quantile_score),
                weight=self.config["weights"]["statistical"] * 0.3,
                description="Matching of quantile values",
                details={
                    "quantiles": quantiles,
                    "synthetic_quantiles": synthetic_quantiles.tolist(),
                    "reference_quantiles": reference_quantiles.tolist(),
                    "quantile_errors": [float(e) for e in quantile_errors]
                }
            ))
        except Exception as e:
            logger.warning(f"Quantile matching evaluation failed: {e}")
        
        return metrics
    
    def _evaluate_temporal_quality(self, synthetic: np.ndarray, reference: np.ndarray) -> List[QualityMetricResult]:
        """Evaluate temporal structure quality."""
        metrics = []
        
        # 1. Autocorrelation preservation
        try:
            # Calculate autocorrelation at different lags
            max_lag = min(20, len(synthetic) // 4)
            lags = range(1, max_lag + 1)
            
            synthetic_acf = []
            reference_acf = []
            
            for lag in lags:
                if lag < len(synthetic):
                    # Synthetic autocorrelation
                    acf_syn = np.corrcoef(synthetic[:-lag], synthetic[lag:])[0, 1]
                    synthetic_acf.append(acf_syn if not np.isnan(acf_syn) else 0)
                    
                    # Reference autocorrelation
                    acf_ref = np.corrcoef(reference[:-lag], reference[lag:])[0, 1]
                    reference_acf.append(acf_ref if not np.isnan(acf_ref) else 0)
            
            # Calculate correlation between autocorrelation functions
            if len(synthetic_acf) > 1 and len(reference_acf) > 1:
                acf_correlation = np.corrcoef(synthetic_acf, reference_acf)[0, 1]
                acf_score = (acf_correlation + 1) / 2  # Convert from [-1, 1] to [0, 1]
            else:
                acf_score = 0.5
            
            metrics.append(QualityMetricResult(
                metric_name="autocorrelation_preservation",
                metric_type=QualityMetricType.TEMPORAL,
                value=float(acf_correlation) if 'acf_correlation' in locals() else 0.0,
                score=float(acf_score),
                weight=self.config["weights"]["temporal"] * 0.4,
                description="Preservation of autocorrelation structure",
                details={
                    "max_lag": max_lag,
                    "synthetic_acf": [float(x) for x in synthetic_acf],
                    "reference_acf": [float(x) for x in reference_acf],
                    "acf_correlation": float(acf_correlation) if 'acf_correlation' in locals() else 0.0
                }
            ))
        except Exception as e:
            logger.warning(f"Autocorrelation evaluation failed: {e}")
        
        # 2. Seasonality detection and comparison
        try:
            # Simple seasonality detection using FFT
            synthetic_fft = np.abs(np.fft.fft(synthetic))
            reference_fft = np.abs(np.fft.fft(reference))
            
            # Focus on lower frequencies (potential seasonality)
            n_freq = len(synthetic_fft) // 4
            synthetic_seasonal = synthetic_fft[:n_freq]
            reference_seasonal = reference_fft[:n_freq]
            
            # Normalize
            synthetic_seasonal = synthetic_seasonal / np.sum(synthetic_seasonal)
            reference_seasonal = reference_seasonal / np.sum(reference_seasonal)
            
            # Calculate similarity
            seasonal_similarity = 1 / (1 + jensenshannon(synthetic_seasonal, reference_seasonal))
            
            metrics.append(QualityMetricResult(
                metric_name="seasonality_preservation",
                metric_type=QualityMetricType.TEMPORAL,
                value=float(1 - seasonal_similarity),
                score=float(seasonal_similarity),
                weight=self.config["weights"]["temporal"] * 0.3,
                description="Preservation of seasonal patterns",
                details={
                    "n_frequencies": n_freq,
                    "seasonal_similarity": float(seasonal_similarity)
                }
            ))
        except Exception as e:
            logger.warning(f"Seasonality evaluation failed: {e}")
        
        # 3. Trend preservation
        try:
            # Fit linear trends
            x_synthetic = np.arange(len(synthetic))
            x_reference = np.arange(len(reference))
            synthetic_trend = np.polyfit(x_synthetic, synthetic, 1)[0]
            reference_trend = np.polyfit(x_reference, reference, 1)[0]
            
            # Calculate trend similarity
            if abs(reference_trend) > 1e-10:
                trend_error = abs(synthetic_trend - reference_trend) / abs(reference_trend)
                trend_score = 1 / (1 + trend_error)
            else:
                trend_score = 1.0 if abs(synthetic_trend) < 1e-10 else 0.0
            
            metrics.append(QualityMetricResult(
                metric_name="trend_preservation",
                metric_type=QualityMetricType.TEMPORAL,
                value=float(trend_error) if 'trend_error' in locals() else 0.0,
                score=float(trend_score),
                weight=self.config["weights"]["temporal"] * 0.3,
                description="Preservation of trend characteristics",
                details={
                    "synthetic_trend": float(synthetic_trend),
                    "reference_trend": float(reference_trend),
                    "trend_error": float(trend_error) if 'trend_error' in locals() else 0.0
                }
            ))
        except Exception as e:
            logger.warning(f"Trend evaluation failed: {e}")
        
        return metrics
    
    def _evaluate_lrd_quality(self, synthetic: np.ndarray, reference: np.ndarray) -> List[QualityMetricResult]:
        """Evaluate long-range dependence quality."""
        metrics = []
        
        # 1. Hurst exponent preservation (if available in metadata)
        try:
            # This would ideally use the actual Hurst estimates
            # For now, we'll use a simplified approach based on variance scaling
            
            # Calculate variance at different scales
            scales = [1, 2, 4, 8, 16]
            synthetic_variances = []
            reference_variances = []
            
            for scale in scales:
                if len(synthetic) >= scale * 10:
                    # Calculate variance at this scale
                    synthetic_scaled = synthetic[::scale]
                    reference_scaled = reference[::scale]
                    
                    synthetic_variances.append(np.var(synthetic_scaled))
                    reference_variances.append(np.var(reference_scaled))
            
            if len(synthetic_variances) > 1 and len(reference_variances) > 1:
                # Calculate scaling similarity
                synthetic_scaling = np.polyfit(np.log(scales[:len(synthetic_variances)]), 
                                            np.log(synthetic_variances), 1)[0]
                reference_scaling = np.polyfit(np.log(scales[:len(reference_variances)]), 
                                             np.log(reference_variances), 1)[0]
                
                scaling_error = abs(synthetic_scaling - reference_scaling) / (abs(reference_scaling) + 1e-10)
                scaling_score = 1 / (1 + scaling_error)
                
                metrics.append(QualityMetricResult(
                    metric_name="scaling_behavior",
                    metric_type=QualityMetricType.LRD,
                    value=float(scaling_error),
                    score=float(scaling_score),
                    weight=self.config["weights"]["lrd"] * 0.4,
                    description="Preservation of scaling behavior",
                    details={
                        "scales": scales[:len(synthetic_variances)],
                        "synthetic_scaling": float(synthetic_scaling),
                        "reference_scaling": float(reference_scaling),
                        "synthetic_variances": [float(v) for v in synthetic_variances],
                        "reference_variances": [float(v) for v in reference_variances]
                    }
                ))
        except Exception as e:
            logger.warning(f"Scaling behavior evaluation failed: {e}")
        
        # 2. Spectral properties
        try:
            # Compare power spectral density
            synthetic_psd = np.abs(np.fft.fft(synthetic)) ** 2
            reference_psd = np.abs(np.fft.fft(reference)) ** 2
            
            # Focus on lower frequencies (LRD typically manifests here)
            n_freq = len(synthetic_psd) // 8
            synthetic_low_freq = synthetic_psd[:n_freq]
            reference_low_freq = reference_psd[:n_freq]
            
            # Normalize
            synthetic_low_freq = synthetic_low_freq / np.sum(synthetic_low_freq)
            reference_low_freq = reference_low_freq / np.sum(reference_low_freq)
            
            # Calculate similarity
            spectral_similarity = 1 / (1 + jensenshannon(synthetic_low_freq, reference_low_freq))
            
            metrics.append(QualityMetricResult(
                metric_name="spectral_properties",
                metric_type=QualityMetricType.LRD,
                value=float(1 - spectral_similarity),
                score=float(spectral_similarity),
                weight=self.config["weights"]["lrd"] * 0.6,
                description="Preservation of spectral properties",
                details={
                    "n_frequencies": n_freq,
                    "spectral_similarity": float(spectral_similarity)
                }
            ))
        except Exception as e:
            logger.warning(f"Spectral properties evaluation failed: {e}")
        
        return metrics
    
    def _evaluate_domain_quality(self, synthetic: np.ndarray, reference: np.ndarray, domain: str) -> List[QualityMetricResult]:
        """Evaluate domain-specific quality metrics."""
        metrics = []
        
        if domain.lower() == "hydrology":
            # Hydrology-specific: extreme value behavior, persistence
            try:
                # Extreme value behavior
                synthetic_extremes = np.percentile(synthetic, [95, 99])
                reference_extremes = np.percentile(reference, [95, 99])
                
                extreme_errors = []
                for i in range(len(synthetic_extremes)):
                    if abs(reference_extremes[i]) > 1e-10:
                        error = abs(synthetic_extremes[i] - reference_extremes[i]) / abs(reference_extremes[i])
                        extreme_errors.append(error)
                
                avg_extreme_error = np.mean(extreme_errors) if extreme_errors else 1.0
                extreme_score = 1 / (1 + avg_extreme_error)
                
                metrics.append(QualityMetricResult(
                    metric_name="extreme_value_behavior",
                    metric_type=QualityMetricType.DOMAIN,
                    value=float(avg_extreme_error),
                    score=float(extreme_score),
                    weight=self.config["weights"]["domain"] * 0.5,
                    description="Preservation of extreme value behavior (hydrology)",
                    details={
                        "synthetic_extremes": [float(x) for x in synthetic_extremes],
                        "reference_extremes": [float(x) for x in reference_extremes],
                        "extreme_errors": [float(e) for e in extreme_errors]
                    }
                ))
            except Exception as e:
                logger.warning(f"Extreme value evaluation failed: {e}")
        
        elif domain.lower() == "financial":
            # Financial-specific: volatility clustering, leverage effects
            try:
                # Volatility clustering (GARCH-like behavior)
                synthetic_returns = np.diff(synthetic)
                reference_returns = np.diff(reference)
                
                # Calculate volatility clustering using squared returns autocorrelation
                if len(synthetic_returns) > 10:
                    synthetic_vol_cluster = np.corrcoef(synthetic_returns[:-1]**2, synthetic_returns[1:]**2)[0, 1]
                    reference_vol_cluster = np.corrcoef(reference_returns[:-1]**2, reference_returns[1:]**2)[0, 1]
                    
                    vol_error = abs(synthetic_vol_cluster - reference_vol_cluster) / (abs(reference_vol_cluster) + 1e-10)
                    vol_score = 1 / (1 + vol_error)
                    
                    metrics.append(QualityMetricResult(
                        metric_name="volatility_clustering",
                        metric_type=QualityMetricType.DOMAIN,
                        value=float(vol_error),
                        score=float(vol_score),
                        weight=self.config["weights"]["domain"] * 0.5,
                        description="Preservation of volatility clustering (financial)",
                        details={
                            "synthetic_vol_clustering": float(synthetic_vol_cluster),
                            "reference_vol_clustering": float(reference_vol_cluster),
                            "vol_error": float(vol_error)
                        }
                    ))
            except Exception as e:
                logger.warning(f"Volatility clustering evaluation failed: {e}")
        
        elif domain.lower() == "biomedical":
            # Biomedical-specific: baseline drift, artifact patterns
            try:
                # Baseline drift assessment
                synthetic_baseline = np.mean(synthetic)
                reference_baseline = np.mean(reference)
                
                baseline_error = abs(synthetic_baseline - reference_baseline) / (abs(reference_baseline) + 1e-10)
                baseline_score = 1 / (1 + baseline_error)
                
                metrics.append(QualityMetricResult(
                    metric_name="baseline_behavior",
                    metric_type=QualityMetricType.DOMAIN,
                    value=float(baseline_error),
                    score=float(baseline_score),
                    weight=self.config["weights"]["domain"] * 0.5,
                    description="Preservation of baseline behavior (biomedical)",
                    details={
                        "synthetic_baseline": float(synthetic_baseline),
                        "reference_baseline": float(reference_baseline),
                        "baseline_error": float(baseline_error)
                    }
                ))
            except Exception as e:
                logger.warning(f"Baseline behavior evaluation failed: {e}")
        
        elif domain.lower() == "climate":
            # Climate-specific: seasonal patterns, long-term trends
            try:
                # Seasonal pattern strength
                synthetic_seasonal_strength = np.std(synthetic) / (np.mean(synthetic) + 1e-10)
                reference_seasonal_strength = np.std(reference) / (np.mean(reference) + 1e-10)
                
                seasonal_error = abs(synthetic_seasonal_strength - reference_seasonal_strength) / (abs(reference_seasonal_strength) + 1e-10)
                seasonal_score = 1 / (1 + seasonal_error)
                
                metrics.append(QualityMetricResult(
                    metric_name="seasonal_pattern_strength",
                    metric_type=QualityMetricType.DOMAIN,
                    value=float(seasonal_error),
                    score=float(seasonal_score),
                    weight=self.config["weights"]["domain"] * 0.5,
                    description="Preservation of seasonal pattern strength (climate)",
                    details={
                        "synthetic_seasonal_strength": float(synthetic_seasonal_strength),
                        "reference_seasonal_strength": float(reference_seasonal_strength),
                        "seasonal_error": float(seasonal_error)
                    }
                ))
            except Exception as e:
                logger.warning(f"Seasonal pattern evaluation failed: {e}")
        
        return metrics
    
    def _calculate_composite_score(self, metrics: List[QualityMetricResult]) -> float:
        """Calculate overall composite quality score."""
        if not metrics:
            return 0.0
        
        # Weighted average of all metric scores
        total_weight = sum(metric.weight for metric in metrics)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(metric.score * metric.weight for metric in metrics)
        composite_score = weighted_sum / total_weight
        
        return float(composite_score)
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        thresholds = self.config["thresholds"]
        
        if score >= thresholds["excellent"]:
            return "excellent"
        elif score >= thresholds["good"]:
            return "good"
        elif score >= thresholds["acceptable"]:
            return "acceptable"
        else:
            return "poor"
    
    def _generate_recommendations(self, metrics: List[QualityMetricResult], overall_score: float) -> List[str]:
        """Generate improvement recommendations based on evaluation results."""
        recommendations = []
        
        # Overall quality recommendations
        if overall_score < 0.5:
            recommendations.append("Overall quality is poor. Consider regenerating synthetic data with different parameters.")
        elif overall_score < 0.7:
            recommendations.append("Quality is acceptable but could be improved. Review generation parameters.")
        
        # Specific metric recommendations
        for metric in metrics:
            if metric.score < 0.5:
                if "distribution" in metric.metric_name.lower():
                    recommendations.append(f"Improve distribution matching: {metric.description}")
                elif "temporal" in metric.metric_name.lower():
                    recommendations.append(f"Enhance temporal structure: {metric.description}")
                elif "lrd" in metric.metric_name.lower():
                    recommendations.append(f"Better preserve LRD properties: {metric.description}")
                elif "domain" in metric.metric_name.lower():
                    recommendations.append(f"Address domain-specific issues: {metric.description}")
        
        # Positive feedback for good scores
        good_metrics = [m for m in metrics if m.score > 0.8]
        if good_metrics:
            recommendations.append(f"Excellent performance in: {', '.join([m.metric_name for m in good_metrics[:3]])}")
        
        return recommendations
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all quality evaluations."""
        if not self.evaluation_history:
            return {"message": "No quality evaluations performed"}
        
        summary = {
            "total_evaluations": len(self.evaluation_history),
            "average_overall_score": np.mean([r.overall_score for r in self.evaluation_history]),
            "quality_level_distribution": {},
            "best_performing_metrics": [],
            "areas_for_improvement": []
        }
        
        # Quality level distribution
        for result in self.evaluation_history:
            level = result.quality_level
            summary["quality_level_distribution"][level] = summary["quality_level_distribution"].get(level, 0) + 1
        
        # Best performing metrics
        all_metrics = []
        for result in self.evaluation_history:
            all_metrics.extend(result.metrics)
        
        if all_metrics:
            metric_scores = {}
            for metric in all_metrics:
                if metric.metric_name not in metric_scores:
                    metric_scores[metric.metric_name] = []
                metric_scores[metric.metric_name].append(metric.score)
            
            avg_metric_scores = {name: np.mean(scores) for name, scores in metric_scores.items()}
            best_metrics = sorted(avg_metric_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            summary["best_performing_metrics"] = [
                {"metric": name, "average_score": float(score)} 
                for name, score in best_metrics
            ]
        
        # Areas for improvement
        if all_metrics:
            worst_metrics = sorted(avg_metric_scores.items(), key=lambda x: x[1])[:5]
            summary["areas_for_improvement"] = [
                {"metric": name, "average_score": float(score)} 
                for name, score in worst_metrics
            ]
        
        return summary
    
    def save_evaluation_result(self, result: QualityEvaluationResult, filepath: Union[str, Path]):
        """Save evaluation result to file."""
        # Convert numpy arrays to lists for JSON serialization
        result_dict = {
            "synthetic_data": result.synthetic_data.tolist(),
            "reference_data": result.reference_data.tolist(),
            "reference_metadata": result.reference_metadata,
            "metrics": [
                {
                    "metric_name": m.metric_name,
                    "metric_type": m.metric_type.value,
                    "value": m.value,
                    "score": m.score,
                    "weight": m.weight,
                    "description": m.description,
                    "details": m.details
                }
                for m in result.metrics
            ],
            "overall_score": result.overall_score,
            "quality_level": result.quality_level,
            "recommendations": result.recommendations,
            "evaluation_date": result.evaluation_date
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Evaluation result saved to {filepath}")
    
    def create_quality_report(self, result: QualityEvaluationResult, save_path: Optional[Union[str, Path]] = None) -> str:
        """Create a comprehensive quality report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("SYNTHETIC DATA QUALITY EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Evaluation Date: {result.evaluation_date}")
        report_lines.append(f"Overall Quality Score: {result.overall_score:.3f} ({result.quality_level.upper()})")
        report_lines.append("")
        
        # Dataset information
        report_lines.append("DATASET INFORMATION")
        report_lines.append("-" * 30)
        report_lines.append(f"Synthetic Data Points: {len(result.synthetic_data):,}")
        report_lines.append(f"Reference Data Points: {len(result.reference_data):,}")
        report_lines.append(f"Reference Domain: {result.reference_metadata.get('domain', 'Unknown')}")
        report_lines.append("")
        
        # Detailed metrics
        report_lines.append("DETAILED METRICS")
        report_lines.append("-" * 30)
        
        # Group metrics by type
        metric_types = {}
        for metric in result.metrics:
            if metric.metric_type not in metric_types:
                metric_types[metric.metric_type] = []
            metric_types[metric.metric_type].append(metric)
        
        for metric_type, type_metrics in metric_types.items():
            report_lines.append(f"\n{metric_type.value.upper()} METRICS:")
            for metric in type_metrics:
                report_lines.append(f"  {metric.metric_name}: {metric.score:.3f} (weight: {metric.weight:.3f})")
                report_lines.append(f"    {metric.description}")
                if metric.details:
                    for key, value in metric.details.items():
                        if isinstance(value, (int, float)):
                            report_lines.append(f"    {key}: {value:.4f}")
                        else:
                            report_lines.append(f"    {key}: {value}")
        
        # Recommendations
        report_lines.append("\n" + "=" * 60)
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("=" * 60)
        
        if result.recommendations:
            for i, rec in enumerate(result.recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        else:
            report_lines.append("No specific recommendations at this time.")
        
        # Footer
        report_lines.append("\n" + "=" * 60)
        report_lines.append("Report generated by SyntheticDataQualityEvaluator")
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Quality report saved to {save_path}")
        
        return report_text


def create_domain_specific_evaluator(domain: str) -> SyntheticDataQualityEvaluator:
    """Create an evaluator configured for a specific domain."""
    config = {
        "statistical_metrics": {
            "distribution_similarity": True,
            "moment_preservation": True,
            "tail_behavior": True,
            "quantile_matching": True
        },
        "temporal_metrics": {
            "autocorrelation": True,
            "seasonality": True,
            "trend_preservation": True,
            "volatility_clustering": True
        },
        "lrd_metrics": {
            "hurst_preservation": True,
            "spectral_properties": True,
            "scaling_behavior": True
        },
        "domain_metrics": {
            domain.lower(): True
        },
        "weights": {
            "statistical": 0.25,
            "temporal": 0.25,
            "lrd": 0.35,
            "domain": 0.15
        },
        "thresholds": {
            "excellent": 0.9,
            "good": 0.7,
            "acceptable": 0.5,
            "poor": 0.3
        }
    }
    
    return SyntheticDataQualityEvaluator(config)
