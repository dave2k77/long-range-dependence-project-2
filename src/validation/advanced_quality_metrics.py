#!/usr/bin/env python3
"""
Advanced Quality Metrics for Synthetic Data

This module provides innovative quality assessment methods including:
- Machine learning-based quality prediction
- Cross-dataset quality assessment
- Advanced LRD-specific metrics
- Multi-objective quality optimization
- Quality uncertainty quantification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import json
import warnings
import logging
from dataclasses import dataclass, asdict

# Machine learning imports
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML-based quality metrics will be disabled.")

# Import our base quality evaluation system
try:
    from .synthetic_data_quality import (
        SyntheticDataQualityEvaluator,
        QualityEvaluationResult,
        QualityMetricResult,
        QualityMetricType
    )
except ImportError:
    # Fallback to absolute imports for demo purposes
    from src.validation.synthetic_data_quality import (
        SyntheticDataQualityEvaluator,
        QualityEvaluationResult,
        QualityMetricResult,
        QualityMetricType
    )

logger = logging.getLogger(__name__)

@dataclass
class MLQualityPrediction:
    """Machine learning-based quality prediction result."""
    predicted_score: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    model_performance: Dict[str, float]
    prediction_uncertainty: float

@dataclass
class CrossDatasetQualityResult:
    """Cross-dataset quality assessment result."""
    dataset_similarity: float
    quality_transfer_score: float
    domain_adaptation_quality: float
    cross_validation_score: float
    transfer_learning_metrics: Dict[str, float]

@dataclass
class AdvancedLRDMetric:
    """Advanced LRD-specific quality metric."""
    metric_name: str
    metric_value: float
    metric_score: float
    confidence_level: float
    theoretical_bounds: Tuple[float, float]
    domain_specificity: str

class AdvancedQualityMetrics:
    """
    Advanced quality assessment methods for synthetic data.
    
    Provides innovative metrics including ML-based prediction,
    cross-dataset assessment, and advanced LRD analysis.
    """
    
    def __init__(self, output_dir: str = "advanced_quality_metrics"):
        """
        Initialize advanced quality metrics system.
        
        Parameters:
        -----------
        output_dir : str
            Directory for storing advanced metric results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize ML models if available
        self.ml_models = {}
        self.feature_scalers = {}
        self.quality_predictor = None
        
        if ML_AVAILABLE:
            self._initialize_ml_models()
        
        # Initialize base evaluator
        self.base_evaluator = SyntheticDataQualityEvaluator()
        
        logger.info(f"Advanced Quality Metrics initialized at {self.output_dir.absolute()}")
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for quality prediction."""
        try:
            # Quality prediction model
            self.quality_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Feature scaler
            self.feature_scaler = StandardScaler()
            
            logger.info("Machine learning models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            ML_AVAILABLE = False
    
    def predict_quality_ml(self, 
                          synthetic_data: np.ndarray,
                          reference_data: np.ndarray,
                          domain: str = "general") -> MLQualityPrediction:
        """
        Predict synthetic data quality using machine learning.
        
        Parameters:
        -----------
        synthetic_data : np.ndarray
            Synthetic data to evaluate
        reference_data : np.ndarray
            Reference data for comparison
        domain : str
            Data domain for domain-specific prediction
            
        Returns:
        --------
        MLQualityPrediction
            ML-based quality prediction with confidence intervals
        """
        if not ML_AVAILABLE or self.quality_predictor is None:
            raise RuntimeError("Machine learning models not available")
        
        try:
            # Extract features from data
            features = self._extract_ml_features(synthetic_data, reference_data, domain)
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            predicted_score = self.quality_predictor.predict(features_scaled)[0]
            
            # Estimate prediction uncertainty using ensemble variance
            predictions = []
            for estimator in self.quality_predictor.estimators_:
                pred = estimator.predict(features_scaled)[0]
                predictions.append(pred)
            
            prediction_std = np.std(predictions)
            confidence_interval = (
                max(0.0, predicted_score - 2 * prediction_std),
                min(1.0, predicted_score + 2 * prediction_std)
            )
            
            # Get feature importance
            feature_names = self._get_feature_names(domain)
            feature_importance = dict(zip(feature_names, 
                                       self.quality_predictor.feature_importances_))
            
            # Model performance metrics
            model_performance = {
                'r2_score': getattr(self.quality_predictor, 'r2_score_', 0.0),
                'prediction_std': float(prediction_std)
            }
            
            return MLQualityPrediction(
                predicted_score=float(predicted_score),
                confidence_interval=confidence_interval,
                feature_importance=feature_importance,
                model_performance=model_performance,
                prediction_uncertainty=float(prediction_std)
            )
            
        except Exception as e:
            logger.error(f"ML quality prediction failed: {e}")
            raise
    
    def _extract_ml_features(self, 
                            synthetic_data: np.ndarray,
                            reference_data: np.ndarray,
                            domain: str) -> np.ndarray:
        """Extract features for machine learning quality prediction."""
        features = []
        
        # Statistical features
        features.extend([
            np.mean(synthetic_data),
            np.std(synthetic_data),
            np.var(synthetic_data),
            np.median(synthetic_data),
            np.percentile(synthetic_data, 25),
            np.percentile(synthetic_data, 75),
            np.max(synthetic_data),
            np.min(synthetic_data)
        ])
        
        # Distribution features
        features.extend([
            np.sum(synthetic_data < np.mean(synthetic_data)) / len(synthetic_data),  # Skewness proxy
            np.sum(np.abs(synthetic_data - np.mean(synthetic_data)) > np.std(synthetic_data)) / len(synthetic_data),  # Kurtosis proxy
        ])
        
        # Spectral features
        try:
            # Power spectral density features
            freqs = np.fft.fftfreq(len(synthetic_data))
            psd = np.abs(np.fft.fft(synthetic_data)) ** 2
            psd = psd[freqs > 0]  # Remove DC component
            
            features.extend([
                np.mean(psd),
                np.std(psd),
                np.percentile(psd, 50),
                np.sum(psd > np.mean(psd)) / len(psd)
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Temporal features
        try:
            # Autocorrelation features
            autocorr = np.correlate(synthetic_data, synthetic_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + 10]  # First 10 lags
            
            features.extend([
                np.mean(autocorr),
                np.std(autocorr),
                autocorr[1] if len(autocorr) > 1 else 0.0,  # Lag-1 autocorrelation
                autocorr[5] if len(autocorr) > 5 else 0.0   # Lag-5 autocorrelation
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Reference data comparison features
        if len(reference_data) > 0:
            try:
                # Scale reference data to match synthetic data length
                if len(reference_data) != len(synthetic_data):
                    ref_resampled = np.interp(
                        np.linspace(0, 1, len(synthetic_data)),
                        np.linspace(0, 1, len(reference_data)),
                        reference_data
                    )
                else:
                    ref_resampled = reference_data
                
                # Comparison features
                features.extend([
                    np.mean(np.abs(synthetic_data - ref_resampled)),
                    np.std(synthetic_data - ref_resampled),
                    np.corrcoef(synthetic_data, ref_resampled)[0, 1] if len(synthetic_data) > 1 else 0.0,
                    np.mean(np.abs(np.diff(synthetic_data) - np.diff(ref_resampled)))
                ])
            except:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Domain-specific features
        domain_features = self._extract_domain_features(synthetic_data, domain)
        features.extend(domain_features)
        
        return np.array(features)
    
    def _extract_domain_features(self, data: np.ndarray, domain: str) -> List[float]:
        """Extract domain-specific features."""
        features = []
        
        if domain == "financial":
            # Financial-specific features
            returns = np.diff(data)
            features.extend([
                np.std(returns),  # Volatility
                np.sum(returns > 0) / len(returns),  # Positive return ratio
                np.mean(np.abs(returns)),  # Mean absolute return
                np.percentile(np.abs(returns), 95)  # 95th percentile of absolute returns
            ])
        elif domain == "hydrology":
            # Hydrology-specific features
            features.extend([
                np.sum(data > np.mean(data)) / len(data),  # Above-mean ratio
                np.max(data) - np.min(data),  # Range
                np.mean(np.diff(data)),  # Mean change
                np.std(np.diff(data))  # Change variability
            ])
        elif domain == "biomedical":
            # Biomedical-specific features
            features.extend([
                np.sum(np.abs(data) > 2 * np.std(data)) / len(data),  # Outlier ratio
                np.mean(np.abs(np.diff(data))),  # Mean absolute change
                np.std(np.abs(np.diff(data))),  # Change variability
                np.sum(data > 0) / len(data)  # Positive value ratio
            ])
        elif domain == "climate":
            # Climate-specific features
            features.extend([
                np.mean(np.diff(data)),  # Trend
                np.std(np.diff(data)),  # Trend variability
                np.sum(data > np.mean(data)) / len(data),  # Above-mean ratio
                np.percentile(data, 90) - np.percentile(data, 10)  # 80% range
            ])
        else:
            # General features
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def _get_feature_names(self, domain: str) -> List[str]:
        """Get feature names for the ML model."""
        base_features = [
            'mean', 'std', 'var', 'median', 'p25', 'p75', 'max', 'min',
            'skewness_proxy', 'kurtosis_proxy',
            'psd_mean', 'psd_std', 'psd_median', 'psd_above_mean_ratio',
            'autocorr_mean', 'autocorr_std', 'lag1_autocorr', 'lag5_autocorr',
            'ref_diff_mean', 'ref_diff_std', 'ref_correlation', 'ref_diff_diff_mean'
        ]
        
        domain_features = {
            'financial': ['volatility', 'positive_return_ratio', 'mean_abs_return', 'return_95th_percentile'],
            'hydrology': ['above_mean_ratio', 'range', 'mean_change', 'change_variability'],
            'biomedical': ['outlier_ratio', 'mean_abs_change', 'change_variability', 'positive_value_ratio'],
            'climate': ['trend', 'trend_variability', 'above_mean_ratio', 'eighty_percent_range']
        }
        
        return base_features + domain_features.get(domain, [0.0] * 4)
    
    def assess_cross_dataset_quality(self,
                                   synthetic_data: np.ndarray,
                                   reference_datasets: Dict[str, np.ndarray],
                                   domain: str = "general") -> CrossDatasetQualityResult:
        """
        Assess quality across multiple reference datasets.
        
        Parameters:
        -----------
        synthetic_data : np.ndarray
            Synthetic data to evaluate
        reference_datasets : Dict[str, np.ndarray]
            Multiple reference datasets for comparison
        domain : str
            Data domain for assessment
            
        Returns:
        --------
        CrossDatasetQualityResult
            Cross-dataset quality assessment results
        """
        try:
            # Calculate quality scores for each reference dataset
            quality_scores = []
            dataset_similarities = []
            
            for dataset_name, reference_data in reference_datasets.items():
                try:
                    # Run quality evaluation
                    quality_result = self.base_evaluator.evaluate_quality(
                        synthetic_data=synthetic_data,
                        reference_data=reference_data,
                        reference_metadata={"domain": domain, "source": dataset_name},
                        domain=domain,
                        normalize_for_comparison=True
                    )
                    
                    quality_scores.append(quality_result.overall_score)
                    
                    # Calculate dataset similarity
                    similarity = self._calculate_dataset_similarity(synthetic_data, reference_data)
                    dataset_similarities.append(similarity)
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate against {dataset_name}: {e}")
                    quality_scores.append(0.0)
                    dataset_similarities.append(0.0)
            
            # Cross-dataset metrics
            avg_quality_score = np.mean(quality_scores)
            quality_transfer_score = np.std(quality_scores)  # Lower is better (more consistent)
            
            # Domain adaptation quality (how well quality transfers across datasets)
            domain_adaptation_quality = 1.0 - quality_transfer_score
            
            # Cross-validation score
            cross_validation_score = np.mean(quality_scores)
            
            # Transfer learning metrics
            transfer_metrics = {
                'quality_consistency': 1.0 - np.std(quality_scores),
                'best_reference_match': np.max(quality_scores),
                'worst_reference_match': np.min(quality_scores),
                'quality_range': np.max(quality_scores) - np.min(quality_scores)
            }
            
            return CrossDatasetQualityResult(
                dataset_similarity=float(np.mean(dataset_similarities)),
                quality_transfer_score=float(quality_transfer_score),
                domain_adaptation_quality=float(domain_adaptation_quality),
                cross_validation_score=float(cross_validation_score),
                transfer_learning_metrics=transfer_metrics
            )
            
        except Exception as e:
            logger.error(f"Cross-dataset quality assessment failed: {e}")
            raise
    
    def _calculate_dataset_similarity(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate similarity between two datasets."""
        try:
            # Resample to same length if necessary
            if len(data1) != len(data2):
                data2_resampled = np.interp(
                    np.linspace(0, 1, len(data1)),
                    np.linspace(0, 1, len(data2)),
                    data2
                )
            else:
                data2_resampled = data2
            
            # Normalize both datasets
            data1_norm = (data1 - np.mean(data1)) / np.std(data1)
            data2_norm = (data2_resampled - np.mean(data2_resampled)) / np.std(data2_resampled)
            
            # Calculate correlation
            correlation = np.corrcoef(data1_norm, data2_norm)[0, 1]
            
            # Calculate distribution similarity (KL divergence proxy)
            hist1, _ = np.histogram(data1_norm, bins=20, density=True)
            hist2, _ = np.histogram(data2_norm, bins=20, density=True)
            
            # Add small epsilon to avoid division by zero
            hist1 = hist1 + 1e-10
            hist2 = hist2 + 1e-10
            
            # Jensen-Shannon divergence (symmetric)
            js_div = 0.5 * (
                np.sum(hist1 * np.log(hist1 / ((hist1 + hist2) / 2))) +
                np.sum(hist2 * np.log(hist2 / ((hist1 + hist2) / 2)))
            )
            
            # Convert to similarity (0-1, higher is more similar)
            js_similarity = 1.0 / (1.0 + js_div)
            
            # Combine correlation and distribution similarity
            overall_similarity = 0.6 * abs(correlation) + 0.4 * js_similarity
            
            return float(overall_similarity)
            
        except Exception as e:
            logger.warning(f"Dataset similarity calculation failed: {e}")
            return 0.0
    
    def calculate_advanced_lrd_metrics(self,
                                     synthetic_data: np.ndarray,
                                     reference_data: np.ndarray,
                                     domain: str = "general") -> List[AdvancedLRDMetric]:
        """
        Calculate advanced LRD-specific quality metrics.
        
        Parameters:
        -----------
        synthetic_data : np.ndarray
            Synthetic data to evaluate
        reference_data : np.ndarray
            Reference data for comparison
        domain : str
            Data domain for domain-specific metrics
            
        Returns:
        --------
        List[AdvancedLRDMetric]
            Advanced LRD quality metrics
        """
        metrics = []
        
        try:
            # 1. Hurst Exponent Consistency
            hurst_metric = self._calculate_hurst_consistency(synthetic_data, reference_data)
            metrics.append(hurst_metric)
            
            # 2. Power Law Scaling
            scaling_metric = self._calculate_power_law_scaling(synthetic_data, reference_data)
            metrics.append(scaling_metric)
            
            # 3. Fractal Dimension Preservation
            fractal_metric = self._calculate_fractal_dimension_preservation(synthetic_data, reference_data)
            metrics.append(fractal_metric)
            
            # 4. Long-Memory Structure
            long_memory_metric = self._calculate_long_memory_structure(synthetic_data, reference_data)
            metrics.append(long_memory_metric)
            
            # 5. Domain-specific LRD metrics
            domain_metrics = self._calculate_domain_lrd_metrics(synthetic_data, reference_data, domain)
            metrics.extend(domain_metrics)
            
        except Exception as e:
            logger.error(f"Advanced LRD metrics calculation failed: {e}")
        
        return metrics
    
    def _calculate_hurst_consistency(self, 
                                   synthetic_data: np.ndarray,
                                   reference_data: np.ndarray) -> AdvancedLRDMetric:
        """Calculate Hurst exponent consistency between synthetic and reference data."""
        try:
            # Simple Hurst estimation using R/S analysis
            def estimate_hurst(data):
                if len(data) < 10:
                    return 0.5
                
                # Calculate R/S for different lags
                lags = np.logspace(1, np.log10(len(data)//4), 10, dtype=int)
                rs_values = []
                
                for lag in lags:
                    if lag < 2:
                        continue
                    
                    # Split data into segments
                    segments = len(data) // lag
                    if segments < 2:
                        continue
                    
                    rs_segments = []
                    for i in range(segments):
                        segment = data[i*lag:(i+1)*lag]
                        if len(segment) < 2:
                            continue
                        
                        # Calculate R (range) and S (standard deviation)
                        segment_mean = np.mean(segment)
                        cumsum = np.cumsum(segment - segment_mean)
                        R = np.max(cumsum) - np.min(cumsum)
                        S = np.std(segment)
                        
                        if S > 0:
                            rs_segments.append(R / S)
                    
                    if rs_segments:
                        rs_values.append(np.mean(rs_segments))
                
                if len(rs_values) < 2:
                    return 0.5
                
                # Fit log-log relationship
                lags = lags[:len(rs_values)]
                if len(lags) < 2:
                    return 0.5
                
                log_lags = np.log(lags)
                log_rs = np.log(rs_values)
                
                # Linear regression
                coeffs = np.polyfit(log_lags, log_rs, 1)
                hurst = coeffs[0]
                
                return max(0.0, min(1.0, hurst))
            
            # Estimate Hurst for both datasets
            hurst_synthetic = estimate_hurst(synthetic_data)
            hurst_reference = estimate_hurst(reference_data)
            
            # Calculate consistency
            hurst_diff = abs(hurst_synthetic - hurst_reference)
            hurst_score = 1.0 / (1.0 + hurst_diff)
            
            # Confidence based on data length
            confidence = min(1.0, len(synthetic_data) / 1000.0)
            
            return AdvancedLRDMetric(
                metric_name="hurst_exponent_consistency",
                metric_value=float(hurst_diff),
                metric_score=float(hurst_score),
                confidence_level=float(confidence),
                theoretical_bounds=(0.0, 1.0),
                domain_specificity="universal"
            )
            
        except Exception as e:
            logger.warning(f"Hurst consistency calculation failed: {e}")
            return AdvancedLRDMetric(
                metric_name="hurst_exponent_consistency",
                metric_value=0.0,
                metric_score=0.0,
                confidence_level=0.0,
                theoretical_bounds=(0.0, 1.0),
                domain_specificity="universal"
            )
    
    def _calculate_power_law_scaling(self,
                                   synthetic_data: np.ndarray,
                                   reference_data: np.ndarray) -> AdvancedLRDMetric:
        """Calculate power law scaling consistency."""
        try:
            # Calculate power spectral density
            def calculate_psd_scaling(data):
                if len(data) < 10:
                    return 0.0, 0.0
                
                # FFT
                freqs = np.fft.fftfreq(len(data))
                psd = np.abs(np.fft.fft(data)) ** 2
                
                # Remove DC component and negative frequencies
                positive_freqs = freqs > 0
                freqs = freqs[positive_freqs]
                psd = psd[positive_freqs]
                
                if len(freqs) < 2:
                    return 0.0, 0.0
                
                # Log-log relationship
                log_freqs = np.log(freqs)
                log_psd = np.log(psd)
                
                # Linear regression
                coeffs = np.polyfit(log_freqs, log_psd, 1)
                slope = coeffs[0]
                
                # R-squared
                y_pred = np.polyval(coeffs, log_freqs)
                ss_res = np.sum((log_psd - y_pred) ** 2)
                ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                return slope, r_squared
            
            # Calculate scaling for both datasets
            slope_synthetic, r2_synthetic = calculate_psd_scaling(synthetic_data)
            slope_reference, r2_reference = calculate_psd_scaling(reference_data)
            
            # Calculate consistency
            slope_diff = abs(slope_synthetic - slope_reference)
            slope_score = 1.0 / (1.0 + slope_diff)
            
            # Confidence based on R-squared values
            confidence = (r2_synthetic + r2_reference) / 2
            
            return AdvancedLRDMetric(
                metric_name="power_law_scaling_consistency",
                metric_value=float(slope_diff),
                metric_score=float(slope_score),
                confidence_level=float(confidence),
                theoretical_bounds=(-5.0, 5.0),
                domain_specificity="universal"
            )
            
        except Exception as e:
            logger.warning(f"Power law scaling calculation failed: {e}")
            return AdvancedLRDMetric(
                metric_name="power_law_scaling_consistency",
                metric_value=0.0,
                metric_score=0.0,
                confidence_level=0.0,
                theoretical_bounds=(-5.0, 5.0),
                domain_specificity="universal"
            )
    
    def _calculate_fractal_dimension_preservation(self,
                                                synthetic_data: np.ndarray,
                                                reference_data: np.ndarray) -> AdvancedLRDMetric:
        """Calculate fractal dimension preservation."""
        try:
            # Simple box-counting dimension estimation
            def estimate_fractal_dimension(data, max_boxes=10):
                if len(data) < 10:
                    return 1.0
                
                # Normalize data to [0, 1]
                data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
                
                box_counts = []
                box_sizes = []
                
                for i in range(1, max_boxes + 1):
                    box_size = 1.0 / i
                    boxes = set()
                    
                    for point in data_norm:
                        box_x = int(point / box_size)
                        boxes.add(box_x)
                    
                    box_counts.append(len(boxes))
                    box_sizes.append(box_size)
                
                # Log-log relationship
                log_sizes = np.log(box_sizes)
                log_counts = np.log(box_counts)
                
                # Linear regression
                coeffs = np.polyfit(log_sizes, log_counts, 1)
                dimension = -coeffs[0]
                
                return max(1.0, min(2.0, dimension))
            
            # Estimate fractal dimension for both datasets
            fd_synthetic = estimate_fractal_dimension(synthetic_data)
            fd_reference = estimate_fractal_dimension(reference_data)
            
            # Calculate preservation
            fd_diff = abs(fd_synthetic - fd_reference)
            fd_score = 1.0 / (1.0 + fd_diff)
            
            # Confidence based on data length
            confidence = min(1.0, len(synthetic_data) / 1000.0)
            
            return AdvancedLRDMetric(
                metric_name="fractal_dimension_preservation",
                metric_value=float(fd_diff),
                metric_score=float(fd_score),
                confidence_level=float(confidence),
                theoretical_bounds=(1.0, 2.0),
                domain_specificity="universal"
            )
            
        except Exception as e:
            logger.warning(f"Fractal dimension calculation failed: {e}")
            return AdvancedLRDMetric(
                metric_name="fractal_dimension_preservation",
                metric_value=0.0,
                metric_score=0.0,
                confidence_level=0.0,
                theoretical_bounds=(1.0, 2.0),
                domain_specificity="universal"
            )
    
    def _calculate_long_memory_structure(self,
                                       synthetic_data: np.ndarray,
                                       reference_data: np.ndarray) -> AdvancedLRDMetric:
        """Calculate long-memory structure preservation."""
        try:
            # Calculate autocorrelation function
            def calculate_long_memory(data, max_lag=50):
                if len(data) < max_lag * 2:
                    return 0.0
                
                # Normalize data
                data_norm = (data - np.mean(data)) / np.std(data)
                
                # Calculate autocorrelation
                autocorr = []
                for lag in range(1, max_lag + 1):
                    if lag >= len(data_norm):
                        break
                    
                    # Calculate correlation at this lag
                    corr = np.corrcoef(data_norm[:-lag], data_norm[lag:])[0, 1]
                    autocorr.append(corr)
                
                if not autocorr:
                    return 0.0
                
                # Calculate long-memory indicator (decay rate)
                lags = np.arange(1, len(autocorr) + 1)
                log_lags = np.log(lags)
                log_autocorr = np.log(np.abs(autocorr))
                
                # Remove infinite values
                valid_mask = np.isfinite(log_autocorr)
                if np.sum(valid_mask) < 2:
                    return 0.0
                
                log_lags = log_lags[valid_mask]
                log_autocorr = log_autocorr[valid_mask]
                
                # Linear regression
                coeffs = np.polyfit(log_lags, log_autocorr, 1)
                decay_rate = coeffs[0]
                
                return decay_rate
            
            # Calculate long-memory for both datasets
            lm_synthetic = calculate_long_memory(synthetic_data)
            lm_reference = calculate_long_memory(reference_data)
            
            # Calculate preservation
            lm_diff = abs(lm_synthetic - lm_reference)
            lm_score = 1.0 / (1.0 + lm_diff)
            
            # Confidence based on data length
            confidence = min(1.0, len(synthetic_data) / 1000.0)
            
            return AdvancedLRDMetric(
                metric_name="long_memory_structure_preservation",
                metric_value=float(lm_diff),
                metric_score=float(lm_score),
                confidence_level=float(confidence),
                theoretical_bounds=(-2.0, 0.0),
                domain_specificity="universal"
            )
            
        except Exception as e:
            logger.warning(f"Long-memory structure calculation failed: {e}")
            return AdvancedLRDMetric(
                metric_name="long_memory_structure_preservation",
                metric_value=0.0,
                metric_score=0.0,
                confidence_level=0.0,
                theoretical_bounds=(-2.0, 0.0),
                domain_specificity="universal"
            )
    
    def _calculate_domain_lrd_metrics(self,
                                    synthetic_data: np.ndarray,
                                    reference_data: np.ndarray,
                                    domain: str) -> List[AdvancedLRDMetric]:
        """Calculate domain-specific LRD metrics."""
        metrics = []
        
        if domain == "financial":
            # Financial volatility clustering
            try:
                returns_synthetic = np.diff(synthetic_data)
                returns_reference = np.diff(reference_data)
                
                # Volatility clustering (autocorrelation of squared returns)
                vol_cluster_synthetic = np.corrcoef(returns_synthetic[:-1]**2, returns_synthetic[1:]**2)[0, 1]
                vol_cluster_reference = np.corrcoef(returns_reference[:-1]**2, returns_reference[1:]**2)[0, 1]
                
                vol_cluster_diff = abs(vol_cluster_synthetic - vol_cluster_reference)
                vol_cluster_score = 1.0 / (1.0 + vol_cluster_diff)
                
                metrics.append(AdvancedLRDMetric(
                    metric_name="volatility_clustering_preservation",
                    metric_value=float(vol_cluster_diff),
                    metric_score=float(vol_cluster_score),
                    confidence_level=0.8,
                    theoretical_bounds=(-1.0, 1.0),
                    domain_specificity="financial"
                ))
            except:
                pass
        
        elif domain == "hydrology":
            # Hydrological persistence
            try:
                # Calculate persistence (how long values stay above/below mean)
                def calculate_persistence(data):
                    mean_val = np.mean(data)
                    above_mean = data > mean_val
                    
                    persistence_lengths = []
                    current_length = 0
                    
                    for val in above_mean:
                        if val:
                            current_length += 1
                        else:
                            if current_length > 0:
                                persistence_lengths.append(current_length)
                                current_length = 0
                    
                    if current_length > 0:
                        persistence_lengths.append(current_length)
                    
                    return np.mean(persistence_lengths) if persistence_lengths else 0
                
                persistence_synthetic = calculate_persistence(synthetic_data)
                persistence_reference = calculate_persistence(reference_data)
                
                persistence_diff = abs(persistence_synthetic - persistence_reference)
                persistence_score = 1.0 / (1.0 + persistence_diff / 10.0)  # Normalize
                
                metrics.append(AdvancedLRDMetric(
                    metric_name="hydrological_persistence_preservation",
                    metric_value=float(persistence_diff),
                    metric_score=float(persistence_score),
                    confidence_level=0.8,
                    theoretical_bounds=(0.0, 50.0),
                    domain_specificity="hydrology"
                ))
            except:
                pass
        
        return metrics
    
    def train_quality_predictor(self, 
                               training_data: List[Tuple[np.ndarray, np.ndarray, float]],
                               domain: str = "general"):
        """
        Train the ML quality predictor on historical data.
        
        Parameters:
        -----------
        training_data : List[Tuple[np.ndarray, np.ndarray, float]]
            List of (synthetic_data, reference_data, quality_score) tuples
        domain : str
            Data domain for domain-specific training
        """
        if not ML_AVAILABLE or self.quality_predictor is None:
            raise RuntimeError("Machine learning models not available")
        
        try:
            # Extract features and labels
            features = []
            labels = []
            
            for synthetic_data, reference_data, quality_score in training_data:
                try:
                    feature_vector = self._extract_ml_features(synthetic_data, reference_data, domain)
                    features.append(feature_vector)
                    labels.append(quality_score)
                except Exception as e:
                    logger.warning(f"Failed to extract features for training sample: {e}")
                    continue
            
            if len(features) < 5:
                raise ValueError("Insufficient training data")
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(labels)
            
            # Split training and validation data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.feature_scaler.fit(X_train)
            X_train_scaled = self.feature_scaler.transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            # Train model
            self.quality_predictor.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.quality_predictor.predict(X_val_scaled)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # Store model performance
            self.quality_predictor.r2_score_ = r2
            self.quality_predictor.mse_ = mse
            
            logger.info(f"Quality predictor trained successfully. R²: {r2:.3f}, MSE: {mse:.3f}")
            
            # Save trained model
            self._save_trained_model(domain)
            
        except Exception as e:
            logger.error(f"Failed to train quality predictor: {e}")
            raise
    
    def _save_trained_model(self, domain: str):
        """Save the trained ML model."""
        try:
            model_dir = self.output_dir / "trained_models"
            model_dir.mkdir(exist_ok=True)
            
            # Save model parameters (simplified)
            model_info = {
                'domain': domain,
                'training_date': datetime.now().isoformat(),
                'model_type': 'RandomForestRegressor',
                'n_estimators': self.quality_predictor.n_estimators,
                'max_depth': self.quality_predictor.max_depth,
                'feature_importance': dict(zip(
                    self._get_feature_names(domain),
                    self.quality_predictor.feature_importances_.tolist()
                )),
                'performance': {
                    'r2_score': getattr(self.quality_predictor, 'r2_score_', 0.0),
                    'mse': getattr(self.quality_predictor, 'mse_', 0.0)
                }
            }
            
            model_file = model_dir / f"quality_predictor_{domain}.json"
            with open(model_file, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"Trained model saved: {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to save trained model: {e}")
    
    def generate_advanced_quality_report(self,
                                       synthetic_data: np.ndarray,
                                       reference_data: np.ndarray,
                                       domain: str = "general") -> Dict[str, Any]:
        """
        Generate comprehensive advanced quality report.
        
        Parameters:
        -----------
        synthetic_data : np.ndarray
            Synthetic data to evaluate
        reference_data : np.ndarray
            Reference data for comparison
        domain : str
            Data domain for assessment
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive advanced quality report
        """
        report = {
            'report_date': datetime.now().isoformat(),
            'domain': domain,
            'data_info': {
                'synthetic_data_length': len(synthetic_data),
                'reference_data_length': len(reference_data),
                'synthetic_data_stats': {
                    'mean': float(np.mean(synthetic_data)),
                    'std': float(np.std(synthetic_data)),
                    'min': float(np.min(synthetic_data)),
                    'max': float(np.max(synthetic_data))
                }
            }
        }
        
        try:
            # ML-based quality prediction
            if ML_AVAILABLE and self.quality_predictor is not None:
                ml_prediction = self.predict_quality_ml(synthetic_data, reference_data, domain)
                report['ml_quality_prediction'] = asdict(ml_prediction)
            
            # Advanced LRD metrics
            lrd_metrics = self.calculate_advanced_lrd_metrics(synthetic_data, reference_data, domain)
            report['advanced_lrd_metrics'] = [
                {
                    'metric_name': m.metric_name,
                    'metric_value': m.metric_value,
                    'metric_score': m.metric_score,
                    'confidence_level': m.confidence_level,
                    'theoretical_bounds': m.theoretical_bounds,
                    'domain_specificity': m.domain_specificity
                }
                for m in lrd_metrics
            ]
            
            # Cross-dataset quality (if multiple reference datasets available)
            if isinstance(reference_data, dict):
                cross_dataset_result = self.assess_cross_dataset_quality(
                    synthetic_data, reference_data, domain
                )
                report['cross_dataset_quality'] = asdict(cross_dataset_result)
            
            # Overall advanced quality score
            if lrd_metrics:
                lrd_scores = [m.metric_score for m in lrd_metrics]
                avg_lrd_score = np.mean(lrd_scores)
                
                if 'ml_quality_prediction' in report:
                    ml_score = report['ml_quality_prediction']['predicted_score']
                    # Combine ML prediction with LRD metrics
                    overall_advanced_score = 0.6 * ml_score + 0.4 * avg_lrd_score
                else:
                    overall_advanced_score = avg_lrd_score
                
                report['overall_advanced_quality_score'] = float(overall_advanced_score)
                report['quality_level'] = self._classify_quality_level(overall_advanced_score)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"advanced_quality_report_{domain}_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Advanced quality report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate advanced quality report: {e}")
            report['error'] = str(e)
        
        return report
    
    def _classify_quality_level(self, score: float) -> str:
        """Classify quality level based on score."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "acceptable"
        else:
            return "poor"
