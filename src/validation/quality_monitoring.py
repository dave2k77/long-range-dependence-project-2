#!/usr/bin/env python3
"""
Automated Quality Monitoring System

This module provides continuous quality assessment for synthetic data generation
pipelines, including real-time monitoring, quality alerts, and trend analysis.

Features:
- Real-time quality monitoring
- Quality trend analysis
- Automated quality alerts
- Quality dashboard generation
- Historical quality tracking
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
from queue import Queue
import warnings

# Import our quality evaluation system
try:
    from .synthetic_data_quality import (
        SyntheticDataQualityEvaluator, 
        create_domain_specific_evaluator,
        QualityEvaluationResult
    )
except ImportError:
    # Fallback to absolute imports for demo purposes
    from src.validation.synthetic_data_quality import (
        SyntheticDataQualityEvaluator, 
        create_domain_specific_evaluator,
        QualityEvaluationResult
    )

logger = logging.getLogger(__name__)

@dataclass
class QualityAlert:
    """Quality alert configuration and status."""
    alert_id: str
    alert_type: str  # 'threshold', 'trend', 'anomaly'
    metric_name: str
    threshold: float
    current_value: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: str
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class QualityTrend:
    """Quality trend analysis result."""
    metric_name: str
    trend_direction: str  # 'improving', 'stable', 'declining'
    trend_strength: float  # 0-1, how strong the trend is
    slope: float  # Linear regression slope
    r_squared: float  # Trend fit quality
    recent_values: List[float]
    timestamps: List[str]

class QualityMonitor:
    """
    Automated quality monitoring system for synthetic data generation.
    
    Provides real-time monitoring, trend analysis, and automated alerts
    for maintaining high-quality synthetic data production.
    """
    
    def __init__(self, 
                 output_dir: str = "quality_monitoring",
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 monitoring_interval: int = 300):  # 5 minutes default
        """
        Initialize the quality monitor.
        
        Parameters:
        -----------
        output_dir : str
            Directory for storing monitoring data and reports
        alert_thresholds : Dict[str, float], optional
            Custom alert thresholds for different metrics
        monitoring_interval : int
            Monitoring interval in seconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "alerts").mkdir(exist_ok=True)
        (self.output_dir / "trends").mkdir(exist_ok=True)
        
        # Initialize monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.quality_queue = Queue()
        self.alert_queue = Queue()
        
        # Set default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'overall_score': 0.5,
            'distribution_similarity': 0.6,
            'moment_preservation': 0.6,
            'spectral_properties': 0.7,
            'seasonality_preservation': 0.7,
            'trend_preservation': 0.6
        }
        
        self.monitoring_interval = monitoring_interval
        
        # Quality history storage
        self.quality_history: List[Dict[str, Any]] = []
        self.alert_history: List[QualityAlert] = []
        
        # Initialize evaluator
        self.evaluator = SyntheticDataQualityEvaluator()
        
        logger.info(f"Quality Monitor initialized at {self.output_dir.absolute()}")
    
    def start_monitoring(self, 
                        data_generator_func: Callable,
                        reference_data: np.ndarray,
                        domain: str = "general",
                        max_history: int = 1000):
        """
        Start continuous quality monitoring.
        
        Parameters:
        -----------
        data_generator_func : Callable
            Function that generates synthetic data for monitoring
        reference_data : np.ndarray
            Reference data for quality comparison
        domain : str
            Data domain for domain-specific evaluation
        max_history : int
            Maximum number of quality evaluations to keep in memory
        """
        if self.is_monitoring:
            logger.warning("Quality monitoring is already running")
            return
        
        self.is_monitoring = True
        self.data_generator_func = data_generator_func
        self.reference_data = reference_data
        self.domain = domain
        self.max_history = max_history
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Quality monitoring started for domain: {domain}")
    
    def stop_monitoring(self):
        """Stop continuous quality monitoring."""
        if not self.is_monitoring:
            logger.warning("Quality monitoring is not running")
            return
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Quality monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        while self.is_monitoring:
            try:
                # Generate synthetic data
                synthetic_data = self.data_generator_func()
                
                # Evaluate quality
                quality_result = self._evaluate_quality(synthetic_data)
                
                # Store in history
                self._store_quality_result(quality_result)
                
                # Check for alerts
                alerts = self._check_alerts(quality_result)
                for alert in alerts:
                    self._process_alert(alert)
                
                # Generate periodic reports
                if len(self.quality_history) % 10 == 0:  # Every 10 evaluations
                    self._generate_monitoring_report()
                
                # Wait for next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _evaluate_quality(self, synthetic_data: np.ndarray) -> QualityEvaluationResult:
        """Evaluate quality of synthetic data."""
        try:
            # Create domain-specific evaluator if available
            if self.domain in ['hydrology', 'financial', 'biomedical', 'climate']:
                evaluator = create_domain_specific_evaluator(self.domain)
            else:
                evaluator = self.evaluator
            
            # Run quality evaluation
            quality_result = evaluator.evaluate_quality(
                synthetic_data=synthetic_data,
                reference_data=self.reference_data,
                reference_metadata={"domain": self.domain, "source": "monitoring_reference"},
                domain=self.domain,
                normalize_for_comparison=True
            )
            
            return quality_result
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            # Return a minimal result indicating failure
            from .synthetic_data_quality import QualityMetricResult, QualityMetricType
            
            failed_metric = QualityMetricResult(
                metric_name="monitoring_evaluation_failed",
                metric_type=QualityMetricType.COMPOSITE,
                value=0.0,
                score=0.0,
                weight=1.0,
                description="Quality evaluation failed during monitoring",
                details={"error": str(e)}
            )
            
            failed_result = QualityEvaluationResult(
                synthetic_data=np.array([]),
                reference_data=np.array([]),
                reference_metadata={},
                metrics=[failed_metric],
                overall_score=0.0,
                quality_level="poor",
                recommendations=[f"Monitoring evaluation failed: {str(e)}"],
                evaluation_date=datetime.now().isoformat(),
                normalization_info={"method": "none", "applied": False}
            )
            
            return failed_result
    
    def _store_quality_result(self, quality_result: QualityEvaluationResult):
        """Store quality evaluation result in history."""
        # Convert to serializable format
        result_dict = {
            'timestamp': quality_result.evaluation_date,
            'overall_score': quality_result.overall_score,
            'quality_level': quality_result.quality_level,
            'metrics': {
                m.metric_name: {
                    'score': m.score,
                    'value': m.value,
                    'weight': m.weight
                }
                for m in quality_result.metrics
            },
            'recommendations': quality_result.recommendations,
            'normalization_info': quality_result.normalization_info
        }
        
        self.quality_history.append(result_dict)
        
        # Limit history size
        if len(self.quality_history) > self.max_history:
            self.quality_history = self.quality_history[-self.max_history:]
        
        # Save to file periodically
        if len(self.quality_history) % 50 == 0:
            self._save_quality_history()
    
    def _check_alerts(self, quality_result: QualityEvaluationResult) -> List[QualityAlert]:
        """Check for quality alerts based on thresholds and trends."""
        alerts = []
        
        # Check overall score threshold
        if quality_result.overall_score < self.alert_thresholds['overall_score']:
            alert = QualityAlert(
                alert_id=f"overall_{int(time.time())}",
                alert_type="threshold",
                metric_name="overall_score",
                threshold=self.alert_thresholds['overall_score'],
                current_value=quality_result.overall_score,
                severity="high" if quality_result.overall_score < 0.3 else "medium",
                message=f"Overall quality score {quality_result.overall_score:.3f} below threshold {self.alert_thresholds['overall_score']}",
                timestamp=datetime.now().isoformat()
            )
            alerts.append(alert)
        
        # Check individual metric thresholds
        for metric in quality_result.metrics:
            metric_name = metric.metric_name
            if metric_name in self.alert_thresholds:
                if metric.score < self.alert_thresholds[metric_name]:
                    alert = QualityAlert(
                        alert_id=f"{metric_name}_{int(time.time())}",
                        alert_type="threshold",
                        metric_name=metric_name,
                        threshold=self.alert_thresholds[metric_name],
                        current_value=metric.score,
                        severity="high" if metric.score < 0.3 else "medium",
                        message=f"{metric_name} score {metric.score:.3f} below threshold {self.alert_thresholds[metric_name]}",
                        timestamp=datetime.now().isoformat()
                    )
                    alerts.append(alert)
        
        # Check for trend-based alerts (if we have enough history)
        if len(self.quality_history) >= 10:
            trend_alerts = self._check_trend_alerts()
            alerts.extend(trend_alerts)
        
        return alerts
    
    def _check_trend_alerts(self) -> List[QualityAlert]:
        """Check for alerts based on quality trends."""
        alerts = []
        
        # Analyze overall score trend
        if len(self.quality_history) >= 10:
            recent_scores = [h['overall_score'] for h in self.quality_history[-10:]]
            trend = self._analyze_trend(recent_scores)
            
            if trend.trend_direction == "declining" and trend.trend_strength > 0.7:
                alert = QualityAlert(
                    alert_id=f"trend_{int(time.time())}",
                    alert_type="trend",
                    metric_name="overall_score",
                    threshold=0.0,  # Not applicable for trends
                    current_value=recent_scores[-1],
                    severity="medium",
                    message=f"Overall quality showing declining trend (strength: {trend.trend_strength:.2f})",
                    timestamp=datetime.now().isoformat()
                )
                alerts.append(alert)
        
        return alerts
    
    def _analyze_trend(self, values: List[float]) -> QualityTrend:
        """Analyze trend in a series of values."""
        if len(values) < 2:
            return QualityTrend(
                metric_name="unknown",
                trend_direction="stable",
                trend_strength=0.0,
                slope=0.0,
                r_squared=0.0,
                recent_values=values,
                timestamps=[]
            )
        
        # Simple linear trend analysis
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Determine trend direction and strength
        if abs(slope) < 0.01:
            trend_direction = "stable"
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = "improving"
            trend_strength = min(abs(slope) * 10, 1.0)  # Scale slope to 0-1
        else:
            trend_direction = "declining"
            trend_strength = min(abs(slope) * 10, 1.0)
        
        return QualityTrend(
            metric_name="unknown",
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            r_squared=r_squared,
            recent_values=values,
            timestamps=[]
        )
    
    def _process_alert(self, alert: QualityAlert):
        """Process and store a quality alert."""
        self.alert_history.append(alert)
        
        # Save alert to file
        alert_file = self.output_dir / "alerts" / f"{alert.alert_id}.json"
        with open(alert_file, 'w') as f:
            json.dump(asdict(alert), f, indent=2)
        
        # Log alert
        logger.warning(f"QUALITY ALERT: {alert.message} (Severity: {alert.severity})")
        
        # Add to queue for external processing
        self.alert_queue.put(alert)
    
    def _generate_monitoring_report(self):
        """Generate periodic monitoring report."""
        if not self.quality_history:
            return
        
        try:
            # Create summary statistics
            recent_history = self.quality_history[-50:]  # Last 50 evaluations
            
            overall_scores = [h['overall_score'] for h in recent_history]
            avg_score = np.mean(overall_scores)
            std_score = np.std(overall_scores)
            min_score = np.min(overall_scores)
            max_score = np.max(overall_scores)
            
            # Quality level distribution
            quality_levels = [h['quality_level'] for h in recent_history]
            level_counts = pd.Series(quality_levels).value_counts()
            
            # Generate report
            report = {
                'report_date': datetime.now().isoformat(),
                'monitoring_period': f"Last {len(recent_history)} evaluations",
                'summary_stats': {
                    'average_quality_score': float(avg_score),
                    'quality_score_std': float(std_score),
                    'min_quality_score': float(min_score),
                    'max_quality_score': float(max_score),
                    'total_evaluations': len(recent_history)
                },
                'quality_level_distribution': level_counts.to_dict(),
                'recent_alerts': len([a for a in self.alert_history if not a.resolved]),
                'trend_analysis': self._analyze_overall_trend()
            }
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / "reports" / f"monitoring_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Monitoring report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate monitoring report: {e}")
    
    def _analyze_overall_trend(self) -> Dict[str, Any]:
        """Analyze overall quality trend."""
        if len(self.quality_history) < 10:
            return {"status": "insufficient_data"}
        
        recent_scores = [h['overall_score'] for h in self.quality_history[-20:]]
        trend = self._analyze_trend(recent_scores)
        
        return {
            'trend_direction': trend.trend_direction,
            'trend_strength': float(trend.trend_strength),
            'slope': float(trend.slope),
            'r_squared': float(trend.r_squared),
            'recent_scores': recent_scores[-5:]  # Last 5 scores
        }
    
    def _save_quality_history(self):
        """Save quality history to file."""
        try:
            history_file = self.output_dir / "logs" / "quality_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.quality_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save quality history: {e}")
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get current quality monitoring summary."""
        if not self.quality_history:
            return {"status": "no_data"}
        
        recent_history = self.quality_history[-10:]  # Last 10 evaluations
        
        return {
            'current_status': 'monitoring' if self.is_monitoring else 'stopped',
            'total_evaluations': len(self.quality_history),
            'recent_average_score': float(np.mean([h['overall_score'] for h in recent_history])),
            'recent_quality_level': recent_history[-1]['quality_level'] if recent_history else 'unknown',
            'active_alerts': len([a for a in self.alert_history if not a.resolved]),
            'last_evaluation': recent_history[-1]['timestamp'] if recent_history else None
        }
    
    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get data for quality dashboard visualization."""
        if not self.quality_history:
            return {"status": "no_data"}
        
        # Prepare time series data
        timestamps = [h['timestamp'] for h in self.quality_history]
        overall_scores = [h['overall_score'] for h in self.quality_history]
        
        # Get metric breakdown for recent evaluations
        recent_history = self.quality_history[-20:]
        metric_scores = {}
        
        if recent_history:
            for metric_name in recent_history[0]['metrics'].keys():
                metric_scores[metric_name] = [
                    h['metrics'][metric_name]['score'] 
                    for h in recent_history 
                    if metric_name in h['metrics']
                ]
        
        return {
            'timestamps': timestamps,
            'overall_scores': overall_scores,
            'metric_scores': metric_scores,
            'quality_levels': [h['quality_level'] for h in self.quality_history],
            'alert_summary': {
                'total_alerts': len(self.alert_history),
                'resolved_alerts': len([a for a in self.alert_history if a.resolved]),
                'active_alerts': len([a for a in self.alert_history if not a.resolved])
            }
        }
