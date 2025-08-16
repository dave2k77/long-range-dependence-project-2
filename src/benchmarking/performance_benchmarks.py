"""
Performance Benchmarking for Long-Range Dependence Estimators

This module provides comprehensive benchmarking tools to compare the performance
of different LRD estimators across various dataset sizes and configurations.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging
import psutil
import os
from dataclasses import dataclass
from pathlib import Path

# Import our estimators
from src.estimators.high_performance import HighPerformanceMFDFAEstimator
from src.estimators.high_performance_dfa import HighPerformanceDFAEstimator
from src.estimators.base import BaseEstimator

# Import quality evaluation system
try:
    from ..validation.synthetic_data_quality import (
        SyntheticDataQualityEvaluator, create_domain_specific_evaluator
    )
except ImportError:
    # Fallback to absolute imports for demo purposes
    from src.validation.synthetic_data_quality import (
        SyntheticDataQualityEvaluator, create_domain_specific_evaluator
    )

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark results with quality metrics."""
    estimator_name: str
    dataset_size: int
    execution_time: float
    memory_peak: float
    memory_final: float
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Optional[Dict[str, Any]] = None
    
    # Quality evaluation results
    quality_score: Optional[float] = None
    quality_level: Optional[str] = None
    quality_metrics: Optional[Dict[str, float]] = None
    quality_recommendations: Optional[List[str]] = None

class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarking for LRD estimators.
    
    This class provides methods to benchmark different estimators across
    various dataset sizes and configurations, measuring execution time,
    memory usage, and accuracy.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
    def generate_synthetic_data(self, size: int, hurst: float = 0.7, 
                                noise_level: float = 0.1) -> np.ndarray:
        """
        Generate synthetic time series with known Hurst exponent.
        
        Args:
            size: Length of the time series
            hurst: Target Hurst exponent
            noise_level: Level of additive noise
            
        Returns:
            Synthetic time series with specified properties
        """
        # Generate fractional Brownian motion using spectral method
        freqs = np.fft.fftfreq(size)
        power_spectrum = np.abs(freqs) ** (-2 * hurst - 1)
        power_spectrum[0] = 0  # Remove DC component
        
        # Generate complex Gaussian noise
        phase = np.random.uniform(0, 2 * np.pi, size)
        amplitude = np.sqrt(power_spectrum) * np.exp(1j * phase)
        
        # Inverse FFT to get time series
        time_series = np.real(np.fft.ifft(amplitude))
        
        # Add noise
        noise = np.random.normal(0, noise_level, size)
        time_series += noise
        
        # Normalize
        time_series = (time_series - np.mean(time_series)) / np.std(time_series)
        
        return time_series
    
    def measure_memory_usage(self) -> Tuple[float, float]:
        """
        Measure current memory usage.
        
        Returns:
            Tuple of (peak_memory_mb, current_memory_mb)
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        current_mb = memory_info.rss / 1024 / 1024
        
        # Get peak memory if available
        try:
            peak_mb = process.memory_info().peak_wset / 1024 / 1024
        except AttributeError:
            peak_mb = current_mb
            
        return peak_mb, current_mb
    
    def _evaluate_data_quality(self, synthetic_data: np.ndarray, 
                              reference_data: np.ndarray, domain: str):
        """
        Evaluate synthetic data quality using our quality evaluation system.
        
        Parameters:
        -----------
        synthetic_data : np.ndarray
            Synthetic data to evaluate
        reference_data : np.ndarray
            Reference data for comparison
        domain : str
            Data domain for domain-specific evaluation
            
        Returns:
        --------
        QualityEvaluationResult
            Quality evaluation results
        """
        try:
            # Create appropriate evaluator
            if domain in ['hydrology', 'financial', 'biomedical', 'climate']:
                evaluator = create_domain_specific_evaluator(domain)
            else:
                evaluator = SyntheticDataQualityEvaluator()
            
            # Run quality evaluation
            quality_result = evaluator.evaluate_quality(
                synthetic_data=synthetic_data,
                reference_data=reference_data,
                reference_metadata={"domain": domain, "source": "benchmark_reference"},
                domain=domain,
                normalize_for_comparison=True
            )
            
            return quality_result
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            raise
    
    def benchmark_estimator(self, estimator_class: type, estimator_name: str,
                           data: np.ndarray, reference_data: Optional[np.ndarray] = None,
                           domain: str = "general", **kwargs) -> BenchmarkResult:
        """
        Benchmark a single estimator on given data with quality evaluation.
        
        Args:
            estimator_class: Class of the estimator to benchmark
            estimator_name: Name identifier for the estimator
            data: Input data for estimation
            reference_data: Reference data for quality evaluation
            domain: Data domain for domain-specific quality evaluation
            **kwargs: Additional arguments for estimator initialization
            
        Returns:
            BenchmarkResult with performance and quality metrics
        """
        dataset_size = len(data)
        
        # Measure initial memory
        initial_peak, initial_memory = self.measure_memory_usage()
        
        try:
            # Initialize estimator
            start_time = time.time()
            estimator = estimator_class(**kwargs)
            
            # Perform estimation
            result = estimator.estimate(data)
            
            # Measure execution time
            execution_time = time.time() - start_time
            
            # Measure final memory
            final_peak, final_memory = self.measure_memory_usage()
            
            # Calculate memory usage
            memory_peak = final_peak - initial_peak
            memory_final = final_memory - initial_memory
            
            # Additional metrics
            additional_metrics = {}
            if hasattr(estimator, 'get_execution_time'):
                additional_metrics['estimator_execution_time'] = estimator.get_execution_time()
            if hasattr(estimator, 'get_memory_usage'):
                additional_metrics['estimator_memory_usage'] = estimator.get_memory_usage()
            
            # QUALITY EVALUATION: Evaluate synthetic data quality if reference data provided
            quality_score = None
            quality_level = None
            quality_metrics = None
            quality_recommendations = None
            
            if reference_data is not None:
                try:
                    quality_result = self._evaluate_data_quality(data, reference_data, domain)
                    quality_score = quality_result.overall_score
                    quality_level = quality_result.quality_level
                    quality_metrics = {m.metric_name: m.score for m in quality_result.metrics}
                    quality_recommendations = quality_result.recommendations
                    
                    # Add quality metrics to additional metrics
                    additional_metrics['quality_score'] = quality_score
                    additional_metrics['quality_level'] = quality_level
                    additional_metrics['quality_metrics'] = quality_metrics
                    
                except Exception as e:
                    logger.warning(f"Quality evaluation failed for {estimator_name}: {e}")
                    additional_metrics['quality_evaluation_error'] = str(e)
            
            return BenchmarkResult(
                estimator_name=estimator_name,
                dataset_size=dataset_size,
                execution_time=execution_time,
                memory_peak=memory_peak,
                memory_final=memory_final,
                success=True,
                additional_metrics=additional_metrics,
                quality_score=quality_score,
                quality_level=quality_level,
                quality_metrics=quality_metrics,
                quality_recommendations=quality_recommendations
            )
            
        except Exception as e:
            logger.warning(f"Benchmark failed for {estimator_name} on dataset size {dataset_size}: {e}")
            return BenchmarkResult(
                estimator_name=estimator_name,
                dataset_size=dataset_size,
                execution_time=0.0,
                memory_peak=0.0,
                memory_final=0.0,
                success=False,
                error_message=str(e)
            )
    
    def run_comprehensive_benchmark(self, dataset_sizes: List[int] = None,
                                   num_runs: int = 3, include_quality: bool = True,
                                   reference_data: Optional[np.ndarray] = None,
                                   domain: str = "general") -> pd.DataFrame:
        """
        Run comprehensive benchmarking across multiple estimators and dataset sizes.
        
        Args:
            dataset_sizes: List of dataset sizes to test
            num_runs: Number of runs per configuration for averaging
            include_quality: Whether to include quality evaluation
            reference_data: Reference data for quality evaluation
            domain: Data domain for quality evaluation
            
        Returns:
            DataFrame with benchmark results including quality metrics
        """
        if dataset_sizes is None:
            dataset_sizes = [100, 500, 1000, 2000, 5000]
        
        estimators = [
            (HighPerformanceMFDFAEstimator, "HighPerformanceMFDFA"),
            (HighPerformanceDFAEstimator, "HighPerformanceDFA"),
        ]
        
        logger.info(f"Starting comprehensive benchmark with {len(estimators)} estimators")
        logger.info(f"Testing dataset sizes: {dataset_sizes}")
        logger.info(f"Running {num_runs} iterations per configuration")
        if include_quality:
            logger.info(f"Including quality evaluation for domain: {domain}")
        
        all_results = []
        
        for size in dataset_sizes:
            logger.info(f"Testing dataset size: {size}")
            
            for estimator_class, estimator_name in estimators:
                logger.info(f"  Benchmarking {estimator_name}")
                
                for run in range(num_runs):
                    # Generate fresh data for each run
                    data = self.generate_synthetic_data(size)
                    
                    # Run benchmark with quality evaluation if requested
                    if include_quality and reference_data is not None:
                        result = self.benchmark_estimator(
                            estimator_class, 
                            estimator_name, 
                            data,
                            reference_data=reference_data,
                            domain=domain
                        )
                    else:
                        result = self.benchmark_estimator(
                            estimator_class, 
                            estimator_name, 
                            data
                        )
                    
                    # Add run information
                    result.additional_metrics = result.additional_metrics or {}
                    result.additional_metrics['run_number'] = run
                    
                    all_results.append(result)
                    
                    # Small delay to ensure clean memory measurement
                    time.sleep(0.1)
        
        # Convert to DataFrame
        df = self.results_to_dataframe(all_results)
        
        # Save results
        self.save_results(df)
        
        return df
    
    def results_to_dataframe(self, results: List[BenchmarkResult]) -> pd.DataFrame:
        """Convert benchmark results to a pandas DataFrame with quality metrics."""
        data = []
        
        for result in results:
            row = {
                'estimator': result.estimator_name,
                'dataset_size': result.dataset_size,
                'execution_time': result.execution_time,
                'memory_peak_mb': result.memory_peak,
                'memory_final_mb': result.memory_final,
                'success': result.success,
                'error_message': result.error_message,
            }
            
            # Add quality metrics
            if result.quality_score is not None:
                row['quality_score'] = result.quality_score
                row['quality_level'] = result.quality_level
                
                # Add individual quality metrics
                if result.quality_metrics:
                    for metric_name, metric_score in result.quality_metrics.items():
                        row[f'quality_{metric_name}'] = metric_score
                
                # Add quality recommendations
                if result.quality_recommendations:
                    row['quality_recommendations'] = '; '.join(result.quality_recommendations[:3])  # Limit to first 3
            
            # Add additional metrics
            if result.additional_metrics:
                for key, value in result.additional_metrics.items():
                    row[f'metric_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_results(self, df: pd.DataFrame, filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Benchmark results saved to: {filepath}")
        
        # Also save as Excel for better formatting
        excel_filepath = filepath.with_suffix('.xlsx')
        df.to_excel(excel_filepath, index=False)
        logger.info(f"Benchmark results also saved to: {excel_filepath}")
    
    def generate_performance_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive performance report."""
        # Filter successful runs
        successful = df[df['success'] == True].copy()
        
        if successful.empty:
            return "No successful benchmark runs to analyze."
        
        # Calculate summary statistics
        summary = successful.groupby(['estimator', 'dataset_size']).agg({
            'execution_time': ['mean', 'std', 'min', 'max'],
            'memory_peak_mb': ['mean', 'std'],
            'memory_final_mb': ['mean', 'std']
        }).round(4)
        
        # Calculate speedup ratios
        speedup_analysis = self._calculate_speedup_analysis(successful)
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("LONG-RANGE DEPENDENCE ESTIMATORS PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 40)
        report.append(summary.to_string())
        report.append("")
        
        report.append("PERFORMANCE ANALYSIS:")
        report.append("-" * 40)
        report.append(speedup_analysis)
        report.append("")
        
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        report.extend(self._generate_recommendations(successful))
        
        return "\n".join(report)
    
    def _calculate_speedup_analysis(self, df: pd.DataFrame) -> str:
        """Calculate and format speedup analysis."""
        analysis = []
        
        # Group by dataset size
        for size in df['dataset_size'].unique():
            size_data = df[df['dataset_size'] == size]
            if len(size_data) < 2:
                continue
                
            analysis.append(f"Dataset Size: {size}")
            
            # Find baseline (fastest estimator)
            baseline = size_data.loc[size_data['execution_time'].idxmin()]
            baseline_name = baseline['estimator']
            baseline_time = baseline['execution_time']
            
            analysis.append(f"  Baseline: {baseline_name} ({baseline_time:.4f}s)")
            
            # Calculate speedup for other estimators
            for _, row in size_data.iterrows():
                if row['estimator'] != baseline_name:
                    speedup = baseline_time / row['execution_time']
                    analysis.append(f"  {row['estimator']}: {speedup:.2f}x {'slower' if speedup < 1 else 'faster'}")
            
            analysis.append("")
        
        return "\n".join(analysis)
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate performance recommendations based on benchmark results."""
        recommendations = []
        
        # Analyze execution time trends
        time_trends = df.groupby(['estimator', 'dataset_size'])['execution_time'].mean().unstack()
        
        # Check for scalability issues
        for estimator in time_trends.index:
            times = time_trends.loc[estimator].dropna()
            if len(times) > 1:
                # Calculate scaling factor
                sizes = times.index
                scaling_factor = np.polyfit(np.log(sizes), np.log(times), 1)[0]
                
                if scaling_factor > 2:
                    recommendations.append(f"⚠️  {estimator} shows poor scalability (O(n^{scaling_factor:.1f}))")
                elif scaling_factor < 1.5:
                    recommendations.append(f"✅ {estimator} shows good scalability (O(n^{scaling_factor:.1f}))")
        
        # Memory usage recommendations
        memory_analysis = df.groupby('estimator')['memory_peak_mb'].mean()
        memory_threshold = 100  # MB
        
        for estimator, avg_memory in memory_analysis.items():
            if avg_memory > memory_threshold:
                recommendations.append(f"⚠️  {estimator} uses high memory ({avg_memory:.1f} MB average)")
            else:
                recommendations.append(f"✅ {estimator} uses reasonable memory ({avg_memory:.1f} MB average)")
        
        # General recommendations
        recommendations.append("")
        recommendations.append("GENERAL RECOMMENDATIONS:")
        recommendations.append("• Use smaller datasets for quick prototyping")
        recommendations.append("• Consider memory constraints for large datasets")
        recommendations.append("• Profile specific bottlenecks for optimization")
        
        return recommendations
    
    def plot_performance_comparison(self, df: pd.DataFrame, save_plots: bool = True):
        """Generate performance comparison plots."""
        successful = df[df['success'] == True].copy()
        
        if successful.empty:
            logger.warning("No successful runs to plot")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. Execution Time vs Dataset Size
        ax1 = axes[0, 0]
        for estimator in successful['estimator'].unique():
            data = successful[successful['estimator'] == estimator]
            times = data.groupby('dataset_size')['execution_time'].mean()
            ax1.plot(times.index, times.values, 'o-', label=estimator, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time vs Dataset Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Memory Usage vs Dataset Size
        ax2 = axes[0, 1]
        for estimator in successful['estimator'].unique():
            data = successful[successful['estimator'] == estimator]
            memory = data.groupby('dataset_size')['memory_peak_mb'].mean()
            ax2.plot(memory.index, memory.values, 's-', label=estimator, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel('Peak Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Dataset Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Speedup Comparison
        ax3 = axes[1, 0]
        # Calculate relative performance
        for size in successful['dataset_size'].unique():
            size_data = successful[successful['dataset_size'] == size]
            baseline_time = size_data['execution_time'].min()
            
            for _, row in size_data.iterrows():
                speedup = baseline_time / row['execution_time']
                ax3.bar(f"{row['estimator']}\n(size={size})", speedup, alpha=0.7)
        
        ax3.set_ylabel('Relative Performance (1.0 = fastest)')
        ax3.set_title('Relative Performance Comparison')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Success Rate
        ax4 = axes[1, 1]
        success_rates = df.groupby('estimator')['success'].agg(['sum', 'count'])
        success_rates['rate'] = success_rates['sum'] / success_rates['count'] * 100
        
        bars = ax4.bar(success_rates.index, success_rates['rate'], alpha=0.7, color='skyblue')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('Estimator Success Rate')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates['rate']):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = self.output_dir / f"performance_plots_{time.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to: {plot_file}")
        
        plt.show()
        
        return fig


def run_quick_benchmark():
    """Run a quick benchmark for demonstration purposes."""
    print("🚀 Starting Quick Performance Benchmark...")
    
    benchmarker = PerformanceBenchmarker()
    
    # Test with smaller dataset sizes for quick results
    dataset_sizes = [100, 500, 1000]
    
    print(f"📊 Testing dataset sizes: {dataset_sizes}")
    print(f"⏱️  Running 2 iterations per configuration...")
    
    # Run benchmark
    results_df = benchmarker.run_comprehensive_benchmark(
        dataset_sizes=dataset_sizes,
        num_runs=2
    )
    
    # Generate report
    report = benchmarker.generate_performance_report(results_df)
    print("\n" + "="*80)
    print("📋 PERFORMANCE REPORT")
    print("="*80)
    print(report)
    
    # Generate plots
    print("\n📈 Generating performance plots...")
    benchmarker.plot_performance_comparison(results_df)
    
    print(f"\n✅ Benchmark completed! Results saved to: {benchmarker.output_dir}")
    
    return results_df


if __name__ == "__main__":
    # Run quick benchmark when script is executed directly
    results = run_quick_benchmark()
