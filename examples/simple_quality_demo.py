#!/usr/bin/env python3
"""
Simple Quality System Demo

This script demonstrates the core quality evaluation functionality
without complex import dependencies.
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_points: int, hurst: float = 0.7) -> np.ndarray:
    """Generate synthetic time series with known Hurst exponent."""
    # Generate fractional Brownian motion using spectral method
    freqs = np.fft.fftfreq(n_points)
    power_spectrum = np.abs(freqs) ** (-2 * hurst - 1)
    power_spectrum[0] = 0  # Remove DC component
    
    # Generate complex Gaussian noise
    phase = np.random.uniform(0, 2 * np.pi, n_points)
    amplitude = np.sqrt(power_spectrum) * np.exp(1j * phase)
    
    # Inverse FFT to get time series
    time_series = np.real(np.fft.ifft(amplitude))
    
    # Normalize
    time_series = (time_series - np.mean(time_series)) / np.std(time_series)
    
    return time_series

def generate_reference_data(domain: str, n_points: int = 1000) -> np.ndarray:
    """Generate reference data for different domains."""
    if domain == "hydrology":
        # Hydrological data: seasonal patterns with long-range dependence
        t = np.linspace(0, 10, n_points)
        seasonal = 10 * np.sin(2 * np.pi * t) + 5 * np.sin(4 * np.pi * t)
        trend = 0.1 * t
        noise = np.random.normal(0, 1, n_points)
        # Add some persistence
        for i in range(1, n_points):
            noise[i] = 0.7 * noise[i-1] + 0.3 * noise[i]
        return seasonal + trend + noise
    
    elif domain == "financial":
        # Financial data: random walk with volatility clustering
        returns = np.random.normal(0, 0.02, n_points)
        # Add volatility clustering
        volatility = np.ones(n_points)
        for i in range(1, n_points):
            volatility[i] = 0.95 * volatility[i-1] + 0.05 * returns[i-1]**2
        returns = returns * np.sqrt(volatility)
        return np.cumsum(returns)
    
    elif domain == "biomedical":
        # Biomedical data: oscillatory with noise
        t = np.linspace(0, 4*np.pi, n_points)
        signal = 5 * np.sin(t) + 2 * np.sin(3*t) + 1.5 * np.sin(5*t)
        noise = np.random.normal(0, 0.5, n_points)
        return signal + noise
    
    else:  # climate
        # Climate data: trend with seasonal and long-term cycles
        t = np.linspace(0, 20, n_points)
        seasonal = 3 * np.sin(2 * np.pi * t) + 1.5 * np.sin(4 * np.pi * t)
        trend = 0.05 * t
        long_cycle = 2 * np.sin(2 * np.pi * t / 10)
        noise = np.random.normal(0, 0.3, n_points)
        return seasonal + trend + long_cycle + noise

def calculate_basic_quality_metrics(synthetic_data: np.ndarray, reference_data: np.ndarray) -> dict:
    """Calculate basic quality metrics between synthetic and reference data."""
    metrics = {}
    
    # Ensure same length for comparison
    min_length = min(len(synthetic_data), len(reference_data))
    if len(synthetic_data) != len(reference_data):
        # Resample reference data to match synthetic data length
        ref_resampled = np.interp(
            np.linspace(0, 1, len(synthetic_data)),
            np.linspace(0, 1, len(reference_data)),
            reference_data
        )
        reference_data = ref_resampled
    
    # Normalize both datasets for fair comparison
    synthetic_norm = (synthetic_data - np.mean(synthetic_data)) / np.std(synthetic_data)
    reference_norm = (reference_data - np.mean(reference_data)) / np.std(reference_data)
    
    # 1. Distribution similarity (histogram overlap)
    try:
        hist_synthetic, _ = np.histogram(synthetic_norm, bins=20, density=True)
        hist_reference, _ = np.histogram(reference_norm, bins=20, density=True)
        
        # Calculate histogram similarity (correlation)
        correlation = np.corrcoef(hist_synthetic, hist_reference)[0, 1]
        metrics['distribution_similarity'] = max(0, correlation) if not np.isnan(correlation) else 0.0
    except:
        metrics['distribution_similarity'] = 0.0
    
    # 2. Moment preservation
    try:
        # Compare means and standard deviations
        mean_diff = abs(np.mean(synthetic_norm) - np.mean(reference_norm))
        std_diff = abs(np.std(synthetic_norm) - np.std(reference_norm))
        
        # Convert to similarity scores (0-1, higher is better)
        mean_similarity = 1.0 / (1.0 + mean_diff)
        std_similarity = 1.0 / (1.0 + std_diff)
        
        metrics['moment_preservation'] = (mean_similarity + std_similarity) / 2
    except:
        metrics['moment_preservation'] = 0.0
    
    # 3. Spectral properties
    try:
        # Calculate power spectral density
        freqs_synthetic = np.fft.fftfreq(len(synthetic_norm))
        psd_synthetic = np.abs(np.fft.fft(synthetic_norm)) ** 2
        
        freqs_reference = np.fft.fftfreq(len(reference_norm))
        psd_reference = np.abs(np.fft.fft(reference_norm)) ** 2
        
        # Remove DC component and negative frequencies
        positive_mask = freqs_synthetic > 0
        freqs_synthetic = freqs_synthetic[positive_mask]
        psd_synthetic = psd_synthetic[positive_mask]
        
        positive_mask = freqs_reference > 0
        freqs_reference = freqs_reference[positive_mask]
        psd_reference = psd_reference[positive_mask]
        
        # Resample to same frequency grid
        if len(freqs_synthetic) != len(freqs_reference):
            psd_reference = np.interp(freqs_synthetic, freqs_reference, psd_reference)
        
        # Calculate spectral similarity
        spectral_correlation = np.corrcoef(psd_synthetic, psd_reference)[0, 1]
        metrics['spectral_properties'] = max(0, spectral_correlation) if not np.isnan(spectral_correlation) else 0.0
    except:
        metrics['spectral_properties'] = 0.0
    
    # 4. Temporal properties
    try:
        # Autocorrelation similarity
        def calculate_autocorr(data, max_lag=10):
            autocorr = []
            for lag in range(1, max_lag + 1):
                if lag < len(data):
                    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0.0)
            return np.array(autocorr)
        
        autocorr_synthetic = calculate_autocorr(synthetic_norm)
        autocorr_reference = calculate_autocorr(reference_norm)
        
        if len(autocorr_synthetic) > 0 and len(autocorr_reference) > 0:
            # Ensure same length
            min_lag = min(len(autocorr_synthetic), len(autocorr_reference))
            autocorr_similarity = np.corrcoef(
                autocorr_synthetic[:min_lag], 
                autocorr_reference[:min_lag]
            )[0, 1]
            metrics['temporal_properties'] = max(0, autocorr_similarity) if not np.isnan(autocorr_similarity) else 0.0
        else:
            metrics['temporal_properties'] = 0.0
    except:
        metrics['temporal_properties'] = 0.0
    
    # 5. Overall quality score (weighted average)
    weights = {
        'distribution_similarity': 0.25,
        'moment_preservation': 0.20,
        'spectral_properties': 0.30,
        'temporal_properties': 0.25
    }
    
    overall_score = sum(metrics[metric] * weight for metric, weight in weights.items())
    metrics['overall_score'] = overall_score
    
    # 6. Quality level classification
    if overall_score >= 0.8:
        quality_level = "excellent"
    elif overall_score >= 0.6:
        quality_level = "good"
    elif overall_score >= 0.4:
        quality_level = "acceptable"
    else:
        quality_level = "poor"
    
    metrics['quality_level'] = quality_level
    
    return metrics

def demonstrate_quality_gates():
    """Demonstrate quality gates concept."""
    logger.info("📋 Demonstrating Quality Gates in Data Submission")
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(1000, hurst=0.7)
    reference_data = generate_reference_data("hydrology", 1000)
    
    # Calculate quality metrics
    quality_metrics = calculate_basic_quality_metrics(synthetic_data, reference_data)
    
    logger.info(f"Quality evaluation result: {quality_metrics['overall_score']:.3f} ({quality_metrics['quality_level']})")
    
    # Check if quality meets threshold (quality gate)
    quality_threshold = 0.5
    if quality_metrics['overall_score'] >= quality_threshold:
        logger.info("✅ Quality gate PASSED - Dataset accepted")
        gate_status = "PASSED"
    else:
        logger.warning("❌ Quality gate FAILED - Dataset rejected")
        gate_status = "FAILED"
    
    # Save quality gate results
    gate_results = {
        'submission_id': 'demo_001',
        'quality_score': quality_metrics['overall_score'],
        'quality_level': quality_metrics['quality_level'],
        'gate_passed': quality_metrics['overall_score'] >= quality_threshold,
        'gate_status': gate_status,
        'threshold': quality_threshold,
        'timestamp': datetime.now().isoformat(),
        'detailed_metrics': quality_metrics
    }
    
    return gate_results

def demonstrate_benchmarking_integration():
    """Demonstrate benchmarking with quality metrics."""
    logger.info("📊 Demonstrating Benchmarking Integration with Quality Metrics")
    
    # Test different dataset sizes
    dataset_sizes = [100, 500, 1000]
    benchmark_results = []
    
    for size in dataset_sizes:
        logger.info(f"Benchmarking dataset size: {size}")
        
        # Generate test data
        synthetic_data = generate_synthetic_data(size, hurst=0.7)
        reference_data = generate_reference_data("hydrology", size)
        
        # Measure execution time
        start_time = datetime.now()
        quality_metrics = calculate_basic_quality_metrics(synthetic_data, reference_data)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Store benchmark result
        result = {
            'dataset_size': size,
            'execution_time': execution_time,
            'quality_score': quality_metrics['overall_score'],
            'quality_level': quality_metrics['quality_level'],
            'memory_usage_mb': len(synthetic_data) * 8 / 1024 / 1024  # Approximate memory usage
        }
        
        benchmark_results.append(result)
        logger.info(f"  Size {size}: Quality = {quality_metrics['overall_score']:.3f}, Time = {execution_time:.4f}s")
    
    return benchmark_results

def demonstrate_quality_monitoring():
    """Demonstrate automated quality monitoring concept."""
    logger.info("🔍 Demonstrating Automated Quality Monitoring")
    
    # Simulate continuous monitoring
    monitoring_results = []
    n_evaluations = 10
    
    for i in range(n_evaluations):
        # Generate data with varying quality (simulate real-world variation)
        quality_factor = 0.6 + 0.3 * np.sin(i * 0.5)  # Varying quality over time
        
        # Generate synthetic data
        synthetic_data = generate_synthetic_data(500, hurst=0.7)
        reference_data = generate_reference_data("hydrology", 500)
        
        # Add quality variation
        synthetic_data = synthetic_data * quality_factor
        
        # Calculate quality
        quality_metrics = calculate_basic_quality_metrics(synthetic_data, reference_data)
        
        monitoring_results.append({
            'evaluation_id': i + 1,
            'timestamp': datetime.now().isoformat(),
            'quality_score': quality_metrics['overall_score'],
            'quality_level': quality_metrics['quality_level'],
            'quality_factor': quality_factor
        })
        
        logger.info(f"  Evaluation {i+1}: Quality = {quality_metrics['overall_score']:.3f} ({quality_metrics['quality_level']})")
    
    # Analyze trends
    quality_scores = [r['quality_score'] for r in monitoring_results]
    avg_quality = np.mean(quality_scores)
    quality_trend = "improving" if quality_scores[-1] > quality_scores[0] else "declining"
    
    monitoring_summary = {
        'total_evaluations': n_evaluations,
        'average_quality': avg_quality,
        'quality_trend': quality_trend,
        'recent_quality': quality_scores[-1],
        'quality_variability': np.std(quality_scores)
    }
    
    logger.info(f"Monitoring summary: Average quality = {avg_quality:.3f}, Trend = {quality_trend}")
    
    return monitoring_results, monitoring_summary

def demonstrate_advanced_metrics():
    """Demonstrate advanced quality metrics concept."""
    logger.info("🧠 Demonstrating Advanced Quality Metrics")
    
    # Generate test data
    synthetic_data = generate_synthetic_data(1000, hurst=0.7)
    reference_data = generate_reference_data("hydrology", 1000)
    
    # Calculate basic metrics first
    basic_metrics = calculate_basic_quality_metrics(synthetic_data, reference_data)
    
    # Advanced LRD-specific metrics
    advanced_metrics = {}
    
    # 1. Hurst exponent consistency
    try:
        def estimate_hurst(data):
            if len(data) < 10:
                return 0.5
            
            # Simple R/S analysis
            lags = np.logspace(1, np.log10(len(data)//4), 5, dtype=int)
            rs_values = []
            
            for lag in lags:
                if lag < 2:
                    continue
                
                segments = len(data) // lag
                if segments < 2:
                    continue
                
                rs_segments = []
                for i in range(segments):
                    segment = data[i*lag:(i+1)*lag]
                    if len(segment) < 2:
                        continue
                    
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
            log_lags = np.log(lags)
            log_rs = np.log(rs_values)
            
            coeffs = np.polyfit(log_lags, log_rs, 1)
            hurst = coeffs[0]
            
            return max(0.0, min(1.0, hurst))
        
        hurst_synthetic = estimate_hurst(synthetic_data)
        hurst_reference = estimate_hurst(reference_data)
        
        hurst_diff = abs(hurst_synthetic - hurst_reference)
        hurst_score = 1.0 / (1.0 + hurst_diff)
        
        advanced_metrics['hurst_exponent_consistency'] = {
            'synthetic_hurst': hurst_synthetic,
            'reference_hurst': hurst_reference,
            'difference': hurst_diff,
            'score': hurst_score
        }
        
    except Exception as e:
        logger.warning(f"Hurst estimation failed: {e}")
        advanced_metrics['hurst_exponent_consistency'] = {'score': 0.0}
    
    # 2. Power law scaling consistency
    try:
        def calculate_psd_scaling(data):
            if len(data) < 10:
                return 0.0, 0.0
            
            freqs = np.fft.fftfreq(len(data))
            psd = np.abs(np.fft.fft(data)) ** 2
            
            positive_freqs = freqs > 0
            freqs = freqs[positive_freqs]
            psd = psd[positive_freqs]
            
            if len(freqs) < 2:
                return 0.0, 0.0
            
            log_freqs = np.log(freqs)
            log_psd = np.log(psd)
            
            coeffs = np.polyfit(log_freqs, log_psd, 1)
            slope = coeffs[0]
            
            return slope, 0.8  # Simplified R-squared
        
        slope_synthetic, _ = calculate_psd_scaling(synthetic_data)
        slope_reference, _ = calculate_psd_scaling(reference_data)
        
        slope_diff = abs(slope_synthetic - slope_reference)
        slope_score = 1.0 / (1.0 + slope_diff)
        
        advanced_metrics['power_law_scaling_consistency'] = {
            'synthetic_slope': slope_synthetic,
            'reference_slope': slope_reference,
            'difference': slope_diff,
            'score': slope_score
        }
        
    except Exception as e:
        logger.warning(f"Power law scaling calculation failed: {e}")
        advanced_metrics['power_law_scaling_consistency'] = {'score': 0.0}
    
    # Overall advanced quality score
    advanced_scores = [m['score'] for m in advanced_metrics.values() if 'score' in m]
    if advanced_scores:
        avg_advanced_score = np.mean(advanced_scores)
        # Combine with basic metrics
        overall_advanced_score = 0.7 * basic_metrics['overall_score'] + 0.3 * avg_advanced_score
    else:
        overall_advanced_score = basic_metrics['overall_score']
    
    advanced_metrics['overall_advanced_score'] = overall_advanced_score
    
    logger.info(f"Advanced metrics calculated: {len(advanced_metrics)} metrics")
    logger.info(f"Overall advanced score: {overall_advanced_score:.3f}")
    
    return advanced_metrics

def main():
    """Main function to run the simple quality demo."""
    print("🚀 Simple Quality System Demo")
    print("=" * 40)
    print("This demo showcases the core quality evaluation concepts:")
    print("1. Quality Gates in Data Submission")
    print("2. Benchmarking Integration with Quality Metrics")
    print("3. Automated Quality Monitoring")
    print("4. Advanced Quality Metrics")
    print("=" * 40)
    
    try:
        # Create output directory
        output_dir = Path("simple_quality_demo")
        output_dir.mkdir(exist_ok=True)
        
        all_results = {}
        
        # Phase 1: Quality Gates
        logger.info("\n" + "="*50)
        gate_results = demonstrate_quality_gates()
        all_results['quality_gates'] = gate_results
        
        # Phase 2: Benchmarking Integration
        logger.info("\n" + "="*50)
        benchmark_results = demonstrate_benchmarking_integration()
        all_results['benchmarking'] = benchmark_results
        
        # Phase 3: Quality Monitoring
        logger.info("\n" + "="*50)
        monitoring_results, monitoring_summary = demonstrate_quality_monitoring()
        all_results['monitoring'] = {
            'results': monitoring_results,
            'summary': monitoring_summary
        }
        
        # Phase 4: Advanced Metrics
        logger.info("\n" + "="*50)
        advanced_metrics = demonstrate_advanced_metrics()
        all_results['advanced_metrics'] = advanced_metrics
        
        # Save all results
        results_file = output_dir / "demo_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Create summary
        summary_file = output_dir / "demo_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("SIMPLE QUALITY SYSTEM DEMO SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Demo completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Results Summary:\n")
            f.write(f"- Quality Gates: {gate_results['gate_status']} (Score: {gate_results['quality_score']:.3f})\n")
            f.write(f"- Benchmarking: {len(benchmark_results)} datasets tested\n")
            f.write(f"- Monitoring: {monitoring_summary['total_evaluations']} evaluations\n")
            f.write(f"- Advanced Metrics: {len(advanced_metrics)} metrics calculated\n")
            f.write(f"\nOutput directory: {output_dir.absolute()}\n")
        
        print("\n✅ Demo completed successfully!")
        print(f"Check the output directory: {output_dir.absolute()}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
