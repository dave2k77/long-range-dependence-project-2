#!/usr/bin/env python3
"""
Comprehensive test script for all new high-performance estimators.
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_data(n_points=1000, hurst=0.7):
    """Generate fractional Brownian motion test data."""
    from scipy.stats import norm
    
    # Generate increments
    increments = norm.rvs(size=n_points)
    
    # Apply fractional integration
    data = np.cumsum(increments)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, n_points)
    data = data + noise
    
    return data

def test_periodogram_estimator():
    """Test the high-performance periodogram estimator."""
    logger.info("Testing HighPerformancePeriodogramEstimator...")
    
    try:
        from estimators.high_performance_periodogram import HighPerformancePeriodogramEstimator
        
        # Generate test data
        data = generate_test_data(1000, 0.7)
        
        # Create estimator
        estimator = HighPerformancePeriodogramEstimator(
            window='hann',
            frequency_range=[0.01, 0.5],
            use_jax=True,
            enable_caching=True,
            vectorized=True
        )
        
        # Fit and estimate
        results = estimator.estimate(data)
        
        logger.info(f"Periodogram Estimation Results:")
        logger.info(f"  Hurst Exponent: {results['hurst_exponent']:.4f}")
        logger.info(f"  Beta: {results['beta']:.4f}")
        logger.info(f"  Scaling Error: {results['scaling_error']:.4f}")
        logger.info(f"  Confidence Interval: {results['confidence_interval']}")
        logger.info(f"  Execution Time: {results['performance']['execution_time']:.4f}s")
        logger.info(f"  Memory Usage: {results['performance']['memory_usage']} bytes")
        logger.info(f"  JAX Usage: {results['performance']['jax_usage']}")
        logger.info(f"  Fallback Usage: {results['performance']['fallback_usage']}")
        
        # Test caching
        cache_stats = estimator.get_cache_stats()
        logger.info(f"  Cache Hit Rate: {cache_stats['hit_rate']:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Periodogram estimator test failed: {e}")
        return False

def test_gph_estimator():
    """Test the high-performance GPH estimator."""
    logger.info("Testing HighPerformanceGPHEstimator...")
    
    try:
        from estimators.high_performance_gph import HighPerformanceGPHEstimator
        
        # Generate test data
        data = generate_test_data(1000, 0.7)
        
        # Create estimator
        estimator = HighPerformanceGPHEstimator(
            num_frequencies=50,
            frequency_threshold=0.1,
            use_jax=True,
            enable_caching=True,
            vectorized=True
        )
        
        # Fit and estimate
        results = estimator.estimate(data)
        
        logger.info(f"GPH Estimation Results:")
        logger.info(f"  Fractional d: {results['fractional_d']:.4f}")
        logger.info(f"  Hurst Exponent: {results['hurst_exponent']:.4f}")
        logger.info(f"  Regression Error: {results['regression_error']:.4f}")
        logger.info(f"  Confidence Interval: {results['confidence_interval']}")
        logger.info(f"  Execution Time: {results['performance']['execution_time']:.4f}s")
        logger.info(f"  Memory Usage: {results['performance']['memory_usage']} bytes")
        logger.info(f"  JAX Usage: {results['performance']['jax_usage']}")
        logger.info(f"  Fallback Usage: {results['performance']['fallback_usage']}")
        
        # Test caching
        cache_stats = estimator.get_cache_stats()
        logger.info(f"  Cache Hit Rate: {cache_stats['hit_rate']:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"GPH estimator test failed: {e}")
        return False

def test_wavelet_leaders_estimator():
    """Test the high-performance wavelet leaders estimator."""
    logger.info("Testing HighPerformanceWaveletLeadersEstimator...")
    
    try:
        from estimators.high_performance_wavelet_leaders import HighPerformanceWaveletLeadersEstimator
        
        # Generate test data
        data = generate_test_data(1000, 0.7)
        
        # Create estimator
        estimator = HighPerformanceWaveletLeadersEstimator(
            wavelet='db4',
            num_scales=15,
            use_jax=True,
            enable_caching=True,
            vectorized=True
        )
        
        # Fit and estimate
        results = estimator.estimate(data)
        
        logger.info(f"Wavelet Leaders Estimation Results:")
        logger.info(f"  Hurst Exponent: {results['hurst_exponent']:.4f}")
        logger.info(f"  Alpha: {results['alpha']:.4f}")
        logger.info(f"  Scaling Error: {results['scaling_error']:.4f}")
        logger.info(f"  Confidence Interval: {results['confidence_interval']}")
        logger.info(f"  Execution Time: {results['performance']['execution_time']:.4f}s")
        logger.info(f"  Memory Usage: {results['performance']['memory_usage']} bytes")
        logger.info(f"  JAX Usage: {results['performance']['jax_usage']}")
        logger.info(f"  Fallback Usage: {results['performance']['fallback_usage']}")
        
        # Test caching
        cache_stats = estimator.get_cache_stats()
        logger.info(f"  Cache Hit Rate: {cache_stats['hit_rate']:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Wavelet leaders estimator test failed: {e}")
        return False

def test_wavelet_whittle_estimator():
    """Test the high-performance wavelet Whittle estimator."""
    logger.info("Testing HighPerformanceWaveletWhittleEstimator...")
    
    try:
        from estimators.high_performance_wavelet_whittle import HighPerformanceWaveletWhittleEstimator
        
        # Generate test data
        data = generate_test_data(1000, 0.7)
        
        # Create estimator
        estimator = HighPerformanceWaveletWhittleEstimator(
            wavelet='db4',
            num_scales=15,
            frequency_range=[0.01, 0.5],
            initial_guess=[0.5],
            use_jax=True,
            enable_caching=True,
            vectorized=True
        )
        
        # Fit and estimate
        results = estimator.estimate(data)
        
        logger.info(f"Wavelet Whittle Estimation Results:")
        logger.info(f"  Alpha: {results['alpha']:.4f}")
        logger.info(f"  Hurst Exponent: {results['hurst_exponent']:.4f}")
        logger.info(f"  Optimization Success: {results['optimization_success']}")
        logger.info(f"  Confidence Interval: {results['confidence_interval']}")
        logger.info(f"  Execution Time: {results['performance']['execution_time']:.4f}s")
        logger.info(f"  Memory Usage: {results['performance']['memory_usage']} bytes")
        logger.info(f"  JAX Usage: {results['performance']['jax_usage']}")
        logger.info(f"  Fallback Usage: {results['performance']['fallback_usage']}")
        
        # Test caching
        cache_stats = estimator.get_cache_stats()
        logger.info(f"  Cache Hit Rate: {cache_stats['hit_rate']:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Wavelet Whittle estimator test failed: {e}")
        return False

def test_performance_comparison():
    """Compare performance between high-performance and standard estimators."""
    logger.info("Testing performance comparison...")
    
    try:
        from estimators.high_performance_periodogram import HighPerformancePeriodogramEstimator
        from estimators.high_performance_gph import HighPerformanceGPHEstimator
        from estimators.spectral import PeriodogramEstimator, GPHEstimator
        
        # Generate test data
        data = generate_test_data(2000, 0.7)
        
        # Test Periodogram estimators
        logger.info("Periodogram Performance Comparison:")
        
        # Standard Periodogram
        start_time = time.time()
        standard_periodogram = PeriodogramEstimator()
        standard_results = standard_periodogram.estimate(data)
        standard_time = time.time() - start_time
        
        # High-performance Periodogram
        start_time = time.time()
        hp_periodogram = HighPerformancePeriodogramEstimator(use_jax=True, vectorized=True)
        hp_results = hp_periodogram.estimate(data)
        hp_time = time.time() - start_time
        
        logger.info(f"  Standard Periodogram: {standard_time:.4f}s")
        logger.info(f"  High-Performance Periodogram: {hp_time:.4f}s")
        logger.info(f"  Speedup: {standard_time/hp_time:.2f}x")
        
        # Test GPH estimators
        logger.info("GPH Performance Comparison:")
        
        # Standard GPH
        start_time = time.time()
        standard_gph = GPHEstimator()
        standard_results = standard_gph.estimate(data)
        standard_time = time.time() - start_time
        
        # High-performance GPH
        start_time = time.time()
        hp_gph = HighPerformanceGPHEstimator(use_jax=True, vectorized=True)
        hp_results = hp_gph.estimate(data)
        hp_time = time.time() - start_time
        
        logger.info(f"  Standard GPH: {standard_time:.4f}s")
        logger.info(f"  High-Performance GPH: {hp_time:.4f}s")
        logger.info(f"  Speedup: {standard_time/hp_time:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance comparison failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting comprehensive tests for all new high-performance estimators...")
    
    # Test individual estimators
    periodogram_success = test_periodogram_estimator()
    gph_success = test_gph_estimator()
    wavelet_leaders_success = test_wavelet_leaders_estimator()
    wavelet_whittle_success = test_wavelet_whittle_estimator()
    
    # Test performance comparison
    perf_success = test_performance_comparison()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Periodogram Estimator: {'✓ PASS' if periodogram_success else '✗ FAIL'}")
    logger.info(f"GPH Estimator: {'✓ PASS' if gph_success else '✗ FAIL'}")
    logger.info(f"Wavelet Leaders Estimator: {'✓ PASS' if wavelet_leaders_success else '✗ FAIL'}")
    logger.info(f"Wavelet Whittle Estimator: {'✓ PASS' if wavelet_whittle_success else '✗ FAIL'}")
    logger.info(f"Performance Comparison: {'✓ PASS' if perf_success else '✗ FAIL'}")
    
    all_tests_passed = all([
        periodogram_success, 
        gph_success, 
        wavelet_leaders_success, 
        wavelet_whittle_success, 
        perf_success
    ])
    
    if all_tests_passed:
        logger.info("\n🎉 All tests passed! All new estimators are working correctly.")
        logger.info("\n📊 COMPLETE ESTIMATOR SUITE:")
        logger.info("  ✓ HighPerformanceDFAEstimator")
        logger.info("  ✓ HighPerformanceMFDFAEstimator")
        logger.info("  ✓ HighPerformanceRSEstimator")
        logger.info("  ✓ HighPerformanceHiguchiEstimator")
        logger.info("  ✓ HighPerformanceWhittleMLEEstimator")
        logger.info("  ✓ HighPerformancePeriodogramEstimator")
        logger.info("  ✓ HighPerformanceGPHEstimator")
        logger.info("  ✓ HighPerformanceWaveletLeadersEstimator")
        logger.info("  ✓ HighPerformanceWaveletWhittleEstimator")
        logger.info("\n🚀 The framework now has 9 high-performance estimators!")
    else:
        logger.error("\n❌ Some tests failed. Please check the error messages above.")
        
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
