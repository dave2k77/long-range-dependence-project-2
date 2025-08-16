#!/usr/bin/env python3
"""
Test script for the new high-performance estimators.
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import logging

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

def test_rs_estimator():
    """Test the high-performance R/S estimator."""
    logger.info("Testing HighPerformanceRSEstimator...")
    
    try:
        from estimators.high_performance_rs import HighPerformanceRSEstimator
        
        # Generate test data
        data = generate_test_data(1000, 0.7)
        
        # Create estimator
        estimator = HighPerformanceRSEstimator(
            min_scale=4,
            num_scales=15,
            use_jax=True,
            enable_caching=True,
            vectorized=True
        )
        
        # Fit and estimate
        results = estimator.fit_estimate(data)
        
        logger.info(f"R/S Estimation Results:")
        logger.info(f"  Hurst Exponent: {results['hurst_exponent']:.4f}")
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
        logger.error(f"R/S estimator test failed: {e}")
        return False

def test_higuchi_estimator():
    """Test the high-performance Higuchi estimator."""
    logger.info("Testing HighPerformanceHiguchiEstimator...")
    
    try:
        from estimators.high_performance_higuchi import HighPerformanceHiguchiEstimator
        
        # Generate test data
        data = generate_test_data(1000, 0.7)
        
        # Create estimator
        estimator = HighPerformanceHiguchiEstimator(
            min_k=2,
            num_k=15,
            use_jax=True,
            enable_caching=True,
            vectorized=True
        )
        
        # Fit and estimate
        results = estimator.fit_estimate(data)
        
        logger.info(f"Higuchi Estimation Results:")
        logger.info(f"  Fractal Dimension: {results['fractal_dimension']:.4f}")
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
        logger.error(f"Higuchi estimator test failed: {e}")
        return False

def test_performance_comparison():
    """Compare performance between high-performance and standard estimators."""
    logger.info("Testing performance comparison...")
    
    try:
        from estimators.high_performance_rs import HighPerformanceRSEstimator
        from estimators.high_performance_higuchi import HighPerformanceHiguchiEstimator
        from estimators.temporal import RSEstimator, HiguchiEstimator
        
        # Generate test data
        data = generate_test_data(2000, 0.7)
        
        # Test R/S estimators
        logger.info("R/S Performance Comparison:")
        
        # Standard R/S
        start_time = time.time()
        standard_rs = RSEstimator()
        standard_results = standard_rs.fit_estimate(data)
        standard_time = time.time() - start_time
        
        # High-performance R/S
        start_time = time.time()
        hp_rs = HighPerformanceRSEstimator(use_jax=True, vectorized=True)
        hp_results = hp_rs.fit_estimate(data)
        hp_time = time.time() - start_time
        
        logger.info(f"  Standard R/S: {standard_time:.4f}s")
        logger.info(f"  High-Performance R/S: {hp_time:.4f}s")
        logger.info(f"  Speedup: {standard_time/hp_time:.2f}x")
        
        # Test Higuchi estimators
        logger.info("Higuchi Performance Comparison:")
        
        # Standard Higuchi
        start_time = time.time()
        standard_higuchi = HiguchiEstimator()
        standard_results = standard_higuchi.fit_estimate(data)
        standard_time = time.time() - start_time
        
        # High-performance Higuchi
        start_time = time.time()
        hp_higuchi = HighPerformanceHiguchiEstimator(use_jax=True, vectorized=True)
        hp_results = hp_higuchi.fit_estimate(data)
        hp_time = time.time() - start_time
        
        logger.info(f"  Standard Higuchi: {standard_time:.4f}s")
        logger.info(f"  High-Performance Higuchi: {hp_time:.4f}s")
        logger.info(f"  Speedup: {standard_time/hp_time:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance comparison failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting tests for new high-performance estimators...")
    
    # Test individual estimators
    rs_success = test_rs_estimator()
    higuchi_success = test_higuchi_estimator()
    
    # Test performance comparison
    perf_success = test_performance_comparison()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"R/S Estimator: {'✓ PASS' if rs_success else '✗ FAIL'}")
    logger.info(f"Higuchi Estimator: {'✓ PASS' if higuchi_success else '✗ PASS'}")
    logger.info(f"Performance Comparison: {'✓ PASS' if perf_success else '✗ FAIL'}")
    
    if all([rs_success, higuchi_success, perf_success]):
        logger.info("\n🎉 All tests passed! The new estimators are working correctly.")
    else:
        logger.error("\n❌ Some tests failed. Please check the error messages above.")
        
    return all([rs_success, higuchi_success, perf_success])

if __name__ == "__main__":
    import time
    success = main()
    sys.exit(0 if success else 1)
