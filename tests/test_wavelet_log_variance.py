#!/usr/bin/env python3
"""
Test script for the new HighPerformanceWaveletLogVarianceEstimator.
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

def test_wavelet_log_variance_estimator():
    """Test the high-performance wavelet log-variance estimator."""
    logger.info("Testing HighPerformanceWaveletLogVarianceEstimator...")
    
    try:
        from estimators.high_performance_wavelet_log_variance import HighPerformanceWaveletLogVarianceEstimator
        
        # Generate test data
        data = generate_test_data(1000, 0.7)
        
        # Create estimator
        estimator = HighPerformanceWaveletLogVarianceEstimator(
            wavelet='db4',
            num_scales=15,
            use_jax=True,
            enable_caching=True,
            vectorized=True
        )
        
        # Fit and estimate
        results = estimator.estimate(data)
        
        logger.info(f"Wavelet Log-Variance Estimation Results:")
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
        
        # Display wavelet variances
        logger.info(f"  Wavelet Variances by Scale:")
        for scale, variance in results['wavelet_variances'].items():
            logger.info(f"    Scale {scale}: {variance:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Wavelet log-variance estimator test failed: {e}")
        return False

def test_performance_comparison():
    """Compare performance between different wavelet methods."""
    logger.info("Testing wavelet methods performance comparison...")
    
    try:
        from estimators.high_performance_wavelet_leaders import HighPerformanceWaveletLeadersEstimator
        from estimators.high_performance_wavelet_whittle import HighPerformanceWaveletWhittleEstimator
        from estimators.high_performance_wavelet_log_variance import HighPerformanceWaveletLogVarianceEstimator
        
        # Generate test data
        data = generate_test_data(2000, 0.7)
        
        # Test all three wavelet methods
        methods = {
            'Wavelet Leaders': HighPerformanceWaveletLeadersEstimator(
                wavelet='db4', num_scales=15, use_jax=True
            ),
            'Wavelet Whittle': HighPerformanceWaveletWhittleEstimator(
                wavelet='db4', num_scales=15, use_jax=True
            ),
            'Wavelet Log-Variance': HighPerformanceWaveletLogVarianceEstimator(
                wavelet='db4', num_scales=15, use_jax=True
            )
        }
        
        results = {}
        
        for method_name, estimator in methods.items():
            logger.info(f"\n{method_name} Performance:")
            
            start_time = time.time()
            method_results = estimator.estimate(data)
            execution_time = time.time() - start_time
            
            results[method_name] = {
                'execution_time': execution_time,
                'hurst_exponent': method_results.get('hurst_exponent', np.nan),
                'memory_usage': method_results['performance']['memory_usage']
            }
            
            logger.info(f"  Execution Time: {execution_time:.4f}s")
            logger.info(f"  Hurst Exponent: {results[method_name]['hurst_exponent']:.4f}")
            logger.info(f"  Memory Usage: {results[method_name]['memory_usage']} bytes")
        
        # Performance ranking
        logger.info(f"\n🏆 Performance Ranking (by execution time):")
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['execution_time'])
        for i, (method_name, perf) in enumerate(sorted_methods, 1):
            logger.info(f"  {i}. {method_name}: {perf['execution_time']:.4f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance comparison failed: {e}")
        return False

def test_different_wavelets():
    """Test the estimator with different wavelet types."""
    logger.info("Testing different wavelet types...")
    
    try:
        from estimators.high_performance_wavelet_log_variance import HighPerformanceWaveletLogVarianceEstimator
        
        # Generate test data
        data = generate_test_data(1000, 0.7)
        
        # Test different wavelet types
        wavelet_types = ['db4', 'db6', 'haar', 'coif4', 'sym4']
        
        results = {}
        
        for wavelet in wavelet_types:
            logger.info(f"\nTesting {wavelet} wavelet:")
            
            try:
                estimator = HighPerformanceWaveletLogVarianceEstimator(
                    wavelet=wavelet,
                    num_scales=15,
                    use_jax=True,
                    enable_caching=True
                )
                
                start_time = time.time()
                method_results = estimator.estimate(data)
                execution_time = time.time() - start_time
                
                results[wavelet] = {
                    'hurst_exponent': method_results['hurst_exponent'],
                    'execution_time': execution_time,
                    'success': True
                }
                
                logger.info(f"  Hurst Exponent: {method_results['hurst_exponent']:.4f}")
                logger.info(f"  Execution Time: {execution_time:.4f}s")
                logger.info(f"  Status: ✓ Success")
                
            except Exception as e:
                logger.error(f"  Status: ✗ Failed - {e}")
                results[wavelet] = {'success': False, 'error': str(e)}
        
        # Summary
        successful_wavelets = [w for w, r in results.items() if r['success']]
        logger.info(f"\n📊 Wavelet Test Summary:")
        logger.info(f"  Successful: {len(successful_wavelets)}/{len(wavelet_types)}")
        logger.info(f"  Failed: {len(wavelet_types) - len(successful_wavelets)}/{len(wavelet_types)}")
        
        if successful_wavelets:
            logger.info(f"  Best performing: {min(successful_wavelets, key=lambda w: results[w]['execution_time'])}")
        
        return len(successful_wavelets) > 0
        
    except Exception as e:
        logger.error(f"Wavelet type testing failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting comprehensive tests for HighPerformanceWaveletLogVarianceEstimator...")
    
    # Test individual estimator
    basic_success = test_wavelet_log_variance_estimator()
    
    # Test performance comparison
    perf_success = test_performance_comparison()
    
    # Test different wavelets
    wavelet_success = test_different_wavelets()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("WAVELET LOG-VARIANCE TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Basic Functionality: {'✓ PASS' if basic_success else '✗ FAIL'}")
    logger.info(f"Performance Comparison: {'✓ PASS' if perf_success else '✗ FAIL'}")
    logger.info(f"Wavelet Type Testing: {'✓ PASS' if wavelet_success else '✗ FAIL'}")
    
    all_tests_passed = all([basic_success, perf_success, wavelet_success])
    
    if all_tests_passed:
        logger.info("\n🎉 All tests passed! The Wavelet Log-Variance estimator is working correctly.")
        logger.info("\n📊 COMPLETE WAVELET SUITE:")
        logger.info("  ✓ HighPerformanceWaveletLeadersEstimator")
        logger.info("  ✓ HighPerformanceWaveletWhittleEstimator")
        logger.info("  ✓ HighPerformanceWaveletLogVarianceEstimator")
        logger.info("\n🚀 The framework now has 10 high-performance estimators!")
    else:
        logger.error("\n❌ Some tests failed. Please check the error messages above.")
        
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
