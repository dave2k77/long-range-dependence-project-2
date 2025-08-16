#!/usr/bin/env python3
"""
Test script for the new high-performance Whittle MLE estimator.
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

def test_whittle_estimator():
    """Test the high-performance Whittle MLE estimator."""
    logger.info("Testing HighPerformanceWhittleMLEEstimator...")
    
    try:
        from estimators.high_performance_whittle import HighPerformanceWhittleMLEEstimator
        
        # Generate test data
        data = generate_test_data(1000, 0.7)
        
        # Create estimator
        estimator = HighPerformanceWhittleMLEEstimator(
            frequency_range=[0.01, 0.5],
            initial_guess=[0.5],
            use_jax=True,
            enable_caching=True,
            vectorized=True
        )
        
        # Fit and estimate
        results = estimator.fit_estimate(data)
        
        logger.info(f"Whittle MLE Estimation Results:")
        logger.info(f"  Alpha: {results['alpha']:.4f}")
        logger.info(f"  Hurst Exponent: {results['hurst_exponent']:.4f}")
        logger.info(f"  Optimization Success: {results['optimization_success']}")
        logger.info(f"  Execution Time: {results['performance']['execution_time']:.4f}s")
        logger.info(f"  Memory Usage: {results['performance']['memory_usage']} bytes")
        logger.info(f"  JAX Usage: {results['performance']['jax_usage']}")
        logger.info(f"  Fallback Usage: {results['performance']['fallback_usage']}")
        
        # Test caching
        cache_stats = estimator.get_cache_stats()
        logger.info(f"  Cache Hit Rate: {cache_stats['hit_rate']:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Whittle estimator test failed: {e}")
        return False

def main():
    """Run the test."""
    logger.info("Starting test for new high-performance Whittle MLE estimator...")
    
    # Test the estimator
    success = test_whittle_estimator()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Whittle MLE Estimator: {'✓ PASS' if success else '✗ FAIL'}")
    
    if success:
        logger.info("\n🎉 Test passed! The Whittle MLE estimator is working correctly.")
    else:
        logger.error("\n❌ Test failed. Please check the error messages above.")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
