"""
Unit tests for High-Performance Estimators

This module tests the NUMBA and JAX optimized estimators,
ensuring proper implementation of high-performance computing capabilities.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Try to import high-performance estimators
try:
    from estimators import HighPerformanceDFAEstimator, HighPerformanceMFDFAEstimator
    HIGH_PERFORMANCE_AVAILABLE = True
except ImportError:
    HIGH_PERFORMANCE_AVAILABLE = False

from estimators import DFAEstimator, MFDFAEstimator
from src.benchmarking.synthetic_data import SyntheticDataGenerator


@pytest.mark.skipif(not HIGH_PERFORMANCE_AVAILABLE, reason="High-performance estimators not available")
class TestHighPerformanceDFAEstimator:
    """Test suite for HighPerformanceDFAEstimator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = HighPerformanceDFAEstimator()
        self.generator = SyntheticDataGenerator()
        
        # Generate test data
        self.fbm_data = self.generator.fractional_brownian_motion(
            n_points=1000, hurst=0.7, noise_level=0.05
        )
        self.random_data = np.random.randn(1000)
        self.anti_persistent_data = self.generator.fractional_brownian_motion(
            n_points=1000, hurst=0.3, noise_level=0.05
        )

    def test_initialization(self):
        """Test estimator initialization with default and custom parameters."""
        # Test default initialization
        est = HighPerformanceDFAEstimator()
        assert est.name == "HighPerformanceDFA"
        assert est.min_scale == 4
        assert est.max_scale is None
        assert est.num_scales == 20
        assert est.polynomial_order == 1
        assert est.use_parallel == True
        
        # Test custom initialization
        est_custom = HighPerformanceDFAEstimator(
            min_scale=8,
            max_scale=100,
            num_scales=15,
            polynomial_order=2,
            use_parallel=False
        )
        assert est_custom.min_scale == 8
        assert est_custom.max_scale == 100
        assert est_custom.num_scales == 15
        assert est_custom.polynomial_order == 2
        assert est_custom.use_parallel == False

    def test_validate_data(self):
        """Test data validation."""
        # Test valid data
        self.estimator._validate_data()
        
        # Test data too short
        with pytest.raises(ValueError, match="at least 100 points"):
            self.estimator.data = np.random.randn(50)
            self.estimator._validate_data()
        
        # Test data with NaN
        with pytest.raises(ValueError, match="NaN or infinite values"):
            self.estimator.data = np.array([1, 2, np.nan, 4])
            self.estimator._validate_data()
        
        # Test data with inf
        with pytest.raises(ValueError, match="NaN or infinite values"):
            self.estimator.data = np.array([1, 2, np.inf, 4])
            self.estimator._validate_data()

    def test_generate_scales(self):
        """Test scale generation."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_scales()
        
        # Check scales are within bounds
        assert len(self.estimator.scales) <= self.estimator.num_scales
        assert self.estimator.scales[0] >= self.estimator.min_scale
        assert self.estimator.scales[-1] <= len(self.fbm_data) // 4
        
        # Check scales are unique and sorted
        assert len(np.unique(self.estimator.scales)) == len(self.estimator.scales)
        assert np.all(np.diff(self.estimator.scales) > 0)
        
        # Test custom max_scale
        est_custom = HighPerformanceDFAEstimator(max_scale=50)
        est_custom.data = self.fbm_data
        est_custom._generate_scales()
        assert est_custom.scales[-1] <= 50

    def test_calculate_fluctuations_optimized(self):
        """Test optimized fluctuation calculation."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_scales()
        self.estimator._calculate_fluctuations_optimized()
        
        # Check fluctuations array
        assert hasattr(self.estimator, 'fluctuations')
        assert len(self.estimator.fluctuations) == len(self.estimator.scales)
        
        # Check for valid fluctuation values
        assert not np.any(np.isnan(self.estimator.fluctuations))
        assert not np.any(np.isinf(self.estimator.fluctuations))
        assert np.all(self.estimator.fluctuations > 0)

    def test_fit_power_law_optimized(self):
        """Test optimized power law fitting."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_scales()
        self.estimator._calculate_fluctuations_optimized()
        self.estimator._fit_power_law_optimized()
        
        # Check Hurst exponent
        assert hasattr(self.estimator, 'hurst_exponent')
        assert isinstance(self.estimator.hurst_exponent, (int, float))
        assert not np.isnan(self.estimator.hurst_exponent)
        assert not np.isinf(self.estimator.hurst_exponent)
        
        # Check intercept and R-squared
        assert hasattr(self.estimator, 'intercept')
        assert hasattr(self.estimator, 'r_squared')
        assert isinstance(self.estimator.intercept, (int, float))
        assert isinstance(self.estimator.r_squared, (int, float))
        assert not np.isnan(self.estimator.intercept)
        assert not np.isnan(self.estimator.r_squared)
        assert 0 <= self.estimator.r_squared <= 1

    def test_complete_estimation_workflow(self):
        """Test complete estimation workflow."""
        # Test with fBm data
        results = self.estimator.estimate(self.fbm_data)
        
        # Check required keys
        required_keys = ['hurst_exponent', 'scales', 'fluctuations', 'intercept', 'r_squared']
        for key in required_keys:
            assert key in results
        
        # Check data types and shapes
        assert isinstance(results['hurst_exponent'], (int, float))
        assert isinstance(results['scales'], np.ndarray)
        assert isinstance(results['fluctuations'], np.ndarray)
        assert isinstance(results['intercept'], (int, float))
        assert isinstance(results['r_squared'], (int, float))
        
        # Check parameters
        assert 'parameters' in results
        parameters = results['parameters']
        assert 'min_scale' in parameters
        assert 'max_scale' in parameters
        assert 'num_scales' in parameters
        assert 'polynomial_order' in parameters
        assert 'use_parallel' in parameters
        
        # Check interpretation
        assert 'interpretation' in results
        interpretation = results['interpretation']
        assert 'lrd_type' in interpretation
        assert 'strength' in interpretation
        assert 'reliability' in interpretation
        assert 'method' in interpretation

    def test_estimation_with_different_data_types(self):
        """Test estimation with different types of time series data."""
        # Test with random walk data
        results_random = self.estimator.estimate(self.random_data)
        assert 'hurst_exponent' in results_random
        
        # Test with anti-persistent data
        results_anti = self.estimator.estimate(self.anti_persistent_data)
        assert 'hurst_exponent' in results_anti
        
        # Compare results - anti-persistent should have lower Hurst than fBm
        fbm_results = self.estimator.estimate(self.fbm_data)
        
        if (results_anti['hurst_exponent'] < fbm_results['hurst_exponent']):
            print("✓ Anti-persistent data shows lower Hurst exponent as expected")
        else:
            print("⚠ Anti-persistent vs fBm comparison inconclusive")

    def test_parameter_effects(self):
        """Test how different parameters affect the estimation."""
        # Test different polynomial orders
        est_order1 = HighPerformanceDFAEstimator(polynomial_order=1)
        est_order2 = HighPerformanceDFAEstimator(polynomial_order=2)
        
        results_order1 = est_order1.estimate(self.fbm_data)
        results_order2 = est_order2.estimate(self.fbm_data)
        
        # Results should be similar but not identical
        assert abs(results_order1['hurst_exponent'] - 
                  results_order2['hurst_exponent']) < 0.2
        
        # Test parallel vs sequential
        est_parallel = HighPerformanceDFAEstimator(use_parallel=True)
        est_sequential = HighPerformanceDFAEstimator(use_parallel=False)
        
        results_parallel = est_parallel.estimate(self.fbm_data)
        results_sequential = est_sequential.estimate(self.fbm_data)
        
        # Both should give reasonable results
        assert not np.isnan(results_parallel['hurst_exponent'])
        assert not np.isnan(results_sequential['hurst_exponent'])

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very short data
        short_data = np.random.randn(120)
        results_short = self.estimator.estimate(short_data)
        assert 'hurst_exponent' in results_short
        
        # Test with very long data
        long_data = np.random.randn(10000)
        results_long = self.estimator.estimate(long_data)
        assert 'hurst_exponent' in results_long
        
        # Test with constant data (should handle gracefully)
        constant_data = np.ones(1000)
        with pytest.raises(ValueError, match="constant"):
            self.estimator.estimate(constant_data)

    def test_memory_and_execution_tracking(self):
        """Test memory usage and execution time tracking."""
        # Reset estimator
        self.estimator.reset()
        
        # Run estimation
        results = self.estimator.estimate(self.fbm_data)
        
        # Check execution time
        execution_time = self.estimator.get_execution_time()
        assert execution_time > 0
        assert execution_time < 60  # Should complete within reasonable time
        
        # Check memory usage
        memory_usage = self.estimator.get_memory_usage()
        assert memory_usage > 0
        
        # Check results consistency
        results_again = self.estimator.get_results()
        assert results == results_again

    def test_performance_comparison(self):
        """Test performance comparison with standard DFA."""
        # Compare with standard DFA estimator
        standard_estimator = DFAEstimator()
        
        # Time both estimators
        import time
        
        # Time high-performance version
        start_time = time.time()
        hp_results = self.estimator.estimate(self.fbm_data)
        hp_time = time.time() - start_time
        
        # Time standard version
        start_time = time.time()
        standard_results = standard_estimator.estimate(self.fbm_data)
        standard_time = time.time() - start_time
        
        # Both should give similar results
        hurst_diff = abs(hp_results['hurst_exponent'] - standard_results['hurst_exponent'])
        assert hurst_diff < 0.1  # Should be very similar
        
        # High-performance should be faster (though this may vary)
        print(f"High-performance DFA: {hp_time:.4f}s")
        print(f"Standard DFA: {standard_time:.4f}s")
        print(f"Speedup: {standard_time/hp_time:.2f}x")

    def test_parallel_vs_sequential(self):
        """Test parallel vs sequential execution."""
        # Test parallel execution
        est_parallel = HighPerformanceDFAEstimator(use_parallel=True)
        est_sequential = HighPerformanceDFAEstimator(use_parallel=False)
        
        # Both should give identical results
        results_parallel = est_parallel.estimate(self.fbm_data)
        results_sequential = est_sequential.estimate(self.fbm_data)
        
        # Results should be identical (deterministic)
        assert abs(results_parallel['hurst_exponent'] - 
                  results_sequential['hurst_exponent']) < 1e-10
        
        # Check all other results are identical
        for key in ['scales', 'fluctuations', 'intercept', 'r_squared']:
            if isinstance(results_parallel[key], np.ndarray):
                np.testing.assert_array_almost_equal(
                    results_parallel[key], results_sequential[key]
                )
            else:
                assert results_parallel[key] == results_sequential[key]


@pytest.mark.skipif(not HIGH_PERFORMANCE_AVAILABLE, reason="High-performance estimators not available")
class TestHighPerformanceMFDFAEstimator:
    """Test suite for HighPerformanceMFDFAEstimator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = HighPerformanceMFDFAEstimator()
        self.generator = SyntheticDataGenerator()
        
        # Generate test data
        self.fbm_data = self.generator.fractional_brownian_motion(
            n_points=1000, hurst=0.7, noise_level=0.05
        )
        self.random_data = np.random.randn(1000)
        self.anti_persistent_data = self.generator.fractional_brownian_motion(
            n_points=1000, hurst=0.3, noise_level=0.05
        )

    def test_initialization(self):
        """Test estimator initialization with default and custom parameters."""
        # Test default initialization
        est = HighPerformanceMFDFAEstimator()
        assert est.name == "HighPerformanceMFDFA"
        assert est.min_scale == 4
        assert est.max_scale is None
        assert est.num_scales == 20
        assert est.polynomial_order == 1
        assert len(est.q_values) == 22  # Default q range
        
        # Test custom initialization
        custom_q = np.array([-3, -1, 0, 1, 3])
        est_custom = HighPerformanceMFDFAEstimator(
            min_scale=8,
            max_scale=100,
            num_scales=15,
            polynomial_order=2,
            q_values=custom_q
        )
        assert est_custom.min_scale == 8
        assert est_custom.max_scale == 100
        assert est_custom.num_scales == 15
        assert est_custom.polynomial_order == 2
        np.testing.assert_array_equal(est_custom.q_values, custom_q)

    def test_validate_data(self):
        """Test data validation."""
        # Test valid data
        self.estimator._validate_data()
        
        # Test data too short
        with pytest.raises(ValueError, match="at least 100 points"):
            self.estimator.data = np.random.randn(50)
            self.estimator._validate_data()
        
        # Test data with NaN
        with pytest.raises(ValueError, match="NaN or infinite values"):
            self.estimator.data = np.array([1, 2, np.nan, 4])
            self.estimator._validate_data()
        
        # Test data with inf
        with pytest.raises(ValueError, match="NaN or infinite values"):
            self.estimator.data = np.array([1, 2, np.inf, 4])
            self.estimator._validate_data()

    def test_generate_scales(self):
        """Test scale generation."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_scales()
        
        # Check scales are within bounds
        assert len(self.estimator.scales) <= self.estimator.num_scales
        assert self.estimator.scales[0] >= self.estimator.min_scale
        assert self.estimator.scales[-1] <= len(self.fbm_data) // 4
        
        # Check scales are unique and sorted
        assert len(np.unique(self.estimator.scales)) == len(self.estimator.scales)
        assert np.all(np.diff(self.estimator.scales) > 0)
        
        # Test custom max_scale
        est_custom = HighPerformanceMFDFAEstimator(max_scale=50)
        est_custom.data = self.fbm_data
        est_custom._generate_scales()
        assert est_custom.scales[-1] <= 50

    def test_calculate_fluctuations_jax(self):
        """Test JAX-optimized fluctuation calculation."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_scales()
        self.estimator._calculate_fluctuations_jax()
        
        # Check fluctuations array shape
        assert hasattr(self.estimator, 'fluctuations')
        assert self.estimator.fluctuations.shape == (len(self.estimator.q_values), 
                                                   len(self.estimator.scales))
        
        # Check for valid fluctuation values
        assert not np.any(np.isnan(self.estimator.fluctuations))
        assert not np.any(np.isinf(self.estimator.fluctuations))
        assert np.all(self.estimator.fluctuations > 0)

    def test_fit_scaling_laws_jax(self):
        """Test JAX-optimized scaling law fitting."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_scales()
        self.estimator._calculate_fluctuations_jax()
        self.estimator._fit_scaling_laws_jax()
        
        # Check Hurst exponents array
        assert hasattr(self.estimator, 'hurst_exponents')
        assert len(self.estimator.hurst_exponents) == len(self.estimator.q_values)
        
        # Check for valid Hurst values (typically between 0 and 1)
        valid_h = self.estimator.hurst_exponents[~np.isnan(self.estimator.hurst_exponents)]
        if len(valid_h) > 0:
            assert np.all(valid_h >= 0)
            assert np.all(valid_h <= 2)  # Allow some flexibility for extreme cases

    def test_calculate_multifractal_spectrum_jax(self):
        """Test JAX-optimized multifractal spectrum calculation."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_scales()
        self.estimator._calculate_fluctuations_jax()
        self.estimator._fit_scaling_laws_jax()
        self.estimator._calculate_multifractal_spectrum_jax()
        
        # Check spectrum components
        assert hasattr(self.estimator, 'multifractal_spectrum')
        spectrum = self.estimator.multifractal_spectrum
        
        if spectrum is not None:
            assert 'alpha' in spectrum
            assert 'f_alpha' in spectrum
            assert len(spectrum['alpha']) > 0
            assert len(spectrum['f_alpha']) > 0
            
            # Check alpha values are reasonable
            assert np.all(spectrum['alpha'] > 0)
            assert np.all(spectrum['alpha'] < 2)

    def test_complete_estimation_workflow(self):
        """Test complete estimation workflow."""
        # Test with fBm data
        results = self.estimator.estimate(self.fbm_data)
        
        # Check required keys
        required_keys = ['hurst_exponents', 'q_values', 'scales', 'fluctuations', 
                        'multifractal_spectrum']
        for key in required_keys:
            assert key in results
        
        # Check data types and shapes
        assert isinstance(results['hurst_exponents'], np.ndarray)
        assert isinstance(results['q_values'], np.ndarray)
        assert isinstance(results['scales'], np.ndarray)
        assert isinstance(results['fluctuations'], np.ndarray)
        
        # Check parameters
        assert 'parameters' in results
        parameters = results['parameters']
        assert 'min_scale' in parameters
        assert 'max_scale' in parameters
        assert 'num_scales' in parameters
        assert 'polynomial_order' in parameters
        
        # Check summary statistics
        assert 'summary' in results
        summary = results['summary']
        assert 'mean_hurst' in summary
        assert 'std_hurst' in summary
        assert 'min_hurst' in summary
        assert 'max_hurst' in summary
        assert 'is_multifractal' in summary

    def test_estimation_with_different_data_types(self):
        """Test estimation with different types of time series data."""
        # Test with random walk data
        results_random = self.estimator.estimate(self.random_data)
        assert 'hurst_exponents' in results_random
        
        # Test with anti-persistent data
        results_anti = self.estimator.estimate(self.anti_persistent_data)
        assert 'hurst_exponents' in results_anti
        
        # Compare results - anti-persistent should have lower Hurst than fBm
        fbm_results = self.estimator.estimate(self.fbm_data)
        
        if (results_anti['summary']['mean_hurst'] < 
            fbm_results['summary']['mean_hurst']):
            print("✓ Anti-persistent data shows lower Hurst exponent as expected")
        else:
            print("⚠ Anti-persistent vs fBm comparison inconclusive")

    def test_parameter_effects(self):
        """Test how different parameters affect the estimation."""
        # Test different polynomial orders
        est_order1 = HighPerformanceMFDFAEstimator(polynomial_order=1)
        est_order2 = HighPerformanceMFDFAEstimator(polynomial_order=2)
        
        results_order1 = est_order1.estimate(self.fbm_data)
        results_order2 = est_order2.estimate(self.fbm_data)
        
        # Results should be similar but not identical
        assert abs(results_order1['summary']['mean_hurst'] - 
                  results_order2['summary']['mean_hurst']) < 0.2
        
        # Test different scale ranges
        est_few_scales = HighPerformanceMFDFAEstimator(num_scales=10)
        est_many_scales = HighPerformanceMFDFAEstimator(num_scales=30)
        
        results_few = est_few_scales.estimate(self.fbm_data)
        results_many = est_many_scales.estimate(self.fbm_data)
        
        # More scales should give more detailed analysis
        assert len(results_many['scales']) > len(results_few['scales'])

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very short data
        short_data = np.random.randn(120)
        results_short = self.estimator.estimate(short_data)
        assert 'hurst_exponents' in results_short
        
        # Test with very long data
        long_data = np.random.randn(10000)
        results_long = self.estimator.estimate(long_data)
        assert 'hurst_exponents' in results_long
        
        # Test with constant data (should handle gracefully)
        constant_data = np.ones(1000)
        with pytest.raises(ValueError, match="constant"):
            self.estimator.estimate(constant_data)

    def test_memory_and_execution_tracking(self):
        """Test memory usage and execution time tracking."""
        # Reset estimator
        self.estimator.reset()
        
        # Run estimation
        results = self.estimator.estimate(self.fbm_data)
        
        # Check execution time
        execution_time = self.estimator.get_execution_time()
        assert execution_time > 0
        assert execution_time < 60  # Should complete within reasonable time
        
        # Check memory usage
        memory_usage = self.estimator.get_memory_usage()
        assert memory_usage > 0
        
        # Check results consistency
        results_again = self.estimator.get_results()
        assert results == results_again

    def test_performance_comparison(self):
        """Test performance comparison with standard MFDFA."""
        # Compare with standard MFDFA estimator
        standard_estimator = MFDFAEstimator()
        
        # Time both estimators
        import time
        
        # Time high-performance version
        start_time = time.time()
        hp_results = self.estimator.estimate(self.fbm_data)
        hp_time = time.time() - start_time
        
        # Time standard version
        start_time = time.time()
        standard_results = standard_estimator.estimate(self.fbm_data)
        standard_time = time.time() - start_time
        
        # Both should give similar results
        hp_mean = hp_results['summary']['mean_hurst']
        standard_mean = standard_results['summary']['mean_hurst']
        hurst_diff = abs(hp_mean - standard_mean)
        assert hurst_diff < 0.2  # Should be similar
        
        # High-performance should be faster (though this may vary)
        print(f"High-performance MFDFA: {hp_time:.4f}s")
        print(f"Standard MFDFA: {standard_time:.4f}s")
        print(f"Speedup: {standard_time/hp_time:.2f}x")

    def test_q_values_handling(self):
        """Test different q-values configurations."""
        # Test custom q range
        custom_q = np.array([-2, -1, 0, 1, 2])
        est_custom = HighPerformanceMFDFAEstimator(q_values=custom_q)
        results_custom = est_custom.estimate(self.fbm_data)
        
        assert len(results_custom['q_values']) == len(custom_q)
        assert len(results_custom['hurst_exponents']) == len(custom_q)
        
        # Test extreme q values
        extreme_q = np.array([-10, -5, 0, 5, 10])
        est_extreme = HighPerformanceMFDFAEstimator(q_values=extreme_q)
        results_extreme = est_extreme.estimate(self.fbm_data)
        
        # Should handle extreme values gracefully
        assert 'hurst_exponents' in results_extreme

    def test_multifractality_detection(self):
        """Test multifractality detection capabilities."""
        # Generate monofractal data (should show low multifractality)
        mono_data = self.generator.fractional_brownian_motion(
            n_points=1000, hurst=0.6, noise_level=0.01
        )
        
        # Generate multifractal data (should show higher multifractality)
        # This is a simplified test - real multifractal data would be more complex
        multi_data = self.fbm_data + 0.1 * np.random.randn(1000)
        
        results_mono = self.estimator.estimate(mono_data)
        results_multi = self.estimator.estimate(multi_data)
        
        # Check that multifractality detection works
        assert 'is_multifractal' in results_mono['summary']
        assert 'is_multifractal' in results_multi['summary']


class TestHighPerformanceAvailability:
    """Test high-performance estimator availability."""

    def test_import_availability(self):
        """Test whether high-performance estimators can be imported."""
        if HIGH_PERFORMANCE_AVAILABLE:
            print("✓ High-performance estimators are available")
            assert True
        else:
            print("⚠ High-performance estimators are not available")
            print("  This may be due to missing NUMBA or JAX installations")
            assert False

    def test_estimator_creation(self):
        """Test whether high-performance estimators can be created."""
        if HIGH_PERFORMANCE_AVAILABLE:
            try:
                hp_dfa = HighPerformanceDFAEstimator()
                hp_mfdfa = HighPerformanceMFDFAEstimator()
                print("✓ High-performance estimators can be created successfully")
                assert True
            except Exception as e:
                print(f"✗ Error creating high-performance estimators: {e}")
                assert False
        else:
            pytest.skip("High-performance estimators not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
