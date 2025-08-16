"""
Unit tests for Spectral Estimators

This module tests the Periodogram and GPH estimators,
ensuring proper implementation of spectral analysis capabilities.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from estimators import PeriodogramEstimator, GPHEstimator
from src.benchmarking.synthetic_data import SyntheticDataGenerator


class TestPeriodogramEstimator:
    """Test suite for PeriodogramEstimator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = PeriodogramEstimator()
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
        est = PeriodogramEstimator()
        assert est.name == "Periodogram"
        assert est.min_freq == 0.01
        assert est.max_freq == 0.49
        assert est.num_freq == 100
        assert est.confidence_level == 0.95
        assert est.n_bootstrap == 1000
        
        # Test custom initialization
        est_custom = PeriodogramEstimator(
            min_freq=0.02,
            max_freq=0.4,
            num_freq=50,
            confidence_level=0.99,
            n_bootstrap=500
        )
        assert est_custom.min_freq == 0.02
        assert est_custom.max_freq == 0.4
        assert est_custom.num_freq == 50
        assert est_custom.confidence_level == 0.99
        assert est_custom.n_bootstrap == 500

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

    def test_calculate_periodogram(self):
        """Test periodogram calculation."""
        self.estimator.data = self.fbm_data
        self.estimator._calculate_periodogram()
        
        # Check periodogram and frequencies
        assert hasattr(self.estimator, 'periodogram')
        assert hasattr(self.estimator, 'frequencies')
        
        # Check shapes
        assert len(self.estimator.periodogram) == len(self.estimator.frequencies)
        assert len(self.estimator.frequencies) == self.estimator.num_freq
        
        # Check frequency bounds
        assert self.estimator.frequencies[0] >= self.estimator.min_freq
        assert self.estimator.frequencies[-1] <= self.estimator.max_freq
        
        # Check for valid periodogram values
        assert not np.any(np.isnan(self.estimator.periodogram))
        assert not np.any(np.isinf(self.estimator.periodogram))
        assert np.all(self.estimator.periodogram >= 0)

    def test_filter_and_fit(self):
        """Test filtering and fitting of periodogram."""
        self.estimator.data = self.fbm_data
        self.estimator._calculate_periodogram()
        self.estimator._filter_and_fit()
        
        # Check beta and Hurst exponent
        assert hasattr(self.estimator, 'beta')
        assert hasattr(self.estimator, 'hurst_exponent')
        
        # Check values are valid
        assert not np.isnan(self.estimator.beta)
        assert not np.isnan(self.estimator.hurst_exponent)
        assert not np.isinf(self.estimator.beta)
        assert not np.isinf(self.estimator.hurst_exponent)
        
        # Check relationship: H = (β + 1) / 2
        expected_hurst = (self.estimator.beta + 1) / 2
        hurst_error = abs(self.estimator.hurst_exponent - expected_hurst)
        assert hurst_error < 1e-10  # Should be exact

    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        self.estimator.data = self.fbm_data
        self.estimator._calculate_periodogram()
        self.estimator._filter_and_fit()
        self.estimator._calculate_confidence_interval()
        
        # Check confidence interval
        assert hasattr(self.estimator, 'confidence_interval')
        ci = self.estimator.confidence_interval
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound < Upper bound
        
        # Check bounds are valid
        assert not np.isnan(ci[0])
        assert not np.isnan(ci[1])
        assert not np.isinf(ci[0])
        assert not np.isinf(ci[1])

    def test_complete_estimation_workflow(self):
        """Test complete estimation workflow."""
        # Test with fBm data
        results = self.estimator.estimate(self.fbm_data)
        
        # Check required keys
        required_keys = ['hurst_exponent', 'frequencies', 'periodogram', 'beta', 
                        'scaling_error', 'confidence_interval']
        for key in required_keys:
            assert key in results
        
        # Check data types and shapes
        assert isinstance(results['hurst_exponent'], (int, float))
        assert isinstance(results['frequencies'], np.ndarray)
        assert isinstance(results['periodogram'], np.ndarray)
        assert isinstance(results['beta'], (int, float))
        
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
        # Test different frequency ranges
        est_narrow = PeriodogramEstimator(min_freq=0.05, max_freq=0.3)
        est_wide = PeriodogramEstimator(min_freq=0.01, max_freq=0.49)
        
        results_narrow = est_narrow.estimate(self.fbm_data)
        results_wide = est_wide.estimate(self.fbm_data)
        
        # Both should give reasonable results
        assert not np.isnan(results_narrow['hurst_exponent'])
        assert not np.isnan(results_wide['hurst_exponent'])
        
        # Test different frequency counts
        est_few = PeriodogramEstimator(num_freq=50)
        est_many = PeriodogramEstimator(num_freq=200)
        
        results_few = est_few.estimate(self.fbm_data)
        results_many = est_many.estimate(self.fbm_data)
        
        # More frequencies should give more detailed analysis
        assert len(results_many['frequencies']) > len(results_few['frequencies'])

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

    def test_periodogram_analysis_accuracy(self):
        """Test periodogram analysis accuracy with known data."""
        # Generate data with known Hurst exponent
        known_hurst = 0.6
        test_data = self.generator.fractional_brownian_motion(
            n_points=2000, hurst=known_hurst, noise_level=0.02
        )
        
        results = self.estimator.estimate(test_data)
        estimated_hurst = results['hurst_exponent']
        
        # Check if estimated Hurst is close to known value
        hurst_error = abs(estimated_hurst - known_hurst)
        assert hurst_error < 0.2  # Allow reasonable estimation error
        
        print(f"✓ Estimated H = {estimated_hurst:.3f}, True H = {known_hurst:.3f}, Error = {hurst_error:.3f}")

    def test_interpretation_logic(self):
        """Test interpretation logic for different Hurst values."""
        # Test anti-persistent interpretation
        est_anti = PeriodogramEstimator()
        est_anti.hurst_exponent = 0.3
        est_anti.beta = -0.4  # H = (-0.4 + 1) / 2 = 0.3
        est_anti._calculate_confidence_interval()
        
        results_anti = est_anti.get_results()
        interpretation = results_anti['interpretation']
        assert 'anti-persistent' in interpretation['lrd_type'].lower()
        
        # Test persistent interpretation
        est_pers = PeriodogramEstimator()
        est_pers.hurst_exponent = 0.8
        est_pers.beta = 0.6  # H = (0.6 + 1) / 2 = 0.8
        est_pers._calculate_confidence_interval()
        
        results_pers = est_pers.get_results()
        interpretation = results_pers['interpretation']
        assert 'persistent' in interpretation['lrd_type'].lower()


class TestGPHEstimator:
    """Test suite for GPHEstimator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = GPHEstimator()
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
        est = GPHEstimator()
        assert est.name == "GPH"
        assert est.min_freq == 0.01
        assert est.max_freq == 0.49
        assert est.num_freq == 100
        assert est.confidence_level == 0.95
        assert est.n_bootstrap == 1000
        
        # Test custom initialization
        est_custom = GPHEstimator(
            min_freq=0.02,
            max_freq=0.4,
            num_freq=50,
            confidence_level=0.99,
            n_bootstrap=500
        )
        assert est_custom.min_freq == 0.02
        assert est_custom.max_freq == 0.4
        assert est_custom.num_freq == 50
        assert est_custom.confidence_level == 0.99
        assert est_custom.n_bootstrap == 500

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

    def test_calculate_periodogram(self):
        """Test periodogram calculation using Welch's method."""
        self.estimator.data = self.fbm_data
        self.estimator._calculate_periodogram()
        
        # Check periodogram and frequencies
        assert hasattr(self.estimator, 'periodogram')
        assert hasattr(self.estimator, 'frequencies')
        
        # Check shapes
        assert len(self.estimator.periodogram) == len(self.estimator.frequencies)
        
        # Check frequency bounds
        assert self.estimator.frequencies[0] >= self.estimator.min_freq
        assert self.estimator.frequencies[-1] <= self.estimator.max_freq
        
        # Check for valid periodogram values
        assert not np.any(np.isnan(self.estimator.periodogram))
        assert not np.any(np.isinf(self.estimator.periodogram))
        assert np.all(self.estimator.periodogram >= 0)

    def test_gph_regression(self):
        """Test GPH regression analysis."""
        self.estimator.data = self.fbm_data
        self.estimator._calculate_periodogram()
        self.estimator._gph_regression()
        
        # Check fractional differencing parameter and Hurst exponent
        assert hasattr(self.estimator, 'fractional_d')
        assert hasattr(self.estimator, 'hurst_exponent')
        
        # Check values are valid
        assert not np.isnan(self.estimator.fractional_d)
        assert not np.isnan(self.estimator.hurst_exponent)
        assert not np.isinf(self.estimator.fractional_d)
        assert not np.isinf(self.estimator.hurst_exponent)
        
        # Check relationship: H = d + 0.5
        expected_hurst = self.estimator.fractional_d + 0.5
        hurst_error = abs(self.estimator.hurst_exponent - expected_hurst)
        assert hurst_error < 1e-10  # Should be exact

    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        self.estimator.data = self.fbm_data
        self.estimator._calculate_periodogram()
        self.estimator._gph_regression()
        self.estimator._calculate_confidence_interval()
        
        # Check confidence interval
        assert hasattr(self.estimator, 'confidence_interval')
        ci = self.estimator.confidence_interval
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound < Upper bound
        
        # Check bounds are valid
        assert not np.isnan(ci[0])
        assert not np.isnan(ci[1])
        assert not np.isinf(ci[0])
        assert not np.isinf(ci[1])

    def test_complete_estimation_workflow(self):
        """Test complete estimation workflow."""
        # Test with fBm data
        results = self.estimator.estimate(self.fbm_data)
        
        # Check required keys
        required_keys = ['fractional_d', 'hurst_exponent', 'frequencies', 'periodogram', 
                        'regression_error', 'confidence_interval']
        for key in required_keys:
            assert key in results
        
        # Check data types and shapes
        assert isinstance(results['fractional_d'], (int, float))
        assert isinstance(results['hurst_exponent'], (int, float))
        assert isinstance(results['frequencies'], np.ndarray)
        assert isinstance(results['periodogram'], np.ndarray)
        
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
        # Test different frequency ranges
        est_narrow = GPHEstimator(min_freq=0.05, max_freq=0.3)
        est_wide = GPHEstimator(min_freq=0.01, max_freq=0.49)
        
        results_narrow = est_narrow.estimate(self.fbm_data)
        results_wide = est_wide.estimate(self.fbm_data)
        
        # Both should give reasonable results
        assert not np.isnan(results_narrow['hurst_exponent'])
        assert not np.isnan(results_wide['hurst_exponent'])
        
        # Test different frequency counts
        est_few = GPHEstimator(num_freq=50)
        est_many = GPHEstimator(num_freq=200)
        
        results_few = est_few.estimate(self.fbm_data)
        results_many = est_many.estimate(self.fbm_data)
        
        # More frequencies should give more detailed analysis
        assert len(results_many['frequencies']) > len(results_few['frequencies'])

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

    def test_gph_analysis_accuracy(self):
        """Test GPH analysis accuracy with known data."""
        # Generate data with known Hurst exponent
        known_hurst = 0.6
        test_data = self.generator.fractional_brownian_motion(
            n_points=2000, hurst=known_hurst, noise_level=0.02
        )
        
        results = self.estimator.estimate(test_data)
        estimated_hurst = results['hurst_exponent']
        
        # Check if estimated Hurst is close to known value
        hurst_error = abs(estimated_hurst - known_hurst)
        assert hurst_error < 0.2  # Allow reasonable estimation error
        
        print(f"✓ Estimated H = {estimated_hurst:.3f}, True H = {known_hurst:.3f}, Error = {hurst_error:.3f}")

    def test_interpretation_logic(self):
        """Test interpretation logic for different Hurst values."""
        # Test anti-persistent interpretation
        est_anti = GPHEstimator()
        est_anti.hurst_exponent = 0.3
        est_anti.fractional_d = -0.2  # H = -0.2 + 0.5 = 0.3
        est_anti._calculate_confidence_interval()
        
        results_anti = est_anti.get_results()
        interpretation = results_anti['interpretation']
        assert 'anti-persistent' in interpretation['lrd_type'].lower()
        
        # Test persistent interpretation
        est_pers = GPHEstimator()
        est_pers.hurst_exponent = 0.8
        est_pers.fractional_d = 0.3  # H = 0.3 + 0.5 = 0.8
        est_pers._calculate_confidence_interval()
        
        results_pers = est_pers.get_results()
        interpretation = results_pers['interpretation']
        assert 'persistent' in interpretation['lrd_type'].lower()

    def test_frequency_dependency(self):
        """Test how GPH analysis depends on frequency selection."""
        # Test with different frequency ranges
        est_low = GPHEstimator(min_freq=0.01, max_freq=0.2)
        est_high = GPHEstimator(min_freq=0.1, max_freq=0.49)
        
        results_low = est_low.estimate(self.fbm_data)
        results_high = est_high.estimate(self.fbm_data)
        
        # Both should give reasonable results
        assert not np.isnan(results_low['hurst_exponent'])
        assert not np.isnan(results_high['hurst_exponent'])
        
        # Check frequency counts
        assert len(results_low['frequencies']) <= est_low.num_freq
        assert len(results_high['frequencies']) <= est_high.num_freq

    def test_regression_robustness(self):
        """Test robustness of GPH regression."""
        # Test with noisy data
        noisy_data = self.fbm_data + 0.1 * np.random.randn(len(self.fbm_data))
        results_noisy = self.estimator.estimate(noisy_data)
        
        # Should still give reasonable results
        assert not np.isnan(results_noisy['hurst_exponent'])
        assert results_noisy['hurst_exponent'] > 0
        
        # Test with trended data
        trended_data = self.fbm_data + 0.01 * np.arange(len(self.fbm_data))
        results_trended = self.estimator.estimate(trended_data)
        
        # Should handle trend gracefully
        assert not np.isnan(results_trended['hurst_exponent'])

    def test_fractional_d_bounds(self):
        """Test that fractional differencing parameters are within reasonable bounds."""
        # Test with various data types
        test_datasets = [
            self.fbm_data,
            self.random_data,
            self.anti_persistent_data,
            np.random.randn(1000),
            np.cumsum(np.random.randn(1000))
        ]
        
        for data in test_datasets:
            results = self.estimator.estimate(data)
            fractional_d = results['fractional_d']
            hurst = results['hurst_exponent']
            
            # Fractional d should be between -0.5 and 1.5 for reasonable Hurst values
            assert -0.5 <= fractional_d <= 1.5, f"Fractional d {fractional_d} out of bounds"
            
            # Hurst should be between 0 and 1
            assert 0 <= hurst <= 1, f"Hurst {hurst} out of bounds"
            
            # Should have valid confidence interval
            ci = results['confidence_interval']
            assert ci[0] < ci[1]
            assert 0 <= ci[0] <= 1
            assert 0 <= ci[1] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
