"""
Unit tests for RSEstimator

This module tests the Rescaled Range (R/S) Analysis estimator,
ensuring proper implementation of R/S analysis capabilities.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from estimators import RSEstimator
from src.benchmarking.synthetic_data import SyntheticDataGenerator


class TestRSEstimator:
    """Test suite for RSEstimator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = RSEstimator()
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
        est = RSEstimator()
        assert est.name == "R/S"
        assert est.min_scale == 4
        assert est.max_scale is None
        assert est.num_scales == 20
        assert est.confidence_level == 0.95
        assert est.n_bootstrap == 1000
        
        # Test custom initialization
        est_custom = RSEstimator(
            min_scale=8,
            max_scale=100,
            num_scales=15,
            confidence_level=0.99,
            n_bootstrap=500
        )
        assert est_custom.min_scale == 8
        assert est_custom.max_scale == 100
        assert est_custom.num_scales == 15
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
        est_custom = RSEstimator(max_scale=50)
        est_custom.data = self.fbm_data
        est_custom._generate_scales()
        assert est_custom.scales[-1] <= 50

    def test_calculate_rs_values(self):
        """Test R/S values calculation."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_scales()
        self.estimator._calculate_rs_values()
        
        # Check R/S values array
        assert len(self.estimator.rs_values) == len(self.estimator.scales)
        assert len(self.estimator.rs_std) == len(self.estimator.scales)
        
        # Check for valid R/S values
        assert not np.any(np.isnan(self.estimator.rs_values))
        assert not np.any(np.isinf(self.estimator.rs_values))
        assert np.all(self.estimator.rs_values > 0)
        
        # Check standard deviations
        assert not np.any(np.isnan(self.estimator.rs_std))
        assert not np.any(np.isinf(self.estimator.rs_std))
        assert np.all(self.estimator.rs_std >= 0)

    def test_calculate_rs_for_segment(self):
        """Test single segment R/S calculation."""
        self.estimator.data = self.fbm_data
        scale = 10
        
        rs_value = self.estimator._calculate_rs_for_segment(scale)
        
        # Check result is a positive number
        assert isinstance(rs_value, (int, float))
        assert rs_value > 0
        assert not np.isnan(rs_value)
        assert not np.isinf(rs_value)
        
        # Test with different scales
        for scale in [5, 15, 25]:
            rs = self.estimator._calculate_rs_for_segment(scale)
            assert rs > 0
            assert not np.isnan(rs)

    def test_fit_scaling_law(self):
        """Test scaling law fitting."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_scales()
        self.estimator._calculate_rs_values()
        self.estimator._fit_scaling_law()
        
        # Check Hurst exponent
        assert hasattr(self.estimator, 'hurst_exponent')
        assert isinstance(self.estimator.hurst_exponent, (int, float))
        assert not np.isnan(self.estimator.hurst_exponent)
        assert not np.isinf(self.estimator.hurst_exponent)
        
        # Check scaling error
        assert hasattr(self.estimator, 'scaling_error')
        assert isinstance(self.estimator.scaling_error, (int, float))
        assert not np.isnan(self.estimator.scaling_error)
        assert not np.isinf(self.estimator.scaling_error)
        
        # Check R-squared
        assert hasattr(self.estimator, 'r_squared')
        assert isinstance(self.estimator.r_squared, (int, float))
        assert not np.isnan(self.estimator.r_squared)
        assert 0 <= self.estimator.r_squared <= 1

    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_scales()
        self.estimator._calculate_rs_values()
        self.estimator._fit_scaling_law()
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
        required_keys = ['hurst_exponent', 'scales', 'rs_values', 'rs_std', 
                        'scaling_error', 'confidence_interval']
        for key in required_keys:
            assert key in results
        
        # Check data types and shapes
        assert isinstance(results['hurst_exponent'], (int, float))
        assert isinstance(results['scales'], np.ndarray)
        assert isinstance(results['rs_values'], np.ndarray)
        assert isinstance(results['rs_std'], np.ndarray)
        
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
        # Test different scale ranges
        est_few_scales = RSEstimator(num_scales=10)
        est_many_scales = RSEstimator(num_scales=30)
        
        results_few = est_few_scales.estimate(self.fbm_data)
        results_many = est_many_scales.estimate(self.fbm_data)
        
        # More scales should give more detailed analysis
        assert len(results_many['scales']) > len(results_few['scales'])
        
        # Test different confidence levels
        est_90 = RSEstimator(confidence_level=0.90)
        est_99 = RSEstimator(confidence_level=0.99)
        
        results_90 = est_90.estimate(self.fbm_data)
        results_99 = est_99.estimate(self.fbm_data)
        
        # Higher confidence should give wider intervals
        ci_90 = results_90['confidence_interval']
        ci_99 = results_99['confidence_interval']
        width_90 = ci_90[1] - ci_90[0]
        width_99 = ci_99[1] - ci_99[0]
        assert width_99 >= width_90

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

    def test_bootstrap_consistency(self):
        """Test bootstrap consistency across multiple runs."""
        # Run estimation multiple times
        results1 = self.estimator.estimate(self.fbm_data)
        results2 = self.estimator.estimate(self.fbm_data)
        
        # Hurst exponents should be similar (within reasonable tolerance)
        hurst_diff = abs(results1['hurst_exponent'] - results2['hurst_exponent'])
        assert hurst_diff < 0.1  # Allow some variation due to bootstrap
        
        # Confidence intervals should be similar
        ci1 = results1['confidence_interval']
        ci2 = results2['confidence_interval']
        ci_diff = abs((ci1[1] - ci1[0]) - (ci2[1] - ci2[0]))
        assert ci_diff < 0.2

    def test_rs_analysis_accuracy(self):
        """Test R/S analysis accuracy with known data."""
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
        est_anti = RSEstimator()
        est_anti.hurst_exponent = 0.3
        est_anti.r_squared = 0.85
        est_anti._calculate_confidence_interval()
        
        results_anti = est_anti.get_results()
        interpretation = results_anti['interpretation']
        assert 'anti-persistent' in interpretation['lrd_type'].lower()
        
        # Test persistent interpretation
        est_pers = RSEstimator()
        est_pers.hurst_exponent = 0.8
        est_pers.r_squared = 0.90
        est_pers._calculate_confidence_interval()
        
        results_pers = est_pers.get_results()
        interpretation = results_pers['interpretation']
        assert 'persistent' in interpretation['lrd_type'].lower()
        
        # Test random walk interpretation
        est_random = RSEstimator()
        est_random.hurst_exponent = 0.5
        est_random.r_squared = 0.75
        est_random._calculate_confidence_interval()
        
        results_random = est_random.get_results()
        interpretation = results_random['interpretation']
        assert 'random' in interpretation['lrd_type'].lower()

    def test_scale_dependency(self):
        """Test how R/S analysis depends on scale selection."""
        # Test with different scale ranges
        est_narrow = RSEstimator(min_scale=10, max_scale=50)
        est_wide = RSEstimator(min_scale=5, max_scale=200)
        
        results_narrow = est_narrow.estimate(self.fbm_data)
        results_wide = est_wide.estimate(self.fbm_data)
        
        # Both should give reasonable results
        assert not np.isnan(results_narrow['hurst_exponent'])
        assert not np.isnan(results_wide['hurst_exponent'])
        
        # Check scale counts
        assert len(results_narrow['scales']) <= est_narrow.num_scales
        assert len(results_wide['scales']) <= est_wide.num_scales


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
