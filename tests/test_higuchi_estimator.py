"""
Unit tests for HiguchiEstimator

This module tests the Higuchi method estimator,
ensuring proper implementation of fractal dimension estimation.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from estimators import HiguchiEstimator
from src.benchmarking.synthetic_data import SyntheticDataGenerator


class TestHiguchiEstimator:
    """Test suite for HiguchiEstimator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = HiguchiEstimator()
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
        est = HiguchiEstimator()
        assert est.name == "Higuchi"
        assert est.min_k == 2
        assert est.max_k == None
        assert est.num_k == 20
        assert est.confidence_level == 0.95
        assert est.n_bootstrap == 1000
        
        # Test custom initialization
        est_custom = HiguchiEstimator(
            min_k=3,
            max_k=50,
            num_k=15,
            confidence_level=0.99,
            n_bootstrap=500
        )
        assert est_custom.min_k == 3
        assert est_custom.max_k == 50
        assert est_custom.num_k == 15
        assert est_custom.confidence_level == 0.99
        assert est_custom.n_bootstrap == 500

    def test_validate_data(self):
        """Test data validation."""
        # Test valid data
        self.estimator.data = self.fbm_data
        self.estimator._validate_data()
        
        # Test data too short
        with pytest.raises(ValueError, match="at least 50 points"):
            self.estimator.data = np.random.randn(49)
            self.estimator._validate_data()
        
        # Reset data for next test
        self.estimator.data = self.fbm_data
        
        # Test data with NaN
        with pytest.raises(ValueError, match="NaN or infinite values"):
            self.estimator.data = np.concatenate([np.random.randn(50), [np.nan]])
            self.estimator._validate_data()
        
        # Reset data for next test
        self.estimator.data = self.fbm_data
        
        # Test data with inf
        with pytest.raises(ValueError, match="NaN or infinite values"):
            self.estimator.data = np.concatenate([np.random.randn(50), [np.inf]])
            self.estimator._validate_data()

    def test_generate_k_values(self):
        """Test k-values generation."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_k_values()
        
        # Check k-values are within bounds
        assert len(self.estimator.k_values) <= self.estimator.num_k
        assert self.estimator.k_values[0] >= self.estimator.min_k
        assert self.estimator.k_values[-1] <= len(self.fbm_data) // 2
        
        # Check k-values are unique and sorted
        assert len(np.unique(self.estimator.k_values)) == len(self.estimator.k_values)
        assert np.all(np.diff(self.estimator.k_values) > 0)
        
        # Test custom max_k
        est_custom = HiguchiEstimator(max_k=30)
        est_custom.data = self.fbm_data
        est_custom._generate_k_values()
        assert est_custom.k_values[-1] <= 30

    def test_calculate_lengths(self):
        """Test length calculation."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_k_values()
        self.estimator._calculate_lengths()
        
        # Check lengths array
        assert len(self.estimator.lengths) == len(self.estimator.k_values)
        assert len(self.estimator.length_std) == len(self.estimator.k_values)
        
        # Check for valid length values
        assert not np.any(np.isnan(self.estimator.lengths))
        assert not np.any(np.isinf(self.estimator.lengths))
        assert np.all(self.estimator.lengths > 0)
        
        # Check standard deviations
        assert not np.any(np.isnan(self.estimator.length_std))
        assert not np.any(np.isinf(self.estimator.length_std))
        assert np.all(self.estimator.length_std >= 0)

    def test_calculate_length_for_k(self):
        """Test single k-value length calculation."""
        self.estimator.data = self.fbm_data
        k = 10
        start_idx = 0
        
        length = self.estimator._calculate_length_for_k(k, start_idx)
        
        # Check result is a positive number
        assert isinstance(length, (int, float))
        assert length > 0
        assert not np.isnan(length)
        assert not np.isinf(length)
        
        # Test with different k values
        for k in [5, 15, 25]:
            length = self.estimator._calculate_length_for_k(k, start_idx)
            assert length > 0
            assert not np.isnan(length)

    def test_fit_scaling_law(self):
        """Test scaling law fitting."""
        self.estimator.data = self.fbm_data
        self.estimator._generate_k_values()
        self.estimator._calculate_lengths()
        self.estimator._fit_scaling_law()
        
        # Check fractal dimension
        assert hasattr(self.estimator, 'fractal_dimension')
        assert isinstance(self.estimator.fractal_dimension, (int, float))
        assert not np.isnan(self.estimator.fractal_dimension)
        assert not np.isinf(self.estimator.fractal_dimension)
        
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
        self.estimator._generate_k_values()
        self.estimator._calculate_lengths()
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
        required_keys = ['fractal_dimension', 'k_values', 'lengths', 'length_std', 
                        'scaling_error', 'confidence_interval']
        for key in required_keys:
            assert key in results
        
        # Check data types and shapes
        assert isinstance(results['fractal_dimension'], (int, float))
        assert isinstance(results['k_values'], np.ndarray)
        assert isinstance(results['lengths'], np.ndarray)
        assert isinstance(results['length_std'], np.ndarray)
        
        # Check interpretation
        assert 'interpretation' in results
        interpretation = results['interpretation']
        assert 'complexity' in interpretation
        assert 'reliability' in interpretation
        assert 'method' in interpretation

    def test_estimation_with_different_data_types(self):
        """Test estimation with different types of time series data."""
        # Test with random walk data
        results_random = self.estimator.estimate(self.random_data)
        assert 'fractal_dimension' in results_random
        
        # Test with anti-persistent data
        results_anti = self.estimator.estimate(self.anti_persistent_data)
        assert 'fractal_dimension' in results_anti
        
        # Compare results - different data types should give different dimensions
        fbm_results = self.estimator.estimate(self.fbm_data)
        
        # All should have valid fractal dimensions
        assert not np.isnan(results_random['fractal_dimension'])
        assert not np.isnan(results_anti['fractal_dimension'])
        assert not np.isnan(fbm_results['fractal_dimension'])

    def test_parameter_effects(self):
        """Test how different parameters affect the estimation."""
        # Test different k ranges
        est_few_k = HiguchiEstimator(num_k=10)
        est_many_k = HiguchiEstimator(num_k=30)
        
        results_few = est_few_k.estimate(self.fbm_data)
        results_many = est_many_k.estimate(self.fbm_data)
        
        # More k values should give more detailed analysis
        assert len(results_many['k_values']) > len(results_few['k_values'])
        
        # Test different confidence levels
        est_90 = HiguchiEstimator(confidence_level=0.90)
        est_99 = HiguchiEstimator(confidence_level=0.99)
        
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
        assert 'fractal_dimension' in results_short
        
        # Test with very long data
        long_data = np.random.randn(10000)
        results_long = self.estimator.estimate(long_data)
        assert 'fractal_dimension' in results_long
        
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
        
        # Fractal dimensions should be similar (within reasonable tolerance)
        dim_diff = abs(results1['fractal_dimension'] - results2['fractal_dimension'])
        assert dim_diff < 0.2  # Allow some variation due to bootstrap
        
        # Confidence intervals should be similar
        ci1 = results1['confidence_interval']
        ci2 = results2['confidence_interval']
        ci_diff = abs((ci1[1] - ci1[0]) - (ci2[1] - ci2[0]))
        assert ci_diff < 0.3

    def test_higuchi_analysis_accuracy(self):
        """Test Higuchi analysis accuracy with known data."""
        # Generate data with known Hurst exponent
        known_hurst = 0.6
        test_data = self.generator.fractional_brownian_motion(
            n_points=2000, hurst=known_hurst, noise_level=0.02
        )
        
        results = self.estimator.estimate(test_data)
        estimated_dimension = results['fractal_dimension']
        
        # Check if estimated dimension is reasonable
        # For fBm, fractal dimension D = 2 - H
        expected_dimension = 2 - known_hurst
        dimension_error = abs(estimated_dimension - expected_dimension)
        assert dimension_error < 0.3  # Allow reasonable estimation error
        
        print(f"✓ Estimated D = {estimated_dimension:.3f}, Expected D = {expected_dimension:.3f}, Error = {dimension_error:.3f}")

    def test_interpretation_logic(self):
        """Test interpretation logic for different fractal dimensions."""
        # Test low complexity interpretation
        est_low = HiguchiEstimator()
        est_low.fractal_dimension = 1.2
        est_low.r_squared = 0.85
        est_low._calculate_confidence_interval()
        
        results_low = est_low.get_results()
        interpretation = results_low['interpretation']
        assert 'low' in interpretation['complexity'].lower()
        
        # Test high complexity interpretation
        est_high = HiguchiEstimator()
        est_high.fractal_dimension = 1.8
        est_high.r_squared = 0.90
        est_high._calculate_confidence_interval()
        
        results_high = est_high.get_results()
        interpretation = results_high['interpretation']
        assert 'high' in interpretation['complexity'].lower()

    def test_k_dependency(self):
        """Test how Higuchi analysis depends on k selection."""
        # Test with different k ranges
        est_narrow = HiguchiEstimator(min_k=5, max_k=25)
        est_wide = HiguchiEstimator(min_k=2, max_k=100)
        
        results_narrow = est_narrow.estimate(self.fbm_data)
        results_wide = est_wide.estimate(self.fbm_data)
        
        # Both should give reasonable results
        assert not np.isnan(results_narrow['fractal_dimension'])
        assert not np.isnan(results_wide['fractal_dimension'])
        
        # Check k counts
        assert len(results_narrow['k_values']) <= est_narrow.num_k
        assert len(results_wide['k_values']) <= est_wide.num_k

    def test_length_calculation_robustness(self):
        """Test robustness of length calculation."""
        # Test with noisy data
        noisy_data = self.fbm_data + 0.1 * np.random.randn(len(self.fbm_data))
        results_noisy = self.estimator.estimate(noisy_data)
        
        # Should still give reasonable results
        assert not np.isnan(results_noisy['fractal_dimension'])
        assert results_noisy['fractal_dimension'] > 0
        
        # Test with trended data
        trended_data = self.fbm_data + 0.01 * np.arange(len(self.fbm_data))
        results_trended = self.estimator.estimate(trended_data)
        
        # Should handle trend gracefully
        assert not np.isnan(results_trended['fractal_dimension'])

    def test_fractal_dimension_bounds(self):
        """Test that fractal dimensions are within reasonable bounds."""
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
            dimension = results['fractal_dimension']
            
            # Fractal dimension should be between 1 and 2 for 1D time series
            assert 1.0 <= dimension <= 2.0, f"Dimension {dimension} out of bounds for data type"
            
            # Should have valid confidence interval
            ci = results['confidence_interval']
            assert ci[0] < ci[1]
            assert 1.0 <= ci[0] <= 2.0
            assert 1.0 <= ci[1] <= 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
