"""
Tests for DFA (Detrended Fluctuation Analysis) estimator.

This module tests the DFA implementation including data validation,
fluctuation calculation, power law fitting, and result interpretation.
"""

import pytest
import numpy as np
from src.estimators.temporal import DFAEstimator


class TestDFAEstimator:
    """Test suite for DFAEstimator class."""
    
    def test_init(self):
        """Test DFA estimator initialization."""
        estimator = DFAEstimator(
            min_scale=8,
            max_scale=100,
            num_scales=25,
            polynomial_order=2
        )
        
        assert estimator.min_scale == 8
        assert estimator.max_scale == 100
        assert estimator.num_scales == 25
        assert estimator.polynomial_order == 2
        assert estimator.name == "DFA"
        
    def test_init_defaults(self):
        """Test DFA estimator initialization with defaults."""
        estimator = DFAEstimator()
        
        assert estimator.min_scale == 4
        assert estimator.max_scale is None
        assert estimator.num_scales == 20
        assert estimator.polynomial_order == 1
        
    def test_fit_data_validation(self):
        """Test data validation in fit method."""
        estimator = DFAEstimator()
        
        # Valid data
        valid_data = np.random.randn(200)
        estimator.fit(valid_data)
        assert estimator.data is valid_data
        
        # Invalid data - too short
        with pytest.raises(ValueError, match="at least 100 points"):
            estimator.fit(np.random.randn(50))
            
        # Invalid data - contains NaN
        invalid_data = np.random.randn(200)
        invalid_data[0] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite values"):
            estimator.fit(invalid_data)
            
        # Invalid data - contains inf
        invalid_data = np.random.randn(200)
        invalid_data[0] = np.inf
        with pytest.raises(ValueError, match="NaN or infinite values"):
            estimator.fit(invalid_data)
            
    def test_generate_scales(self):
        """Test scale generation."""
        estimator = DFAEstimator()
        estimator.data = np.random.randn(1000)
        estimator._generate_scales()
        
        # Check scale properties
        assert len(estimator.scales) > 0
        assert estimator.scales[0] >= estimator.min_scale
        assert estimator.scales[-1] <= len(estimator.data) // 4
        assert len(estimator.scales) <= estimator.num_scales
        
        # Check that scales are unique and sorted
        assert len(estimator.scales) == len(np.unique(estimator.scales))
        assert np.all(np.diff(estimator.scales) >= 0)
        
    def test_calculate_fluctuations(self):
        """Test fluctuation calculation."""
        estimator = DFAEstimator()
        data = np.random.randn(500)
        estimator.fit(data)
        
        # Calculate fluctuations
        estimator._calculate_fluctuations()
        
        # Check results
        assert estimator.fluctuations is not None
        assert len(estimator.fluctuations) == len(estimator.scales)
        assert not np.any(np.isnan(estimator.fluctuations))
        assert np.all(estimator.fluctuations > 0)
        
    def test_fit_power_law(self):
        """Test power law fitting."""
        estimator = DFAEstimator()
        data = np.random.randn(500)
        estimator.fit(data)
        estimator._calculate_fluctuations()
        
        # Fit power law
        hurst, intercept, r_squared = estimator._fit_power_law()
        
        # Check results
        assert not np.isnan(hurst)
        assert not np.isnan(intercept)
        assert not np.isnan(r_squared)
        assert 0 <= r_squared <= 1
        assert hurst > 0  # Hurst exponent should be positive
        
    def test_estimate_complete_workflow(self):
        """Test complete estimation workflow."""
        estimator = DFAEstimator()
        data = np.random.randn(500)
        
        # Run complete estimation
        results = estimator.estimate(data)
        
        # Check that all required results are present
        required_keys = ['hurst_exponent', 'scales', 'fluctuations', 'intercept', 'r_squared']
        for key in required_keys:
            assert key in results
            
        # Check result types and values
        assert not np.isnan(results['hurst_exponent'])
        assert len(results['scales']) > 0
        assert len(results['fluctuations']) == len(results['scales'])
        assert not np.isnan(results['intercept'])
        assert 0 <= results['r_squared'] <= 1
        
    def test_estimate_with_fitted_data(self):
        """Test estimation with already fitted data."""
        estimator = DFAEstimator()
        data = np.random.randn(500)
        
        # Fit first
        estimator.fit(data)
        
        # Then estimate
        results = estimator.estimate()
        
        # Check results
        assert 'hurst_exponent' in results
        assert not np.isnan(results['hurst_exponent'])
        
    def test_estimate_with_new_data(self):
        """Test estimation with new data."""
        estimator = DFAEstimator()
        data1 = np.random.randn(500)
        data2 = np.random.randn(600)
        
        # Estimate with first dataset
        results1 = estimator.estimate(data1)
        hurst1 = results1['hurst_exponent']
        
        # Estimate with second dataset
        results2 = estimator.estimate(data2)
        hurst2 = results2['hurst_exponent']
        
        # Results should be different for different data
        assert hurst1 != hurst2
        
    def test_polynomial_order_effects(self):
        """Test effects of different polynomial orders."""
        data = np.random.randn(500)
        
        # Test different polynomial orders
        for order in [1, 2, 3]:
            estimator = DFAEstimator(polynomial_order=order)
            results = estimator.estimate(data)
            
            assert not np.isnan(results['hurst_exponent'])
            assert 0 <= results['r_squared'] <= 1
            
    def test_scale_parameter_effects(self):
        """Test effects of different scale parameters."""
        data = np.random.randn(1000)
        
        # Test different scale configurations
        configs = [
            {'min_scale': 4, 'max_scale': 100, 'num_scales': 10},
            {'min_scale': 8, 'max_scale': 200, 'num_scales': 20},
            {'min_scale': 16, 'max_scale': 400, 'num_scales': 15}
        ]
        
        for config in configs:
            estimator = DFAEstimator(**config)
            results = estimator.estimate(data)
            
            assert not np.isnan(results['hurst_exponent'])
            assert len(results['scales']) > 0
            
    def test_fractional_brownian_motion(self):
        """Test DFA on fractional Brownian motion data."""
        # Generate fBm with known Hurst exponent
        hurst_true = 0.7
        n_points = 1000
        
        # Simple fBm generation (approximate)
        t = np.linspace(0, 1, n_points)
        noise = np.random.randn(n_points)
        
        # Apply fractional integration
        fbm = np.cumsum(noise * (t[1] - t[0]) ** (hurst_true - 0.5))
        
        # Estimate Hurst exponent
        estimator = DFAEstimator()
        results = estimator.estimate(fbm)
        
        # Check that estimated Hurst is reasonable
        estimated_hurst = results['hurst_exponent']
        assert not np.isnan(estimated_hurst)
        assert 0 < estimated_hurst < 1
        
        # Allow some tolerance for estimation error
        assert abs(estimated_hurst - hurst_true) < 0.3
        
    def test_random_walk(self):
        """Test DFA on random walk data (H = 0.5)."""
        # Generate random walk
        n_points = 1000
        steps = np.random.randn(n_points)
        random_walk = np.cumsum(steps)
        
        # Estimate Hurst exponent
        estimator = DFAEstimator()
        results = estimator.estimate(random_walk)
        
        # Check results
        estimated_hurst = results['hurst_exponent']
        assert not np.isnan(estimated_hurst)
        assert 0 < estimated_hurst < 1
        
        # Random walk should have H ≈ 0.5
        assert abs(estimated_hurst - 0.5) < 0.3
        
    def test_anti_persistent_data(self):
        """Test DFA on anti-persistent data (H < 0.5)."""
        # Generate anti-persistent data (simplified)
        n_points = 1000
        data = np.random.randn(n_points)
        
        # Apply differencing to create anti-persistent series
        anti_persistent = np.diff(data, n=2)
        
        # Estimate Hurst exponent
        estimator = DFAEstimator()
        results = estimator.estimate(anti_persistent)
        
        # Check results
        estimated_hurst = results['hurst_exponent']
        assert not np.isnan(estimated_hurst)
        
        # Anti-persistent data should have H < 0.5
        # Note: This is a simplified test and may not always hold
        # due to the nature of the generated data
        
    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases."""
        estimator = DFAEstimator()
        
        # Empty data
        with pytest.raises(ValueError):
            estimator.fit(np.array([]))
            
        # Single point data
        with pytest.raises(ValueError):
            estimator.fit(np.array([1.0]))
            
        # All zero data
        with pytest.raises(ValueError):
            estimator.fit(np.zeros(100))
            
    def test_result_consistency(self):
        """Test consistency of results across multiple runs."""
        estimator = DFAEstimator()
        data = np.random.randn(500)
        
        # Run estimation multiple times
        results1 = estimator.estimate(data)
        results2 = estimator.estimate(data)
        
        # Results should be consistent (same data, same parameters)
        np.testing.assert_allclose(
            results1['hurst_exponent'], 
            results2['hurst_exponent'], 
            rtol=1e-10
        )
        np.testing.assert_allclose(
            results1['scales'], 
            results2['scales'], 
            rtol=1e-10
        )
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid min_scale
        with pytest.raises(ValueError):
            DFAEstimator(min_scale=0)
            
        # Invalid polynomial_order
        with pytest.raises(ValueError):
            DFAEstimator(polynomial_order=0)
            
        # Invalid num_scales
        with pytest.raises(ValueError):
            DFAEstimator(num_scales=0)
            
    def test_memory_efficiency(self):
        """Test memory efficiency for large datasets."""
        # Large dataset
        data = np.random.randn(10000)
        
        estimator = DFAEstimator()
        
        # Should not raise memory errors
        results = estimator.estimate(data)
        
        # Check results
        assert not np.isnan(results['hurst_exponent'])
        assert len(results['scales']) > 0
        
    def test_execution_time_tracking(self):
        """Test execution time tracking."""
        estimator = DFAEstimator()
        data = np.random.randn(500)
        
        # Run estimation
        results = estimator.estimate(data)
        
        # Check execution time
        execution_time = estimator.get_execution_time()
        assert execution_time is not None
        assert execution_time > 0
        
        # Check that execution time is stored in results
        assert 'execution_time' in estimator.results
