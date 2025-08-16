"""
Tests for BaseEstimator class.

This module tests the abstract base class that all LRD estimators
must inherit from, ensuring consistent interface and functionality.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from src.estimators.base import BaseEstimator


class MockEstimator(BaseEstimator):
    """Mock estimator for testing BaseEstimator functionality."""
    
    def fit(self, data, **kwargs):
        self.data = data
        return self
        
    def estimate(self, data=None, **kwargs):
        if data is not None:
            self.fit(data, **kwargs)
        return {'result': 'test', 'data_length': len(self.data)}


class TestBaseEstimator:
    """Test suite for BaseEstimator class."""
    
    def test_init(self):
        """Test estimator initialization."""
        estimator = MockEstimator(name="TestEstimator", param1=10, param2="test")
        
        assert estimator.name == "TestEstimator"
        assert estimator.parameters == {'param1': 10, 'param2': 'test'}
        assert estimator.results == {}
        assert estimator.execution_time is None
        assert estimator.memory_usage is None
        
    def test_init_default_name(self):
        """Test estimator initialization with default name."""
        estimator = MockEstimator()
        assert estimator.name == "MockEstimator"
        
    def test_fit_estimate(self):
        """Test fit_estimate method."""
        estimator = MockEstimator()
        data = np.random.randn(100)
        
        start_time = time.time()
        results = estimator.fit_estimate(data)
        end_time = time.time()
        
        assert results['result'] == 'test'
        assert results['data_length'] == 100
        assert estimator.execution_time is not None
        assert estimator.execution_time > 0
        assert estimator.execution_time <= (end_time - start_time) + 0.1
        assert estimator.results == results
        
    def test_fit_estimate_error_handling(self):
        """Test fit_estimate error handling."""
        estimator = MockEstimator()
        
        # Mock the fit method to raise an exception
        with patch.object(estimator, 'fit', side_effect=ValueError("Test error")):
            with pytest.raises(ValueError, match="Test error"):
                estimator.fit_estimate(np.random.randn(100))
                
    def test_get_results(self):
        """Test get_results method."""
        estimator = MockEstimator()
        estimator.results = {'test': 'value', 'number': 42}
        
        results = estimator.get_results()
        assert results == {'test': 'value', 'number': 42}
        
    def test_get_execution_time(self):
        """Test get_execution_time method."""
        estimator = MockEstimator()
        
        # Initially None
        assert estimator.get_execution_time() is None
        
        # After fit_estimate
        data = np.random.randn(100)
        estimator.fit_estimate(data)
        
        execution_time = estimator.get_execution_time()
        assert execution_time is not None
        assert execution_time > 0
        
    def test_get_memory_usage(self):
        """Test get_memory_usage method."""
        estimator = MockEstimator()
        
        # Initially None
        assert estimator.get_memory_usage() is None
        
        # Set memory usage
        estimator.memory_usage = 1024
        assert estimator.get_memory_usage() == 1024
        
    def test_reset(self):
        """Test reset method."""
        estimator = MockEstimator()
        
        # Set some state
        estimator.results = {'test': 'value'}
        estimator.execution_time = 1.5
        estimator.memory_usage = 1024
        estimator.data = np.random.randn(100)
        
        # Reset
        estimator.reset()
        
        assert estimator.results == {}
        assert estimator.execution_time is None
        assert estimator.memory_usage is None
        assert estimator.data is None
        
    def test_abstract_methods(self):
        """Test that abstract methods are properly defined."""
        # Should not be able to instantiate BaseEstimator directly
        with pytest.raises(TypeError):
            BaseEstimator()
            
    def test_estimator_inheritance(self):
        """Test that MockEstimator properly inherits from BaseEstimator."""
        estimator = MockEstimator()
        
        assert isinstance(estimator, BaseEstimator)
        assert hasattr(estimator, 'fit')
        assert hasattr(estimator, 'estimate')
        assert hasattr(estimator, 'fit_estimate')
        assert hasattr(estimator, 'get_results')
        assert hasattr(estimator, 'get_execution_time')
        assert hasattr(estimator, 'reset')
        
    def test_parameter_storage(self):
        """Test parameter storage and retrieval."""
        estimator = MockEstimator(
            scale_min=4,
            scale_max=100,
            num_scales=20,
            custom_param="test_value"
        )
        
        assert estimator.parameters['scale_min'] == 4
        assert estimator.parameters['scale_max'] == 100
        assert estimator.parameters['num_scales'] == 20
        assert estimator.parameters['custom_param'] == "test_value"
        
    def test_data_validation_in_fit(self):
        """Test that data is properly stored in fit method."""
        estimator = MockEstimator()
        data = np.random.randn(200)
        
        estimator.fit(data)
        assert estimator.data is data
        
    def test_data_validation_in_estimate(self):
        """Test that data is properly handled in estimate method."""
        estimator = MockEstimator()
        data = np.random.randn(200)
        
        results = estimator.estimate(data)
        assert estimator.data is data
        assert results['data_length'] == 200
        
    def test_multiple_fit_calls(self):
        """Test multiple fit calls behavior."""
        estimator = MockEstimator()
        data1 = np.random.randn(100)
        data2 = np.random.randn(200)
        
        # First fit
        estimator.fit(data1)
        assert estimator.data is data1
        
        # Second fit
        estimator.fit(data2)
        assert estimator.data is data2
        
    def test_parameter_modification(self):
        """Test parameter modification after initialization."""
        estimator = MockEstimator(param1=10)
        assert estimator.parameters['param1'] == 10
        
        # Modify parameter
        estimator.parameters['param1'] = 20
        assert estimator.parameters['param1'] == 20
        
    def test_name_modification(self):
        """Test name modification after initialization."""
        estimator = MockEstimator(name="OriginalName")
        assert estimator.name == "OriginalName"
        
        # Modify name
        estimator.name = "ModifiedName"
        assert estimator.name == "ModifiedName"
        
    def test_results_modification(self):
        """Test results modification."""
        estimator = MockEstimator()
        assert estimator.results == {}
        
        # Add results
        estimator.results['new_result'] = 'new_value'
        assert estimator.results['new_result'] == 'new_value'
        
        # Modify existing result
        estimator.results['new_result'] = 'modified_value'
        assert estimator.results['new_result'] == 'modified_value'
        
    def test_execution_time_accuracy(self):
        """Test execution time measurement accuracy."""
        estimator = MockEstimator()
        data = np.random.randn(100)
        
        # Measure time manually
        start_time = time.time()
        results = estimator.fit_estimate(data)
        manual_time = time.time() - start_time
        
        # Check that estimator's time is reasonable
        estimator_time = estimator.get_execution_time()
        assert abs(estimator_time - manual_time) < 0.1  # Allow 100ms tolerance
        
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        estimator = MockEstimator()
        
        # Initially None
        assert estimator.get_memory_usage() is None
        
        # Set memory usage
        estimator.memory_usage = 2048
        assert estimator.get_memory_usage() == 2048
        
        # Reset should clear memory usage
        estimator.reset()
        assert estimator.get_memory_usage() is None
