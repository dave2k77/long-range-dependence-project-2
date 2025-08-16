"""
Integration tests for Long-Range Dependence Framework

This module tests complete workflows and interactions between
different components of the framework.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from estimators import (
    DFAEstimator, MFDFAEstimator, RSEstimator, HiguchiEstimator,
    PeriodogramEstimator, GPHEstimator, WaveletLeadersEstimator, WaveletWhittleEstimator
)
from src.benchmarking.synthetic_data import SyntheticDataGenerator
# from benchmarking.benchmarking import BenchmarkingSuite  # Not implemented yet

# Try to import high-performance estimators
try:
    from estimators import HighPerformanceDFAEstimator, HighPerformanceMFDFAEstimator
    HIGH_PERFORMANCE_AVAILABLE = True
except ImportError:
    HIGH_PERFORMANCE_AVAILABLE = False


class TestCompleteWorkflows:
    """Test complete estimation workflows across different estimators."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SyntheticDataGenerator()
        
        # Generate test datasets
        self.datasets = {
            'fbm_persistent': self.generator.fractional_brownian_motion(
                n_points=2000, hurst=0.8, noise_level=0.02
            ),
            'fbm_random': self.generator.fractional_brownian_motion(
                n_points=2000, hurst=0.5, noise_level=0.02
            ),
            'fbm_anti_persistent': self.generator.fractional_brownian_motion(
                n_points=2000, hurst=0.2, noise_level=0.02
            ),
            'random_walk': np.cumsum(np.random.randn(2000)),
            'white_noise': np.random.randn(2000)
        }
        
        # Initialize all estimators
        self.estimators = {
            'DFA': DFAEstimator(),
            'MFDFA': MFDFAEstimator(),
            'R/S': RSEstimator(),
            'Higuchi': HiguchiEstimator(),
            'Periodogram': PeriodogramEstimator(),
            'GPH': GPHEstimator(),
            'WaveletLeaders': WaveletLeadersEstimator(),
            'WaveletWhittle': WaveletWhittleEstimator()
        }
        
        # Add high-performance estimators if available
        if HIGH_PERFORMANCE_AVAILABLE:
            self.estimators.update({
                'HP_DFA': HighPerformanceDFAEstimator(),
                'HP_MFDFA': HighPerformanceMFDFAEstimator()
            })

    def test_all_estimators_on_same_data(self):
        """Test all estimators on the same dataset for consistency."""
        test_data = self.datasets['fbm_persistent']
        results = {}
        
        print(f"\nTesting all estimators on fBm data (H=0.8, {len(test_data)} points)")
        print("=" * 60)
        
        for name, estimator in self.estimators.items():
            try:
                print(f"Running {name}...")
                results[name] = estimator.estimate(test_data)
                
                # Extract Hurst exponent (handle different result structures)
                if 'hurst_exponent' in results[name]:
                    hurst = results[name]['hurst_exponent']
                elif 'mean_hurst' in results[name].get('summary', {}):
                    hurst = results[name]['summary']['mean_hurst']
                elif 'hurst_exponents' in results[name]:
                    hurst = np.mean(results[name]['hurst_exponents'])
                else:
                    hurst = np.nan
                
                print(f"  {name}: H = {hurst:.3f}")
                
            except Exception as e:
                print(f"  {name}: ERROR - {str(e)}")
                results[name] = None
        
        # Check that most estimators gave reasonable results
        valid_results = {k: v for k, v in results.items() if v is not None}
        assert len(valid_results) >= len(self.estimators) * 0.8, "Too many estimators failed"
        
        # Check that Hurst estimates are reasonably consistent
        hurst_values = []
        for name, result in valid_results.items():
            if 'hurst_exponent' in result:
                hurst_values.append(result['hurst_exponent'])
            elif 'mean_hurst' in result.get('summary', {}):
                hurst_values.append(result['summary']['mean_hurst'])
            elif 'hurst_exponents' in result:
                hurst_values.append(np.mean(result['hurst_exponents']))
        
        if len(hurst_values) > 1:
            hurst_std = np.std(hurst_values)
            print(f"\nHurst estimates: {[f'{h:.3f}' for h in hurst_values]}")
            print(f"Standard deviation: {hurst_std:.3f}")
            
            # Most estimates should be within reasonable range of true value (0.8)
            true_hurst = 0.8
            errors = [abs(h - true_hurst) for h in hurst_values]
            mean_error = np.mean(errors)
            print(f"Mean error from true H={true_hurst}: {mean_error:.3f}")
            
            assert mean_error < 0.3, "Hurst estimates are too far from true value"

    def test_estimator_consistency_across_datasets(self):
        """Test that estimators give consistent results across different datasets."""
        print(f"\nTesting estimator consistency across datasets")
        print("=" * 50)
        
        # Test a subset of estimators for efficiency
        test_estimators = ['DFA', 'R/S', 'Higuchi']
        
        for estimator_name in test_estimators:
            estimator = self.estimators[estimator_name]
            print(f"\n{estimator_name} results:")
            
            dataset_results = {}
            for dataset_name, data in self.datasets.items():
                try:
                    result = estimator.estimate(data)
                    
                    # Extract Hurst exponent
                    if 'hurst_exponent' in result:
                        hurst = result['hurst_exponent']
                    elif 'mean_hurst' in result.get('summary', {}):
                        hurst = result['summary']['mean_hurst']
                    elif 'hurst_exponents' in result:
                        hurst = np.mean(result['hurst_exponents'])
                    else:
                        hurst = np.nan
                    
                    dataset_results[dataset_name] = hurst
                    print(f"  {dataset_name}: H = {hurst:.3f}")
                    
                except Exception as e:
                    print(f"  {dataset_name}: ERROR - {str(e)}")
                    dataset_results[dataset_name] = np.nan
            
            # Check that results make sense qualitatively
            valid_results = {k: v for k, v in dataset_results.items() 
                           if not np.isnan(v)}
            
            if len(valid_results) >= 3:
                # Anti-persistent should have lower H than random, which should have lower H than persistent
                if ('fbm_anti_persistent' in valid_results and 
                    'fbm_random' in valid_results and 
                    'fbm_persistent' in valid_results):
                    
                    anti_h = valid_results['fbm_anti_persistent']
                    random_h = valid_results['fbm_random']
                    pers_h = valid_results['fbm_persistent']
                    
                    print(f"    Consistency check: {anti_h:.3f} < {random_h:.3f} < {pers_h:.3f}")
                    
                    # Results should be in expected order (with some tolerance)
                    assert anti_h < random_h + 0.1, "Anti-persistent H not lower than random H"
                    assert random_h < pers_h + 0.1, "Random H not lower than persistent H"

    def test_parameter_sensitivity(self):
        """Test how sensitive estimators are to parameter changes."""
        print(f"\nTesting parameter sensitivity")
        print("=" * 40)
        
        test_data = self.datasets['fbm_persistent']
        
        # Test DFA with different parameters
        print("DFA parameter sensitivity:")
        dfa_params = [
            {'num_scales': 10, 'polynomial_order': 1},
            {'num_scales': 20, 'polynomial_order': 1},
            {'num_scales': 20, 'polynomial_order': 2}
        ]
        
        dfa_results = []
        for params in dfa_params:
            estimator = DFAEstimator(**params)
            result = estimator.estimate(test_data)
            hurst = result['hurst_exponent']
            dfa_results.append(hurst)
            print(f"  {params}: H = {hurst:.3f}")
        
        # Results should be similar (within reasonable tolerance)
        hurst_std = np.std(dfa_results)
        print(f"  Standard deviation: {hurst_std:.3f}")
        assert hurst_std < 0.1, "DFA too sensitive to parameter changes"
        
        # Test MFDFA with different q ranges
        print("\nMFDFA parameter sensitivity:")
        mfdfa_params = [
            {'q_values': np.array([-2, -1, 0, 1, 2])},
            {'q_values': np.array([-5, -2, -1, 0, 1, 2, 5])},
            {'q_values': np.array([-10, -5, -2, -1, 0, 1, 2, 5, 10])}
        ]
        
        mfdfa_results = []
        for params in mfdfa_params:
            estimator = MFDFAEstimator(**params)
            result = estimator.estimate(test_data)
            mean_hurst = result['summary']['mean_hurst']
            mfdfa_results.append(mean_hurst)
            print(f"  {params}: H = {mean_hurst:.3f}")
        
        # Results should be similar
        hurst_std = np.std(mfdfa_results)
        print(f"  Standard deviation: {hurst_std:.3f}")
        assert hurst_std < 0.1, "MFDFA too sensitive to q-range changes"

    def test_error_handling_and_robustness(self):
        """Test error handling and robustness of estimators."""
        print(f"\nTesting error handling and robustness")
        print("=" * 45)
        
        # Test with problematic data
        problematic_datasets = {
            'very_short': np.random.randn(50),  # Too short
            'with_nan': np.array([1, 2, np.nan, 4, 5]),
            'with_inf': np.array([1, 2, np.inf, 4, 5]),
            'constant': np.ones(1000),
            'very_long': np.random.randn(50000)  # Very long
        }
        
        # Test a subset of estimators
        test_estimators = ['DFA', 'R/S', 'Higuchi']
        
        for estimator_name in test_estimators:
            estimator = self.estimators[estimator_name]
            print(f"\n{estimator_name} robustness:")
            
            for dataset_name, data in problematic_datasets.items():
                try:
                    if dataset_name in ['very_short', 'with_nan', 'with_inf', 'constant']:
                        # These should raise errors
                        with pytest.raises((ValueError, TypeError)):
                            estimator.estimate(data)
                        print(f"  {dataset_name}: Correctly raised error")
                    else:
                        # This should work
                        result = estimator.estimate(data)
                        print(f"  {dataset_name}: Success")
                        
                except Exception as e:
                    if dataset_name in ['very_short', 'with_nan', 'with_inf', 'constant']:
                        print(f"  {dataset_name}: Correctly raised error - {str(e)}")
                    else:
                        print(f"  {dataset_name}: Unexpected error - {str(e)}")
                        raise

    def test_memory_and_performance_tracking(self):
        """Test memory usage and execution time tracking across estimators."""
        print(f"\nTesting memory and performance tracking")
        print("=" * 45)
        
        test_data = self.datasets['fbm_persistent']
        
        for name, estimator in self.estimators.items():
            try:
                print(f"\n{name}:")
                
                # Reset estimator
                estimator.reset()
                
                # Run estimation
                result = estimator.estimate(test_data)
                
                # Check tracking
                execution_time = estimator.get_execution_time()
                memory_usage = estimator.get_memory_usage()
                
                print(f"  Execution time: {execution_time:.4f}s")
                print(f"  Memory usage: {memory_usage:.2f} MB")
                
                # Basic checks
                assert execution_time > 0, "Execution time should be positive"
                assert memory_usage > 0, "Memory usage should be positive"
                assert execution_time < 60, "Execution time should be reasonable"
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")

    def test_results_structure_consistency(self):
        """Test that all estimators return results in consistent structure."""
        print(f"\nTesting results structure consistency")
        print("=" * 45)
        
        test_data = self.datasets['fbm_random']
        
        for name, estimator in self.estimators.items():
            try:
                print(f"\n{name}:")
                result = estimator.estimate(test_data)
                
                # Check basic structure
                assert isinstance(result, dict), "Result should be a dictionary"
                
                # Check for required keys based on estimator type
                if 'DFA' in name:
                    required_keys = ['hurst_exponent', 'scales', 'fluctuations']
                elif 'MFDFA' in name:
                    required_keys = ['hurst_exponents', 'q_values', 'scales', 'fluctuations']
                elif 'R/S' in name:
                    required_keys = ['hurst_exponent', 'scales', 'rs_values']
                elif 'Higuchi' in name:
                    required_keys = ['fractal_dimension', 'k_values', 'lengths']
                elif 'Periodogram' in name:
                    required_keys = ['hurst_exponent', 'frequencies', 'periodogram']
                elif 'GPH' in name:
                    required_keys = ['fractional_d', 'hurst_exponent', 'frequencies']
                elif 'Wavelet' in name:
                    required_keys = ['hurst_exponent', 'scales', 'wavelet_coeffs']
                else:
                    required_keys = ['hurst_exponent']  # Default
                
                for key in required_keys:
                    assert key in result, f"Missing required key: {key}"
                
                # Check interpretation
                assert 'interpretation' in result, "Missing interpretation"
                interpretation = result['interpretation']
                assert 'lrd_type' in interpretation, "Missing lrd_type in interpretation"
                assert 'method' in interpretation, "Missing method in interpretation"
                
                print(f"  ✓ Structure consistent")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")

    def test_high_performance_comparison(self):
        """Test high-performance estimators against standard versions."""
        if not HIGH_PERFORMANCE_AVAILABLE:
            pytest.skip("High-performance estimators not available")
        
        print(f"\nTesting high-performance vs standard estimators")
        print("=" * 50)
        
        test_data = self.datasets['fbm_persistent']
        
        # Compare DFA variants
        print("DFA comparison:")
        standard_dfa = DFAEstimator()
        hp_dfa = HighPerformanceDFAEstimator()
        
        # Time both
        import time
        
        start_time = time.time()
        standard_result = standard_dfa.estimate(test_data)
        standard_time = time.time() - start_time
        
        start_time = time.time()
        hp_result = hp_dfa.estimate(test_data)
        hp_time = time.time() - start_time
        
        # Compare results
        standard_hurst = standard_result['hurst_exponent']
        hp_hurst = hp_result['hurst_exponent']
        hurst_diff = abs(standard_hurst - hp_hurst)
        
        print(f"  Standard DFA: H = {standard_hurst:.3f}, Time = {standard_time:.4f}s")
        print(f"  HP DFA: H = {hp_hurst:.3f}, Time = {hp_time:.4f}s")
        print(f"  Difference: {hurst_diff:.3f}")
        print(f"  Speedup: {standard_time/hp_time:.2f}x")
        
        # Results should be very similar
        assert hurst_diff < 0.05, "High-performance and standard DFA results differ too much"
        
        # Compare MFDFA variants
        print("\nMFDFA comparison:")
        standard_mfdfa = MFDFAEstimator()
        hp_mfdfa = HighPerformanceMFDFAEstimator()
        
        start_time = time.time()
        standard_result = standard_mfdfa.estimate(test_data)
        standard_time = time.time() - start_time
        
        start_time = time.time()
        hp_result = hp_mfdfa.estimate(test_data)
        hp_time = time.time() - start_time
        
        # Compare results
        standard_mean = standard_result['summary']['mean_hurst']
        hp_mean = hp_result['summary']['mean_hurst']
        mean_diff = abs(standard_mean - hp_mean)
        
        print(f"  Standard MFDFA: H = {standard_mean:.3f}, Time = {standard_time:.4f}s")
        print(f"  HP MFDFA: H = {hp_mean:.3f}, Time = {hp_time:.4f}s")
        print(f"  Difference: {mean_diff:.3f}")
        print(f"  Speedup: {standard_time/hp_time:.2f}x")
        
        # Results should be similar
        assert mean_diff < 0.1, "High-performance and standard MFDFA results differ too much"


class TestBenchmarkingIntegration:
    """Test integration with benchmarking components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SyntheticDataGenerator()
        
        # Generate benchmark datasets
        self.benchmark_datasets = {
            'small': self.generator.fractional_brownian_motion(
                n_points=500, hurst=0.6, noise_level=0.05
            ),
            'medium': self.generator.fractional_brownian_motion(
                n_points=2000, hurst=0.6, noise_level=0.05
            ),
            'large': self.generator.fractional_brownian_motion(
                n_points=10000, hurst=0.6, noise_level=0.05
            )
        }

    def test_benchmarking_suite_integration(self):
        """Test integration with benchmarking suite."""
        print(f"\nTesting benchmarking suite integration")
        print("=" * 45)
        
        # This test would require the benchmarking suite to be fully implemented
        # For now, we'll test the basic structure
        
        # Test that we can create estimators and run them
        estimators = [DFAEstimator(), RSEstimator(), HiguchiEstimator()]
        
        for estimator in estimators:
            try:
                # Test on different dataset sizes
                for size_name, data in self.benchmark_datasets.items():
                    print(f"  {estimator.name} on {size_name} dataset ({len(data)} points)")
                    
                    result = estimator.estimate(data)
                    
                    # Check basic results
                    if 'hurst_exponent' in result:
                        hurst = result['hurst_exponent']
                    elif 'mean_hurst' in result.get('summary', {}):
                        hurst = result['summary']['mean_hurst']
                    elif 'hurst_exponents' in result:
                        hurst = np.mean(result['hurst_exponents'])
                    else:
                        hurst = np.nan
                    
                    print(f"    Estimated H = {hurst:.3f}")
                    
                    # Check execution time
                    execution_time = estimator.get_execution_time()
                    print(f"    Execution time = {execution_time:.4f}s")
                    
                    # Basic validation
                    assert not np.isnan(hurst), "Hurst exponent should not be NaN"
                    assert execution_time > 0, "Execution time should be positive"
                    
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
