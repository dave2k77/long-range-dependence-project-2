"""
Benchmark Runner for Long-Range Dependence Estimators

This module provides the main benchmarking framework that runs performance
tests on all available estimators and generates comprehensive reports.
"""

import numpy as np
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

# Try relative import first, fall back to absolute if needed
try:
    from ..estimators import (
        DFAEstimator, MFDFAEstimator, RSEstimator, HiguchiEstimator,
        WhittleMLEEstimator, PeriodogramEstimator, GPHEstimator,
        WaveletLeadersEstimator, WaveletWhittleEstimator
    )
except ImportError:
    # Fall back to absolute imports
    from src.estimators import (
        DFAEstimator, MFDFAEstimator, RSEstimator, HiguchiEstimator,
        WhittleMLEEstimator, PeriodogramEstimator, GPHEstimator,
        WaveletLeadersEstimator, WaveletWhittleEstimator
    )
from .synthetic_data import SyntheticDataGenerator
from .performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Main benchmark runner for long-range dependence estimators.
    
    This class orchestrates comprehensive performance testing of all
    available estimators across different datasets and conditions.
    """
    
    def __init__(self, output_dir: str = "benchmarks", n_jobs: int = -1):
        """
        Initialize the benchmark runner.
        
        Parameters
        ----------
        output_dir : str
            Directory to save benchmark results
        n_jobs : int
            Number of parallel jobs (-1 for all available cores)
        """
        self.output_dir = output_dir
        self.n_jobs = n_jobs if n_jobs > 0 else psutil.cpu_count()
        
        # Initialize estimators
        self.estimators = self._initialize_estimators()
        
        # Initialize synthetic data generator
        self.data_generator = SyntheticDataGenerator()
        
        # Initialize performance metrics
        self.metrics = PerformanceMetrics()
        
        # Benchmark results storage
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Benchmark runner initialized with {self.n_jobs} parallel jobs")
    
    def _initialize_estimators(self) -> Dict[str, Any]:
        """Initialize all available estimators."""
        estimators = {
            'DFA': DFAEstimator(),
            'MFDFA': MFDFAEstimator(),
            'R/S': RSEstimator(),
            'Higuchi': HiguchiEstimator(),
            'WhittleMLE': WhittleMLEEstimator(),
            'Periodogram': PeriodogramEstimator(),
            'GPH': GPHEstimator(),
            'WaveletLeaders': WaveletLeadersEstimator(),
            'WaveletWhittle': WaveletWhittleEstimator()
        }
        
        logger.info(f"Initialized {len(estimators)} estimators")
        return estimators
    
    def run_comprehensive_benchmark(self, 
                                  dataset_sizes: List[int] = None,
                                  hurst_values: List[float] = None,
                                  noise_levels: List[float] = None,
                                  num_repetitions: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across multiple parameters.
        
        Parameters
        ----------
        dataset_sizes : List[int]
            List of dataset sizes to test
        hurst_values : List[float]
            List of Hurst exponents to test
        noise_levels : List[float]
            List of noise levels to test
        num_repetitions : int
            Number of repetitions for each configuration
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive benchmark results
        """
        if dataset_sizes is None:
            dataset_sizes = [100, 500, 1000, 2000]
        if hurst_values is None:
            hurst_values = [0.3, 0.5, 0.7, 0.9]
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.2, 0.5]
        
        logger.info("Starting comprehensive benchmark...")
        logger.info(f"Testing {len(dataset_sizes)} dataset sizes, "
                   f"{len(hurst_values)} Hurst values, "
                   f"{len(noise_levels)} noise levels, "
                   f"{num_repetitions} repetitions")
        
        start_time = time.time()
        
        # Generate test configurations
        test_configs = self._generate_test_configurations(
            dataset_sizes, hurst_values, noise_levels, num_repetitions
        )
        
        # Run benchmarks
        benchmark_results = self._run_benchmarks_parallel(test_configs)
        
        # Aggregate results
        aggregated_results = self._aggregate_results(benchmark_results)
        
        # Calculate performance metrics
        performance_analysis = self._analyze_performance(aggregated_results)
        
        # Generate summary
        summary = self._generate_benchmark_summary(
            aggregated_results, performance_analysis, start_time
        )
        
        # Save results
        self._save_results(aggregated_results, performance_analysis, summary)
        
        self.results = {
            'aggregated_results': aggregated_results,
            'performance_analysis': performance_analysis,
            'summary': summary
        }
        
        logger.info("Comprehensive benchmark completed")
        return self.results
    
    def _generate_test_configurations(self, dataset_sizes: List[int],
                                    hurst_values: List[float],
                                    noise_levels: List[float],
                                    num_repetitions: int) -> List[Dict[str, Any]]:
        """Generate all test configurations."""
        configs = []
        
        for size in dataset_sizes:
            for hurst in hurst_values:
                for noise in noise_levels:
                    for rep in range(num_repetitions):
                        config = {
                            'dataset_size': size,
                            'hurst_value': hurst,
                            'noise_level': noise,
                            'repetition': rep,
                            'config_id': f"size_{size}_hurst_{hurst}_noise_{noise}_rep_{rep}"
                        }
                        configs.append(config)
        
        return configs
    
    def _run_benchmarks_parallel(self, test_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run benchmarks in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(self._run_single_benchmark, config): config
                for config in test_configs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if len(results) % 10 == 0:
                        logger.info(f"Completed {len(results)}/{len(test_configs)} benchmarks")
                        
                except Exception as e:
                    logger.error(f"Benchmark failed for config {config['config_id']}: {str(e)}")
                    # Add failed result for tracking
                    results.append({
                        'config': config,
                        'status': 'failed',
                        'error': str(e),
                        'results': {}
                    })
        
        return results
    
    def _run_single_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark configuration."""
        try:
            # Generate synthetic data
            data = self.data_generator.generate_fractional_brownian_motion(
                size=config['dataset_size'],
                hurst=config['hurst_value'],
                noise_level=config['noise_level']
            )
            
            # Run all estimators
            estimator_results = {}
            
            for name, estimator in self.estimators.items():
                try:
                    # Reset estimator
                    estimator.reset()
                    
                    # Measure memory before
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Run estimation
                    start_time = time.time()
                    results = estimator.fit_estimate(data)
                    execution_time = time.time() - start_time
                    
                    # Measure memory after
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_used = memory_after - memory_before
                    
                    # Store results
                    estimator_results[name] = {
                        'results': results,
                        'execution_time': execution_time,
                        'memory_used': memory_used,
                        'status': 'success'
                    }
                    
                except Exception as e:
                    estimator_results[name] = {
                        'results': {},
                        'execution_time': None,
                        'memory_used': None,
                        'status': 'failed',
                        'error': str(e)
                    }
            
            return {
                'config': config,
                'status': 'success',
                'estimator_results': estimator_results,
                'true_hurst': config['hurst_value'],
                'data_size': len(data)
            }
            
        except Exception as e:
            return {
                'config': config,
                'status': 'failed',
                'error': str(e),
                'estimator_results': {}
            }
    
    def _aggregate_results(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate benchmark results by estimator and configuration."""
        aggregated = {}
        
        for result in benchmark_results:
            if result['status'] == 'failed':
                continue
                
            config = result['config']
            config_key = f"size_{config['dataset_size']}_hurst_{config['hurst_value']}_noise_{config['noise_level']}"
            
            if config_key not in aggregated:
                aggregated[config_key] = {
                    'config': {
                        'dataset_size': config['dataset_size'],
                        'hurst_value': config['hurst_value'],
                        'noise_level': config['noise_level']
                    },
                    'estimator_results': {},
                    'num_repetitions': 0
                }
            
            aggregated[config_key]['num_repetitions'] += 1
            
            for estimator_name, estimator_result in result['estimator_results'].items():
                if estimator_name not in aggregated[config_key]['estimator_results']:
                    aggregated[config_key]['estimator_results'][estimator_name] = {
                        'execution_times': [],
                        'memory_usage': [],
                        'hurst_estimates': [],
                        'alpha_estimates': [],
                        'success_count': 0,
                        'failure_count': 0
                    }
                
                if estimator_result['status'] == 'success':
                    aggregated[config_key]['estimator_results'][estimator_name]['success_count'] += 1
                    aggregated[config_key]['estimator_results'][estimator_name]['execution_times'].append(
                        estimator_result['execution_time']
                    )
                    aggregated[config_key]['estimator_results'][estimator_name]['memory_usage'].append(
                        estimator_result['memory_used']
                    )
                    
                    # Extract estimates
                    if 'hurst_exponent' in estimator_result['results']:
                        aggregated[config_key]['estimator_results'][estimator_name]['hurst_estimates'].append(
                            estimator_result['results']['hurst_exponent']
                        )
                    if 'alpha' in estimator_result['results']:
                        aggregated[config_key]['estimator_results'][estimator_name]['alpha_estimates'].append(
                            estimator_result['results']['alpha']
                        )
                else:
                    aggregated[config_key]['estimator_results'][estimator_name]['failure_count'] += 1
        
        return aggregated
    
    def _analyze_performance(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics for all estimators."""
        return self.metrics.calculate_comprehensive_metrics(aggregated_results)
    
    def _generate_benchmark_summary(self, aggregated_results: Dict[str, Any],
                                  performance_analysis: Dict[str, Any],
                                  start_time: float) -> Dict[str, Any]:
        """Generate a summary of the benchmark results."""
        total_time = time.time() - start_time
        
        summary = {
            'benchmark_info': {
                'total_execution_time': total_time,
                'total_configurations': len(aggregated_results),
                'estimators_tested': list(self.estimators.keys()),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'performance_overview': {
                'best_estimator': performance_analysis.get('best_estimator', 'N/A'),
                'most_robust': performance_analysis.get('most_robust', 'N/A'),
                'fastest': performance_analysis.get('fastest', 'N/A'),
                'most_memory_efficient': performance_analysis.get('most_memory_efficient', 'N/A')
            },
            'data_summary': {
                'dataset_sizes_tested': list(set(
                    config['config']['dataset_size'] 
                    for config in aggregated_results.values()
                )),
                'hurst_values_tested': list(set(
                    config['config']['hurst_value'] 
                    for config in aggregated_results.values()
                )),
                'noise_levels_tested': list(set(
                    config['config']['noise_level'] 
                    for config in aggregated_results.values()
                ))
            }
        }
        
        return summary
    
    def _save_results(self, aggregated_results: Dict[str, Any],
                     performance_analysis: Dict[str, Any],
                     summary: Dict[str, Any]):
        """Save benchmark results to files."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save aggregated results
        results_file = os.path.join(self.output_dir, f'benchmark_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2, default=str)
        
        # Save performance analysis
        analysis_file = os.path.join(self.output_dir, f'performance_analysis_{timestamp}.json')
        with open(analysis_file, 'w') as f:
            json.dump(performance_analysis, f, indent=2, default=str)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, f'benchmark_summary_{timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get the latest benchmark results."""
        return self.results
    
    def run_quick_benchmark(self, dataset_size: int = 1000, 
                           hurst_value: float = 0.7) -> Dict[str, Any]:
        """
        Run a quick benchmark with a single configuration.
        
        Parameters
        ----------
        dataset_size : int
            Size of the dataset to test
        hurst_value : float
            Hurst exponent to test
            
        Returns
        -------
        Dict[str, Any]
            Quick benchmark results
        """
        config = {
            'dataset_size': dataset_size,
            'hurst_value': hurst_value,
            'noise_level': 0.0,
            'repetition': 0,
            'config_id': f"quick_size_{dataset_size}_hurst_{hurst_value}"
        }
        
        result = self._run_single_benchmark(config)
        
        return {
            'config': config,
            'result': result
        }
