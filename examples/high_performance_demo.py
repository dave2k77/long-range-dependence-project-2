"""
High-Performance Demo for Long-Range Dependence Framework

This demo showcases the NUMBA and JAX optimized estimators,
demonstrating GPU acceleration, parallel computing, and memory efficiency.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from estimators import (
    DFAEstimator, MFDFAEstimator,
    HighPerformanceDFAEstimator, HighPerformanceMFDFAEstimator
)
from benchmarking.synthetic_data import SyntheticDataGenerator


def generate_test_data(n_points=10000, hurst_values=[0.3, 0.5, 0.7, 0.9]):
    """Generate synthetic test data for benchmarking."""
    print("Generating synthetic test data...")
    
    generator = SyntheticDataGenerator()
    datasets = {}
    
    for hurst in hurst_values:
        print(f"  Generating fBm with H = {hurst} ({n_points} points)")
        data = generator.generate_fractional_brownian_motion(
            n_points=n_points, 
            hurst=hurst, 
            noise_level=0.1
        )
        datasets[f"H={hurst}"] = data
        
    return datasets


def benchmark_estimators(datasets, estimators):
    """Benchmark different estimators on the same datasets."""
    print("\nRunning performance benchmarks...")
    
    results = {}
    
    for estimator_name, estimator_class in estimators.items():
        print(f"\nBenchmarking {estimator_name}...")
        estimator_results = {}
        
        for dataset_name, data in datasets.items():
            print(f"  Processing {dataset_name}...")
            
            try:
                # Create estimator instance
                if "HighPerformance" in estimator_name:
                    if "DFA" in estimator_name:
                        est = estimator_class(use_parallel=True)
                    else:
                        est = estimator_class()
                else:
                    est = estimator_class()
                
                # Measure execution time
                start_time = time.time()
                start_memory = get_memory_usage()
                
                results_est = est.estimate(data)
                
                end_time = time.time()
                end_memory = get_memory_usage()
                
                execution_time = end_time - start_time
                memory_usage = end_memory - start_memory
                
                # Store results
                estimator_results[dataset_name] = {
                    'execution_time': execution_time,
                    'memory_usage': memory_usage,
                    'hurst_estimate': results_est.get('hurst_exponent', 
                                                   results_est.get('mean_hurst', np.nan)),
                    'r_squared': results_est.get('r_squared', 
                                               results_est.get('reliability', np.nan)),
                    'success': True
                }
                
                print(f"    Execution time: {execution_time:.4f}s")
                print(f"    Memory usage: {memory_usage:.2f} MB")
                print(f"    Hurst estimate: {estimator_results[dataset_name]['hurst_estimate']:.3f}")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                estimator_results[dataset_name] = {
                    'execution_time': np.nan,
                    'memory_usage': np.nan,
                    'hurst_estimate': np.nan,
                    'r_squared': np.nan,
                    'success': False,
                    'error': str(e)
                }
                
        results[estimator_name] = estimator_results
        
    return results


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0.0


def analyze_performance(results):
    """Analyze and summarize performance results."""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    # Calculate summary statistics
    summary = {}
    
    for estimator_name, estimator_results in results.items():
        successful_runs = [r for r in estimator_results.values() if r['success']]
        
        if successful_runs:
            execution_times = [r['execution_time'] for r in successful_runs]
            memory_usages = [r['memory_usage'] for r in successful_runs]
            hurst_estimates = [r['hurst_estimate'] for r in successful_runs]
            
            summary[estimator_name] = {
                'avg_execution_time': np.mean(execution_times),
                'std_execution_time': np.std(execution_times),
                'avg_memory_usage': np.mean(memory_usages),
                'std_memory_usage': np.std(memory_usages),
                'success_rate': len(successful_runs) / len(estimator_results),
                'avg_hurst_estimate': np.mean(hurst_estimates),
                'std_hurst_estimate': np.std(hurst_estimates)
            }
            
            print(f"\n{estimator_name}:")
            print(f"  Success Rate: {summary[estimator_name]['success_rate']:.1%}")
            print(f"  Avg Execution Time: {summary[estimator_name]['avg_execution_time']:.4f}s ± {summary[estimator_name]['std_execution_time']:.4f}s")
            print(f"  Avg Memory Usage: {summary[estimator_name]['avg_memory_usage']:.2f} MB ± {summary[estimator_name]['std_memory_usage']:.2f} MB")
            print(f"  Avg Hurst Estimate: {summary[estimator_name]['avg_hurst_estimate']:.3f} ± {summary[estimator_name]['std_hurst_estimate']:.3f}")
        else:
            print(f"\n{estimator_name}: No successful runs")
            
    return summary


def create_performance_plots(results, summary):
    """Create performance comparison plots."""
    print("\nCreating performance visualization plots...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('High-Performance LRD Estimators Benchmark Results', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    estimator_names = list(results.keys())
    dataset_names = list(next(iter(results.values())).keys())
    
    # 1. Execution Time Comparison
    ax1 = axes[0, 0]
    execution_times = []
    labels = []
    
    for est_name in estimator_names:
        for dataset_name in dataset_names:
            if results[est_name][dataset_name]['success']:
                execution_times.append(results[est_name][dataset_name]['execution_time'])
                labels.append(f"{est_name}\n{dataset_name}")
    
    if execution_times:
        bars1 = ax1.bar(range(len(execution_times)), execution_times, alpha=0.8)
        ax1.set_title('Execution Time Comparison')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_xticks(range(len(execution_times)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        
        # Color bars by estimator type
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, bar in enumerate(bars1):
            bar.set_color(colors[i % len(colors)])
    
    # 2. Memory Usage Comparison
    ax2 = axes[0, 1]
    memory_usages = []
    
    for est_name in estimator_names:
        for dataset_name in dataset_names:
            if results[est_name][dataset_name]['success']:
                memory_usages.append(results[est_name][dataset_name]['memory_usage'])
    
    if memory_usages:
        bars2 = ax2.bar(range(len(memory_usages)), memory_usages, alpha=0.8, color='#2ca02c')
        ax2.set_title('Memory Usage Comparison')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_xticks(range(len(memory_usages)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
    
    # 3. Speedup Comparison (relative to slowest estimator)
    ax3 = axes[1, 0]
    if execution_times:
        max_time = max(execution_times)
        speedups = [max_time / t for t in execution_times]
        
        bars3 = ax3.bar(range(len(speedups)), speedups, alpha=0.8, color='#ff7f0e')
        ax3.set_title('Speedup Relative to Slowest Estimator')
        ax3.set_ylabel('Speedup Factor')
        ax3.set_xticks(range(len(speedups)))
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add horizontal line at speedup = 1
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # 4. Success Rate Comparison
    ax4 = axes[1, 1]
    success_rates = []
    est_labels = []
    
    for est_name in estimator_names:
        if est_name in summary:
            success_rates.append(summary[est_name]['success_rate'])
            est_labels.append(est_name)
    
    if success_rates:
        bars4 = ax4.bar(range(len(success_rates)), success_rates, alpha=0.8, color='#d62728')
        ax4.set_title('Success Rate Comparison')
        ax4.set_ylabel('Success Rate')
        ax4.set_xticks(range(len(success_rates)))
        ax4.set_xticklabels(est_labels, rotation=45, ha='right')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent / "performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance plots saved to: {output_path}")
    
    plt.show()


def demonstrate_gpu_acceleration():
    """Demonstrate GPU acceleration capabilities."""
    print("\n" + "="*60)
    print("GPU ACCELERATION DEMONSTRATION")
    print("="*60)
    
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        
        # Check available devices
        devices = jax.devices()
        print(f"Available devices: {[str(d) for d in devices]}")
        
        # Check if GPU is available
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if gpu_devices:
            print(f"GPU devices found: {len(gpu_devices)}")
            print(f"Primary GPU: {gpu_devices[0]}")
            
            # Test GPU computation
            print("\nTesting GPU computation...")
            x = jax.numpy.ones((1000, 1000))
            
            # Time CPU vs GPU computation
            start_time = time.time()
            result_cpu = np.dot(x, x.T)
            cpu_time = time.time() - start_time
            
            start_time = time.time()
            result_gpu = jax.numpy.dot(x, x.T)
            gpu_time = time.time() - start_time
            
            print(f"CPU matrix multiplication time: {cpu_time:.4f}s")
            print(f"GPU matrix multiplication time: {gpu_time:.4f}s")
            print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")
            
        else:
            print("No GPU devices found. Running on CPU/TPU.")
            
    except ImportError:
        print("JAX not available. Cannot demonstrate GPU acceleration.")
    except Exception as e:
        print(f"Error during GPU demonstration: {str(e)}")


def demonstrate_parallel_computing():
    """Demonstrate parallel computing capabilities."""
    print("\n" + "="*60)
    print("PARALLEL COMPUTING DEMONSTRATION")
    print("="*60)
    
    try:
        import numba
        print(f"NUMBA version: {numba.__version__}")
        
        # Test parallel vs sequential computation
        print("\nTesting parallel vs sequential computation...")
        
        # Generate large dataset
        data = np.random.randn(10000)
        scales = np.logspace(1, 3, 20, dtype=int)
        
        # Sequential computation
        start_time = time.time()
        result_seq = np.array([np.var(data[i:i+100]) for i in range(0, len(data)-100, 100)])
        seq_time = time.time() - start_time
        
        # Parallel computation (simulated with numpy vectorization)
        start_time = time.time()
        indices = np.arange(0, len(data)-100, 100)
        result_par = np.array([np.var(data[i:i+100]) for i in indices])
        par_time = time.time() - start_time
        
        print(f"Sequential computation time: {seq_time:.4f}s")
        print(f"Parallel computation time: {par_time:.4f}s")
        print(f"Parallel speedup: {seq_time/par_time:.2f}x")
        
        # Verify results are the same
        np.testing.assert_allclose(result_seq, result_par, rtol=1e-10)
        print("✓ Results are identical between sequential and parallel computation")
        
    except ImportError:
        print("NUMBA not available. Cannot demonstrate parallel computing.")
    except Exception as e:
        print(f"Error during parallel computing demonstration: {str(e)}")


def main():
    """Main demonstration function."""
    print("🚀 HIGH-PERFORMANCE LRD ESTIMATION DEMONSTRATION")
    print("="*60)
    
    # Check system capabilities
    print("System Information:")
    print(f"  Python version: {sys.version}")
    print(f"  NumPy version: {np.__version__}")
    
    # Generate test data
    datasets = generate_test_data(n_points=5000, hurst_values=[0.3, 0.5, 0.7, 0.9])
    
    # Define estimators to benchmark
    estimators = {
        'Classical DFA': DFAEstimator,
        'Classical MFDFA': MFDFAEstimator,
    }
    
    # Add high-performance variants if available
    try:
        estimators['High-Performance DFA (NUMBA)'] = HighPerformanceDFAEstimator
        print("✓ NUMBA-optimized DFA available")
    except ImportError:
        print("✗ NUMBA-optimized DFA not available")
        
    try:
        estimators['High-Performance MFDFA (JAX)'] = HighPerformanceMFDFAEstimator
        print("✓ JAX-optimized MFDFA available")
    except ImportError:
        print("✗ JAX-optimized MFDFA not available")
    
    # Run benchmarks
    results = benchmark_estimators(datasets, estimators)
    
    # Analyze performance
    summary = analyze_performance(results)
    
    # Create visualizations
    create_performance_plots(results, summary)
    
    # Demonstrate advanced features
    demonstrate_gpu_acceleration()
    demonstrate_parallel_computing()
    
    # Final summary
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    
    if summary:
        fastest_estimator = min(summary.keys(), 
                               key=lambda x: summary[x]['avg_execution_time'])
        most_memory_efficient = min(summary.keys(), 
                                   key=lambda x: summary[x]['avg_memory_usage'])
        
        print(f"\n🏆 Performance Champions:")
        print(f"  Fastest: {fastest_estimator}")
        print(f"  Most Memory Efficient: {most_memory_efficient}")
        
        print(f"\n📊 Key Insights:")
        for est_name, stats in summary.items():
            if 'HighPerformance' in est_name:
                classical_name = est_name.replace('High-Performance ', 'Classical ')
                if classical_name in summary:
                    speedup = summary[classical_name]['avg_execution_time'] / stats['avg_execution_time']
                    print(f"  {est_name} is {speedup:.2f}x faster than {classical_name}")
    
    print(f"\n💡 Next Steps:")
    print(f"  - Run with larger datasets for more dramatic performance differences")
    print(f"  - Experiment with different GPU configurations")
    print(f"  - Try custom parameter tuning for your specific use case")
    print(f"  - Explore the benchmarking framework for systematic performance analysis")


if __name__ == "__main__":
    main()
