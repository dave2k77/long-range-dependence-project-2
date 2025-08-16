#!/usr/bin/env python3
"""
Quick Performance Benchmark Runner

This script runs performance benchmarks on the LRD estimators
to compare their performance across different dataset sizes.
"""

import sys
import os
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from benchmarking.performance_benchmarks import run_quick_benchmark, PerformanceBenchmarker

def main():
    """Main benchmark execution function."""
    print("🚀 Long-Range Dependence Estimators Performance Benchmark")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run quick benchmark
        print("\n📊 Starting performance benchmark...")
        results_df = run_quick_benchmark()
        
        print("\n✅ Benchmark completed successfully!")
        print(f"📁 Results saved to: benchmark_results/")
        
        # Show summary
        print("\n📋 QUICK SUMMARY:")
        print("-" * 40)
        
        successful = results_df[results_df['success'] == True]
        if not successful.empty:
            # Show average execution times
            avg_times = successful.groupby('estimator')['execution_time'].mean()
            print("Average Execution Times:")
            for estimator, time in avg_times.items():
                print(f"  {estimator}: {time:.4f}s")
            
            # Show average memory usage
            avg_memory = successful.groupby('estimator')['memory_peak_mb'].mean()
            print("\nAverage Peak Memory Usage:")
            for estimator, memory in avg_memory.items():
                print(f"  {estimator}: {memory:.1f} MB")
            
            # Show success rates
            success_rates = results_df.groupby('estimator')['success'].agg(['sum', 'count'])
            success_rates['rate'] = success_rates['sum'] / success_rates['count'] * 100
            print("\nSuccess Rates:")
            for estimator in success_rates.index:
                rate = success_rates.loc[estimator, 'rate']
                print(f"  {estimator}: {rate:.1f}%")
        else:
            print("❌ No successful benchmark runs!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        logging.error(f"Benchmark error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
