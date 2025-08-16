#!/usr/bin/env python3
"""
Direct Demo: Synthetic Data Generation and Analysis

This script directly demonstrates the core functionality without complex imports.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    """Main demonstration function."""
    print("🚀 Starting Direct Demo: Synthetic Data Generation and Analysis")
    print("=" * 70)
    
    # Create data directories if they don't exist
    os.makedirs("data/synthetic", exist_ok=True)
    os.makedirs("data/submissions", exist_ok=True)
    os.makedirs("data/submissions/benchmarks", exist_ok=True)
    os.makedirs("data/submissions/benchmarks/leaderboard", exist_ok=True)
    
    print("\n1. Data directories created/verified")
    
    # Generate synthetic data directly
    print("\n2. Generating synthetic datasets...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate different types of data with different Hurst exponents
    datasets = []
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    
    for i, hurst in enumerate(hurst_values):
        print(f"   Generating dataset {i+1}/4 with Hurst exponent {hurst}")
        
        # Generate fractional Gaussian noise (simplified)
        n_points = 1000
        
        # Create time series with long-range dependence
        # This is a simplified version - in practice you'd use more sophisticated methods
        time_series = np.random.normal(0, 1, n_points)
        
        # Apply some basic transformations to simulate LRD
        # (This is a simplified approach - real LRD generation is more complex)
        if hurst > 0.5:
            # Positive correlation for H > 0.5
            for j in range(1, n_points):
                time_series[j] += 0.1 * time_series[j-1]
        elif hurst < 0.5:
            # Negative correlation for H < 0.5
            for j in range(1, n_points):
                time_series[j] -= 0.1 * time_series[j-1]
        
        # Add some noise
        noise = np.random.normal(0, 0.05, n_points)
        time_series += noise
        
        # Save data
        data_file = f"data/synthetic/dataset_{i+1}.npy"
        np.save(data_file, time_series)
        
        datasets.append({
            'data': time_series,
            'hurst': hurst,
            'id': f'dataset_{i+1}',
            'file': data_file
        })
        
        print(f"   Saved {data_file}")
    
    # Create a simple leaderboard
    print("\n3. Creating benchmark results and leaderboard...")
    
    # Simulate some benchmark results
    estimators = ["GPH", "R/S", "DMA", "DFA"]
    benchmark_results = []
    
    for dataset in datasets:
        dataset_id = dataset['id']
        true_hurst = dataset['hurst']
        
        for estimator in estimators:
            # Simulate estimation results
            estimated_hurst = true_hurst + np.random.normal(0, 0.05)
            execution_time = np.random.uniform(0.1, 2.0)
            memory_usage = np.random.uniform(10, 100)
            
            # Calculate accuracy metrics
            mse = (estimated_hurst - true_hurst) ** 2
            mae = abs(estimated_hurst - true_hurst)
            
            # Calculate a simple leaderboard score (lower is better)
            leaderboard_score = mse + 0.1 * execution_time + 0.01 * memory_usage
            
            benchmark_results.append({
                'estimator_name': estimator,
                'dataset_name': dataset_id,
                'true_hurst': true_hurst,
                'estimated_hurst': estimated_hurst,
                'mse': mse,
                'mae': mae,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'leaderboard_score': leaderboard_score
            })
            
            print(f"   {estimator} on {dataset_id}: H_est={estimated_hurst:.3f}, MSE={mse:.4f}")
    
    # Sort by leaderboard score
    benchmark_results.sort(key=lambda x: x['leaderboard_score'])
    
    # Display leaderboard
    print("\n4. 🏆 LEADERBOARD:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Estimator':<15} {'Dataset':<15} {'Score':<10} {'MSE':<10} {'Time':<10}")
    print("-" * 80)
    
    for i, entry in enumerate(benchmark_results[:10], 1):  # Show top 10
        print(f"{i:<4} {entry['estimator_name']:<15} {entry['dataset_name']:<15} "
              f"{entry['leaderboard_score']:<10.3f} {entry['mse']:<10.4f} "
              f"{entry['execution_time']:<10.3f}")
    
    # Generate summary statistics
    print("\n5. 📊 ESTIMATOR PERFORMANCE SUMMARY:")
    print("-" * 50)
    
    estimators_summary = {}
    for entry in benchmark_results:
        estimator = entry['estimator_name']
        if estimator not in estimators_summary:
            estimators_summary[estimator] = []
        estimators_summary[estimator].append(entry['mse'])
    
    for estimator, mse_values in estimators_summary.items():
        avg_mse = np.mean(mse_values)
        std_mse = np.std(mse_values)
        print(f"{estimator:<15}: MSE = {avg_mse:.4f} ± {std_mse:.4f}")
    
    # Export leaderboard to CSV
    print("\n6. Exporting leaderboard to CSV...")
    
    df = pd.DataFrame(benchmark_results)
    csv_file = "data/submissions/benchmarks/leaderboard/leaderboard.csv"
    df.to_csv(csv_file, index=False)
    print(f"   Leaderboard exported to: {csv_file}")
    
    # Create a simple visualization
    print("\n7. Generating visualization...")
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot 1: True vs Estimated Hurst
        plt.subplot(2, 2, 1)
        for estimator in estimators:
            estimator_data = [r for r in benchmark_results if r['estimator_name'] == estimator]
            true_hursts = [r['true_hurst'] for r in estimator_data]
            est_hursts = [r['estimated_hurst'] for r in estimator_data]
            plt.scatter(true_hursts, est_hursts, label=estimator, alpha=0.7)
        
        plt.plot([0.2, 1.0], [0.2, 1.0], 'k--', alpha=0.5, label='Perfect Estimation')
        plt.xlabel('True Hurst Exponent')
        plt.ylabel('Estimated Hurst Exponent')
        plt.title('True vs Estimated Hurst Exponents')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: MSE by Estimator
        plt.subplot(2, 2, 2)
        mse_by_estimator = []
        estimator_names = []
        for estimator in estimators:
            estimator_data = [r for r in benchmark_results if r['estimator_name'] == estimator]
            mse_values = [r['mse'] for r in estimator_data]
            mse_by_estimator.append(mse_values)
            estimator_names.append(estimator)
        
        plt.boxplot(mse_by_estimator, labels=estimator_names)
        plt.ylabel('Mean Squared Error')
        plt.title('MSE Distribution by Estimator')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Execution Time by Estimator
        plt.subplot(2, 2, 3)
        time_by_estimator = []
        for estimator in estimators:
            estimator_data = [r for r in benchmark_results if r['estimator_name'] == estimator]
            time_values = [r['execution_time'] for r in estimator_data]
            time_by_estimator.append(time_values)
        
        plt.boxplot(time_by_estimator, labels=estimator_names)
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time Distribution by Estimator')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Sample time series
        plt.subplot(2, 2, 4)
        for i, dataset in enumerate(datasets[:2]):  # Show first 2 datasets
            plt.plot(dataset['data'][:200], label=f'Dataset {i+1} (H={dataset["hurst"]})', alpha=0.7)
        plt.xlabel('Time Point')
        plt.ylabel('Value')
        plt.title('Sample Time Series')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        viz_file = "data/submissions/benchmarks/leaderboard/leaderboard_visualization.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Visualization saved to: {viz_file}")
        
    except Exception as e:
        print(f"   Visualization generation failed: {e}")
    
    print("\n✅ Demo completed successfully!")
    print("\n📁 Generated files:")
    print(f"   - Synthetic data: data/synthetic/")
    print(f"   - Leaderboard CSV: {csv_file}")
    print(f"   - Visualization: {viz_file if 'viz_file' in locals() else 'N/A'}")
    
    # Show some sample data
    print("\n📊 Sample Data Preview:")
    print(f"   Generated {len(datasets)} datasets with {len(benchmark_results)} benchmark results")
    print(f"   Best performing estimator: {benchmark_results[0]['estimator_name']}")
    print(f"   Best MSE: {benchmark_results[0]['mse']:.4f}")

if __name__ == "__main__":
    main()
