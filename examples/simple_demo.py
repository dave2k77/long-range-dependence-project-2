#!/usr/bin/env python3
"""
Simple Demo: Synthetic Data Generation and Leaderboard

This script demonstrates the core functionality of:
1. Generating synthetic data
2. Submitting benchmark results
3. Viewing the leaderboard

Run this from the project root directory.
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

# Import our modules
from data_generation.synthetic_data_generator import SyntheticDataGenerator, DataSpecification, DomainType
from data_submission.dataset_submission import DatasetSubmissionManager
from data_submission.benchmark_submission import BenchmarkSubmissionManager

def main():
    """Main demonstration function."""
    print("🚀 Starting Simple Demo: Synthetic Data Generation and Leaderboard")
    print("=" * 70)
    
    # Initialize managers
    print("\n1. Initializing managers...")
    dataset_manager = DatasetSubmissionManager()
    benchmark_manager = BenchmarkSubmissionManager()
    
    # Generate synthetic data
    print("\n2. Generating synthetic datasets...")
    generator = SyntheticDataGenerator(random_seed=42)
    
    # Generate different types of data
    datasets = []
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    
    for i, hurst in enumerate(hurst_values):
        print(f"   Generating dataset {i+1}/4 with Hurst exponent {hurst}")
        
        spec = DataSpecification(
            n_points=1000,
            hurst_exponent=hurst,
            domain_type=DomainType.GENERAL,
            confound_strength=0.1,
            noise_level=0.05
        )
        
        data = generator.generate_data(spec)
        datasets.append({
            'data': data,
            'hurst': hurst,
            'id': f'dataset_{i+1}'
        })
    
    # Submit synthetic datasets
    print("\n3. Submitting synthetic datasets...")
    for dataset in datasets:
        dataset_id = dataset['id']
        data = dataset['data']
        
        # Save data to file
        data_file = f"data/synthetic/{dataset_id}.npy"
        np.save(data_file, data['data'])
        
        # Submit to dataset manager
        submission = dataset_manager.submit_synthetic_dataset(
            submitter_name="Demo User",
            submitter_email="demo@example.com",
            dataset_name=f"Synthetic Dataset {dataset_id}",
            dataset_description=f"Generated synthetic data with Hurst exponent {dataset['hurst']}",
            dataset_version="1.0",
            license="MIT",
            file_paths={"data": data_file},
            dataset_properties=data['properties']
        )
        
        print(f"   Submitted {dataset_id}: {submission.submission_id}")
    
    # Submit benchmark results
    print("\n4. Submitting benchmark results...")
    
    # Simulate some benchmark results
    estimators = ["GPH", "R/S", "DMA", "DFA"]
    
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
            
            benchmark_result = benchmark_manager.submit_benchmark_results(
                submitter_name="Demo User",
                submitter_email="demo@example.com",
                benchmark_name=f"Demo Benchmark {dataset_id}",
                benchmark_description=f"Benchmarking {estimator} on {dataset_id}",
                benchmark_version="1.0",
                dataset_name=dataset_id,
                estimators_tested=[estimator],
                performance_metrics={
                    "estimated_hurst": estimated_hurst,
                    "true_hurst": true_hurst,
                    "mse": mse,
                    "mae": mae,
                    "execution_time": execution_time,
                    "memory_usage": memory_usage
                }
            )
            
            print(f"   Submitted benchmark: {estimator} on {dataset_id}")
    
    # View leaderboard
    print("\n5. Viewing leaderboard...")
    leaderboard = benchmark_manager.get_leaderboard()
    
    if leaderboard:
        print("\n🏆 LEADERBOARD:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Estimator':<15} {'Dataset':<15} {'Score':<10} {'MSE':<10} {'Time':<10}")
        print("-" * 80)
        
        for i, entry in enumerate(leaderboard[:10], 1):  # Show top 10
            print(f"{i:<4} {entry['estimator_name']:<15} {entry['dataset_name']:<15} "
                  f"{entry['leaderboard_score']:<10.3f} {entry['mse']:<10.3f} "
                  f"{entry['execution_time']:<10.3f}")
    else:
        print("   No leaderboard entries found.")
    
    # Generate benchmark comparison
    print("\n6. Generating benchmark comparison...")
    comparison = benchmark_manager.get_benchmark_comparison()
    
    if comparison:
        print(f"   Generated comparison with {len(comparison)} datasets")
        
        # Show summary statistics
        estimators_summary = {}
        for entry in comparison:
            estimator = entry['estimator_name']
            if estimator not in estimators_summary:
                estimators_summary[estimator] = []
            estimators_summary[estimator].append(entry['mse'])
        
        print("\n📊 ESTIMATOR PERFORMANCE SUMMARY:")
        print("-" * 50)
        for estimator, mse_values in estimators_summary.items():
            avg_mse = np.mean(mse_values)
            std_mse = np.std(mse_values)
            print(f"{estimator:<15}: MSE = {avg_mse:.4f} ± {std_mse:.4f}")
    
    # Export leaderboard
    print("\n7. Exporting leaderboard...")
    csv_file = benchmark_manager.export_leaderboard_csv()
    if csv_file:
        print(f"   Leaderboard exported to: {csv_file}")
    
    # Generate visualization
    print("\n8. Generating leaderboard visualization...")
    try:
        viz_file = benchmark_manager.generate_leaderboard_visualization()
        if viz_file:
            print(f"   Visualization saved to: {viz_file}")
    except Exception as e:
        print(f"   Visualization generation failed: {e}")
    
    print("\n✅ Demo completed successfully!")
    print("\n📁 Generated files:")
    print(f"   - Synthetic data: data/synthetic/")
    print(f"   - Leaderboard CSV: {csv_file if 'csv_file' in locals() else 'N/A'}")
    print(f"   - Visualization: {viz_file if 'viz_file' in locals() else 'N/A'}")

if __name__ == "__main__":
    main()
