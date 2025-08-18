#!/usr/bin/env python3
"""
Realistic Datasets Demo: Analysis and Comparison

This script demonstrates analysis of the realistic datasets we downloaded:
1. Nile River flow data (hydrology)
2. Sunspot activity data (astronomy/climate)
3. Dow Jones Industrial Average (financial)
4. EEG sample data (biomedical)
5. Temperature data (climate)

We'll analyze their properties and compare with synthetic data.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

def load_realistic_datasets():
    """Load all realistic datasets and their metadata."""
    print("📊 Loading realistic datasets...")
    
    datasets = {}
    realistic_dir = Path("data/realistic")
    
    # Find all .npy files
    for file_path in realistic_dir.glob("*.npy"):
        if file_path.name.endswith('.npy') and not file_path.name.endswith('_dates.npy'):
            dataset_name = file_path.stem
            metadata_file = file_path.parent / f"{dataset_name}_metadata.json"
            
            if metadata_file.exists():
                try:
                    # Load data
                    data = np.load(file_path)
                    
                    # Load metadata
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    datasets[dataset_name] = {
                        'data': data,
                        'metadata': metadata,
                        'file_path': str(file_path)
                    }
                    
                    print(f"   ✅ Loaded {dataset_name}: {len(data):,} points")
                    
                except Exception as e:
                    print(f"   ⚠️ Error loading {dataset_name}: {e}")
    
    return datasets

def analyze_dataset_properties(datasets):
    """Analyze basic properties of each dataset."""
    print("\n🔍 Analyzing dataset properties...")
    
    analysis_results = {}
    
    for name, dataset_info in datasets.items():
        data = dataset_info['data']
        metadata = dataset_info['metadata']
        
        # Basic statistics
        stats = {
            'n_points': len(data),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'range': float(np.max(data) - np.min(data)),
            'cv': float(np.std(data) / np.abs(np.mean(data))) if np.mean(data) != 0 else 0,
            'skewness': float(np.mean(((data - np.mean(data)) / np.std(data)) ** 3)),
            'kurtosis': float(np.mean(((data - np.mean(data)) / np.std(data)) ** 4))
        }
        
        # Check for missing/infinite values
        stats['has_nan'] = bool(np.any(np.isnan(data)))
        stats['has_inf'] = bool(np.any(np.isinf(data)))
        stats['has_negative'] = bool(np.any(data < 0))
        
        # Basic LRD indicators (simplified)
        # Autocorrelation at lag 1
        if len(data) > 1:
            autocorr_lag1 = np.corrcoef(data[:-1], data[1:])[0, 1]
            stats['autocorr_lag1'] = float(autocorr_lag1) if not np.isnan(autocorr_lag1) else 0
        else:
            stats['autocorr_lag1'] = 0
        
        analysis_results[name] = {
            'metadata': metadata,
            'statistics': stats
        }
        
        print(f"   📊 {name}:")
        print(f"      Points: {stats['n_points']:,}")
        print(f"      Mean: {stats['mean']:.4f}")
        print(f"      Std: {stats['std']:.4f}")
        print(f"      Range: {stats['range']:.4f}")
        print(f"      CV: {stats['cv']:.4f}")
        print(f"      Autocorr(1): {stats['autocorr_lag1']:.4f}")
        print()
    
    return analysis_results

def create_visualizations(datasets, analysis_results):
    """Create comprehensive visualizations of the datasets."""
    print("🎨 Creating visualizations...")
    
    try:
        # Create a large figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Realistic Datasets Analysis', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Plot each dataset
        for i, (name, dataset_info) in enumerate(datasets.items()):
            if i >= len(axes_flat):
                break
                
            ax = axes_flat[i]
            data = dataset_info['data']
            metadata = dataset_info['metadata']
            
            # Plot time series
            ax.plot(data, alpha=0.8, linewidth=0.8)
            ax.set_title(f"{metadata['name']}\n({metadata['domain']})", fontsize=10)
            ax.set_xlabel('Time Point')
            ax.set_ylabel(metadata.get('units', 'Value'))
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats = analysis_results[name]['statistics']
            stats_text = f"n={stats['n_points']:,}\nμ={stats['mean']:.2f}\nσ={stats['std']:.2f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Remove empty subplot if we have odd number of datasets
        if len(datasets) < len(axes_flat):
            axes_flat[-1].set_visible(False)
        
        plt.tight_layout()
        
        # Save the plot
        viz_file = "data/realistic/datasets_visualization.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Visualization saved to: {viz_file}")
        return viz_file
        
    except Exception as e:
        print(f"   ⚠️ Visualization generation failed: {e}")
        return None

def compare_with_synthetic_data(realistic_datasets, analysis_results):
    """Compare realistic datasets with synthetic data properties."""
    print("\n🔄 Comparing with synthetic data...")
    
    # Load synthetic data for comparison
    synthetic_dir = Path("data/synthetic")
    synthetic_datasets = {}
    
    if synthetic_dir.exists():
        for file_path in synthetic_dir.glob("*.npy"):
            dataset_name = file_path.stem
            data = np.load(file_path)
            synthetic_datasets[dataset_name] = {
                'data': data,
                'n_points': len(data)
            }
    
    if not synthetic_datasets:
        print("   ⚠️ No synthetic datasets found for comparison")
        return
    
    print(f"   📊 Found {len(synthetic_datasets)} synthetic datasets")
    
    # Compare properties
    comparison = {
        'realistic': {},
        'synthetic': {},
        'summary': {}
    }
    
    # Analyze realistic datasets
    for name, dataset_info in realistic_datasets.items():
        data = dataset_info['data']
        stats = analysis_results[name]['statistics']
        
        comparison['realistic'][name] = {
            'n_points': stats['n_points'],
            'mean': stats['mean'],
            'std': stats['std'],
            'cv': stats['cv'],
            'autocorr_lag1': stats['autocorr_lag1']
        }
    
    # Analyze synthetic datasets
    for name, dataset_info in synthetic_datasets.items():
        data = dataset_info['data']
        
        comparison['synthetic'][name] = {
            'n_points': len(data),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'cv': float(np.std(data) / np.abs(np.mean(data))) if np.mean(data) != 0 else 0,
            'autocorr_lag1': float(np.corrcoef(data[:-1], data[1:])[0, 1]) if len(data) > 1 else 0
        }
    
    # Create summary statistics
    realistic_stats = list(comparison['realistic'].values())
    synthetic_stats = list(comparison['synthetic'].values())
    
    comparison['summary'] = {
        'realistic_count': len(realistic_stats),
        'synthetic_count': len(synthetic_stats),
        'realistic_avg_points': np.mean([s['n_points'] for s in realistic_stats]),
        'synthetic_avg_points': np.mean([s['n_points'] for s in synthetic_stats]),
        'realistic_avg_cv': np.mean([s['cv'] for s in realistic_stats]),
        'synthetic_avg_cv': np.mean([s['cv'] for s in synthetic_stats]),
        'realistic_avg_autocorr': np.mean([s['autocorr_lag1'] for s in realistic_stats]),
        'synthetic_avg_autocorr': np.mean([s['autocorr_lag1'] for s in synthetic_stats])
    }
    
    # Print comparison
    print("\n📈 COMPARISON SUMMARY:")
    print("-" * 50)
    print(f"Realistic datasets: {comparison['summary']['realistic_count']}")
    print(f"Synthetic datasets: {comparison['summary']['synthetic_count']}")
    print()
    print(f"Average points per dataset:")
    print(f"  Realistic: {comparison['summary']['realistic_avg_points']:,.0f}")
    print(f"  Synthetic: {comparison['summary']['synthetic_avg_points']:,.0f}")
    print()
    print(f"Average coefficient of variation:")
    print(f"  Realistic: {comparison['summary']['realistic_avg_cv']:.4f}")
    print(f"  Synthetic: {comparison['summary']['synthetic_avg_cv']:.4f}")
    print()
    print(f"Average autocorrelation at lag 1:")
    print(f"  Realistic: {comparison['summary']['realistic_avg_autocorr']:.4f}")
    print(f"  Synthetic: {comparison['summary']['synthetic_avg_autocorr']:.4f}")
    
    # Save comparison
    comparison_file = "data/realistic/datasets_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"\n💾 Comparison saved to: {comparison_file}")
    return comparison

def generate_analysis_report(analysis_results, comparison):
    """Generate a comprehensive analysis report."""
    print("\n📝 Generating analysis report...")
    
    report = {
        "analysis_date": datetime.now().isoformat(),
        "total_realistic_datasets": len(analysis_results),
        "analysis_results": analysis_results,
        "comparison": comparison,
        "recommendations": []
    }
    
    # Generate recommendations based on analysis
    recommendations = []
    
    # Check for datasets with strong autocorrelation (potential LRD)
    high_autocorr_datasets = []
    for name, result in analysis_results.items():
        if abs(result['statistics']['autocorr_lag1']) > 0.5:
            high_autocorr_datasets.append(name)
    
    if high_autocorr_datasets:
        recommendations.append({
            "type": "high_autocorrelation",
            "message": f"Datasets with strong autocorrelation (potential LRD): {', '.join(high_autocorr_datasets)}",
            "datasets": high_autocorr_datasets
        })
    
    # Check for datasets with high variability
    high_cv_datasets = []
    for name, result in analysis_results.items():
        if result['statistics']['cv'] > 1.0:
            high_cv_datasets.append(name)
    
    if high_cv_datasets:
        recommendations.append({
            "type": "high_variability",
            "message": f"Datasets with high variability (CV > 1): {', '.join(high_cv_datasets)}",
            "datasets": high_cv_datasets
        })
    
    # Check for datasets with different scales
    scales = [result['statistics']['std'] for result in analysis_results.values()]
    if max(scales) / min(scales) > 100:
        recommendations.append({
            "type": "scale_differences",
            "message": "Large differences in data scales detected - consider normalization for analysis",
            "datasets": list(analysis_results.keys())
        })
    
    report["recommendations"] = recommendations
    
    # Save report
    report_file = "data/realistic/analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"   ✅ Analysis report saved to: {report_file}")
    
    # Print recommendations
    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 30)
        for rec in recommendations:
            print(f"• {rec['message']}")
    
    return report

def main():
    """Main function to run the realistic datasets analysis."""
    print("🚀 Starting Realistic Datasets Analysis")
    print("=" * 50)
    
    # Load datasets
    realistic_datasets = load_realistic_datasets()
    
    if not realistic_datasets:
        print("❌ No realistic datasets found!")
        return
    
    print(f"\n✅ Loaded {len(realistic_datasets)} realistic datasets")
    
    # Analyze properties
    analysis_results = analyze_dataset_properties(realistic_datasets)
    
    # Create visualizations
    viz_file = create_visualizations(realistic_datasets, analysis_results)
    
    # Compare with synthetic data
    comparison = compare_with_synthetic_data(realistic_datasets, analysis_results)
    
    # Generate analysis report
    report = generate_analysis_report(analysis_results, comparison)
    
    # Final summary
    print(f"\n🎉 Analysis completed successfully!")
    print(f"📊 Analyzed {len(realistic_datasets)} realistic datasets")
    print(f"📁 Results saved in: data/realistic/")
    
    if viz_file:
        print(f"🎨 Visualization: {viz_file}")
    
    print("\n🎯 Next steps:")
    print("   • Use these datasets to test your LRD estimators")
    print("   • Compare performance between realistic and synthetic data")
    print("   • Validate your methods on real-world time series")
    print("   • Consider domain-specific preprocessing for each dataset type")

if __name__ == "__main__":
    main()


