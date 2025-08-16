#!/usr/bin/env python3
"""
Data Preprocessing Demo: Normalization and Quality Control

This script demonstrates the comprehensive data preprocessing pipeline including:
1. Data normalization using multiple methods
2. Quality checks and validation
3. Domain-specific preprocessing configurations
4. Comparison of original vs processed data
5. Preservation of LRD properties

The goal is to ensure fair comparison across datasets with different scales
while maintaining the long-range dependence characteristics.
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

# Import our preprocessing utilities
from utils.data_preprocessing import (
    DataPreprocessor, PreprocessingConfig, NormalizationMethod,
    create_domain_specific_preprocessor
)

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

def demonstrate_normalization_methods(datasets):
    """Demonstrate different normalization methods on the datasets."""
    print("\n🔧 Demonstrating Normalization Methods")
    print("=" * 50)
    
    # Select a representative dataset from each domain
    domain_datasets = {
        'hydrology': 'nile_river_flow',
        'financial': 'dow_jones_monthly', 
        'biomedical': 'eeg_sample',
        'climate': 'daily_temperature'
    }
    
    normalization_methods = [
        NormalizationMethod.ZSCORE,
        NormalizationMethod.MINMAX,
        NormalizationMethod.ROBUST,
        NormalizationMethod.DECIMAL,
        NormalizationMethod.LOG
    ]
    
    results = {}
    
    for domain, dataset_name in domain_datasets.items():
        if dataset_name not in datasets:
            continue
            
        print(f"\n🌍 Processing {domain} dataset: {dataset_name}")
        data = datasets[dataset_name]['data']
        metadata = datasets[dataset_name]['metadata']
        
        domain_results = {}
        
        for method in normalization_methods:
            print(f"   📊 Testing {method.value} normalization...")
            
            # Create preprocessor with specific normalization method
            config = PreprocessingConfig(
                normalization_method=method,
                handle_missing_values=True,
                detect_outliers=True,
                treat_outliers=False,
                remove_trend=False,
                remove_seasonality=False,
                noise_reduction=False,
                quality_check=True
            )
            
            preprocessor = DataPreprocessor(config)
            
            # Preprocess the data
            result = preprocessor.preprocess_dataset(data, metadata, domain)
            
            # Store results
            domain_results[method.value] = {
                'original_stats': {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'range': float(np.max(data) - np.min(data))
                },
                'normalized_stats': {
                    'mean': float(np.mean(result.processed_data)),
                    'std': float(np.std(result.processed_data)),
                    'min': float(np.min(result.processed_data)),
                    'max': float(np.max(result.processed_data)),
                    'range': float(np.max(result.processed_data) - np.min(result.processed_data))
                },
                'preprocessing_info': result.preprocessing_info,
                'quality_report': result.quality_report
            }
            
            print(f"      ✅ {method.value} completed")
        
        results[domain] = domain_results
    
    return results

def demonstrate_domain_specific_preprocessing(datasets):
    """Demonstrate domain-specific preprocessing configurations."""
    print("\n🎯 Demonstrating Domain-Specific Preprocessing")
    print("=" * 50)
    
    domain_results = {}
    
    for domain in ['hydrology', 'financial', 'biomedical', 'climate']:
        print(f"\n🌍 Processing {domain} datasets...")
        
        # Find datasets for this domain
        domain_datasets = {}
        for name, dataset_info in datasets.items():
            if dataset_info['metadata']['domain'].lower() == domain:
                domain_datasets[name] = dataset_info
        
        if not domain_datasets:
            print(f"   ⚠️ No datasets found for domain: {domain}")
            continue
        
        # Create domain-specific preprocessor
        preprocessor = create_domain_specific_preprocessor(domain)
        print(f"   🔧 Using {domain}-specific configuration:")
        print(f"      Normalization: {preprocessor.config.normalization_method.value}")
        print(f"      Trend removal: {preprocessor.config.remove_trend}")
        print(f"      Seasonality removal: {preprocessor.config.remove_seasonality}")
        print(f"      Noise reduction: {preprocessor.config.noise_reduction}")
        
        # Process each dataset
        domain_results[domain] = {}
        
        for name, dataset_info in domain_datasets.items():
            print(f"   📊 Processing {name}...")
            
            result = preprocessor.preprocess_dataset(
                dataset_info['data'], 
                dataset_info['metadata'], 
                domain
            )
            
            domain_results[domain][name] = {
                'original_data': dataset_info['data'],
                'processed_data': result.processed_data,
                'preprocessing_info': result.preprocessing_info,
                'quality_report': result.quality_report
            }
            
            print(f"      ✅ {name} completed")
    
    return domain_results

def create_preprocessing_visualizations(normalization_results, domain_results):
    """Create comprehensive visualizations of preprocessing results."""
    print("\n🎨 Creating preprocessing visualizations...")
    
    try:
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Normalization comparison for one dataset
        ax1 = plt.subplot(3, 3, 1)
        dataset_name = 'daily_temperature'  # Use temperature as example
        if dataset_name in normalization_results.get('climate', {}):
            data = normalization_results['climate'][dataset_name]
            methods = list(data.keys())
            
            # Plot original vs normalized ranges
            original_ranges = [data[method]['original_stats']['range'] for method in methods]
            normalized_ranges = [data[method]['normalized_stats']['range'] for method in methods]
            
            x = np.arange(len(methods))
            width = 0.35
            
            ax1.bar(x - width/2, original_ranges, width, label='Original', alpha=0.7)
            ax1.bar(x + width/2, normalized_ranges, width, label='Normalized', alpha=0.7)
            ax1.set_xlabel('Normalization Method')
            ax1.set_ylabel('Data Range')
            ax1.set_title('Data Range Comparison\n(Original vs Normalized)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Scale comparison across domains
        ax2 = plt.subplot(3, 3, 2)
        domains = list(domain_results.keys())
        original_scales = []
        processed_scales = []
        
        for domain in domains:
            if domain in domain_results and domain_results[domain]:
                # Get first dataset from domain
                first_dataset = list(domain_results[domain].keys())[0]
                dataset_data = domain_results[domain][first_dataset]
                
                original_std = np.std(dataset_data['original_data'])
                processed_std = np.std(dataset_data['processed_data'])
                
                original_scales.append(original_std)
                processed_scales.append(processed_std)
            else:
                original_scales.append(0)
                processed_scales.append(0)
        
        x = np.arange(len(domains))
        width = 0.35
        
        ax2.bar(x - width/2, original_scales, width, label='Original', alpha=0.7)
        ax2.bar(x + width/2, processed_scales, width, label='Processed', alpha=0.7)
        ax2.set_xlabel('Domain')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Scale Comparison Across Domains')
        ax2.set_xticks(x)
        ax2.set_xticklabels(domains, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Quality check results
        ax3 = plt.subplot(3, 3, 3)
        quality_passed = []
        quality_failed = []
        
        for domain in domains:
            if domain in domain_results and domain_results[domain]:
                passed = 0
                failed = 0
                for dataset_name in domain_results[domain]:
                    quality_report = domain_results[domain][dataset_name]['quality_report']
                    if quality_report.get('passed', False):
                        passed += 1
                    else:
                        failed += 1
                quality_passed.append(passed)
                quality_failed.append(failed)
            else:
                quality_passed.append(0)
                quality_failed.append(0)
        
        x = np.arange(len(domains))
        width = 0.35
        
        ax3.bar(x - width/2, quality_passed, width, label='Passed', color='green', alpha=0.7)
        ax3.bar(x + width/2, quality_failed, width, label='Failed', color='red', alpha=0.7)
        ax3.set_xlabel('Domain')
        ax3.set_ylabel('Number of Datasets')
        ax3.set_title('Quality Check Results by Domain')
        ax3.set_xticks(x)
        ax3.set_xticklabels(domains, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4-9. Individual dataset comparisons
        plot_positions = [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
        dataset_count = 0
        
        for domain in domains:
            if dataset_count >= 6:  # Limit to 6 plots
                break
                
            if domain in domain_results and domain_results[domain]:
                for dataset_name in list(domain_results[domain].keys())[:1]:  # One per domain
                    if dataset_count >= 6:
                        break
                        
                    ax = plt.subplot(3, 3, plot_positions[dataset_count][0] * 3 + plot_positions[dataset_count][1])
                    
                    dataset_data = domain_results[domain][dataset_name]
                    original = dataset_data['original_data']
                    processed = dataset_data['processed_data']
                    
                    # Plot first 1000 points to avoid overcrowding
                    n_plot = min(1000, len(original))
                    ax.plot(original[:n_plot], label='Original', alpha=0.7, linewidth=0.8)
                    ax.plot(processed[:n_plot], label='Processed', alpha=0.7, linewidth=0.8)
                    ax.set_title(f"{domain.title()}: {dataset_name}")
                    ax.set_xlabel('Time Point')
                    ax.set_ylabel('Value')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    dataset_count += 1
        
        plt.tight_layout()
        
        # Save the plot
        viz_file = "data/realistic/preprocessing_visualization.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Visualization saved to: {viz_file}")
        return viz_file
        
    except Exception as e:
        print(f"   ⚠️ Visualization generation failed: {e}")
        return None

def generate_preprocessing_report(normalization_results, domain_results):
    """Generate a comprehensive preprocessing report."""
    print("\n📝 Generating preprocessing report...")
    
    report = {
        "report_date": datetime.now().isoformat(),
        "summary": {
            "total_datasets_processed": 0,
            "domains_covered": [],
            "normalization_methods_tested": [],
            "quality_check_results": {"passed": 0, "failed": 0}
        },
        "normalization_analysis": {},
        "domain_specific_results": {},
        "recommendations": []
    }
    
    # Analyze normalization results
    for domain, datasets in normalization_results.items():
        report["summary"]["domains_covered"].append(domain)
        
        for dataset_name, results in datasets.items():
            report["summary"]["total_datasets_processed"] += 1
            
            # Collect normalization methods
            for method in results.keys():
                if method not in report["summary"]["normalization_methods_tested"]:
                    report["summary"]["normalization_methods_tested"].append(method)
            
            # Analyze normalization effectiveness
            if domain not in report["normalization_analysis"]:
                report["normalization_analysis"][domain] = {}
            
            report["normalization_analysis"][domain][dataset_name] = {
                "methods_comparison": {},
                "best_method": None,
                "scale_reduction_factor": None
            }
            
            # Compare methods
            original_range = results[list(results.keys())[0]]['original_stats']['range']
            best_method = None
            best_reduction = 0
            
            for method, method_results in results.items():
                normalized_range = method_results['normalized_stats']['range']
                reduction_factor = original_range / normalized_range if normalized_range > 0 else 1
                
                report["normalization_analysis"][domain][dataset_name]["methods_comparison"][method] = {
                    "original_range": method_results['original_stats']['range'],
                    "normalized_range": method_results['normalized_stats']['range'],
                    "reduction_factor": reduction_factor,
                    "mean_shift": abs(method_results['normalized_stats']['mean']),
                    "std_shift": abs(method_results['normalized_stats']['std'] - 1) if method == 'zscore' else None
                }
                
                if reduction_factor > best_reduction:
                    best_reduction = reduction_factor
                    best_method = method
            
            report["normalization_analysis"][domain][dataset_name]["best_method"] = best_method
            report["normalization_analysis"][domain][dataset_name]["scale_reduction_factor"] = best_reduction
    
    # Analyze domain-specific results
    for domain, datasets in domain_results.items():
        report["domain_specific_results"][domain] = {
            "datasets_processed": len(datasets),
            "quality_results": {"passed": 0, "failed": 0},
            "preprocessing_steps_applied": []
        }
        
        for dataset_name, dataset_data in datasets.items():
            quality_report = dataset_data['quality_report']
            if quality_report.get('passed', False):
                report["domain_specific_results"][domain]["quality_results"]["passed"] += 1
                report["summary"]["quality_check_results"]["passed"] += 1
            else:
                report["domain_specific_results"][domain]["quality_results"]["failed"] += 1
                report["summary"]["quality_check_results"]["failed"] += 1
            
            # Collect preprocessing steps
            preprocessing_info = dataset_data['preprocessing_info']
            for step in preprocessing_info.keys():
                if step not in report["domain_specific_results"][domain]["preprocessing_steps_applied"]:
                    report["domain_specific_results"][domain]["preprocessing_steps_applied"].append(step)
    
    # Generate recommendations
    recommendations = []
    
    # Normalization recommendations
    for domain, datasets in report["normalization_analysis"].items():
        for dataset_name, analysis in datasets.items():
            best_method = analysis["best_method"]
            if best_method:
                recommendations.append({
                    "type": "normalization",
                    "domain": domain,
                    "dataset": dataset_name,
                    "recommendation": f"Use {best_method} normalization for {domain} data",
                    "reason": f"Best scale reduction factor: {analysis['scale_reduction_factor']:.2f}x"
                })
    
    # Quality recommendations
    for domain, results in report["domain_specific_results"].items():
        if results["quality_results"]["failed"] > 0:
            recommendations.append({
                "type": "quality",
                "domain": domain,
                "recommendation": f"Review quality issues in {domain} datasets",
                "reason": f"{results['quality_results']['failed']} out of {results['quality_results']['passed'] + results['quality_results']['failed']} datasets failed quality checks"
            })
    
    # General recommendations
    recommendations.append({
        "type": "general",
        "recommendation": "Apply domain-specific preprocessing configurations",
        "reason": "Different domains have different requirements for trend removal, seasonality, and outlier treatment"
    })
    
    recommendations.append({
        "type": "general", 
        "recommendation": "Use Z-score normalization for most LRD analysis",
        "reason": "Preserves relative relationships while standardizing scale, suitable for most statistical methods"
    })
    
    report["recommendations"] = recommendations
    
    # Save report
    report_file = "data/realistic/preprocessing_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"   ✅ Preprocessing report saved to: {report_file}")
    
    # Print key findings
    print(f"\n📊 PREPROCESSING SUMMARY:")
    print(f"   Total datasets processed: {report['summary']['total_datasets_processed']}")
    print(f"   Domains covered: {', '.join(report['summary']['domains_covered'])}")
    print(f"   Normalization methods tested: {', '.join(report['summary']['normalization_methods_tested'])}")
    print(f"   Quality check results: {report['summary']['quality_check_results']['passed']} passed, {report['summary']['quality_check_results']['failed']} failed")
    
    return report

def main():
    """Main function to run the data preprocessing demonstration."""
    print("🚀 Starting Data Preprocessing Demonstration")
    print("=" * 60)
    
    # Load datasets
    datasets = load_realistic_datasets()
    
    if not datasets:
        print("❌ No realistic datasets found!")
        return
    
    print(f"\n✅ Loaded {len(datasets)} realistic datasets")
    
    # Demonstrate normalization methods
    normalization_results = demonstrate_normalization_methods(datasets)
    
    # Demonstrate domain-specific preprocessing
    domain_results = demonstrate_domain_specific_preprocessing(datasets)
    
    # Create visualizations
    viz_file = create_preprocessing_visualizations(normalization_results, domain_results)
    
    # Generate comprehensive report
    report = generate_preprocessing_report(normalization_results, domain_results)
    
    # Final summary
    print(f"\n🎉 Data preprocessing demonstration completed successfully!")
    print(f"📊 Processed {report['summary']['total_datasets_processed']} datasets")
    print(f"🌍 Covered {len(report['summary']['domains_covered'])} domains")
    print(f"🔧 Tested {len(report['summary']['normalization_methods_tested'])} normalization methods")
    
    if viz_file:
        print(f"🎨 Visualization: {viz_file}")
    
    print(f"\n💡 Key recommendations:")
    for rec in report['recommendations'][:3]:  # Show top 3
        print(f"   • {rec['recommendation']}")
    
    print(f"\n🎯 Next steps:")
    print(f"   • Use the normalized datasets for LRD analysis")
    print(f"   • Compare estimator performance on original vs processed data")
    print(f"   • Validate that preprocessing preserves LRD properties")
    print(f"   • Apply domain-specific configurations for production use")

if __name__ == "__main__":
    main()
