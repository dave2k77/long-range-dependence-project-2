#!/usr/bin/env python3
"""
Synthetic Data Quality Evaluation Demo

This script demonstrates the TSGBench-inspired synthetic data quality evaluation
system. It shows how to:

1. Evaluate synthetic data quality against realistic reference data
2. Identify areas for improvement in synthetic data generation
3. Generate comprehensive quality reports
4. Use domain-specific evaluation criteria
5. Track quality improvements over time

The goal is to ensure synthetic data closely matches realistic parameters
while maintaining intended LRD properties.
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

# Import our quality evaluation utilities
from validation.synthetic_data_quality import (
    SyntheticDataQualityEvaluator, create_domain_specific_evaluator,
    QualityMetricType
)

# Import synthetic data generation
from data_generation.synthetic_data_generator import SyntheticDataGenerator

def load_realistic_datasets():
    """Load realistic datasets for quality evaluation."""
    print("📊 Loading realistic datasets for quality evaluation...")
    
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
                    
                    print(f"   ✅ Loaded {dataset_name}: {len(data):,} points ({metadata['domain']})")
                    
                except Exception as e:
                    print(f"   ⚠️ Error loading {dataset_name}: {e}")
    
    return datasets

def generate_synthetic_datasets_for_evaluation():
    """Generate synthetic datasets with different quality levels for evaluation."""
    print("\n🔧 Generating synthetic datasets for quality evaluation...")
    
    generator = SyntheticDataGenerator()
    synthetic_datasets = {}
    
    # 1. High-quality synthetic data (should score well)
    print("   📊 Generating high-quality synthetic data...")
    
    # Hydrology: High-quality hydrology data
    from data_generation.synthetic_data_generator import DataSpecification, DomainType, ConfoundType
    
    hydrology_spec = DataSpecification(
        n_points=100,  # Match Nile River data length
        hurst_exponent=0.7,
        domain_type=DomainType.HYDROLOGY,
        confound_strength=0.1,
        noise_level=0.01
    )
    hydrology_data = generator.generate_data(hydrology_spec, [ConfoundType.SEASONALITY])
    synthetic_datasets['hydrology_high'] = {
        'data': hydrology_data['data'],
        'metadata': {'domain': 'hydrology', 'quality_level': 'high'},
        'description': 'High-quality hydrology data'
    }
    
    # Financial: High-quality financial data
    financial_spec = DataSpecification(
        n_points=168,  # Match DJIA data length
        hurst_exponent=0.55,
        domain_type=DomainType.FINANCIAL,
        confound_strength=0.2,
        noise_level=0.02
    )
    financial_data = generator.generate_data(financial_spec, [ConfoundType.VOLATILITY_CLUSTERING])
    synthetic_datasets['financial_high'] = {
        'data': financial_data['data'],
        'metadata': {'domain': 'financial', 'quality_level': 'high'},
        'description': 'High-quality financial data'
    }
    
    # 2. Medium-quality synthetic data (should score moderately)
    print("   📊 Generating medium-quality synthetic data...")
    
    hydrology_medium_spec = DataSpecification(
        n_points=100,  # Match Nile River data length
        hurst_exponent=0.7,
        domain_type=DomainType.HYDROLOGY,
        confound_strength=0.3,
        noise_level=0.1  # Higher noise
    )
    hydrology_medium_data = generator.generate_data(hydrology_medium_spec, [ConfoundType.SEASONALITY, ConfoundType.NOISE])
    synthetic_datasets['hydrology_medium'] = {
        'data': hydrology_medium_data['data'],
        'metadata': {'domain': 'hydrology', 'quality_level': 'medium'},
        'description': 'Medium-quality hydrology data'
    }
    
    # 3. Low-quality synthetic data (should score poorly)
    print("   📊 Generating low-quality synthetic data...")
    
    # White noise (should score poorly for LRD preservation)
    white_noise = np.random.normal(0, 1, 100)  # Match reference data length
    synthetic_datasets['white_noise_poor'] = {
        'data': white_noise,
        'metadata': {'domain': 'general', 'quality_level': 'poor'},
        'description': 'White noise (poor LRD preservation)'
    }
    
    print(f"   ✅ Generated {len(synthetic_datasets)} synthetic datasets")
    return synthetic_datasets

def demonstrate_quality_evaluation(realistic_datasets, synthetic_datasets):
    """Demonstrate comprehensive quality evaluation."""
    print("\n🔍 Demonstrating Quality Evaluation")
    print("=" * 50)
    
    # Create evaluators for different domains
    evaluators = {
        'hydrology': create_domain_specific_evaluator('hydrology'),
        'financial': create_domain_specific_evaluator('financial'),
        'general': SyntheticDataQualityEvaluator()
    }
    
    evaluation_results = {}
    
    # Evaluate each synthetic dataset against appropriate reference data
    for synth_name, synth_info in synthetic_datasets.items():
        print(f"\n📊 Evaluating {synth_name}...")
        
        # Find appropriate reference data
        domain = synth_info['metadata']['domain']
        reference_data = None
        reference_metadata = None
        
        # Find best matching reference dataset
        for ref_name, ref_info in realistic_datasets.items():
            if ref_info['metadata']['domain'].lower() == domain.lower():
                reference_data = ref_info['data']
                reference_metadata = ref_info['metadata']
                break
        
        if reference_data is None:
            # Use first available reference data if no domain match
            ref_name = list(realistic_datasets.keys())[0]
            reference_data = realistic_datasets[ref_name]['data']
            reference_metadata = realistic_datasets[ref_name]['metadata']
            print(f"   ⚠️ No domain match found, using {ref_name} as reference")
        
        # Select appropriate evaluator
        if domain in evaluators:
            evaluator = evaluators[domain]
        else:
            evaluator = evaluators['general']
        
        # Perform quality evaluation
        try:
            result = evaluator.evaluate_quality(
                synthetic_data=synth_info['data'],
                reference_data=reference_data,
                reference_metadata=reference_metadata,
                domain=domain
            )
            
            evaluation_results[synth_name] = result
            
            print(f"   ✅ Quality Score: {result.overall_score:.3f} ({result.quality_level})")
            print(f"      Best metrics: {', '.join([m.metric_name for m in result.metrics if m.score > 0.8][:3])}")
            
            if result.recommendations:
                print(f"      Top recommendation: {result.recommendations[0]}")
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
    
    return evaluation_results

def create_quality_visualizations(evaluation_results):
    """Create comprehensive visualizations of quality evaluation results."""
    print("\n🎨 Creating quality evaluation visualizations...")
    
    try:
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Overall quality scores comparison
        ax1 = plt.subplot(3, 3, 1)
        dataset_names = list(evaluation_results.keys())
        quality_scores = [evaluation_results[name].overall_score for name in dataset_names]
        quality_levels = [evaluation_results[name].quality_level for name in dataset_names]
        
        # Color code by quality level
        colors = []
        for level in quality_levels:
            if level == 'excellent':
                colors.append('green')
            elif level == 'good':
                colors.append('blue')
            elif level == 'acceptable':
                colors.append('orange')
            else:
                colors.append('red')
        
        bars = ax1.bar(range(len(dataset_names)), quality_scores, color=colors, alpha=0.7)
        ax1.set_xlabel('Synthetic Dataset')
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Overall Quality Scores by Dataset')
        ax1.set_xticks(range(len(dataset_names)))
        ax1.set_xticklabels([name.replace('_', '\n') for name in dataset_names], rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, quality_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Quality level distribution
        ax2 = plt.subplot(3, 3, 2)
        quality_counts = {}
        for level in quality_levels:
            quality_counts[level] = quality_counts.get(level, 0) + 1
        
        if quality_counts:
            levels = list(quality_counts.keys())
            counts = list(quality_counts.values())
            colors = ['green', 'blue', 'orange', 'red']
            
            ax2.pie(counts, labels=levels, autopct='%1.0f%%', colors=colors[:len(levels)])
            ax2.set_title('Quality Level Distribution')
        
        # 3. Metric performance comparison
        ax3 = plt.subplot(3, 3, 3)
        
        # Collect all unique metric names
        all_metric_names = set()
        for result in evaluation_results.values():
            for metric in result.metrics:
                all_metric_names.add(metric.metric_name)
        
        metric_names = sorted(list(all_metric_names))
        
        # Calculate average scores for each metric
        metric_scores = {}
        for metric_name in metric_names:
            scores = []
            for result in evaluation_results.values():
                for metric in result.metrics:
                    if metric.metric_name == metric_name:
                        scores.append(metric.score)
            if scores:
                metric_scores[metric_name] = np.mean(scores)
        
        if metric_scores:
            metric_names_plot = list(metric_scores.keys())
            avg_scores = list(metric_scores.values())
            
            bars = ax3.bar(range(len(metric_names_plot)), avg_scores, alpha=0.7)
            ax3.set_xlabel('Quality Metric')
            ax3.set_ylabel('Average Score')
            ax3.set_title('Average Performance by Metric')
            ax3.set_xticks(range(len(metric_names_plot)))
            ax3.set_xticklabels([name.replace('_', '\n') for name in metric_names_plot], rotation=45, ha='right')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
        
        # 4-6. Individual dataset quality breakdowns (limit to 3 to avoid subplot overflow)
        plot_positions = [(2, 1), (2, 2), (2, 3)]
        dataset_count = 0
        
        for synth_name, result in evaluation_results.items():
            if dataset_count >= 3:  # Limit to 3 plots to avoid subplot overflow
                break
                
            ax = plt.subplot(3, 3, plot_positions[dataset_count][0] * 3 + plot_positions[dataset_count][1])
            
            # Group metrics by type
            metric_types = {}
            for metric in result.metrics:
                if metric.metric_type not in metric_types:
                    metric_types[metric.metric_type] = []
                metric_types[metric.metric_type].append(metric)
            
            # Plot metric scores by type
            type_names = []
            type_scores = []
            type_colors = []
            
            for metric_type, metrics in metric_types.items():
                avg_score = np.mean([m.score for m in metrics])
                type_names.append(metric_type.value)
                type_scores.append(avg_score)
                
                # Color code by metric type
                if metric_type == QualityMetricType.STATISTICAL:
                    type_colors.append('blue')
                elif metric_type == QualityMetricType.TEMPORAL:
                    type_colors.append('green')
                elif metric_type == QualityMetricType.LRD:
                    type_colors.append('red')
                elif metric_type == QualityMetricType.DOMAIN:
                    type_colors.append('orange')
                else:
                    type_colors.append('gray')
            
            bars = ax.bar(range(len(type_names)), type_scores, color=type_colors, alpha=0.7)
            ax.set_title(f"{synth_name.replace('_', '\n')}\nQuality: {result.overall_score:.3f}")
            ax.set_xlabel('Metric Type')
            ax.set_ylabel('Average Score')
            ax.set_xticks(range(len(type_names)))
            ax.set_xticklabels([name.replace('_', '\n') for name in type_names], rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            dataset_count += 1
        
        plt.tight_layout()
        
        # Save the plot
        viz_file = "data/realistic/quality_evaluation_visualization.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Visualization saved to: {viz_file}")
        return viz_file
        
    except Exception as e:
        print(f"   ⚠️ Visualization generation failed: {e}")
        return None

def generate_quality_reports(evaluation_results):
    """Generate comprehensive quality reports for all evaluations."""
    print("\n📝 Generating quality reports...")
    
    reports_dir = Path("data/realistic/quality_reports")
    reports_dir.mkdir(exist_ok=True)
    
    generated_reports = []
    
    for synth_name, result in evaluation_results.items():
        print(f"   📊 Generating report for {synth_name}...")
        
        # Create detailed report using the evaluator method
        from validation.synthetic_data_quality import SyntheticDataQualityEvaluator
        evaluator = SyntheticDataQualityEvaluator()
        report_text = evaluator.create_quality_report(result)
        
        # Save report
        report_file = reports_dir / f"{synth_name}_quality_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        # Save JSON result
        json_file = reports_dir / f"{synth_name}_quality_result.json"
        evaluator.save_evaluation_result(result, json_file)
        
        generated_reports.append({
            'dataset': synth_name,
            'text_report': str(report_file),
            'json_result': str(json_file),
            'overall_score': result.overall_score,
            'quality_level': result.quality_level
        })
        
        print(f"      ✅ Report saved: {report_file.name}")
    
    # Generate summary report
    summary_report = create_summary_report(generated_reports)
    summary_file = reports_dir / "quality_evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary_report)
    
    print(f"   ✅ Summary report saved: {summary_file.name}")
    
    return generated_reports

def create_summary_report(reports):
    """Create a summary report of all quality evaluations."""
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("SYNTHETIC DATA QUALITY EVALUATION SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Datasets Evaluated: {len(reports)}")
    report_lines.append("")
    
    # Overall statistics
    scores = [r['overall_score'] for r in reports]
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 30)
    report_lines.append(f"Average Quality Score: {np.mean(scores):.3f}")
    report_lines.append(f"Best Quality Score: {np.max(scores):.3f}")
    report_lines.append(f"Worst Quality Score: {np.min(scores):.3f}")
    report_lines.append(f"Quality Score Std Dev: {np.std(scores):.3f}")
    report_lines.append("")
    
    # Quality level distribution
    quality_levels = [r['quality_level'] for r in reports]
    level_counts = {}
    for level in quality_levels:
        level_counts[level] = level_counts.get(level, 0) + 1
    
    report_lines.append("QUALITY LEVEL DISTRIBUTION")
    report_lines.append("-" * 30)
    for level, count in sorted(level_counts.items()):
        percentage = (count / len(reports)) * 100
        report_lines.append(f"{level.title()}: {count} ({percentage:.1f}%)")
    report_lines.append("")
    
    # Individual dataset results
    report_lines.append("INDIVIDUAL DATASET RESULTS")
    report_lines.append("-" * 30)
    
    # Sort by quality score (best first)
    sorted_reports = sorted(reports, key=lambda x: x['overall_score'], reverse=True)
    
    for i, report in enumerate(sorted_reports, 1):
        report_lines.append(f"{i}. {report['dataset']}")
        report_lines.append(f"   Quality Score: {report['overall_score']:.3f}")
        report_lines.append(f"   Quality Level: {report['quality_level'].title()}")
        report_lines.append(f"   Report: {Path(report['text_report']).name}")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 30)
    
    # Overall recommendations based on results
    if np.mean(scores) < 0.5:
        report_lines.append("• Overall quality is poor. Review synthetic data generation parameters.")
        report_lines.append("• Consider implementing more sophisticated generation methods.")
    elif np.mean(scores) < 0.7:
        report_lines.append("• Quality is acceptable but could be improved.")
        report_lines.append("• Focus on improving the lowest-scoring metrics.")
    else:
        report_lines.append("• Overall quality is good. Continue current generation approach.")
        report_lines.append("• Consider fine-tuning for even better results.")
    
    # Specific recommendations
    poor_datasets = [r for r in reports if r['overall_score'] < 0.5]
    if poor_datasets:
        report_lines.append("• Poor performing datasets requiring attention:")
        for dataset in poor_datasets:
            report_lines.append(f"  - {dataset['dataset']} (score: {dataset['overall_score']:.3f})")
    
    excellent_datasets = [r for r in reports if r['overall_score'] >= 0.9]
    if excellent_datasets:
        report_lines.append("• Excellent performing datasets (use as templates):")
        for dataset in excellent_datasets:
            report_lines.append(f"  - {dataset['dataset']} (score: {dataset['overall_score']:.3f})")
    
    # Footer
    report_lines.append("\n" + "=" * 80)
    report_lines.append("End of Summary Report")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)

def demonstrate_quality_improvement():
    """Demonstrate how quality evaluation can guide improvements."""
    print("\n🚀 Demonstrating Quality Improvement Process")
    print("=" * 50)
    
    generator = SyntheticDataGenerator()
    
    # Start with poor quality data
    print("   📊 Generating initial (poor quality) synthetic data...")
    from data_generation.synthetic_data_generator import DataSpecification, DomainType, ConfoundType
    
    initial_spec = DataSpecification(
        n_points=100,  # Match Nile River data length
        hurst_exponent=0.7,
        domain_type=DomainType.HYDROLOGY,
        confound_strength=0.5,
        noise_level=0.5  # High noise
    )
    initial_data = generator.generate_data(initial_spec, [ConfoundType.NOISE, ConfoundType.HEAVY_TAILS])
    
    # Create evaluator
    evaluator = SyntheticDataQualityEvaluator()
    
    # Use Nile River data as reference
    nile_data = np.load("data/realistic/nile_river_flow.npy")
    nile_metadata = {"domain": "hydrology", "source": "Nile River"}
    
    # Evaluate initial quality
    print("   🔍 Evaluating initial quality...")
    initial_result = evaluator.evaluate_quality(
        initial_data['data'], nile_data, nile_metadata, "hydrology"
    )
    
    print(f"      Initial Quality Score: {initial_result.overall_score:.3f} ({initial_result.quality_level})")
    
    # Improve based on recommendations
    print("   🔧 Improving synthetic data based on evaluation...")
    
    # Generate improved data with lower noise and better parameters
    improved_spec = DataSpecification(
        n_points=100,  # Match Nile River data length
        hurst_exponent=0.7,
        domain_type=DomainType.HYDROLOGY,
        confound_strength=0.1,
        noise_level=0.05  # Lower noise
    )
    improved_data = generator.generate_data(improved_spec, [ConfoundType.SEASONALITY])
    
    # Evaluate improved quality
    print("   🔍 Evaluating improved quality...")
    improved_result = evaluator.evaluate_quality(
        improved_data['data'], nile_data, nile_metadata, "hydrology"
    )
    
    print(f"      Improved Quality Score: {improved_result.overall_score:.3f} ({improved_result.quality_level})")
    
    # Show improvement
    improvement = improved_result.overall_score - initial_result.overall_score
    print(f"      Quality Improvement: +{improvement:.3f}")
    
    if improvement > 0:
        print("   ✅ Quality improvement successful!")
    else:
        print("   ⚠️ Quality did not improve. Review improvement strategy.")
    
    return {
        'initial': initial_result,
        'improved': improved_result,
        'improvement': improvement
    }

def main():
    """Main function to run the synthetic data quality evaluation demonstration."""
    print("🚀 Starting Synthetic Data Quality Evaluation Demonstration")
    print("=" * 70)
    
    # Load realistic datasets
    realistic_datasets = load_realistic_datasets()
    
    if not realistic_datasets:
        print("❌ No realistic datasets found! Please run realistic datasets demo first.")
        return
    
    print(f"\n✅ Loaded {len(realistic_datasets)} realistic datasets")
    
    # Generate synthetic datasets for evaluation
    synthetic_datasets = generate_synthetic_datasets_for_evaluation()
    
    # Perform comprehensive quality evaluation
    evaluation_results = demonstrate_quality_evaluation(realistic_datasets, synthetic_datasets)
    
    # Create visualizations
    viz_file = create_quality_visualizations(evaluation_results)
    
    # Generate quality reports
    generated_reports = generate_quality_reports(evaluation_results)
    
    # Demonstrate quality improvement process
    improvement_results = demonstrate_quality_improvement()
    
    # Final summary
    print(f"\n🎉 Synthetic Data Quality Evaluation completed successfully!")
    print(f"📊 Evaluated {len(evaluation_results)} synthetic datasets")
    print(f"📁 Quality reports saved in: data/realistic/quality_reports/")
    
    if viz_file:
        print(f"🎨 Visualization: {viz_file}")
    
    # Print key findings
    scores = [r.overall_score for r in evaluation_results.values()]
    print(f"\n📊 KEY FINDINGS:")
    print(f"   Average Quality Score: {np.mean(scores):.3f}")
    print(f"   Best Dataset: {max(evaluation_results.items(), key=lambda x: x[1].overall_score)[0]}")
    print(f"   Worst Dataset: {min(evaluation_results.items(), key=lambda x: x[1].overall_score)[0]}")
    
    # Quality improvement demonstration
    if improvement_results['improvement'] > 0:
        print(f"   Quality Improvement Demonstrated: +{improvement_results['improvement']:.3f}")
    
    print(f"\n💡 Key insights:")
    print(f"   • Use quality evaluation to guide synthetic data generation")
    print(f"   • Focus on improving lowest-scoring metrics")
    print(f"   • Domain-specific evaluation provides targeted feedback")
    print(f"   • Regular quality assessment ensures consistent data quality")
    
    print(f"\n🎯 Next steps:")
    print(f"   • Integrate quality evaluation into synthetic data generation pipeline")
    print(f"   • Use quality scores to automatically tune generation parameters")
    print(f"   • Implement continuous quality monitoring for production use")
    print(f"   • Extend domain-specific metrics for specialized applications")

if __name__ == "__main__":
    main()
