#!/usr/bin/env python3
"""
Comprehensive Synthetic Data Generation Demonstration

This script demonstrates the complete synthetic data generation system for
Long-Range Dependence Analysis, including:

1. Base Models: ARFIMA, fBm, fGn
2. Realistic Confounds: Non-stationarity, artifacts, baseline drift
3. Domain-specific patterns: EEG, Hydrology/Climate, Financial
4. Dataset specifications and submission protocols
5. Benchmark dataset generation

Usage:
    python comprehensive_synthetic_data_demo.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generation.synthetic_data_generator import (
    SyntheticDataGenerator, 
    DataSpecification, 
    DataType, 
    ConfoundType, 
    DomainType,
    create_standard_dataset_specifications,
    generate_benchmark_dataset
)
from data_generation.dataset_specifications import (
    DatasetSpecification as DatasetSpec,
    DatasetMetadata,
    DatasetProperties,
    ConfoundDescription,
    BenchmarkProtocol,
    DatasetFormat,
    DomainCategory
)

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class ComprehensiveSyntheticDataDemo:
    """Comprehensive demonstration of synthetic data generation capabilities."""
    
    def __init__(self, output_dir: str = "demo_outputs"):
        """Initialize the demonstration."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize generator with reproducible seed
        self.generator = SyntheticDataGenerator(random_seed=42)
        
        # Create subdirectories
        (self.output_dir / "datasets").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "specifications").mkdir(exist_ok=True)
        
        print("🚀 Comprehensive Synthetic Data Generation Demo")
        print("=" * 60)
        print(f"Output directory: {self.output_dir.absolute()}")
        print()
    
    def demonstrate_base_models(self):
        """Demonstrate the three base models: ARFIMA, fBm, fGn."""
        print("📊 Demonstrating Base Models")
        print("-" * 40)
        
        # Generate base models with different Hurst exponents
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        n_points = 2000
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle("Base Models: ARFIMA, fBm, fGn with Different Hurst Exponents", fontsize=16)
        
        for i, hurst in enumerate(hurst_values):
            # ARFIMA
            spec_arfima = DataSpecification(
                n_points=n_points,
                hurst_exponent=hurst,
                d_parameter=hurst - 0.5,
                ar_coeffs=[0.3, -0.2],
                ma_coeffs=[0.1],
                domain_type=DomainType.GENERAL
            )
            arfima_data = self.generator.generate_data(spec_arfima)
            
            # fBm
            spec_fbm = DataSpecification(
                n_points=n_points,
                hurst_exponent=hurst,
                domain_type=DomainType.GENERAL
            )
            fbm_data = self.generator.generate_data(spec_fbm)
            
            # fGn
            spec_fgn = DataSpecification(
                n_points=n_points,
                hurst_exponent=hurst,
                domain_type=DomainType.GENERAL
            )
            fgn_data = self.generator.generate_data(spec_fgn)
            
            # Plot results
            axes[0, i].plot(arfima_data['data'][:500], linewidth=1)
            axes[0, i].set_title(f'ARFIMA (H={hurst:.1f})')
            axes[0, i].set_ylabel('Value')
            if i == 0:
                axes[0, i].set_ylabel('ARFIMA')
            
            axes[1, i].plot(fbm_data['data'][:500], linewidth=1)
            axes[1, i].set_title(f'fBm (H={hurst:.1f})')
            if i == 0:
                axes[1, i].set_ylabel('fBm')
            
            axes[2, i].plot(fgn_data['data'][:500], linewidth=1)
            axes[2, i].set_title(f'fGn (H={hurst:.1f})')
            axes[2, i].set_xlabel('Time')
            if i == 0:
                axes[2, i].set_ylabel('fGn')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "base_models_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Generated base models comparison plot")
        print()
    
    def demonstrate_confounds(self):
        """Demonstrate various realistic confounds."""
        print("🔧 Demonstrating Realistic Confounds")
        print("-" * 40)
        
        # Generate clean base data
        base_spec = DataSpecification(
            n_points=2000,
            hurst_exponent=0.7,
            domain_type=DomainType.GENERAL
        )
        base_data = self.generator.generate_data(base_spec)
        
        # Apply different confounds
        confound_types = [
            ConfoundType.NON_STATIONARITY,
            ConfoundType.HEAVY_TAILS,
            ConfoundType.BASELINE_DRIFT,
            ConfoundType.ARTIFACTS,
            ConfoundType.SEASONALITY,
            ConfoundType.TREND_CHANGES
        ]
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle("Realistic Confounds Applied to Base Data", fontsize=16)
        
        for i, confound in enumerate(confound_types):
            row, col = i // 2, i % 2
            
            # Apply single confound
            contaminated_data = self.generator.generate_data(
                base_spec, 
                confounds=[confound]
            )
            
            axes[row, col].plot(base_data['data'][:500], alpha=0.7, label='Clean', linewidth=1)
            axes[row, col].plot(contaminated_data['data'][:500], alpha=0.9, label=f'{confound.value}', linewidth=1)
            axes[row, col].set_title(f'{confound.value.replace("_", " ").title()}')
            axes[row, col].legend()
            axes[row, col].set_xlabel('Time')
            if col == 0:
                axes[row, col].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "confounds_demonstration.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Generated confounds demonstration plot")
        print()
    
    def demonstrate_domain_specific_patterns(self):
        """Demonstrate domain-specific data generation."""
        print("🌍 Demonstrating Domain-Specific Patterns")
        print("-" * 40)
        
        # Generate domain-specific datasets
        domains = [
            ('eeg_resting', 'EEG Resting State'),
            ('hydrology_daily', 'Hydrology Daily'),
            ('climate_monthly', 'Climate Monthly'),
            ('financial_daily', 'Financial Daily')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Domain-Specific Synthetic Data Patterns", fontsize=16)
        
        for i, (spec_name, title) in enumerate(domains):
            row, col = i // 2, i % 2
            
            # Generate data with domain-appropriate confounds
            data = generate_benchmark_dataset(self.generator, spec_name)
            
            axes[row, col].plot(data['data'][:1000], linewidth=1)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Time')
            if col == 0:
                axes[row, col].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "domain_specific_patterns.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Generated domain-specific patterns plot")
        print()
    
    def demonstrate_dataset_specifications(self):
        """Demonstrate dataset specification and submission system."""
        print("📋 Demonstrating Dataset Specifications")
        print("-" * 40)
        
        # Create a comprehensive dataset specification
        metadata = DatasetMetadata(
            name="Synthetic EEG Benchmark Dataset",
            description="High-quality synthetic EEG data for LRD estimator benchmarking",
            author="Demo User",
            contact="demo@example.com",
            creation_date=datetime.now().isoformat(),
            version="1.0.0",
            license="MIT",
            citation="Demo Dataset for LRD Analysis",
            keywords=["EEG", "synthetic", "long-range dependence", "benchmark"],
            references=["Demo Reference 1", "Demo Reference 2"]
        )
        
        properties = DatasetProperties(
            n_points=5000,
            n_variables=1,
            sampling_frequency=100.0,
            time_unit="seconds",
            missing_values=False,
            outliers=True,
            mean=0.0,
            std=1.0,
            hurst_exponent=0.7,
            is_stationary=False,
            has_seasonality=False,
            has_trends=True
        )
        
        confounds = ConfoundDescription(
            non_stationarity=True,
            heavy_tails=False,
            baseline_drift=True,
            artifacts=True,
            seasonality=False,
            trend_changes=True,
            volatility_clustering=False,
            regime_changes=False,
            jumps=False,
            measurement_noise=True,
            missing_data=False,
            outliers=True,
            confound_details={
                "baseline_drift": "Slow linear trend",
                "artifacts": "Random spikes with 1% probability",
                "measurement_noise": "Gaussian noise with 5% amplitude"
            }
        )
        
        benchmark_protocol = BenchmarkProtocol(
            name="EEG LRD Estimator Benchmark",
            description="Comprehensive evaluation of LRD estimators on synthetic EEG data",
            estimators_to_test=["DFA", "GPH", "Higuchi", "RS", "Wavelet"],
            performance_metrics=["RMSE", "MAE", "Bias", "Variance"],
            validation_methods=["Cross-validation", "Bootstrap", "Monte Carlo"],
            cross_validation_folds=5,
            bootstrap_samples=1000,
            confidence_level=0.95
        )
        
        # Create complete specification
        dataset_spec = DatasetSpec(
            metadata=metadata,
            properties=properties,
            confounds=confounds,
            benchmark_protocol=benchmark_protocol,
            data_format=DatasetFormat.NUMPY,
            validation_status="validated"
        )
        
        # Save specification
        spec_path = self.output_dir / "specifications" / "eeg_benchmark_spec.json"
        dataset_spec.to_json(str(spec_path))
        
        print(f"✅ Created dataset specification: {spec_path}")
        print(f"   - Dataset: {metadata.name}")
        print(f"   - Points: {properties.n_points}")
        print(f"   - Hurst: {properties.hurst_exponent}")
        print(f"   - Confounds: {sum([getattr(confounds, attr) for attr in dir(confounds) if not attr.startswith('_') and isinstance(getattr(confounds, attr), bool)])}")
        print()
        
        return dataset_spec
    
    def demonstrate_benchmark_generation(self):
        """Demonstrate benchmark dataset generation."""
        print("🏆 Demonstrating Benchmark Dataset Generation")
        print("-" * 40)
        
        # Get all available specifications
        specs = create_standard_dataset_specifications()
        
        # Generate benchmark datasets
        benchmark_results = {}
        
        for spec_name in specs.keys():
            print(f"   Generating {spec_name}...")
            
            try:
                data = generate_benchmark_dataset(self.generator, spec_name)
                benchmark_results[spec_name] = {
                    'data_length': len(data['data']),
                    'hurst_exponent': data['specification'].hurst_exponent,
                    'domain_type': data['specification'].domain_type.value,
                    'confounds_applied': [c.value for c in data['confounds_applied']],
                    'metadata': data['metadata']
                }
                
                # Save data
                data_path = self.output_dir / "datasets" / f"{spec_name}.npy"
                np.save(data_path, data['data'])
                
                print(f"     ✅ Saved {len(data['data'])} points to {data_path}")
                
            except Exception as e:
                print(f"     ❌ Error generating {spec_name}: {e}")
        
        # Create summary
        summary_path = self.output_dir / "benchmark_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        print(f"\n✅ Generated {len(benchmark_results)} benchmark datasets")
        print(f"   Summary saved to: {summary_path}")
        print()
        
        return benchmark_results
    
    def demonstrate_performance_analysis(self):
        """Demonstrate performance analysis capabilities."""
        print("📈 Demonstrating Performance Analysis")
        print("-" * 40)
        
        # Generate datasets with different characteristics
        datasets = {}
        
        # Clean data
        clean_spec = DataSpecification(
            n_points=1000,
            hurst_exponent=0.7,
            domain_type=DomainType.GENERAL,
            confound_strength=0.0,
            noise_level=0.0
        )
        datasets['clean'] = self.generator.generate_data(clean_spec)
        
        # Noisy data
        noisy_spec = DataSpecification(
            n_points=1000,
            hurst_exponent=0.7,
            domain_type=DomainType.GENERAL,
            confound_strength=0.0,
            noise_level=0.2
        )
        datasets['noisy'] = self.generator.generate_data(noisy_spec)
        
        # Confounded data
        confounded_spec = DataSpecification(
            n_points=1000,
            hurst_exponent=0.7,
            domain_type=DomainType.GENERAL,
            confound_strength=0.3,
            noise_level=0.1
        )
        datasets['confounded'] = self.generator.generate_data(
            confounded_spec,
            confounds=[ConfoundType.NON_STATIONARITY, ConfoundType.HEAVY_TAILS]
        )
        
        # Analyze performance characteristics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Performance Analysis: Data Quality vs. Complexity", fontsize=16)
        
        for i, (name, data) in enumerate(datasets.items()):
            row, col = i // 2, i % 2
            
            # Plot time series
            axes[row, col].plot(data['data'], linewidth=1)
            axes[row, col].set_title(f'{name.title()} Data')
            axes[row, col].set_xlabel('Time')
            if col == 0:
                axes[row, col].set_ylabel('Value')
            
            # Add statistics
            stats_text = f"Mean: {np.mean(data['data']):.3f}\nStd: {np.std(data['data']):.3f}\nH: {data['specification'].hurst_exponent:.3f}"
            axes[row, col].text(0.02, 0.98, stats_text, transform=axes[row, col].transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Performance comparison plot
        data_quality = ['Clean', 'Noisy', 'Confounded']
        complexity_scores = [1.0, 2.0, 4.0]  # Relative complexity scores
        
        axes[1, 1].bar(data_quality, complexity_scores, color=['green', 'orange', 'red'])
        axes[1, 1].set_title('Data Complexity Comparison')
        axes[1, 1].set_ylabel('Complexity Score')
        axes[1, 1].set_ylim(0, 5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Generated performance analysis plots")
        print()
    
    def run_comprehensive_demo(self):
        """Run the complete demonstration."""
        print("🎯 Starting Comprehensive Synthetic Data Generation Demo")
        print("=" * 60)
        
        try:
            # 1. Base Models
            self.demonstrate_base_models()
            
            # 2. Confounds
            self.demonstrate_confounds()
            
            # 3. Domain-specific patterns
            self.demonstrate_domain_specific_patterns()
            
            # 4. Dataset specifications
            self.demonstrate_dataset_specifications()
            
            # 5. Benchmark generation
            self.demonstrate_benchmark_generation()
            
            # 6. Performance analysis
            self.demonstrate_performance_analysis()
            
            print("🎉 Demo completed successfully!")
            print(f"📁 All outputs saved to: {self.output_dir.absolute()}")
            print()
            print("Generated files:")
            print(f"   📊 Plots: {self.output_dir / 'plots'}")
            print(f"   📁 Datasets: {self.output_dir / 'datasets'}")
            print(f"   📋 Specifications: {self.output_dir / 'specifications'}")
            print()
            print("Next steps:")
            print("   1. Review the generated plots and data")
            print("   2. Use the specifications for your own datasets")
            print("   3. Run benchmarks with the generated data")
            print("   4. Customize parameters for your research needs")
            
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the demonstration."""
    demo = ComprehensiveSyntheticDataDemo()
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    main()
