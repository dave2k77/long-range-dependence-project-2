#!/usr/bin/env python3
"""
High-Performance Synthetic Data Generation Demonstration

This script demonstrates high-performance synthetic data generation capabilities
for large-scale benchmarking and research, including:

1. Parallel data generation
2. Memory-efficient processing
3. Large-scale dataset creation
4. Performance profiling
5. Batch processing optimization

Usage:
    python high_performance_synthetic_data_demo.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import multiprocessing as mp
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generation.synthetic_data_generator import (
    SyntheticDataGenerator, 
    DataSpecification, 
    DomainType,
    create_standard_dataset_specifications
)
from data_generation.dataset_specifications import (
    DatasetSpecification as DatasetSpec,
    DatasetMetadata,
    DatasetProperties,
    ConfoundDescription,
    BenchmarkProtocol,
    DatasetFormat
)

# Suppress warnings for performance
warnings.filterwarnings('ignore')

class HighPerformanceSyntheticDataDemo:
    """High-performance demonstration of synthetic data generation."""
    
    def __init__(self, output_dir: str = "high_performance_outputs"):
        """Initialize the high-performance demonstration."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_metrics = {}
        self.memory_usage = []
        
        # Create subdirectories
        (self.output_dir / "large_datasets").mkdir(exist_ok=True)
        (self.output_dir / "performance_plots").mkdir(exist_ok=True)
        (self.output_dir / "profiles").mkdir(exist_ok=True)
        
        print("🚀 High-Performance Synthetic Data Generation Demo")
        print("=" * 60)
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"CPU cores: {mp.cpu_count()}")
        print(f"Available memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print()
    
    def monitor_performance(self, func):
        """Decorator to monitor function performance."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            self.performance_metrics[func.__name__] = {
                'execution_time': execution_time,
                'memory_used': memory_used,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"⏱️  {func.__name__}: {execution_time:.3f}s, Memory: {memory_used:+.1f}MB")
            
            return result
        return wrapper
    
    def generate_large_scale_datasets(self, n_points: int = 100000, n_datasets: int = 10):
        """Generate large-scale datasets efficiently."""
        print(f"📊 Generating {n_datasets} large datasets with {n_points:,} points each")
        
        # Initialize generator
        generator = SyntheticDataGenerator(random_seed=42)
        
        # Create specifications for different Hurst exponents
        hurst_values = np.linspace(0.1, 0.9, n_datasets)
        
        datasets = {}
        for i, hurst in enumerate(hurst_values):
            spec = DataSpecification(
                n_points=n_points,
                hurst_exponent=float(hurst),
                domain_type=DomainType.GENERAL,
                confound_strength=0.1,
                noise_level=0.05
            )
            
            print(f"   Generating dataset {i+1}/{n_datasets} (H={hurst:.3f})...")
            data = generator.generate_data(spec)
            
            # Save efficiently using numpy
            filename = f"large_dataset_H{hurst:.3f}_{n_points}.npy"
            filepath = self.output_dir / "large_datasets" / filename
            np.save(filepath, data['data'])
            
            datasets[f"dataset_{i+1}"] = {
                'hurst': hurst,
                'filepath': str(filepath),
                'data_length': len(data['data']),
                'file_size_mb': filepath.stat().st_size / (1024**2)
            }
        
        # Save metadata
        metadata_path = self.output_dir / "large_datasets" / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(datasets, f, indent=2, default=str)
        
        print(f"✅ Generated {n_datasets} large datasets")
        print(f"   Total data points: {n_datasets * n_points:,}")
        print(f"   Total file size: {sum(d['file_size_mb'] for d in datasets.values()):.1f} MB")
        
        return datasets
    
    def parallel_data_generation(self, n_points: int = 50000, n_datasets: int = 20):
        """Generate datasets in parallel for improved performance."""
        print(f"🔄 Generating {n_datasets} datasets in parallel ({n_points:,} points each)")
        
        # Prepare arguments for parallel processing
        hurst_values = np.linspace(0.1, 0.9, n_datasets)
        args_list = [(i, hurst, n_points) for i, hurst in enumerate(hurst_values)]
        
        # Use multiprocessing for parallel generation
        with mp.Pool(processes=min(mp.cpu_count(), n_datasets)) as pool:
            results = pool.map(self._generate_single_dataset_worker, args_list)
        
        # Save results
        datasets = {}
        for dataset_id, hurst, data in results:
            filename = f"parallel_dataset_{dataset_id}_H{hurst:.3f}_{n_points}.npy"
            filepath = self.output_dir / "large_datasets" / filename
            np.save(filepath, data)
            
            datasets[f"parallel_{dataset_id}"] = {
                'hurst': hurst,
                'filepath': str(filepath),
                'data_length': len(data),
                'file_size_mb': filepath.stat().st_size / (1024**2)
            }
        
        print(f"✅ Generated {n_datasets} datasets in parallel")
        return datasets
    
    def _generate_single_dataset_worker(self, args):
        """Worker function for parallel data generation."""
        dataset_id, hurst, n_points = args
        generator = SyntheticDataGenerator(random_seed=42 + dataset_id)
        
        spec = DataSpecification(
            n_points=n_points,
            hurst_exponent=float(hurst),
            domain_type=DomainType.GENERAL,
            confound_strength=0.1,
            noise_level=0.05
        )
        
        data = generator.generate_data(spec)
        return dataset_id, hurst, data['data']
    
    def generate_mixed_domain_benchmark(self):
        """Generate a comprehensive mixed-domain benchmark dataset."""
        print("🌍 Generating mixed-domain benchmark dataset")
        
        # Get standard specifications
        specs = create_standard_dataset_specifications()
        generator = SyntheticDataGenerator(random_seed=42)
        
        benchmark_datasets = {}
        total_points = 0
        
        for spec_name, spec in specs.items():
            print(f"   Generating {spec_name}...")
            
            # Generate data
            data = generator.generate_data(spec)
            
            # Save with metadata
            filename = f"benchmark_{spec_name}.npy"
            filepath = self.output_dir / "large_datasets" / filename
            np.save(filepath, data['data'])
            
            # Save metadata separately
            metadata_filename = f"benchmark_{spec_name}_metadata.json"
            metadata_filepath = self.output_dir / "large_datasets" / metadata_filename
            
            metadata = {
                'specification': {
                    'n_points': spec.n_points,
                    'hurst_exponent': spec.hurst_exponent,
                    'domain_type': spec.domain_type.value,
                    'confound_strength': spec.confound_strength,
                    'noise_level': spec.noise_level
                },
                'generated_data': {
                    'data_length': len(data['data']),
                    'filepath': str(filepath),
                    'file_size_mb': filepath.stat().st_size / (1024**2)
                }
            }
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            benchmark_datasets[spec_name] = metadata
            total_points += len(data['data'])
        
        print(f"✅ Generated mixed-domain benchmark with {len(benchmark_datasets)} datasets")
        print(f"   Total data points: {total_points:,}")
        
        return benchmark_datasets
    
    def memory_efficient_processing(self, n_points: int = 1000000):
        """Demonstrate memory-efficient processing of very large datasets."""
        print(f"💾 Memory-efficient processing of {n_points:,} points")
        
        # Generate data in chunks to manage memory
        chunk_size = 100000
        n_chunks = n_points // chunk_size
        
        generator = SyntheticDataGenerator(random_seed=42)
        
        # Process in chunks
        processed_chunks = []
        
        for chunk_idx in range(n_chunks):
            print(f"   Processing chunk {chunk_idx + 1}/{n_chunks}...")
            
            # Generate chunk
            spec = DataSpecification(
                n_points=chunk_size,
                hurst_exponent=0.7,
                domain_type=DomainType.GENERAL,
                confound_strength=0.1,
                noise_level=0.05
            )
            
            chunk_data = generator.generate_data(spec)
            processed_chunks.append(chunk_data['data'])
            
            # Monitor memory usage
            memory_usage = psutil.Process().memory_info().rss / (1024**2)
            self.memory_usage.append({
                'chunk': chunk_idx + 1,
                'memory_mb': memory_usage,
                'timestamp': datetime.now().isoformat()
            })
        
        # Combine chunks efficiently
        print("   Combining chunks...")
        combined_data = np.concatenate(processed_chunks)
        
        # Save combined dataset
        output_path = self.output_dir / "large_datasets" / f"memory_efficient_{n_points}.npy"
        np.save(output_path, combined_data)
        
        print(f"✅ Memory-efficient processing completed")
        print(f"   Final dataset size: {len(combined_data):,} points")
        print(f"   File size: {output_path.stat().st_size / (1024**2):.1f} MB")
        
        return combined_data
    
    def analyze_performance(self):
        """Analyze and visualize performance metrics."""
        print("📈 Analyzing performance metrics")
        
        if not self.performance_metrics:
            print("   No performance metrics available")
            return
        
        # Create performance summary
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("High-Performance Synthetic Data Generation Analysis", fontsize=16)
        
        # Execution time comparison
        functions = list(self.performance_metrics.keys())
        execution_times = [self.performance_metrics[f]['execution_time'] for f in functions]
        
        axes[0, 0].bar(functions, execution_times, color='skyblue')
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        memory_used = [self.performance_metrics[f]['memory_used'] for f in functions]
        
        axes[0, 1].bar(functions, memory_used, color='lightcoral')
        axes[0, 1].set_title('Memory Usage Comparison')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Performance efficiency (points per second)
        total_points = {
            'generate_large_scale_datasets': 100000 * 10,
            'parallel_data_generation': 50000 * 20,
            'generate_mixed_domain_benchmark': 5000 * 15,  # Approximate
            'memory_efficient_processing': 1000000
        }
        
        efficiency = []
        for f in functions:
            if f in total_points:
                points_per_second = total_points[f] / self.performance_metrics[f]['execution_time']
                efficiency.append(points_per_second)
            else:
                efficiency.append(0)
        
        axes[1, 0].bar(functions, efficiency, color='lightgreen')
        axes[1, 0].set_title('Generation Efficiency')
        axes[1, 0].set_ylabel('Points per Second')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage over time (if available)
        if self.memory_usage:
            chunks = [m['chunk'] for m in self.memory_usage]
            memory = [m['memory_mb'] for m in self.memory_usage]
            
            axes[1, 1].plot(chunks, memory, 'o-', color='purple', linewidth=2)
            axes[1, 1].set_title('Memory Usage Over Time')
            axes[1, 1].set_xlabel('Chunk Number')
            axes[1, 1].set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_plots" / "performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save performance summary
        summary_path = self.output_dir / "profiles" / "performance_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
        
        print(f"✅ Performance analysis completed")
        print(f"   Summary saved to: {summary_path}")
    
    def run_high_performance_demo(self):
        """Run the complete high-performance demonstration."""
        print("🎯 Starting High-Performance Synthetic Data Generation Demo")
        print("=" * 60)
        
        try:
            # 1. Large-scale dataset generation
            self.generate_large_scale_datasets(n_points=100000, n_datasets=10)
            
            # 2. Parallel data generation
            self.parallel_data_generation(n_points=50000, n_datasets=20)
            
            # 3. Mixed-domain benchmark
            self.generate_mixed_domain_benchmark()
            
            # 4. Memory-efficient processing
            self.memory_efficient_processing(n_points=1000000)
            
            # 5. Performance analysis
            self.analyze_performance()
            
            print("🎉 High-performance demo completed successfully!")
            print(f"📁 All outputs saved to: {self.output_dir.absolute()}")
            print()
            print("Generated files:")
            print(f"   📊 Large datasets: {self.output_dir / 'large_datasets'}")
            print(f"   📈 Performance plots: {self.output_dir / 'performance_plots'}")
            print(f"   📋 Performance profiles: {self.output_dir / 'profiles'}")
            print()
            print("Performance highlights:")
            for func_name, metrics in self.performance_metrics.items():
                print(f"   {func_name}: {metrics['execution_time']:.3f}s, {metrics['memory_used']:+.1f}MB")
            
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the high-performance demonstration."""
    demo = HighPerformanceSyntheticDataDemo()
    demo.run_high_performance_demo()


if __name__ == "__main__":
    main()
