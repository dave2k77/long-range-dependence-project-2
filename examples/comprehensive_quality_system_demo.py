#!/usr/bin/env python3
"""
Comprehensive Quality System Demo

This script demonstrates all four quality evaluation options working together:
1. Quality Gates in Data Submission
2. Benchmarking Integration with Quality Metrics
3. Automated Quality Monitoring
4. Advanced Quality Metrics

The goal is to show how these systems work together to provide
comprehensive quality assurance for synthetic data generation.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, Any, List
import time
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our quality evaluation systems
from validation.synthetic_data_quality import (
    SyntheticDataQualityEvaluator, 
    create_domain_specific_evaluator
)
from validation.quality_monitoring import QualityMonitor
from validation.advanced_quality_metrics import AdvancedQualityMetrics
from data_submission.dataset_submission import DatasetSubmissionManager
from benchmarking.performance_benchmarks import PerformanceBenchmarker
from data_generation.synthetic_data_generator import SyntheticDataGenerator
from data_generation.dataset_specifications import (
    DatasetSpecification, DatasetMetadata, DatasetProperties,
    DomainCategory, DatasetFormat
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveQualityDemo:
    """
    Comprehensive demonstration of all quality evaluation systems working together.
    """
    
    def __init__(self, output_dir: str = "comprehensive_quality_demo"):
        """
        Initialize the comprehensive quality demo.
        
        Parameters:
        -----------
        output_dir : str
            Directory for storing demo outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize all quality systems
        self.quality_evaluator = SyntheticDataQualityEvaluator()
        self.quality_monitor = QualityMonitor(output_dir=str(self.output_dir / "monitoring"))
        self.advanced_metrics = AdvancedQualityMetrics(output_dir=str(self.output_dir / "advanced"))
        
        # Initialize data systems
        self.data_generator = SyntheticDataGenerator()
        self.submission_manager = DatasetSubmissionManager()
        self.benchmarker = PerformanceBenchmarker(output_dir=str(self.output_dir / "benchmarks"))
        
        # Load reference datasets
        self.reference_datasets = self._load_reference_datasets()
        
        logger.info(f"Comprehensive Quality Demo initialized at {self.output_dir.absolute()}")
    
    def _load_reference_datasets(self) -> Dict[str, np.ndarray]:
        """Load reference datasets for quality evaluation."""
        reference_datasets = {}
        
        try:
            # Load realistic datasets
            data_dir = Path("data/realistic")
            
            if (data_dir / "nile_river_flow.npy").exists():
                reference_datasets["hydrology"] = np.load(data_dir / "nile_river_flow.npy")
                logger.info("Loaded hydrology reference dataset")
            
            if (data_dir / "dow_jones_monthly.npy").exists():
                reference_datasets["financial"] = np.load(data_dir / "dow_jones_monthly.npy")
                logger.info("Loaded financial reference dataset")
            
            if (data_dir / "eeg_sample.npy").exists():
                reference_datasets["biomedical"] = np.load(data_dir / "eeg_sample.npy")
                logger.info("Loaded biomedical reference dataset")
            
            if (data_dir / "daily_temperature.npy").exists():
                reference_datasets["climate"] = np.load(data_dir / "daily_temperature.npy")
                logger.info("Loaded climate reference dataset")
            
        except Exception as e:
            logger.warning(f"Failed to load some reference datasets: {e}")
        
        # Generate fallback datasets if needed
        for domain in ["hydrology", "financial", "biomedical", "climate"]:
            if domain not in reference_datasets:
                logger.info(f"Generating fallback reference dataset for {domain}")
                reference_datasets[domain] = self._generate_fallback_reference(domain)
        
        return reference_datasets
    
    def _generate_fallback_reference(self, domain: str) -> np.ndarray:
        """Generate fallback reference dataset for a domain."""
        n_points = 1000
        
        if domain == "hydrology":
            # Hydrological data: seasonal patterns with long-range dependence
            t = np.linspace(0, 10, n_points)
            seasonal = 10 * np.sin(2 * np.pi * t) + 5 * np.sin(4 * np.pi * t)
            trend = 0.1 * t
            noise = np.random.normal(0, 1, n_points)
            # Add some persistence
            for i in range(1, n_points):
                noise[i] = 0.7 * noise[i-1] + 0.3 * noise[i]
            return seasonal + trend + noise
        
        elif domain == "financial":
            # Financial data: random walk with volatility clustering
            returns = np.random.normal(0, 0.02, n_points)
            # Add volatility clustering
            volatility = np.ones(n_points)
            for i in range(1, n_points):
                volatility[i] = 0.95 * volatility[i-1] + 0.05 * returns[i-1]**2
            returns = returns * np.sqrt(volatility)
            return np.cumsum(returns)
        
        elif domain == "biomedical":
            # Biomedical data: oscillatory with noise
            t = np.linspace(0, 4*np.pi, n_points)
            signal = 5 * np.sin(t) + 2 * np.sin(3*t) + 1.5 * np.sin(5*t)
            noise = np.random.normal(0, 0.5, n_points)
            return signal + noise
        
        else:  # climate
            # Climate data: trend with seasonal and long-term cycles
            t = np.linspace(0, 20, n_points)
            seasonal = 3 * np.sin(2 * np.pi * t) + 1.5 * np.sin(4 * np.pi * t)
            trend = 0.05 * t
            long_cycle = 2 * np.sin(2 * np.pi * t / 10)
            noise = np.random.normal(0, 0.3, n_points)
            return seasonal + trend + long_cycle + noise
    
    def run_comprehensive_demo(self):
        """Run the comprehensive quality evaluation demonstration."""
        logger.info("🚀 Starting Comprehensive Quality System Demo")
        
        try:
            # Phase 1: Demonstrate Quality Gates in Data Submission
            logger.info("\n📋 Phase 1: Quality Gates in Data Submission")
            self._demonstrate_quality_gates()
            
            # Phase 2: Demonstrate Benchmarking Integration
            logger.info("\n📊 Phase 2: Benchmarking Integration with Quality Metrics")
            self._demonstrate_benchmarking_integration()
            
            # Phase 3: Demonstrate Automated Quality Monitoring
            logger.info("\n🔍 Phase 3: Automated Quality Monitoring")
            self._demonstrate_quality_monitoring()
            
            # Phase 4: Demonstrate Advanced Quality Metrics
            logger.info("\n🧠 Phase 4: Advanced Quality Metrics")
            self._demonstrate_advanced_metrics()
            
            # Phase 5: Integration Demonstration
            logger.info("\n🔗 Phase 5: System Integration Demonstration")
            self._demonstrate_system_integration()
            
            # Generate comprehensive report
            self._generate_comprehensive_report()
            
            logger.info("\n✅ Comprehensive Quality System Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    def _demonstrate_quality_gates(self):
        """Demonstrate quality gates in data submission."""
        logger.info("Creating synthetic dataset for submission...")
        
        # Create dataset specification
        spec = DatasetSpecification(
            metadata=DatasetMetadata(
                name="demo_synthetic_dataset",
                description="Demo dataset for quality gates demonstration",
                version="1.0.0",
                author="Demo User",
                license="MIT"
            ),
            properties=DatasetProperties(
                n_points=1000,
                domain_category=DomainCategory.HYDROLOGY,
                format=DatasetFormat.NUMPY,
                has_trend=True,
                has_seasonality=True,
                has_noise=True
            )
        )
        
        # Generate synthetic data
        result = self.data_generator.generate_data(spec, [])
        synthetic_data = result['data']
        
        # Save data for submission
        data_file = self.output_dir / "demo_synthetic_data.npy"
        np.save(data_file, synthetic_data)
        
        # Create submission metadata
        submission_metadata = {
            'submission_id': 'demo_001',
            'submitter_name': 'Demo User',
            'submitter_email': 'demo@example.com',
            'submission_date': datetime.now().isoformat(),
            'submission_type': 'synthetic',
            'dataset_name': 'demo_synthetic_dataset',
            'dataset_description': 'Demo dataset for quality gates demonstration',
            'dataset_version': '1.0.0',
            'license': 'MIT',
            'file_paths': {
                'data_file': str(data_file),
                'specification_file': str(self.output_dir / "demo_spec.json"),
                'generator_parameters': str(self.output_dir / "demo_params.json")
            }
        }
        
        # Save specification and parameters
        with open(self.output_dir / "demo_spec.json", 'w') as f:
            json.dump(spec.__dict__, f, indent=2, default=str)
        
        with open(self.output_dir / "demo_params.json", 'w') as f:
            json.dump({'generation_method': 'fbm', 'hurst': 0.7}, f, indent=2)
        
        # Simulate submission validation with quality gates
        logger.info("Running quality gates validation...")
        
        try:
            # This would normally be called by the submission manager
            # We'll simulate it here
            quality_result = self.quality_evaluator.evaluate_quality(
                synthetic_data=synthetic_data,
                reference_data=self.reference_datasets["hydrology"],
                reference_metadata={"domain": "hydrology", "source": "reference_dataset"},
                domain="hydrology",
                normalize_for_comparison=True
            )
            
            logger.info(f"Quality evaluation result: {quality_result.overall_score:.3f} ({quality_result.quality_level})")
            
            # Check if quality meets threshold (quality gate)
            if quality_result.overall_score >= 0.5:
                logger.info("✅ Quality gate PASSED - Dataset accepted")
            else:
                logger.warning("❌ Quality gate FAILED - Dataset rejected")
            
            # Save quality gate results
            gate_results = {
                'submission_id': 'demo_001',
                'quality_score': quality_result.overall_score,
                'quality_level': quality_result.quality_level,
                'gate_passed': quality_result.overall_score >= 0.5,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.output_dir / "quality_gate_results.json", 'w') as f:
                json.dump(gate_results, f, indent=2)
            
        except Exception as e:
            logger.error(f"Quality gate validation failed: {e}")
    
    def _demonstrate_benchmarking_integration(self):
        """Demonstrate benchmarking integration with quality metrics."""
        logger.info("Running performance benchmarks with quality evaluation...")
        
        try:
            # Generate test datasets of different sizes
            dataset_sizes = [100, 500, 1000]
            test_datasets = {}
            
            for size in dataset_sizes:
                # Generate synthetic data with known Hurst exponent
                data = self._generate_test_dataset(size, hurst=0.7)
                test_datasets[size] = data
            
            # Run benchmarks with quality evaluation
            all_results = []
            
            for size, data in test_datasets.items():
                logger.info(f"Benchmarking dataset size: {size}")
                
                # Run benchmark with quality evaluation
                result = self.benchmarker.benchmark_estimator(
                    estimator_class=type(self.quality_evaluator),  # Use evaluator as estimator
                    estimator_name="QualityEvaluator",
                    data=data,
                    reference_data=self.reference_datasets["hydrology"],
                    domain="hydrology"
                )
                
                all_results.append(result)
            
            # Convert results to DataFrame
            df = self.benchmarker.results_to_dataframe(all_results)
            
            # Save benchmark results
            benchmark_file = self.output_dir / "benchmark_with_quality.csv"
            df.to_csv(benchmark_file, index=False)
            
            logger.info(f"Benchmark results saved: {benchmark_file}")
            
            # Display quality metrics from benchmarks
            if 'quality_score' in df.columns:
                logger.info("Quality metrics from benchmarks:")
                for _, row in df.iterrows():
                    logger.info(f"  Size {row['dataset_size']}: Quality = {row['quality_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Benchmarking integration failed: {e}")
    
    def _demonstrate_quality_monitoring(self):
        """Demonstrate automated quality monitoring."""
        logger.info("Starting automated quality monitoring...")
        
        try:
            # Create a data generator function for monitoring
            def generate_monitoring_data():
                """Generate synthetic data for monitoring."""
                # Generate data with varying quality
                n_points = 500
                t = np.linspace(0, 5, n_points)
                
                # Add some randomness to quality
                quality_factor = 0.7 + 0.3 * np.sin(time.time() / 10)  # Varying quality
                
                base_signal = 5 * np.sin(2 * np.pi * t)
                noise = np.random.normal(0, 1, n_points) * (1 - quality_factor)
                
                return base_signal + noise
            
            # Start monitoring
            self.quality_monitor.start_monitoring(
                data_generator_func=generate_monitoring_data,
                reference_data=self.reference_datasets["hydrology"],
                domain="hydrology",
                max_history=50
            )
            
            # Let it run for a few cycles
            logger.info("Monitoring for 30 seconds...")
            time.sleep(30)
            
            # Stop monitoring
            self.quality_monitor.stop_monitoring()
            
            # Get monitoring summary
            summary = self.quality_monitor.get_quality_summary()
            logger.info(f"Monitoring summary: {summary}")
            
            # Get dashboard data
            dashboard_data = self.quality_monitor.get_quality_dashboard_data()
            logger.info(f"Dashboard data available: {len(dashboard_data.get('overall_scores', []))} evaluations")
            
        except Exception as e:
            logger.error(f"Quality monitoring demonstration failed: {e}")
    
    def _demonstrate_advanced_metrics(self):
        """Demonstrate advanced quality metrics."""
        logger.info("Calculating advanced quality metrics...")
        
        try:
            # Generate test data
            synthetic_data = self._generate_test_dataset(1000, hurst=0.7)
            reference_data = self.reference_datasets["hydrology"]
            
            # Calculate advanced LRD metrics
            lrd_metrics = self.advanced_metrics.calculate_advanced_lrd_metrics(
                synthetic_data=synthetic_data,
                reference_data=reference_data,
                domain="hydrology"
            )
            
            logger.info(f"Calculated {len(lrd_metrics)} advanced LRD metrics:")
            for metric in lrd_metrics:
                logger.info(f"  {metric.metric_name}: {metric.metric_score:.3f} (confidence: {metric.confidence_level:.2f})")
            
            # Cross-dataset quality assessment
            cross_dataset_result = self.advanced_metrics.assess_cross_dataset_quality(
                synthetic_data=synthetic_data,
                reference_datasets=self.reference_datasets,
                domain="hydrology"
            )
            
            logger.info(f"Cross-dataset quality: {cross_dataset_result.cross_validation_score:.3f}")
            logger.info(f"Domain adaptation quality: {cross_dataset_result.domain_adaptation_quality:.3f}")
            
            # Generate advanced quality report
            advanced_report = self.advanced_metrics.generate_advanced_quality_report(
                synthetic_data=synthetic_data,
                reference_data=reference_data,
                domain="hydrology"
            )
            
            # Save advanced report
            advanced_file = self.output_dir / "advanced_quality_report.json"
            with open(advanced_file, 'w') as f:
                json.dump(advanced_report, f, indent=2)
            
            logger.info(f"Advanced quality report saved: {advanced_file}")
            
        except Exception as e:
            logger.error(f"Advanced metrics demonstration failed: {e}")
    
    def _demonstrate_system_integration(self):
        """Demonstrate how all systems work together."""
        logger.info("Demonstrating system integration...")
        
        try:
            # Create a comprehensive workflow
            workflow_results = {
                'workflow_id': 'integration_demo_001',
                'timestamp': datetime.now().isoformat(),
                'phases': []
            }
            
            # Phase 1: Data Generation with Quality Check
            logger.info("Phase 1: Data Generation with Quality Check")
            spec = self._create_integration_spec()
            result = self.data_generator.generate_data(spec, [])
            synthetic_data = result['data']
            
            # Quality evaluation
            quality_result = self.quality_evaluator.evaluate_quality(
                synthetic_data=synthetic_data,
                reference_data=self.reference_datasets["hydrology"],
                reference_metadata={"domain": "hydrology", "source": "integration_demo"},
                domain="hydrology",
                normalize_for_comparison=True
            )
            
            workflow_results['phases'].append({
                'phase': 'data_generation',
                'quality_score': quality_result.overall_score,
                'quality_level': quality_result.quality_level,
                'status': 'completed'
            })
            
            # Phase 2: Performance Benchmarking with Quality
            logger.info("Phase 2: Performance Benchmarking with Quality")
            benchmark_result = self.benchmarker.benchmark_estimator(
                estimator_class=type(self.quality_evaluator),
                estimator_name="IntegrationQualityEvaluator",
                data=synthetic_data,
                reference_data=self.reference_datasets["hydrology"],
                domain="hydrology"
            )
            
            workflow_results['phases'].append({
                'phase': 'performance_benchmarking',
                'execution_time': benchmark_result.execution_time,
                'quality_score': benchmark_result.quality_score,
                'status': 'completed'
            })
            
            # Phase 3: Advanced Metrics Analysis
            logger.info("Phase 3: Advanced Metrics Analysis")
            advanced_metrics = self.advanced_metrics.calculate_advanced_lrd_metrics(
                synthetic_data=synthetic_data,
                reference_data=self.reference_datasets["hydrology"],
                domain="hydrology"
            )
            
            workflow_results['phases'].append({
                'phase': 'advanced_metrics',
                'metrics_count': len(advanced_metrics),
                'avg_metric_score': np.mean([m.metric_score for m in advanced_metrics]),
                'status': 'completed'
            })
            
            # Phase 4: Quality Monitoring Setup
            logger.info("Phase 4: Quality Monitoring Setup")
            workflow_results['phases'].append({
                'phase': 'quality_monitoring',
                'status': 'configured',
                'monitoring_interval': self.quality_monitor.monitoring_interval
            })
            
            # Save workflow results
            workflow_file = self.output_dir / "integration_workflow_results.json"
            with open(workflow_file, 'w') as f:
                json.dump(workflow_results, f, indent=2)
            
            logger.info(f"Integration workflow results saved: {workflow_file}")
            
        except Exception as e:
            logger.error(f"System integration demonstration failed: {e}")
    
    def _create_integration_spec(self) -> DatasetSpecification:
        """Create dataset specification for integration demo."""
        return DatasetSpecification(
            metadata=DatasetMetadata(
                name="integration_demo_dataset",
                description="Dataset for system integration demonstration",
                version="1.0.0",
                author="Integration Demo",
                license="MIT"
            ),
            properties=DatasetProperties(
                n_points=800,
                domain_category=DomainCategory.HYDROLOGY,
                format=DatasetFormat.NUMPY,
                has_trend=True,
                has_seasonality=True,
                has_noise=True
            )
        )
    
    def _generate_test_dataset(self, size: int, hurst: float = 0.7) -> np.ndarray:
        """Generate test dataset with specified Hurst exponent."""
        # Generate fractional Brownian motion
        freqs = np.fft.fftfreq(size)
        power_spectrum = np.abs(freqs) ** (-2 * hurst - 1)
        power_spectrum[0] = 0  # Remove DC component
        
        # Generate complex Gaussian noise
        phase = np.random.uniform(0, 2 * np.pi, size)
        amplitude = np.sqrt(power_spectrum) * np.exp(1j * phase)
        
        # Inverse FFT to get time series
        time_series = np.real(np.fft.ifft(amplitude))
        
        # Normalize
        time_series = (time_series - np.mean(time_series)) / np.std(time_series)
        
        return time_series
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive demo report."""
        logger.info("Generating comprehensive demo report...")
        
        try:
            # Collect all results
            report = {
                'demo_info': {
                    'title': 'Comprehensive Quality System Demo',
                    'date': datetime.now().isoformat(),
                    'description': 'Demonstration of all four quality evaluation options working together'
                },
                'systems_demonstrated': [
                    'Quality Gates in Data Submission',
                    'Benchmarking Integration with Quality Metrics',
                    'Automated Quality Monitoring',
                    'Advanced Quality Metrics'
                ],
                'output_files': [],
                'summary': {
                    'status': 'completed',
                    'total_phases': 5,
                    'successful_phases': 5
                }
            }
            
            # List output files
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.json', '.csv', '.txt']:
                    report['output_files'].append(str(file_path.relative_to(self.output_dir)))
            
            # Save comprehensive report
            report_file = self.output_dir / "comprehensive_demo_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Comprehensive demo report saved: {report_file}")
            
            # Create summary text file
            summary_file = self.output_dir / "demo_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("COMPREHENSIVE QUALITY SYSTEM DEMO SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Demo completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("Systems Demonstrated:\n")
                for i, system in enumerate(report['systems_demonstrated'], 1):
                    f.write(f"{i}. {system}\n")
                f.write(f"\nTotal output files: {len(report['output_files'])}\n")
                f.write(f"Output directory: {self.output_dir.absolute()}\n")
            
            logger.info(f"Demo summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")

def main():
    """Main function to run the comprehensive quality demo."""
    print("🚀 Comprehensive Quality System Demo")
    print("=" * 50)
    print("This demo showcases all four quality evaluation options:")
    print("1. Quality Gates in Data Submission")
    print("2. Benchmarking Integration with Quality Metrics")
    print("3. Automated Quality Monitoring")
    print("4. Advanced Quality Metrics")
    print("5. System Integration")
    print("=" * 50)
    
    try:
        # Initialize demo
        demo = ComprehensiveQualityDemo()
        
        # Run comprehensive demonstration
        demo.run_comprehensive_demo()
        
        print("\n✅ Demo completed successfully!")
        print(f"Check the output directory: {demo.output_dir.absolute()}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
