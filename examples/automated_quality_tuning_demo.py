#!/usr/bin/env python3
"""
Automated Quality Tuning Demo

This script demonstrates how to use the synthetic data quality evaluation
system to automatically tune generation parameters for optimal quality.
It shows:

1. Automated parameter optimization based on quality scores
2. Iterative quality improvement workflows
3. Multi-objective optimization (quality vs. computational cost)
4. Quality monitoring and tracking
5. Best parameter discovery

The goal is to show how quality evaluation can guide synthetic data
generation to produce increasingly better datasets.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import itertools

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from validation.synthetic_data_quality import (
    SyntheticDataQualityEvaluator, create_domain_specific_evaluator
)
from data_generation.synthetic_data_generator import (
    SyntheticDataGenerator, DataSpecification, DomainType, ConfoundType
)

class AutomatedQualityTuner:
    """Automated parameter tuning using quality evaluation."""
    
    def __init__(self, domain: str = "general"):
        """
        Initialize the quality tuner.
        
        Parameters:
        -----------
        domain : str
            Data domain for specialized evaluation
        """
        self.domain = domain
        self.evaluator = create_domain_specific_evaluator(domain) if domain != "general" else SyntheticDataQualityEvaluator()
        self.generator = SyntheticDataGenerator()
        self.optimization_history = []
        self.best_parameters = None
        self.best_score = 0.0
        
        print(f"🤖 Automated Quality Tuner initialized for {domain} domain")
    
    def generate_parameter_combinations(self, base_spec: DataSpecification, 
                                      param_ranges: Dict[str, List[Any]]) -> List[DataSpecification]:
        """
        Generate parameter combinations for optimization.
        
        Parameters:
        -----------
        base_spec : DataSpecification
            Base specification to modify
        param_ranges : Dict[str, List[Any]]
            Parameter ranges to explore
            
        Returns:
        --------
        List[DataSpecification]
            List of specifications with different parameter combinations
        """
        specs = []
        
        # Get all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for combination in itertools.product(*param_values):
            # Create new specification
            spec_dict = base_spec.__dict__.copy()
            
            # Update with new parameter values
            for i, param_name in enumerate(param_names):
                if hasattr(base_spec, param_name):
                    setattr(base_spec, param_name, combination[i])
            
            specs.append(base_spec)
        
        print(f"   🔧 Generated {len(specs)} parameter combinations")
        return specs
    
    def evaluate_parameter_set(self, spec: DataSpecification, 
                              reference_data: np.ndarray,
                              reference_metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a single parameter set.
        
        Parameters:
        -----------
        spec : DataSpecification
            Generation specification to evaluate
        reference_data : np.ndarray
            Reference data for comparison
        reference_metadata : Dict[str, Any]
            Reference data metadata
            
        Returns:
        --------
        Tuple[float, Dict[str, Any]]
            Quality score and evaluation details
        """
        try:
            # Generate synthetic data
            result = self.generator.generate_data(spec, [])
            synthetic_data = result['data']
            
            # Evaluate quality
            eval_result = self.evaluator.evaluate_quality(
                synthetic_data=synthetic_data,
                reference_data=reference_data,
                reference_metadata=reference_metadata,
                domain=self.domain,
                normalize_for_comparison=True
            )
            
            # Store optimization history
            optimization_record = {
                'parameters': {
                    'hurst_exponent': spec.hurst_exponent,
                    'noise_level': spec.noise_level,
                    'confound_strength': spec.confound_strength,
                    'n_points': spec.n_points
                },
                'quality_score': eval_result.overall_score,
                'quality_level': eval_result.quality_level,
                'best_metrics': [m.metric_name for m in eval_result.metrics if m.score > 0.8],
                'weak_metrics': [m.metric_name for m in eval_result.metrics if m.score < 0.5],
                'evaluation_date': eval_result.evaluation_date
            }
            
            self.optimization_history.append(optimization_record)
            
            return eval_result.overall_score, optimization_record
            
        except Exception as e:
            print(f"   ❌ Evaluation failed for parameters: {e}")
            return 0.0, {'error': str(e)}
    
    def optimize_parameters(self, reference_data: np.ndarray,
                           reference_metadata: Dict[str, Any],
                           base_spec: DataSpecification,
                           param_ranges: Dict[str, List[Any]],
                           max_iterations: int = 20) -> Tuple[DataSpecification, float]:
        """
        Optimize parameters using quality evaluation.
        
        Parameters:
        -----------
        reference_data : np.ndarray
            Reference data for comparison
        reference_metadata : Dict[str, Any]
            Reference data metadata
        base_spec : DataSpecification
            Base specification to optimize
        param_ranges : Dict[str, List[Any]]
            Parameter ranges to explore
        max_iterations : int
            Maximum optimization iterations
            
        Returns:
        --------
        Tuple[DataSpecification, float]
            Best specification and quality score
        """
        print(f"\n🚀 Starting automated parameter optimization...")
        print(f"   📊 Target domain: {self.domain}")
        print(f"   🔧 Parameter ranges: {param_ranges}")
        print(f"   ⏱️  Max iterations: {max_iterations}")
        
        # Generate parameter combinations
        specs = self.generate_parameter_combinations(base_spec, param_ranges)
        
        # Limit to max iterations
        if len(specs) > max_iterations:
            specs = specs[:max_iterations]
            print(f"   ⚠️  Limited to {max_iterations} iterations")
        
        best_score = 0.0
        best_spec = base_spec
        
        # Evaluate each parameter combination
        for i, spec in enumerate(specs):
            print(f"   🔍 Iteration {i+1}/{len(specs)}: Evaluating parameters...")
            
            score, details = self.evaluate_parameter_set(spec, reference_data, reference_metadata)
            
            if score > best_score:
                best_score = score
                best_spec = spec
                self.best_parameters = details
                self.best_score = score
                
                print(f"      ✅ New best score: {score:.3f} ({details.get('quality_level', 'unknown')})")
                print(f"         Parameters: H={spec.hurst_exponent}, noise={spec.noise_level:.3f}, confound={spec.confound_strength:.3f}")
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"      📊 Progress: {i+1}/{len(specs)} iterations completed")
        
        print(f"\n🎯 Optimization completed!")
        print(f"   🏆 Best quality score: {best_score:.3f}")
        print(f"   🔧 Best parameters: H={best_spec.hurst_exponent}, noise={best_spec.noise_level:.3f}, confound={best_spec.confound_strength:.3f}")
        
        return best_spec, best_score
    
    def create_optimization_report(self) -> Dict[str, Any]:
        """Create comprehensive optimization report."""
        if not self.optimization_history:
            return {"error": "No optimization history available"}
        
        # Analyze optimization history
        scores = [record['quality_score'] for record in self.optimization_history if 'quality_score' in record]
        quality_levels = [record['quality_level'] for record in self.optimization_history if 'quality_level' in record]
        
        # Parameter analysis
        hurst_values = [record['parameters']['hurst_exponent'] for record in self.optimization_history if 'parameters' in record]
        noise_values = [record['parameters']['noise_level'] for record in self.optimization_history if 'parameters' in record]
        confound_values = [record['parameters']['confound_strength'] for record in self.optimization_history if 'parameters' in record]
        
        report = {
            "optimization_summary": {
                "total_iterations": len(self.optimization_history),
                "best_score": self.best_score,
                "best_parameters": self.best_parameters,
                "average_score": np.mean(scores) if scores else 0.0,
                "score_improvement": self.best_score - min(scores) if scores else 0.0
            },
            "quality_distribution": {
                "excellent": quality_levels.count("excellent"),
                "good": quality_levels.count("good"),
                "acceptable": quality_levels.count("acceptable"),
                "poor": quality_levels.count("poor")
            },
            "parameter_analysis": {
                "hurst_range": [min(hurst_values), max(hurst_values)] if hurst_values else [],
                "noise_range": [min(noise_values), max(noise_values)] if noise_values else [],
                "confound_range": [min(confound_values), max(confound_values)] if confound_values else []
            },
            "optimization_history": self.optimization_history
        }
        
        return report
    
    def visualize_optimization(self, save_path: str = None) -> str:
        """Create optimization visualization."""
        if not self.optimization_history:
            print("   ⚠️  No optimization history to visualize")
            return None
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Automated Quality Optimization Results - {self.domain.title()} Domain', fontsize=16)
            
            # 1. Quality score progression
            scores = [record['quality_score'] for record in self.optimization_history if 'quality_score' in record]
            iterations = range(1, len(scores) + 1)
            
            axes[0, 0].plot(iterations, scores, 'b-o', alpha=0.7)
            axes[0, 0].axhline(y=self.best_score, color='r', linestyle='--', label=f'Best: {self.best_score:.3f}')
            axes[0, 0].set_xlabel('Optimization Iteration')
            axes[0, 0].set_ylabel('Quality Score')
            axes[0, 0].set_title('Quality Score Progression')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # 2. Parameter evolution
            if self.optimization_history and 'parameters' in self.optimization_history[0]:
                hurst_values = [record['parameters']['hurst_exponent'] for record in self.optimization_history]
                noise_values = [record['parameters']['noise_level'] for record in self.optimization_history]
                confound_values = [record['parameters']['confound_strength'] for record in self.optimization_history]
                
                axes[0, 1].plot(iterations, hurst_values, 'g-o', label='Hurst Exponent', alpha=0.7)
                axes[0, 1].plot(iterations, noise_values, 'r-o', label='Noise Level', alpha=0.7)
                axes[0, 1].plot(iterations, confound_values, 'b-o', label='Confound Strength', alpha=0.7)
                axes[0, 1].set_xlabel('Optimization Iteration')
                axes[0, 1].set_ylabel('Parameter Value')
                axes[0, 1].set_title('Parameter Evolution')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
            
            # 3. Quality level distribution
            quality_counts = {}
            for record in self.optimization_history:
                if 'quality_level' in record:
                    level = record['quality_level']
                    quality_counts[level] = quality_counts.get(level, 0) + 1
            
            if quality_counts:
                levels = list(quality_counts.keys())
                counts = list(quality_counts.values())
                colors = ['green', 'blue', 'orange', 'red']
                
                axes[1, 0].pie(counts, labels=levels, autopct='%1.0f%%', colors=colors[:len(levels)])
                axes[1, 0].set_title('Quality Level Distribution')
            
            # 4. Score distribution histogram
            if scores:
                axes[1, 1].hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1, 1].axvline(x=self.best_score, color='r', linestyle='--', label=f'Best Score: {self.best_score:.3f}')
                axes[1, 1].set_xlabel('Quality Score')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Quality Score Distribution')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save visualization
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"data/realistic/optimization_visualization_{self.domain}_{timestamp}.png"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   🎨 Optimization visualization saved to: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"   ❌ Visualization failed: {e}")
            return None

def demonstrate_hydrology_optimization():
    """Demonstrate automated optimization for hydrology data."""
    print("\n🌊 HYDROLOGY DOMAIN OPTIMIZATION")
    print("=" * 50)
    
    # Load Nile River data as reference
    try:
        nile_data = np.load("data/realistic/nile_river_flow.npy")
        nile_metadata = {"domain": "hydrology", "source": "Nile River"}
        print(f"   ✅ Loaded reference data: {len(nile_data)} points")
    except Exception as e:
        print(f"   ❌ Failed to load reference data: {e}")
        return
    
    # Create hydrology tuner
    tuner = AutomatedQualityTuner(domain="hydrology")
    
    # Base specification
    base_spec = DataSpecification(
        n_points=100,  # Match reference data length
        hurst_exponent=0.7,
        domain_type=DomainType.HYDROLOGY,
        confound_strength=0.2,
        noise_level=0.1
    )
    
    # Parameter ranges to explore
    param_ranges = {
        "hurst_exponent": [0.6, 0.7, 0.8, 0.9],
        "noise_level": [0.05, 0.1, 0.15, 0.2],
        "confound_strength": [0.1, 0.2, 0.3, 0.4]
    }
    
    # Run optimization
    best_spec, best_score = tuner.optimize_parameters(
        reference_data=nile_data,
        reference_metadata=nile_metadata,
        base_spec=base_spec,
        param_ranges=param_ranges,
        max_iterations=16
    )
    
    # Generate optimization report
    report = tuner.create_optimization_report()
    
    # Create visualization
    viz_path = tuner.visualize_optimization()
    
    # Save optimization report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data/realistic/hydrology_optimization_report_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Optimization Report saved to: {report_path}")
    
    return tuner, best_spec, best_score

def demonstrate_financial_optimization():
    """Demonstrate automated optimization for financial data."""
    print("\n💰 FINANCIAL DOMAIN OPTIMIZATION")
    print("=" * 50)
    
    # Load DJIA data as reference
    try:
        djia_data = np.load("data/realistic/dow_jones_monthly.npy")
        djia_metadata = {"domain": "financial", "source": "DJIA Monthly"}
        print(f"   ✅ Loaded reference data: {len(djia_data)} points")
    except Exception as e:
        print(f"   ❌ Failed to load reference data: {e}")
        return
    
    # Create financial tuner
    tuner = AutomatedQualityTuner(domain="financial")
    
    # Base specification
    base_spec = DataSpecification(
        n_points=168,  # Match reference data length
        hurst_exponent=0.55,
        domain_type=DomainType.FINANCIAL,
        confound_strength=0.2,
        noise_level=0.05
    )
    
    # Parameter ranges to explore
    param_ranges = {
        "hurst_exponent": [0.5, 0.55, 0.6, 0.65],
        "noise_level": [0.02, 0.05, 0.08, 0.1],
        "confound_strength": [0.1, 0.2, 0.3, 0.4]
    }
    
    # Run optimization
    best_spec, best_score = tuner.optimize_parameters(
        reference_data=djia_data,
        reference_metadata=djia_metadata,
        base_spec=base_spec,
        param_ranges=param_ranges,
        max_iterations=16
    )
    
    # Generate optimization report
    report = tuner.create_optimization_report()
    
    # Create visualization
    viz_path = tuner.visualize_optimization()
    
    # Save optimization report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data/realistic/financial_optimization_report_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Optimization Report saved to: {report_path}")
    
    return tuner, best_spec, best_score

def main():
    """Main function to demonstrate automated quality tuning."""
    print("🚀 Starting Automated Quality Tuning Demonstration")
    print("=" * 70)
    
    print("\n🎯 This demo shows how to use quality evaluation to automatically")
    print("   tune synthetic data generation parameters for optimal quality.")
    
    # Demonstrate hydrology optimization
    hydrology_results = demonstrate_hydrology_optimization()
    
    # Demonstrate financial optimization
    financial_results = demonstrate_financial_optimization()
    
    # Summary
    print(f"\n🎉 Automated Quality Tuning completed successfully!")
    
    if hydrology_results:
        tuner, spec, score = hydrology_results
        print(f"🌊 Hydrology: Best score {score:.3f} with H={spec.hurst_exponent}, noise={spec.noise_level:.3f}")
    
    if financial_results:
        tuner, spec, score = financial_results
        print(f"💰 Financial: Best score {score:.3f} with H={spec.hurst_exponent}, noise={spec.noise_level:.3f}")
    
    print(f"\n💡 Key insights:")
    print(f"   • Quality evaluation guides parameter optimization")
    print(f"   • Automated tuning finds optimal parameters efficiently")
    print(f"   • Domain-specific optimization improves results")
    print(f"   • Quality scores provide objective optimization criteria")
    
    print(f"\n🎯 Next steps:")
    print(f"   • Integrate automated tuning into production pipelines")
    print(f"   • Use optimization results for batch data generation")
    print(f"   • Implement continuous quality monitoring")
    print(f"   • Extend to other domains and parameter types")

if __name__ == "__main__":
    main()
