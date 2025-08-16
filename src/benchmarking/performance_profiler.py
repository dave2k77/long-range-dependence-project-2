"""
Performance Profiler for Long-Range Dependence Estimators

This module provides detailed performance profiling to identify bottlenecks
and optimization opportunities in our estimators.
"""

import time
import numpy as np
import cProfile
import pstats
import io
from typing import Dict, List, Tuple, Any, Optional, Callable
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import functools

logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """
    Comprehensive performance profiler for LRD estimators.
    
    This class provides methods to profile different parts of the estimation
    process and identify optimization opportunities.
    """
    
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiling_results = {}
        
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile a single function execution.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with profiling results
        """
        # Create profiler
        pr = cProfile.Profile()
        
        # Start profiling
        start_time = time.time()
        pr.enable()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logger.error(f"Function execution failed: {e}")
        
        # Stop profiling
        pr.disable()
        end_time = time.time()
        
        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        stats_output = s.getvalue()
        
        # Calculate timing
        execution_time = end_time - start_time
        
        # Store results
        profiling_result = {
            'function_name': func.__name__,
            'execution_time': execution_time,
            'success': success,
            'result': result,
            'stats_output': stats_output,
            'call_count': pr.getstats()[0].callcount if pr.getstats() else 0,
            'total_time': pr.getstats()[0].totaltime if pr.getstats() else 0,
            'inline_time': pr.getstats()[0].inlinetime if pr.getstats() else 0
        }
        
        self.profiling_results[func.__name__] = profiling_result
        
        return profiling_result
    
    def profile_estimator_components(self, estimator, data: np.ndarray) -> Dict[str, Any]:
        """
        Profile individual components of an estimator.
        
        Args:
            estimator: Estimator instance to profile
            data: Input data for estimation
            
        Returns:
            Dictionary with component profiling results
        """
        component_results = {}
        
        # Profile scale generation
        if hasattr(estimator, '_generate_scales'):
            logger.info("Profiling scale generation...")
            component_results['scale_generation'] = self.profile_function(
                estimator._generate_scales
            )
        
        # Profile fluctuation calculation
        if hasattr(estimator, '_calculate_fluctuations_numpy'):
            logger.info("Profiling numpy fluctuation calculation...")
            component_results['fluctuation_calculation'] = self.profile_function(
                estimator._calculate_fluctuations_numpy
            )
        
        # Profile power law fitting
        if hasattr(estimator, '_fit_power_law_numpy'):
            logger.info("Profiling numpy power law fitting...")
            component_results['power_law_fitting'] = self.profile_function(
                estimator._fit_power_law_numpy
            )
        
        # Profile complete estimation
        logger.info("Profiling complete estimation...")
        component_results['complete_estimation'] = self.profile_function(
            estimator.estimate, data
        )
        
        return component_results
    
    def analyze_bottlenecks(self, profiling_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze profiling results to identify bottlenecks.
        
        Args:
            profiling_results: Results from profiling
            
        Returns:
            List of bottleneck analysis results
        """
        bottlenecks = []
        
        for func_name, result in profiling_results.items():
            if not result['success']:
                continue
                
            # Calculate efficiency metrics
            total_time = result['total_time']
            call_count = result['call_count']
            
            if call_count > 0:
                avg_time_per_call = total_time / call_count
                efficiency_score = 1.0 / (avg_time_per_call + 1e-10)
                
                bottleneck_info = {
                    'function_name': func_name,
                    'total_execution_time': result['execution_time'],
                    'total_profiled_time': total_time,
                    'call_count': call_count,
                    'avg_time_per_call': avg_time_per_call,
                    'efficiency_score': efficiency_score,
                    'bottleneck_level': self._classify_bottleneck(avg_time_per_call),
                    'optimization_priority': self._calculate_optimization_priority(
                        avg_time_per_call, call_count
                    )
                }
                
                bottlenecks.append(bottleneck_info)
        
        # Sort by optimization priority
        bottlenecks.sort(key=lambda x: x['optimization_priority'], reverse=True)
        
        return bottlenecks
    
    def _classify_bottleneck(self, avg_time_per_call: float) -> str:
        """Classify bottleneck severity."""
        if avg_time_per_call < 0.001:
            return "Low"
        elif avg_time_per_call < 0.01:
            return "Medium"
        elif avg_time_per_call < 0.1:
            return "High"
        else:
            return "Critical"
    
    def _calculate_optimization_priority(self, avg_time_per_call: float, call_count: int) -> float:
        """Calculate optimization priority score."""
        # Higher priority for functions that are called frequently and take long
        time_factor = avg_time_per_call * 1000  # Convert to milliseconds
        frequency_factor = min(call_count / 100, 1.0)  # Normalize call count
        
        return time_factor * frequency_factor
    
    def generate_optimization_report(self, bottlenecks: List[Dict[str, Any]]) -> str:
        """Generate optimization recommendations report."""
        if not bottlenecks:
            return "No bottlenecks identified."
        
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append("BOTTLENECK ANALYSIS:")
        report.append("-" * 40)
        
        for i, bottleneck in enumerate(bottlenecks[:10], 1):  # Top 10
            report.append(f"{i}. {bottleneck['function_name']}")
            report.append(f"   Bottleneck Level: {bottleneck['bottleneck_level']}")
            report.append(f"   Avg Time per Call: {bottleneck['avg_time_per_call']:.6f}s")
            report.append(f"   Call Count: {bottleneck['call_count']}")
            report.append(f"   Optimization Priority: {bottleneck['optimization_priority']:.2f}")
            report.append("")
        
        report.append("OPTIMIZATION RECOMMENDATIONS:")
        report.append("-" * 40)
        
        # Group by bottleneck level
        critical_bottlenecks = [b for b in bottlenecks if b['bottleneck_level'] == 'Critical']
        high_bottlenecks = [b for b in bottlenecks if b['bottleneck_level'] == 'High']
        
        if critical_bottlenecks:
            report.append("🚨 CRITICAL BOTTLENECKS (Fix Immediately):")
            for bottleneck in critical_bottlenecks:
                report.append(f"   • {bottleneck['function_name']}: Consider algorithm redesign")
            report.append("")
        
        if high_bottlenecks:
            report.append("⚠️  HIGH BOTTLENECKS (Optimize Soon):")
            for bottleneck in high_bottlenecks:
                report.append(f"   • {bottleneck['function_name']}: Profile and optimize loops")
            report.append("")
        
        # General recommendations
        report.append("🔧 GENERAL OPTIMIZATION STRATEGIES:")
        report.append("   • Use vectorized operations where possible")
        report.append("   • Implement caching for repeated calculations")
        report.append("   • Consider parallel processing for independent operations")
        report.append("   • Profile memory usage and optimize allocations")
        
        return "\n".join(report)
    
    def plot_bottleneck_analysis(self, bottlenecks: List[Dict[str, Any]], save_plots: bool = True):
        """Generate bottleneck analysis plots."""
        if not bottlenecks:
            logger.warning("No bottlenecks to plot")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Bottleneck Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        func_names = [b['function_name'] for b in bottlenecks]
        avg_times = [b['avg_time_per_call'] for b in bottlenecks]
        call_counts = [b['call_count'] for b in bottlenecks]
        priority_scores = [b['optimization_priority'] for b in bottlenecks]
        bottleneck_levels = [b['bottleneck_level'] for b in bottlenecks]
        
        # 1. Average Time per Call
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(func_names)), avg_times, alpha=0.7)
        ax1.set_xlabel('Function')
        ax1.set_ylabel('Average Time per Call (seconds)')
        ax1.set_title('Function Performance Bottlenecks')
        ax1.set_xticks(range(len(func_names)))
        ax1.set_xticklabels(func_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Color bars by bottleneck level
        colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Critical': 'red'}
        for bar, level in zip(bars1, bottleneck_levels):
            bar.set_color(colors.get(level, 'blue'))
        
        # 2. Call Count vs Average Time
        ax2 = axes[0, 1]
        scatter = ax2.scatter(call_counts, avg_times, c=priority_scores, 
                             s=100, alpha=0.7, cmap='viridis')
        ax2.set_xlabel('Call Count')
        ax2.set_ylabel('Average Time per Call (seconds)')
        ax2.set_title('Call Frequency vs Performance')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Optimization Priority')
        
        # 3. Optimization Priority
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(func_names)), priority_scores, alpha=0.7, color='skyblue')
        ax3.set_xlabel('Function')
        ax3.set_ylabel('Optimization Priority Score')
        ax3.set_title('Optimization Priority Ranking')
        ax3.set_xticks(range(len(func_names)))
        ax3.set_xticklabels(func_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Bottleneck Level Distribution
        ax4 = axes[1, 1]
        level_counts = {}
        for level in bottleneck_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        if level_counts:
            levels = list(level_counts.keys())
            counts = list(level_counts.values())
            colors = [colors.get(level, 'blue') for level in levels]
            
            bars4 = ax4.bar(levels, counts, color=colors, alpha=0.7)
            ax4.set_xlabel('Bottleneck Level')
            ax4.set_ylabel('Number of Functions')
            ax4.set_title('Bottleneck Severity Distribution')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars4, counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = self.output_dir / f"bottleneck_analysis_{time.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Bottleneck analysis plots saved to: {plot_file}")
        
        plt.show()
        
        return fig


def profile_estimator_performance(estimator_class, data_sizes: List[int] = None):
    """Profile estimator performance across different data sizes."""
    if data_sizes is None:
        data_sizes = [100, 500, 1000]
    
    profiler = PerformanceProfiler()
    
    print("🔍 Starting Performance Profiling...")
    print(f"📊 Testing data sizes: {data_sizes}")
    
    all_bottlenecks = []
    
    for size in data_sizes:
        print(f"\n📏 Profiling dataset size: {size}")
        
        # Generate test data
        data = np.random.randn(size)
        
        # Create estimator
        estimator = estimator_class()
        
        # Profile components
        component_results = profiler.profile_estimator_components(estimator, data)
        
        # Analyze bottlenecks
        bottlenecks = profiler.analyze_bottlenecks(component_results)
        all_bottlenecks.extend(bottlenecks)
        
        print(f"   ✅ Profiling completed for size {size}")
    
    # Generate comprehensive report
    print("\n📋 GENERATING OPTIMIZATION REPORT...")
    report = profiler.generate_optimization_report(all_bottlenecks)
    print(report)
    
    # Generate plots
    print("\n📈 Generating bottleneck analysis plots...")
    profiler.plot_bottleneck_analysis(all_bottlenecks)
    
    print(f"\n✅ Profiling completed! Results saved to: {profiler.output_dir}")
    
    return profiler, all_bottlenecks


if __name__ == "__main__":
    # Example usage
    from src.estimators.high_performance_dfa import HighPerformanceDFAEstimator
    
    print("🚀 Performance Profiler Demo")
    print("=" * 40)
    
    # Profile DFA estimator
    profiler, bottlenecks = profile_estimator_performance(HighPerformanceDFAEstimator)
