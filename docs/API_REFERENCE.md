# 📚 API Reference - Long-Range Dependence Framework

**Complete API documentation for all estimators, utilities, and benchmarking tools**

## 📋 **Table of Contents**

1. [Core Estimators](#core-estimators)
2. [Utility Classes](#utility-classes)
3. [Benchmarking Tools](#benchmarking-tools)
4. [Validation Framework](#validation-framework)
5. [Performance Monitoring](#performance-monitoring)

---

## 🎯 **Core Estimators**

### **HighPerformanceDFAEstimator**

The high-performance Detrended Fluctuation Analysis (DFA) estimator with automatic optimization backends.

#### **Class Definition**

```python
class HighPerformanceDFAEstimator(BaseEstimator):
    """
    High-performance DFA estimator using NUMBA and JAX.
    
    This estimator provides multiple optimization strategies:
    - NUMBA for CPU optimization and parallelization
    - JAX for GPU acceleration and automatic differentiation
    - Memory-efficient processing for large datasets
    - Parallel processing across multiple cores/GPUs
    """
```

#### **Constructor Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"HighPerformanceDFA"` | Name identifier for the estimator |
| `optimization_backend` | `str` | `"auto"` | Backend: `'numba'`, `'jax'`, `'numpy'`, or `'auto'` |
| `use_gpu` | `bool` | `True` | Whether to use GPU acceleration |
| `memory_efficient` | `bool` | `True` | Whether to use memory-efficient processing |
| `min_scale` | `int` | `4` | Minimum scale for analysis |
| `max_scale` | `int` | `None` | Maximum scale (default: `len(data)//4`) |
| `num_scales` | `int` | `20` | Number of scales to analyze |
| `polynomial_order` | `int` | `1` | Order of polynomial for detrending |
| `batch_size` | `int` | `1000` | Batch size for processing |

#### **Core Methods**

##### **`estimate(data: np.ndarray, **kwargs) -> Dict[str, Any]`**

Main estimation method that performs DFA analysis.

**Parameters:**
- `data`: Input time series data as numpy array
- `**kwargs`: Additional estimation parameters

**Returns:**
```python
{
    'hurst_exponent': float,           # Estimated Hurst exponent
    'r_squared': float,                # R-squared value of the fit
    'slope': float,                    # Slope from power law fit
    'intercept': float,                # Intercept from power law fit
    'std_error': float,                # Standard error of estimate
    'scales': np.ndarray,              # Array of scales used
    'fluctuations': np.ndarray,        # Fluctuation values
    'method': str,                     # Estimation method name
    'optimization_backend': str,       # Backend actually used
    'performance_metrics': Dict        # Timing and memory info
}
```

##### **`fit(data: np.ndarray, **kwargs) -> 'HighPerformanceDFAEstimator'`**

Fit the estimator to data (required by base class).

**Parameters:**
- `data`: Input time series data
- `**kwargs`: Additional fitting parameters

**Returns:** Self for method chaining

##### **`get_performance_summary() -> Dict[str, Any]`**

Get comprehensive performance summary including cache statistics.

**Returns:**
```python
{
    'estimator_name': str,
    'optimization_backend': str,
    'use_gpu': bool,
    'memory_efficient': bool,
    'performance_metrics': Dict,
    'memory_summary': Dict,
    'parallel_summary': Dict,
    'cache_performance': Dict,
    'data_size': int,
    'scales_count': int,
    'optimization_features': Dict
}
```

##### **`get_cache_stats() -> Dict[str, Any]`**

Get caching performance statistics.

**Returns:**
```python
{
    'cache_hits': int,                 # Number of cache hits
    'cache_misses': int,               # Number of cache misses
    'total_requests': int,             # Total cache requests
    'hit_rate': float,                 # Cache hit rate (0.0 to 1.0)
    'cache_size': int,                 # Current cache size
    'cache_efficiency': str            # Formatted hit rate percentage
}
```

#### **Performance Features**

- **🚀 Vectorized Operations**: NumPy vectorization for maximum speed
- **💾 Smart Caching**: Scale generation caching (50%+ hit rate)
- **🧠 Memory Pools**: Efficient memory management for large datasets
- **⚡ Parallel Processing**: Multi-core CPU utilization
- **📈 Performance Monitoring**: Real-time metrics and profiling

---

### **HighPerformanceMFDFAEstimator**

The high-performance Multifractal Detrended Fluctuation Analysis (MFDFA) estimator.

#### **Class Definition**

```python
class HighPerformanceMFDFAEstimator(BaseEstimator):
    """
    High-performance MFDFA estimator using JAX and NumPy.
    
    This estimator provides:
    - Multifractal analysis with q-value support
    - JAX acceleration with NumPy fallbacks
    - Comprehensive multifractal spectrum calculation
    - Statistical validation of multifractality
    """
```

#### **Constructor Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"HighPerformanceMFDFA"` | Name identifier for the estimator |
| `num_scales` | `int` | `20` | Number of scales for analysis |
| `q_values` | `np.ndarray` | `np.arange(-3, 4, 0.5)` | Array of q-values for multifractal analysis |
| `polynomial_order` | `int` | `1` | Order of polynomial for detrending |
| `optimization_backend` | `str` | `"auto"` | Optimization backend to use |

#### **Core Methods**

##### **`estimate(data: np.ndarray, **kwargs) -> Dict[str, Any]`**

Perform complete MFDFA analysis.

**Parameters:**
- `data`: Input time series data as numpy array
- `**kwargs`: Additional estimation parameters

**Returns:**
```python
{
    'hurst_exponents': np.ndarray,     # Hurst exponents for each q-value
    'q_values': np.ndarray,            # Q-values used in analysis
    'scales': np.ndarray,              # Scales used in analysis
    'fluctuations': np.ndarray,        # Fluctuation functions
    'multifractal_spectrum': Dict,     # Alpha and f(alpha) values
    'summary': Dict,                   # Statistical summary
    'method': str,                     # Estimation method name
    'performance_metrics': Dict        # Timing and memory info
}
```

##### **`get_multifractal_summary() -> Dict[str, Any]`**

Get comprehensive multifractal analysis summary.

**Returns:**
```python
{
    'mean_hurst': float,               # Mean Hurst exponent
    'hurst_range': float,              # Range of Hurst exponents
    'is_multifractal': bool,           # Multifractality test result
    'multifractal_strength': float,    # Strength of multifractality
    'spectrum_width': float,           # Width of multifractal spectrum
    'spectrum_asymmetry': float        # Asymmetry of spectrum
}
```

---

## 🔧 **Utility Classes**

### **MemoryManager**

Efficient memory management with pooling and monitoring.

#### **Key Methods**

```python
class MemoryManager:
    def create_memory_pool(self, name: str, size: int, dtype: np.dtype) -> None
    def get_from_pool(self, name: str, size: int) -> Optional[np.ndarray]
    def return_to_pool(self, name: str) -> None
    def get_memory_summary(self) -> Dict[str, Any]
    def cleanup_memory(self, aggressive: bool = False) -> None
```

### **ParallelProcessor**

Multi-core and GPU parallel processing utilities.

#### **Key Methods**

```python
class ParallelProcessor:
    def __init__(self, n_jobs: int = -1, backend: str = "auto")
    def parallel_map(self, func: Callable, data: List) -> List
    def get_performance_summary(self) -> Dict[str, Any]
    @property
    def available_gpus(self) -> List[str]
```

### **JAXOptimizer**

JAX-based optimization utilities with GPU acceleration.

#### **Key Methods**

```python
class JAXOptimizer:
    def __init__(self, device: str = "auto")
    def fast_linregress(self, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[float, float, float]
    def fast_detrend(self, data: jnp.ndarray, degree: int = 1) -> jnp.ndarray
    def fast_rms(self, data: jnp.ndarray, axis: int = None) -> jnp.ndarray
```

### **NumbaOptimizer**

NUMBA-based optimization utilities for CPU acceleration.

#### **Key Methods**

```python
class NumbaOptimizer:
    def __init__(self, use_gpu: bool = False)
    def fast_linregress(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]
    def fast_detrend(self, data: np.ndarray, degree: int = 1) -> np.ndarray
    def fast_rms(self, data: np.ndarray) -> float
```

---

## 📊 **Benchmarking Tools**

### **PerformanceBenchmarker**

Comprehensive performance benchmarking across multiple estimators and dataset sizes.

#### **Key Methods**

```python
class PerformanceBenchmarker:
    def __init__(self, estimators: List[BaseEstimator], 
                 dataset_sizes: List[int], iterations: int = 3)
    
    def run_benchmark(self) -> pd.DataFrame
    def generate_report(self) -> str
    def plot_performance(self, save_plots: bool = True) -> None
    def save_results(self, output_dir: str = "benchmark_results") -> None
```

#### **Usage Example**

```python
from src.benchmarking.performance_benchmarks import PerformanceBenchmarker
from src.estimators.high_performance_dfa import HighPerformanceDFAEstimator
from src.estimators.high_performance import HighPerformanceMFDFAEstimator

# Create estimators
estimators = [
    HighPerformanceDFAEstimator(),
    HighPerformanceMFDFAEstimator()
]

# Run benchmark
benchmarker = PerformanceBenchmarker(
    estimators=estimators,
    dataset_sizes=[100, 500, 1000],
    iterations=3
)

results = benchmarker.run_benchmark()
benchmarker.generate_report()
benchmarker.plot_performance()
```

### **PerformanceProfiler**

Detailed performance profiling to identify bottlenecks and optimization opportunities.

#### **Key Methods**

```python
class PerformanceProfiler:
    def __init__(self, output_dir: str = "profiling_results")
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]
    def profile_estimator_components(self, estimator, data: np.ndarray) -> Dict[str, Any]
    def analyze_bottlenecks(self, profiling_results: Dict[str, Any]) -> List[Dict[str, Any]]
    def generate_optimization_report(self, bottlenecks: List[Dict[str, Any]]) -> str
    def plot_bottleneck_analysis(self, bottlenecks: List[Dict[str, Any]], save_plots: bool = True) -> None
```

#### **Usage Example**

```python
from src.benchmarking.performance_profiler import profile_estimator_performance
from src.estimators.high_performance_dfa import HighPerformanceDFAEstimator

# Profile estimator performance
profiler, bottlenecks = profile_estimator_performance(
    HighPerformanceDFAEstimator,
    [500, 1000, 2000]
)

# Generate optimization report
report = profiler.generate_optimization_report(bottlenecks)
print(report)

# Plot bottleneck analysis
profiler.plot_bottleneck_analysis(bottlenecks)
```

---

## ✅ **Validation Framework**

### **Statistical Validation**

```python
from src.validation import validate_estimator, bootstrap_confidence_intervals

# Validate estimator accuracy
validation_result = validate_estimator(
    estimator=HighPerformanceDFAEstimator(),
    data=fbm_data,
    expected_hurst=0.7,
    tolerance=0.1
)

# Bootstrap confidence intervals
ci_result = bootstrap_confidence_intervals(
    estimator=HighPerformanceDFAEstimator(),
    data=data,
    n_bootstrap=1000,
    confidence_level=0.95
)
```

### **Robustness Testing**

```python
from src.validation import test_robustness

# Test estimator robustness
robustness_result = test_robustness(
    estimator=HighPerformanceDFAEstimator(),
    data=data,
    noise_levels=[0.0, 0.1, 0.2, 0.5],
    n_iterations=100
)
```

---

## 📈 **Performance Monitoring**

### **Real-time Metrics**

```python
# Get comprehensive performance summary
summary = estimator.get_performance_summary()

print(f"Optimization Features: {summary['optimization_features']}")
print(f"Cache Performance: {summary['cache_performance']}")
print(f"Memory Usage: {summary['memory_summary']}")
print(f"Parallel Performance: {summary['parallel_summary']}")
```

### **Memory Monitoring**

```python
# Monitor memory usage
memory_manager = estimator.memory_manager
memory_summary = memory_manager.get_memory_summary()

print(f"Current Memory: {memory_summary['current_memory_mb']:.2f} MB")
print(f"Peak Memory: {memory_summary['peak_memory_mb']:.2f} MB")
print(f"Memory Pools: {memory_summary['pool_count']}")
```

### **Cache Performance**

```python
# Monitor cache performance
cache_stats = estimator.get_cache_stats()

print(f"Cache Hit Rate: {cache_stats['cache_efficiency']}")
print(f"Cache Hits: {cache_stats['cache_hits']}")
print(f"Cache Misses: {cache_stats['cache_misses']}")
print(f"Cache Size: {cache_stats['cache_size']}")
```

---

## 🔍 **Error Handling & Debugging**

### **Common Error Types**

1. **JAX Compilation Errors**: Expected behavior - automatically falls back to NumPy
2. **Memory Issues**: Enable `memory_efficient=True` for large datasets
3. **Performance Issues**: Use `optimization_backend='numpy'` for maximum reliability

### **Debug Mode**

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Create estimator with debug info
estimator = HighPerformanceDFAEstimator()
estimator.estimate(data)  # Will show detailed optimization decisions
```

### **Performance Debugging**

```python
# Profile specific components
from src.benchmarking.performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()
component_results = profiler.profile_estimator_components(estimator, data)

# Analyze bottlenecks
bottlenecks = profiler.analyze_bottlenecks(component_results)
optimization_report = profiler.generate_optimization_report(bottlenecks)
print(optimization_report)
```

---

## 📝 **Best Practices**

### **Estimator Selection**

- **HighPerformanceDFA**: Use for fast, lightweight DFA analysis
- **HighPerformanceMFDFA**: Use for comprehensive multifractal analysis
- **Backend Selection**: Use `'numpy'` for maximum reliability, `'auto'` for best performance

### **Performance Optimization**

- **Memory Efficiency**: Enable for datasets > 1GB
- **Caching**: Leverage scale generation caching for repeated analyses
- **Parallel Processing**: Use for batch processing of multiple datasets

### **Error Handling**

- **Graceful Fallbacks**: All estimators automatically fall back to reliable methods
- **Logging**: Monitor logs for optimization decisions and fallback usage
- **Validation**: Always validate results with known test datasets

---

## 🔗 **Related Documentation**

- [Quick Start Guide](README.md)
- [Performance Benchmarks](PERFORMANCE_BENCHMARK_ANALYSIS.md)
- [Installation Guide](INSTALLATION.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)

---

*This API reference covers the core functionality. For advanced usage and examples, see the [examples/](examples/) directory.*
