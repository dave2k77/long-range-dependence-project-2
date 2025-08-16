# Long-Range Dependence Benchmarking Framework

A comprehensive and reproducible benchmarking framework for detecting and characterising long-range dependence in time series data.

## Project Overview

This framework implements classical long-range dependence estimators with statistical validation, efficiency optimization, and comprehensive benchmarking capabilities.

## Features

### 1. Classical Long-Range-Dependence Estimators
- **Temporal Methods**: DFA, MFDFA, R/S, Higuchi
- **Spectral Methods**: Whittle MLE, Periodogram, GPH
- **Wavelet Methods**: Wavelet Leaders, Wavelet Whittle
- **High-Performance Variants**: GPU-accelerated and parallelized implementations

### 2. Statistical Validation
- Hypothesis Testing and CI Bootstrapping
- Out-Of-Distribution (OOD) and Marginal Distribution Testing
- Monte Carlo Significance Testing
- Robustness Testing and Sensitivity Analysis
- Cross Validation Mechanisms

### 3. High-Performance Computing Framework
- **NUMBA Integration**: CPU optimization and parallelization
- **JAX Integration**: GPU acceleration and automatic differentiation
- **Memory Management**: Efficient memory pooling and optimization
- **Parallel Processing**: Multi-core, GPU, and distributed computing
- **Auto-optimization**: Automatic backend selection for optimal performance

### 4. Performance Benchmarking
- Error/Uncertainty Quantification
- Accuracy and Efficiency (time and memory)
- Head-to-head performance comparison
- Performance Leaderboard
- Full performance benchmarking analysis
- GPU vs CPU performance analysis

## Project Structure

```
long-range-dependence-project-2/
├── src/                           # Source code
│   ├── estimators/                # LRD estimation algorithms
│   ├── validation/                # Statistical validation methods
│   ├── benchmarking/              # Performance benchmarking tools
│   └── utils/                     # Utility functions
├── tests/                         # Test suite
├── benchmarks/                    # Benchmark results and datasets
├── docs/                          # Documentation
├── examples/                      # Usage examples
└── requirements.txt               # Dependencies
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `python -m pytest tests/`
3. Run benchmarks: `python -m src.benchmarking.run_benchmarks`
4. Run high-performance demo: `python examples/high_performance_demo.py`

### High-Performance Features

The framework now includes advanced optimization capabilities:

- **NUMBA**: CPU optimization with parallel processing and JIT compilation
- **JAX**: GPU acceleration with automatic differentiation and vectorization
- **Memory Management**: Efficient memory pooling and cache optimization
- **Parallel Processing**: Multi-core CPU and multi-GPU support
- **Auto-optimization**: Automatic selection of optimal backend based on hardware

### GPU Requirements

For GPU acceleration:
- CUDA-compatible GPU (NVIDIA)
- JAX with GPU support
- Sufficient GPU memory for large datasets

## Development Status

- [x] Project structure setup
- [x] Core estimator implementations
- [x] Statistical validation framework
- [x] Benchmarking infrastructure
- [x] High-performance computing framework (NUMBA + JAX)
- [x] Memory management and optimization
- [x] Parallel processing utilities
- [x] GPU acceleration support
- [x] High-performance DFA estimator
- [x] Documentation and examples
- [ ] Testing suite
- [ ] Additional high-performance estimators
- [ ] Performance optimization and tuning

## Contributing

This project follows test-driven development principles. Please ensure all code includes appropriate tests and documentation.

## License

[Add your license here]
