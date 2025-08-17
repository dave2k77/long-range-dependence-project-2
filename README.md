# Long-Range Dependence Benchmarking Framework

A comprehensive framework for benchmarking long-range dependence estimators with integrated synthetic data quality evaluation.

## 🚀 Features

### Core Framework
- **Multiple Estimator Implementations**: DFA, MFDFA, R/S, Higuchi, GPH, Whittle MLE, Periodogram, Wavelet-based methods
- **High-Performance Optimizations**: JAX, Numba, parallel processing support
- **Comprehensive Benchmarking**: Performance metrics, memory profiling, scalability analysis
- **Advanced Validation**: Bootstrap methods, hypothesis testing, robustness analysis

### Quality Evaluation System (TSGBench-Inspired)
- **Statistical Quality Metrics**: Distribution matching, correlation analysis, trend preservation
- **Temporal Quality Metrics**: Autocorrelation, stationarity, volatility clustering
- **Domain-Specific Evaluation**: Hydrology, financial, biomedical, climate domains
- **Data Normalization**: Z-score, min-max, and robust normalization for fair comparison
- **Machine Learning Quality Prediction**: Random Forest-based quality scoring

### Integrated Quality & Performance Benchmarking
- **Combined Analysis**: Quality evaluation + estimator performance in one pipeline
- **Synthetic & Realistic Datasets**: Comprehensive testing across data types
- **H-Value Comparison**: Ground truth vs estimated Hurst exponents with accuracy analysis
- **Advanced Visualizations**: Multi-panel plots with improved formatting and readability

### Quality System Options (All Implemented)
1. **Quality Gates in Data Submission** ✅
2. **Benchmarking Integration with Quality Metrics** ✅
3. **Automated Quality Monitoring** ✅
4. **Advanced Quality Metrics** ✅

## 📊 Quick Start

### Basic Quality Evaluation
```python
from src.validation.synthetic_data_quality import SyntheticDataQualityEvaluator

evaluator = SyntheticDataQualityEvaluator()
result = evaluator.evaluate_quality(
    synthetic_data=your_synthetic_data,
    reference_data=your_reference_data,
    domain="financial"
)
print(f"Quality Score: {result.overall_score:.3f}")
```

### Comprehensive Benchmarking
```python
python examples/comprehensive_quality_benchmark_demo.py
```

This will run:
- Quality evaluation on all datasets
- Estimator performance testing
- H-value comparison analysis
- Comprehensive visualizations

### Domain-Specific Evaluation
```python
from src.validation.synthetic_data_quality import create_domain_specific_evaluator

financial_evaluator = create_domain_specific_evaluator("financial")
result = financial_evaluator.evaluate_quality(synthetic_data, reference_data)
```

## 🏗️ Architecture

```
src/
├── estimators/           # High-performance estimator implementations
│   ├── temporal.py      # DFA, MFDFA, R/S, Higuchi methods
│   ├── spectral.py      # Whittle MLE, Periodogram, GPH methods
│   ├── wavelet.py       # Wavelet Leaders, Wavelet Whittle methods
│   └── high_performance_*.py  # JAX-optimized variants
├── validation/          # Quality evaluation system
│   ├── synthetic_data_quality.py      # Core quality evaluator
│   ├── quality_monitoring.py          # Automated monitoring
│   ├── advanced_quality_metrics.py    # ML-based quality prediction
│   ├── bootstrap.py                   # Bootstrap validation
│   ├── hypothesis_testing.py         # Statistical testing
│   ├── robustness.py                 # Robustness analysis
│   └── cross_validation.py           # Cross-validation methods
├── benchmarking/        # Performance benchmarking framework
│   ├── benchmark_runner.py           # Main benchmarking engine
│   ├── performance_metrics.py        # Performance measurement
│   └── leaderboard.py                # Performance ranking system
├── data_generation/     # Synthetic data generation
│   ├── synthetic_data_generator.py   # Data generation engine
│   └── dataset_specifications.py     # Dataset configuration
├── data_submission/     # Quality-gated submission system
│   ├── dataset_submission.py         # Dataset submission & validation
│   ├── estimator_submission.py       # Estimator submission
│   ├── benchmark_submission.py       # Benchmark result submission
│   └── validation.py                 # Submission validation
├── utils/               # Utility functions and helpers
└── data_processing/     # Data preprocessing and normalization
```

## 📈 Quality Metrics

### Statistical Quality
- **Distribution Matching**: KS test, histogram comparison
- **Correlation Analysis**: Pearson, Spearman, Kendall correlations
- **Trend Preservation**: Linear trend analysis, slope comparison
- **Volatility Clustering**: GARCH-like volatility analysis

### Temporal Quality
- **Autocorrelation**: Lag-1 autocorrelation preservation
- **Stationarity**: ADF test, variance stability
- **Seasonality**: Periodogram analysis, seasonal decomposition
- **Long-Range Dependence**: Hurst exponent consistency

### Advanced Metrics
- **Machine Learning Quality Prediction**: Random Forest-based quality scoring
- **Cross-Dataset Assessment**: Consistency across multiple datasets
- **Domain-Specific Metrics**: Tail behavior, extreme value analysis
- **Bootstrap Validation**: Confidence intervals and uncertainty quantification

## 🏆 Leaderboard System

The framework includes a comprehensive leaderboard system for competitive performance ranking:

- **Automated Scoring**: Combines accuracy, execution time, and memory usage
- **Performance Ranking**: Competitive comparison across estimators
- **Visualization**: Interactive charts and performance plots
- **Export Options**: CSV and visualization exports
- **Historical Tracking**: Performance evolution over time

### Leaderboard Features
- **Multi-Metric Scoring**: Balanced evaluation of accuracy vs. efficiency
- **Dataset-Specific Rankings**: Performance comparison across different data types
- **Estimator Comparison**: Side-by-side performance analysis
- **Trend Analysis**: Performance improvement tracking

## 📤 Data Submission System

A comprehensive submission system for datasets, estimators, and benchmark results:

### Dataset Submission
- **Quality Gates**: Automatic quality evaluation during submission
- **Validation Pipeline**: Comprehensive data validation
- **Metadata Management**: Rich dataset descriptions and provenance
- **Version Control**: Dataset versioning and history

### Estimator Submission
- **Code Validation**: Automatic code quality checks
- **Performance Testing**: Automated performance evaluation
- **Documentation Requirements**: Comprehensive documentation standards
- **Integration Testing**: Compatibility verification

### Benchmark Submission
- **Result Validation**: Automatic result verification
- **Performance Metrics**: Comprehensive performance analysis
- **Leaderboard Integration**: Automatic ranking updates
- **Reproducibility**: Full experiment documentation

## 🔄 Data Processing Pipeline

A comprehensive data processing pipeline with normalization and quality evaluation:

### Preprocessing
- **Data Cleaning**: Automatic outlier detection and handling
- **Normalization**: Z-score, min-max, and robust normalization
- **Quality Assessment**: Statistical quality metrics
- **Domain-Specific Processing**: Tailored for different data types

### Validation
- **Statistical Validation**: Distribution and correlation checks
- **Temporal Validation**: Autocorrelation and stationarity
- **Quality Scoring**: Automated quality assessment
- **Threshold Management**: Configurable quality gates

### Integration
- **Seamless Workflow**: End-to-end data processing
- **Quality Monitoring**: Real-time quality tracking
- **Performance Optimization**: Efficient processing for large datasets
- **Reproducibility**: Full pipeline documentation

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd long-range-dependence-project-2

# Install dependencies
pip install -r requirements.txt

# Run tests
python run_tests.py
```

## 📚 Examples

### Core Demos
```bash
# Quality evaluation
python examples/synthetic_data_quality_demo.py

# Comprehensive benchmark
python examples/comprehensive_quality_benchmark_demo.py

# Automated quality tuning
python examples/automated_quality_tuning_demo.py

# High-performance estimation
python examples/high_performance_demo.py

# Realistic datasets
python examples/realistic_datasets_demo.py
```

### Advanced Demos
```bash
# Data preprocessing pipeline
python examples/data_preprocessing_demo.py

# Submission system
python examples/submission_system_demo.py

# Synthetic data generation
python examples/synthetic_data_demo.py

# Direct benchmarking
python examples/direct_demo.py

# Simple workflow
python examples/simple_demo.py
```

### Specialized Demos
```bash
# Quality system integration
python examples/comprehensive_quality_system_demo.py

# High-performance synthetic data
python examples/high_performance_synthetic_data_demo.py

# Simple quality demo
python examples/simple_quality_demo.py
```

## 📊 Output Structure

```
comprehensive_quality_benchmark/
├── results/             # CSV and Excel results
├── plots/              # Quality and estimator visualizations
├── reports/            # Summary reports
└── h_comparison/       # H-value analysis
    ├── h_comparison_visualization_*.png
    ├── r2_vs_accuracy_by_estimator_*.png
    ├── time_vs_accuracy_by_estimator_*.png
    └── h_comparison_report_*.txt
```

## 🎯 Use Cases

- **Research**: Evaluate synthetic data quality for long-range dependence studies
- **Development**: Test estimator performance on quality-controlled datasets
- **Validation**: Ensure synthetic data preserves key statistical properties
- **Benchmarking**: Compare estimators across different data quality levels
- **Production**: High-performance estimation for large-scale time series analysis

## 🔬 Estimators Supported

### **Temporal Methods**
- **DFA (Detrended Fluctuation Analysis)**: High-performance implementation with polynomial detrending
- **MFDFA (Multifractal DFA)**: Multifractal analysis with q-order estimation
- **R/S (Rescaled Range)**: Classic Hurst exponent estimation with proper scaling
- **Higuchi**: Fractal dimension estimation for time series

### **Spectral Methods**
- **Whittle MLE**: Maximum likelihood estimation in frequency domain
- **Periodogram**: Power spectral density analysis
- **GPH (Geweke-Porter-Hudak)**: Semi-parametric estimation method

### **Wavelet Methods**
- **Wavelet Leaders**: Wavelet coefficient analysis for LRD estimation
- **Wavelet Whittle**: Wavelet-based Whittle likelihood optimization

### **High-Performance Variants**
- **JAX-Optimized**: All estimators have high-performance JAX variants for 10-100x speedup
- **Parallel Processing**: Multi-core support for large datasets
- **Memory Optimization**: Efficient memory usage tracking and optimization

## 📈 Performance Features

- **Parallel Processing**: Multi-core estimator evaluation
- **Memory Optimization**: Efficient memory usage for large datasets
- **Scalability**: Handles datasets from 100 to 1,000,000+ points
- **Profiling**: Detailed performance and memory analysis
- **JAX Integration**: GPU acceleration and vectorized operations
- **Numba Support**: JIT compilation for critical numerical operations

## 🔍 Validation & Testing

- **Comprehensive Test Suite**: 180+ tests covering all estimators and edge cases
- **Bootstrap Methods**: Confidence interval estimation and uncertainty quantification
- **Hypothesis Testing**: Statistical significance testing for estimates
- **Robustness Analysis**: Edge case handling and error recovery
- **Cross-Validation**: Robustness assessment across different data conditions

## 📊 Project Status

### **Current Implementation Status** ✅
- **Core Framework**: 100% complete with modular architecture
- **Estimators**: 9 base estimators + 10 high-performance variants
- **Quality System**: All 4 quality system options implemented
- **Benchmarking**: Comprehensive performance evaluation framework
- **Submission System**: Complete data/estimator/benchmark submission
- **Leaderboard**: Automated performance ranking system
- **Data Processing**: Full pipeline with normalization and validation

### **Test Coverage** 📈
- **Total Tests**: 180+ tests
- **Wavelet Estimators**: 31/31 (100% passing) ✅
- **Overall Framework**: Comprehensive coverage across all components
- **Integration Tests**: End-to-end workflow validation

### **Performance Features** 🚀
- **JAX Integration**: 10-100x speedup for high-performance variants
- **Parallel Processing**: Multi-core support for large datasets
- **Memory Optimization**: Efficient memory usage tracking
- **Scalability**: Handles datasets from 100 to 1,000,000+ points

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the TSGBench framework for synthetic data quality evaluation
- Built on established long-range dependence estimation methods
- Enhanced with modern Python performance optimization techniques
- JAX and Numba integration for high-performance computing
