# Long-Range Dependence Benchmarking Framework

A comprehensive framework for benchmarking long-range dependence estimators with integrated synthetic data quality evaluation.

## 🚀 Features

### Core Framework
- **Multiple Estimator Implementations**: DFA, R/S, Higuchi, GPH, Whittle, Wavelet-based methods
- **High-Performance Optimizations**: JAX, Numba, parallel processing support
- **Comprehensive Benchmarking**: Performance metrics, memory profiling, scalability analysis

### Quality Evaluation System (TSGBench-Inspired)
- **Statistical Quality Metrics**: Distribution matching, correlation analysis, trend preservation
- **Temporal Quality Metrics**: Autocorrelation, stationarity, volatility clustering
- **Domain-Specific Evaluation**: Hydrology, financial, biomedical, climate domains
- **Data Normalization**: Z-score, min-max, and robust normalization for fair comparison

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
├── validation/          # Quality evaluation system
│   ├── synthetic_data_quality.py      # Core quality evaluator
│   ├── quality_monitoring.py          # Automated monitoring
│   └── advanced_quality_metrics.py    # ML-based quality prediction
├── benchmarking/        # Performance benchmarking framework
├── data_generation/     # Synthetic data generation
└── data_submission/     # Quality-gated submission system
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

### Quality Evaluation Demo
```bash
python examples/synthetic_data_quality_demo.py
```

### Comprehensive Benchmark
```bash
python examples/comprehensive_quality_benchmark_demo.py
```

### Automated Quality Tuning
```bash
python examples/automated_quality_tuning_demo.py
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

## 🔬 Estimators Supported

- **DFA (Detrended Fluctuation Analysis)**: High-performance implementation
- **R/S (Rescaled Range)**: Classic Hurst exponent estimation
- **Higuchi**: Fractal dimension estimation
- **GPH (Geweke-Porter-Hudak)**: Semi-parametric estimation
- **Whittle MLE**: Maximum likelihood estimation
- **Wavelet Methods**: Wavelet-based Hurst estimation

## 📈 Performance Features

- **Parallel Processing**: Multi-core estimator evaluation
- **Memory Optimization**: Efficient memory usage for large datasets
- **Scalability**: Handles datasets from 100 to 1,000,000+ points
- **Profiling**: Detailed performance and memory analysis

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
