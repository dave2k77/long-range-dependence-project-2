# Long-Range Dependence Benchmarking Framework

A comprehensive framework for benchmarking long-range dependence estimators with synthetic data generation and quality evaluation.

## Features

### Core Framework
- **Long-Range Dependence Estimators**: Multiple estimators including DFA, GPH, Higuchi, Periodogram, R/S, Wavelet methods, and Whittle
- **Synthetic Data Generation**: Configurable synthetic data with known Hurst exponents
- **Performance Benchmarking**: Execution time and memory usage analysis
- **Quality Evaluation**: TSGBench-inspired synthetic data quality assessment

### Quality Evaluation System
- **Quality Gates in Data Submission**: Automated quality checks during dataset submission
- **Benchmarking Integration**: Quality metrics integrated with performance benchmarks
- **Automated Quality Monitoring**: Real-time quality assessment with alerts and trend analysis
- **Advanced Quality Metrics**: ML-based quality prediction and cross-dataset assessment

### Data Processing
- **Data Preprocessing**: Normalization, detrending, and quality filtering
- **Validation**: Comprehensive dataset and estimator validation
- **Submission System**: Structured data and estimator submission pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.benchmarking import PerformanceBenchmark
from src.data_generation import SyntheticDataGenerator

# Generate synthetic data
generator = SyntheticDataGenerator()
data = generator.generate_data(specification)

# Run benchmarks
benchmark = PerformanceBenchmark()
results = benchmark.run_comprehensive_benchmark(data)
```

### Quality Evaluation

```python
from src.validation import SyntheticDataQualityEvaluator

# Evaluate synthetic data quality
evaluator = SyntheticDataQualityEvaluator()
quality_result = evaluator.evaluate_quality(
    synthetic_data=synthetic_data,
    reference_data=reference_data,
    domain="financial"
)
```

### Quality Gates

```python
from src.data_submission import DatasetSubmission

# Submit dataset with automatic quality checks
submission = DatasetSubmission()
result = submission.submit_synthetic_dataset(
    data=data,
    specification=spec,
    metadata=metadata
)
```

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Synthetic Data Quality Evaluation](SYNTHETIC_DATA_QUALITY_EVALUATION_SUMMARY.md)
- [Final Implementation Summary](FINAL_IMPLEMENTATION_SUMMARY.md)
- [Project Status](PROJECT_STATUS_FINAL.md)

## Examples

See the `examples/` directory for comprehensive demonstrations:
- `synthetic_data_quality_demo.py`: Quality evaluation system demo
- `simple_quality_demo.py`: Simplified quality system concepts demo
- `comprehensive_demo.py`: Full framework demonstration

## Testing

```bash
python run_tests.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
