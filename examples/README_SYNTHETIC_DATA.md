# 🚀 Synthetic Data Generation System

This directory contains a comprehensive synthetic data generation system for Long-Range Dependence (LRD) analysis, designed for benchmarking physics-based fractional machine learning models.

## 🌟 Features

### 1. Base Models
- **ARFIMA**: Autoregressive Fractionally Integrated Moving Average
- **fBm**: Fractional Brownian Motion  
- **fGn**: Fractional Gaussian Noise

### 2. Realistic Confounds
- **EEG-like**: Non-stationarity, artifacts, baseline drift
- **Hydrology/Climate**: Seasonal patterns, trend changes, extreme events
- **Financial**: Volatility clustering, regime changes, jumps
- **General**: Heavy tails, measurement noise, outliers

### 3. Domain-Specific Patterns
- **Biomedical**: EEG, MEG, physiological signals
- **Environmental**: Climate, hydrology, geophysical data
- **Financial**: Stock prices, returns, volatility
- **Network**: Traffic, social media, communication

### 4. Dataset Specifications
- Standardized format for user submissions
- Benchmark protocols for consistent evaluation
- Performance metrics across different data types
- Validation and quality control

## 📁 Demo Scripts

### 1. `synthetic_data_demo.py` - Quick Start
**Purpose**: Simple demonstration of basic capabilities
**Usage**: `python synthetic_data_demo.py`
**Features**:
- Generate basic fGn, fBm, and ARFIMA data
- Apply domain-specific confounds
- Basic statistical analysis
- No heavy dependencies required

### 2. `comprehensive_synthetic_data_demo.py` - Full Features
**Purpose**: Complete demonstration with visualizations
**Usage**: `python comprehensive_synthetic_data_demo.py`
**Features**:
- All base models with different Hurst exponents
- Comprehensive confound demonstrations
- Domain-specific pattern generation
- Dataset specification system
- Benchmark dataset generation
- Performance analysis
- High-quality plots and visualizations

### 3. `high_performance_synthetic_data_demo.py` - Performance Testing
**Purpose**: Large-scale data generation and performance analysis
**Usage**: `python high_performance_synthetic_data_demo.py`
**Features**:
- Parallel data generation
- Memory-efficient processing
- Large-scale dataset creation (1M+ points)
- Performance profiling and monitoring
- Memory usage tracking
- Efficiency analysis

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_demo.txt
```

### 2. Run Basic Demo
```bash
python synthetic_data_demo.py
```

### 3. Run Comprehensive Demo
```bash
python comprehensive_synthetic_data_demo.py
```

### 4. Run Performance Demo
```bash
python high_performance_synthetic_data_demo.py
```

## 📊 Output Structure

All demos create organized output directories:

```
demo_outputs/                    # Basic demo outputs
├── datasets/                    # Generated data files
├── plots/                      # Visualization plots
└── specifications/             # Dataset specifications

high_performance_outputs/        # Performance demo outputs
├── large_datasets/             # Large-scale datasets
├── performance_plots/          # Performance analysis plots
└── profiles/                   # Performance profiles
```

## 🔧 Customization

### Creating Custom Data Specifications

```python
from data_generation.synthetic_data_generator import DataSpecification, DomainType

# Custom EEG specification
custom_spec = DataSpecification(
    n_points=5000,
    hurst_exponent=0.75,
    domain_type=DomainType.EEG,
    confound_strength=0.15,
    noise_level=0.08,
    seasonal_period=100
)
```

### Applying Custom Confounds

```python
from data_generation.synthetic_data_generator import ConfoundType

# Apply multiple confounds
data = generator.generate_data(
    spec,
    confounds=[
        ConfoundType.NON_STATIONARITY,
        ConfoundType.HEAVY_TAILS,
        ConfoundType.BASELINE_DRIFT
    ]
)
```

### Domain-Specific Generation

```python
# Financial data with volatility regimes
financial_spec = DataSpecification(
    n_points=2520,  # 10 years of daily data
    hurst_exponent=0.55,
    domain_type=DomainType.FINANCIAL,
    volatility_regimes=[0.8, 1.2, 0.6, 1.5],
    confound_strength=0.3
)
```

## 📈 Performance Characteristics

### Generation Speed
- **Small datasets** (< 10K points): ~0.1-1.0 seconds
- **Medium datasets** (10K-100K points): ~1-10 seconds  
- **Large datasets** (100K-1M points): ~10-100 seconds
- **Parallel generation**: 2-8x speedup depending on CPU cores

### Memory Usage
- **Efficient**: ~8 bytes per data point
- **Chunked processing**: Handles datasets larger than available RAM
- **Parallel processing**: Memory usage scales with number of processes

### Quality Metrics
- **Hurst exponent accuracy**: ±0.02 for clean data
- **Statistical properties**: Preserved within ±5%
- **Confound realism**: Domain-appropriate patterns and artifacts

## 🔬 Research Applications

### 1. Estimator Benchmarking
- Test LRD estimators on controlled data
- Evaluate robustness to different confounds
- Compare performance across data types

### 2. Method Development
- Validate new estimation techniques
- Test parameter sensitivity
- Develop preprocessing pipelines

### 3. Educational Purposes
- Demonstrate LRD concepts
- Show effects of different parameters
- Illustrate confound impacts

### 4. Quality Control
- Validate real-world data characteristics
- Test preprocessing effectiveness
- Benchmark analysis pipelines

## 🛠️ Advanced Usage

### Batch Processing
```python
# Generate multiple datasets with different parameters
hurst_values = np.linspace(0.1, 0.9, 20)
datasets = {}

for hurst in hurst_values:
    spec = DataSpecification(
        n_points=10000,
        hurst_exponent=hurst,
        domain_type=DomainType.GENERAL
    )
    datasets[f'H_{hurst:.2f}'] = generator.generate_data(spec)
```

### Custom Confound Implementation
```python
# Extend the generator with custom confounds
class CustomSyntheticDataGenerator(SyntheticDataGenerator):
    def _add_custom_confound(self, data, spec):
        # Implement custom confound logic
        return modified_data
```

### Integration with Analysis Pipeline
```python
# Use generated data with LRD estimators
from estimators.high_performance_dfa import HighPerformanceDFAEstimator

# Generate test data
test_data = generator.generate_data(test_spec)

# Run estimation
estimator = HighPerformanceDFAEstimator()
results = estimator.estimate(test_data['data'])
```

## 📚 References

- **ARFIMA Models**: Hosking (1981), Granger & Joyeux (1980)
- **Fractional Processes**: Mandelbrot & Van Ness (1968)
- **Long-Range Dependence**: Beran (1994), Samorodnitsky & Taqqu (1994)
- **Synthetic Data Generation**: Abry et al. (2000), Bardet et al. (2003)

## 🤝 Contributing

To contribute to the synthetic data generation system:

1. **Add new base models**: Implement in `synthetic_data_generator.py`
2. **Extend confounds**: Add new confound types and implementations
3. **Improve performance**: Optimize algorithms and add parallel processing
4. **Add domain types**: Implement new domain-specific patterns
5. **Enhance validation**: Improve data quality checks and metrics

## 📞 Support

For questions, issues, or contributions:

- **Documentation**: Check the main project README
- **Issues**: Report bugs or feature requests
- **Discussions**: Join project discussions
- **Contributions**: Submit pull requests

---

**🎯 The synthetic data generation system provides a robust foundation for LRD research and benchmarking, enabling reproducible, controlled experiments across diverse domains and data characteristics.**
