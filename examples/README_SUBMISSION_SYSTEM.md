# 🚀 Data Submission System

This directory contains a comprehensive submission system for the Long-Range Dependence Analysis Framework, enabling users to submit datasets, estimators, and benchmark results.

## 🌟 Overview

The submission system provides a complete workflow for:
- **Dataset Submissions**: Raw, processed, and synthetic data
- **Estimator Submissions**: New LRD estimation algorithms
- **Benchmark Submissions**: Performance evaluation results
- **Validation & Quality Control**: Automated checks and reports
- **Leaderboard Management**: Performance ranking and comparison

## 📁 System Architecture

```
data/
├── synthetic/          # Generated synthetic datasets
├── raw/               # User-uploaded raw data
├── realistic/         # Processed user data
└── submissions/       # Submission metadata and tracking
    ├── datasets/      # Dataset submission metadata
    ├── estimators/    # Estimator submission metadata
    └── benchmarks/    # Benchmark submission metadata
        ├── results/   # Benchmark result files
        └── leaderboard/ # Performance rankings
```

## 🔧 Core Components

### 1. Dataset Submission Manager
- **Raw Data**: Upload unprocessed datasets (CSV, JSON, NPY, etc.)
- **Processed Data**: Submit preprocessed data with specifications
- **Synthetic Data**: Submit generated data with parameters
- **Validation**: Automatic quality checks and format validation

### 2. Estimator Submission Manager
- **Code Validation**: Syntax and interface compliance checking
- **Functionality Testing**: Basic operation verification
- **Integration**: Seamless framework integration
- **Documentation**: Automatic documentation generation

### 3. Benchmark Submission Manager
- **Performance Metrics**: Execution time, memory, accuracy
- **Leaderboard**: Automated ranking and scoring
- **Comparison**: Multi-estimator performance analysis
- **Visualization**: Charts and reports generation

### 4. Validation System
- **Data Validator**: Quality metrics and statistical checks
- **Estimator Validator**: Interface compliance and testing
- **Quality Reports**: Comprehensive validation summaries

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_demo.txt
```

### 2. Run the Complete Demo
```bash
python submission_system_demo.py
```

### 3. Individual Demonstrations
```bash
# Dataset submissions only
python -c "
from submission_system_demo import SubmissionSystemDemo
demo = SubmissionSystemDemo()
demo.demonstrate_dataset_submissions()
"

# Estimator submissions only
python -c "
from submission_system_demo import SubmissionSystemDemo
demo = SubmissionSystemDemo()
demo.demonstrate_estimator_submissions()
"
```

## 📊 Dataset Submissions

### Raw Dataset Submission
```python
from data_submission.dataset_submission import DatasetSubmissionManager

manager = DatasetSubmissionManager()

submission_id = manager.submit_raw_dataset(
    file_path="path/to/your/data.csv",
    submitter_name="Your Name",
    submitter_email="your.email@example.com",
    dataset_name="My Dataset",
    dataset_description="Description of your dataset",
    dataset_version="1.0.0",
    license="MIT",
    keywords=["EEG", "LRD", "neuroscience"],
    references=["Your Reference 2024"]
)
```

### Processed Dataset Submission
```python
import numpy as np
from data_generation.dataset_specifications import DatasetSpecification

# Your processed data
data = np.random.randn(1000)

# Create specification
spec = DatasetSpecification(
    n_points=1000,
    hurst_exponent=0.7,
    domain_type="eeg"
)

submission_id = manager.submit_processed_dataset(
    data=data,
    submitter_name="Your Name",
    submitter_email="your.email@example.com",
    dataset_name="Processed EEG Data",
    dataset_description="Preprocessed EEG data for LRD analysis",
    dataset_specification=spec,
    dataset_version="1.0.0"
)
```

### Synthetic Dataset Submission
```python
from data_generation.synthetic_data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(random_seed=42)
data = generator.generate_data(spec)

generator_params = {
    'hurst_exponent': 0.7,
    'confound_strength': 0.1,
    'noise_level': 0.05
}

submission_id = manager.submit_synthetic_dataset(
    generator_name="SyntheticDataGenerator",
    generator_parameters=generator_params,
    submitter_name="Your Name",
    submitter_email="your.email@example.com",
    dataset_name="Synthetic EEG Data",
    dataset_description="Synthetic EEG data with controlled properties",
    dataset_specification=spec,
    dataset_version="1.0.0"
)
```

## 🔧 Estimator Submissions

### Creating an Estimator
```python
# my_estimator.py
import numpy as np
from typing import Dict, Any

class MyLRDEstimator:
    """My custom LRD estimator."""
    
    def __init__(self):
        self.name = "My LRD Estimator"
        self.description = "A custom estimator for LRD analysis"
        self.version = "1.0.0"
        self.author = "Your Name"
    
    def estimate(self, data: np.ndarray) -> Dict[str, Any]:
        """Estimate Hurst exponent."""
        # Your estimation logic here
        hurst_estimate = 0.5  # Placeholder
        
        return {
            'hurst_exponent': hurst_estimate,
            'method': 'my_method',
            'confidence': 0.95
        }
    
    def validate_data(self, data: np.ndarray) -> bool:
        """Validate input data."""
        return len(data) > 100 and not np.any(np.isnan(data))
```

### Submitting an Estimator
```python
from data_submission.estimator_submission import EstimatorSubmissionManager

manager = EstimatorSubmissionManager()

submission_id = manager.submit_estimator(
    estimator_file="my_estimator.py",
    submitter_name="Your Name",
    submitter_email="your.email@example.com",
    estimator_name="My LRD Estimator",
    estimator_description="A custom estimator for LRD analysis",
    estimator_version="1.0.0",
    license="MIT",
    citation="Your Paper 2024",
    keywords=["LRD", "custom", "estimation"],
    references=["Your Paper 2024"],
    dependencies=["numpy", "scipy"],
    test_data=np.random.randn(1000)
)
```

## 🏆 Benchmark Submissions

### Submitting Benchmark Results
```python
from data_submission.benchmark_submission import BenchmarkSubmissionManager

manager = BenchmarkSubmissionManager()

performance_metrics = {
    'MyEstimator': {
        'execution_time': 0.15,
        'memory_usage': 25.5,
        'accuracy': 0.92,
        'rmse': 0.08,
        'mae': 0.06
    },
    'BaselineEstimator': {
        'execution_time': 0.08,
        'memory_usage': 15.2,
        'accuracy': 0.89,
        'rmse': 0.11,
        'mae': 0.09
    }
}

submission_id = manager.submit_benchmark_results(
    benchmark_name="My Estimator Benchmark",
    benchmark_description="Comprehensive evaluation of my estimator",
    dataset_name="Test Dataset",
    estimators_tested=list(performance_metrics.keys()),
    performance_metrics=performance_metrics,
    submitter_name="Your Name",
    submitter_email="your.email@example.com",
    benchmark_version="1.0.0"
)
```

## 🔍 Validation and Quality Control

### Data Validation
```python
from data_submission.validation import DataValidator

validator = DataValidator()

# Validate your dataset
result = validator.validate_dataset(
    data=your_data,
    data_type="eeg"  # or "financial", "hydrology", "climate"
)

if result.is_valid:
    print("✅ Dataset validation passed")
else:
    print("❌ Dataset validation failed:")
    for error in result.errors:
        print(f"   - {error}")

# Check warnings
for warning in result.warnings:
    print(f"⚠️ {warning}")
```

### Estimator Validation
```python
from data_submission.validation import EstimatorValidator

validator = EstimatorValidator()

# Validate your estimator
result = validator.validate_estimator(
    estimator_file="my_estimator.py",
    test_data=np.random.randn(1000)
)

if result.is_valid:
    print("✅ Estimator validation passed")
else:
    print("❌ Estimator validation failed:")
    for error in result.errors:
        print(f"   - {error}")

# Generate validation report
report = validator.generate_validation_report(result)
print(report)
```

## 📈 Leaderboard and Rankings

### View Leaderboard
```python
from data_submission.benchmark_submission import BenchmarkSubmissionManager

manager = BenchmarkSubmissionManager()

# Get top 10 submissions
leaderboard = manager.get_leaderboard(top_n=10)

for i, entry in enumerate(leaderboard):
    print(f"{i+1}. {entry['benchmark_name']} - Score: {entry['leaderboard_score']:.4f}")
    print(f"    Dataset: {entry['dataset_name']}")
    print(f"    Submitter: {entry['submitter_name']}")
    print()
```

### Export Results
```python
# Export to CSV
csv_path = manager.export_leaderboard_csv()

# Generate visualization
viz_path = manager.generate_leaderboard_visualization()

print(f"Leaderboard exported to: {csv_path}")
print(f"Visualization saved to: {viz_path}")
```

## 📊 Submission Statistics

### Get Overall Statistics
```python
# Dataset statistics
dataset_stats = manager.get_submission_statistics()
print(f"Total dataset submissions: {dataset_stats['total_submissions']}")
print(f"By type: {dataset_stats['by_type']}")
print(f"By status: {dataset_stats['by_status']}")

# Estimator statistics
estimator_stats = estimator_manager.get_submission_statistics()
print(f"Total estimator submissions: {estimator_stats['total_submissions']}")
print(f"Approved estimators: {estimator_stats['approved_estimators']}")

# Benchmark statistics
benchmark_stats = benchmark_manager.get_submission_statistics()
print(f"Total benchmark submissions: {benchmark_stats['total_submissions']}")
print(f"Average leaderboard score: {benchmark_stats['average_leaderboard_score']:.4f}")
```

## 🛠️ Advanced Usage

### Custom Validation Rules
```python
from data_submission.validation import DataValidator

class CustomDataValidator(DataValidator):
    def __init__(self):
        super().__init__()
        # Customize quality thresholds
        self.quality_thresholds.update({
            'min_data_points': 500,  # Require more data points
            'max_missing_ratio': 0.05,  # Stricter missing data limit
            'max_outlier_ratio': 0.02  # Stricter outlier limit
        })
    
    def _validate_domain_specific(self, data, data_type, result):
        """Add custom domain-specific validation."""
        if data_type == "eeg":
            # Custom EEG validation logic
            if len(data) < 1000:
                result.warnings.append("EEG data should have at least 1000 points")
        
        super()._validate_domain_specific(data, data_type, result)
```

### Batch Processing
```python
# Submit multiple datasets
dataset_files = ["data1.csv", "data2.csv", "data3.csv"]
submission_ids = []

for file_path in dataset_files:
    submission_id = manager.submit_raw_dataset(
        file_path=file_path,
        submitter_name="Your Name",
        submitter_email="your.email@example.com",
        dataset_name=f"Dataset {len(submission_ids) + 1}",
        dataset_description=f"Description for dataset {len(submission_ids) + 1}",
        dataset_version="1.0.0"
    )
    submission_ids.append(submission_id)

print(f"Submitted {len(submission_ids)} datasets")
```

### Integration with Analysis Pipeline
```python
# Submit dataset and immediately use it
submission_id = manager.submit_processed_dataset(
    data=your_data,
    submitter_name="Your Name",
    submitter_email="your.email@example.com",
    dataset_name="Analysis Dataset",
    dataset_description="Dataset for immediate analysis",
    dataset_specification=spec,
    dataset_version="1.0.0"
)

# Get the submitted data
submission = manager.get_submission(submission_id)
data_file = submission.file_paths['data_file']
loaded_data = np.load(data_file)

# Use in your analysis
from estimators.high_performance_dfa import HighPerformanceDFAEstimator
estimator = HighPerformanceDFAEstimator()
result = estimator.estimate(loaded_data)
```

## 📋 Submission Requirements

### Dataset Requirements
- **Raw Data**: CSV, JSON, NPY, NPZ, or compressed archives
- **Processed Data**: NumPy arrays with full specifications
- **Synthetic Data**: Generated data with parameters and specifications
- **Metadata**: Name, description, version, license, citation
- **Quality**: Minimum 100 data points, reasonable missing data ratio

### Estimator Requirements
- **Interface**: Must implement `estimate(data)` method
- **Attributes**: Must have `name` and `description` attributes
- **Validation**: Should include `validate_data(data)` method
- **Documentation**: Proper docstrings and comments
- **Testing**: Should work with provided test data

### Benchmark Requirements
- **Metrics**: Execution time, memory usage, accuracy
- **Comparisons**: Multiple estimators tested
- **Validation**: Results must be reasonable and complete
- **Metadata**: Benchmark name, description, dataset used

## 🔒 Security and Privacy

### Data Protection
- All submissions are stored locally
- No external data transmission
- User information is kept private
- Data validation prevents malicious uploads

### Access Control
- Submission tracking with unique IDs
- User attribution for all submissions
- Validation status tracking
- Approval workflow for estimators

## 📚 API Reference

### DatasetSubmissionManager
```python
class DatasetSubmissionManager:
    def submit_raw_dataset(self, file_path, submitter_name, submitter_email, 
                          dataset_name, dataset_description, **kwargs) -> str
    
    def submit_processed_dataset(self, data, submitter_name, submitter_email,
                               dataset_name, dataset_description, 
                               dataset_specification, **kwargs) -> str
    
    def submit_synthetic_dataset(self, generator_name, generator_parameters,
                                submitter_name, submitter_email, dataset_name,
                                dataset_description, dataset_specification, **kwargs) -> str
    
    def get_submission(self, submission_id) -> Optional[SubmissionMetadata]
    
    def list_submissions(self, submission_type=None) -> List[SubmissionMetadata]
    
    def delete_submission(self, submission_id) -> bool
    
    def get_submission_statistics(self) -> Dict[str, Any]
```

### EstimatorSubmissionManager
```python
class EstimatorSubmissionManager:
    def submit_estimator(self, estimator_file, submitter_name, submitter_email,
                        estimator_name, estimator_description, **kwargs) -> str
    
    def approve_estimator(self, submission_id) -> bool
    
    def get_submission(self, submission_id) -> Optional[EstimatorSubmissionMetadata]
    
    def list_submissions(self, status=None) -> List[EstimatorSubmissionMetadata]
    
    def generate_documentation(self, submission_id) -> str
```

### BenchmarkSubmissionManager
```python
class BenchmarkSubmissionManager:
    def submit_benchmark_results(self, benchmark_name, benchmark_description,
                               dataset_name, estimators_tested, performance_metrics,
                               submitter_name, submitter_email, **kwargs) -> str
    
    def get_leaderboard(self, top_n=None) -> List[Dict[str, Any]]
    
    def get_benchmark_comparison(self, submission_ids) -> Dict[str, Any]
    
    def generate_benchmark_report(self, submission_id) -> str
    
    def export_leaderboard_csv(self, output_path=None) -> str
    
    def generate_leaderboard_visualization(self, output_path=None) -> str
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# Make sure src is in your Python path
import sys
sys.path.insert(0, "path/to/your/project/src")
```

#### 2. File Permission Errors
```bash
# Ensure write permissions to data directory
chmod -R 755 data/
```

#### 3. Validation Failures
- Check data format and size requirements
- Verify estimator interface compliance
- Ensure benchmark metrics are complete
- Review validation error messages

#### 4. Memory Issues
- Use chunked processing for large datasets
- Monitor memory usage during validation
- Consider data compression for large files

### Getting Help

1. **Check Logs**: All operations are logged with detailed information
2. **Validation Reports**: Review comprehensive validation summaries
3. **Error Messages**: Detailed error descriptions with suggestions
4. **Documentation**: Refer to this README and code comments

## 🤝 Contributing

### Adding New Features
1. **Extend Validators**: Add new validation rules and checks
2. **New Submission Types**: Support additional data or estimator formats
3. **Enhanced Reporting**: Improve documentation and visualization
4. **Performance Optimization**: Optimize validation and processing

### Code Style
- Follow PEP 8 guidelines
- Include comprehensive docstrings
- Add type hints for all functions
- Write unit tests for new functionality

### Testing
```bash
# Run validation tests
python -m pytest tests/test_validation.py

# Run submission tests
python -m pytest tests/test_submissions.py

# Run integration tests
python -m pytest tests/test_integration.py
```

## 📄 License

This submission system is part of the Long-Range Dependence Analysis Framework and is licensed under the MIT License.

## 🙏 Acknowledgments

- **Contributors**: All users who submit datasets, estimators, and benchmarks
- **Research Community**: For feedback and suggestions on system design
- **Open Source**: Built on top of excellent open-source libraries

---

**🎯 The submission system provides a robust foundation for collaborative LRD research, enabling researchers to share data, algorithms, and results in a standardized, validated, and trackable manner.**
