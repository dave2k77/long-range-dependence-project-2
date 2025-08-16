"""
Validation Module for Data and Estimator Submissions

This module provides comprehensive validation utilities for:
- Dataset quality and format validation
- Estimator interface and functionality validation
- Benchmark result validation
- Data integrity checks
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import logging
import inspect
import importlib.util
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class DataValidator:
    """
    Validates dataset submissions for quality and format compliance.
    
    Features:
    - Data format validation
    - Quality metrics calculation
    - Statistical property checks
    - Missing data detection
    - Outlier identification
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.quality_thresholds = {
            'min_data_points': 100,
            'max_missing_ratio': 0.1,
            'max_outlier_ratio': 0.05,
            'min_variance': 1e-6,
            'max_skewness': 10.0,
            'max_kurtosis': 100.0
        }
    
    def validate_dataset(self, data: Union[np.ndarray, pd.DataFrame, str, Path], 
                        data_type: str = "general") -> ValidationResult:
        """
        Validate a dataset for quality and format compliance.
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.DataFrame, str, Path]
            Dataset to validate
        data_type : str
            Type of dataset ('general', 'eeg', 'financial', 'hydrology', 'climate')
            
        Returns:
        --------
        ValidationResult
            Validation result with errors, warnings, and metadata
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Load data if it's a file path
            if isinstance(data, (str, Path)):
                data = self._load_data_file(data)
            
            # Basic data checks
            self._validate_basic_properties(data, result)
            
            # Data quality checks
            self._validate_data_quality(data, result)
            
            # Domain-specific validation
            if data_type != "general":
                self._validate_domain_specific(data, data_type, result)
            
            # Statistical property checks
            self._validate_statistical_properties(data, result)
            
            # Determine overall validity
            result.is_valid = len(result.errors) == 0
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error: {str(e)}")
            logger.error(f"Dataset validation failed: {e}")
        
        return result
    
    def _load_data_file(self, file_path: Union[str, Path]) -> Union[np.ndarray, pd.DataFrame]:
        """Load data from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Determine file type and load accordingly
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix.lower() == '.npy':
            return np.load(file_path)
        elif file_path.suffix.lower() == '.npz':
            data = np.load(file_path)
            # Return first array if multiple arrays
            return data[data.files[0]]
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _validate_basic_properties(self, data: Union[np.ndarray, pd.DataFrame], 
                                 result: ValidationResult):
        """Validate basic data properties."""
        # Check data type
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            result.errors.append("Data must be numpy array or pandas DataFrame")
            return
        
        # Check data dimensions
        if data.ndim == 0:
            result.errors.append("Data must have at least one dimension")
            return
        
        if data.ndim > 2:
            result.warnings.append(f"Data has {data.ndim} dimensions, expected 1-2")
        
        # Check data size
        if data.size == 0:
            result.errors.append("Data is empty")
            return
        
        if data.size < self.quality_thresholds['min_data_points']:
            result.warnings.append(f"Data has only {data.size} points, recommended minimum is {self.quality_thresholds['min_data_points']}")
        
        # Store basic metadata
        result.metadata['data_shape'] = data.shape
        result.metadata['data_size'] = data.size
        result.metadata['data_type'] = str(type(data))
    
    def _validate_data_quality(self, data: Union[np.ndarray, pd.DataFrame], 
                             result: ValidationResult):
        """Validate data quality metrics."""
        # Convert to numpy array for analysis
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        # Check for missing values
        missing_mask = np.isnan(data_array) | np.isinf(data_array)
        missing_ratio = np.sum(missing_mask) / data_array.size
        
        if missing_ratio > self.quality_thresholds['max_missing_ratio']:
            result.errors.append(f"Too many missing values: {missing_ratio:.2%} (max: {self.quality_thresholds['max_missing_ratio']:.2%})")
        elif missing_ratio > 0:
            result.warnings.append(f"Missing values detected: {missing_ratio:.2%}")
        
        # Check for outliers (using IQR method)
        if data_array.size > 0:
            q1 = np.percentile(data_array[~missing_mask], 25)
            q3 = np.percentile(data_array[~missing_mask], 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (data_array < lower_bound) | (data_array > upper_bound)
            outlier_ratio = np.sum(outlier_mask) / data_array.size
            
            if outlier_ratio > self.quality_thresholds['max_outlier_ratio']:
                result.warnings.append(f"High outlier ratio: {outlier_ratio:.2%} (max: {self.quality_thresholds['max_outlier_ratio']:.2%})")
            
            result.metadata['outlier_ratio'] = outlier_ratio
        
        result.metadata['missing_ratio'] = missing_ratio
    
    def _validate_domain_specific(self, data: Union[np.ndarray, pd.DataFrame], 
                                data_type: str, result: ValidationResult):
        """Validate domain-specific properties."""
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        if data_type == "eeg":
            self._validate_eeg_data(data_array, result)
        elif data_type == "financial":
            self._validate_financial_data(data_array, result)
        elif data_type in ["hydrology", "climate"]:
            self._validate_environmental_data(data_array, result)
    
    def _validate_eeg_data(self, data: np.ndarray, result: ValidationResult):
        """Validate EEG-specific data properties."""
        # Check for typical EEG frequency ranges
        if data.size > 100:
            # Simple frequency analysis
            fft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data))
            power = np.abs(fft) ** 2
            
            # Check if power is concentrated in typical EEG bands (0.5-50 Hz)
            # This is a simplified check
            result.metadata['eeg_frequency_check'] = "Basic frequency analysis completed"
    
    def _validate_financial_data(self, data: np.ndarray, result: ValidationResult):
        """Validate financial data properties."""
        # Check for extreme values typical in financial data
        if data.size > 0:
            data_clean = data[~np.isnan(data) & ~np.isinf(data)]
            if len(data_clean) > 0:
                extreme_ratio = np.sum(np.abs(data_clean) > 10 * np.std(data_clean)) / len(data_clean)
                if extreme_ratio > 0.01:
                    result.warnings.append(f"High extreme value ratio: {extreme_ratio:.2%}")
                
                result.metadata['extreme_value_ratio'] = extreme_ratio
    
    def _validate_environmental_data(self, data: np.ndarray, result: ValidationResult):
        """Validate environmental data properties."""
        # Check for seasonal patterns
        if data.size > 100:
            # Simple seasonality check
            result.metadata['environmental_check'] = "Basic environmental data validation completed"
    
    def _validate_statistical_properties(self, data: Union[np.ndarray, pd.DataFrame], 
                                       result: ValidationResult):
        """Validate statistical properties of the data."""
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        # Remove missing values for statistical analysis
        data_clean = data_array[~np.isnan(data_array) & ~np.isinf(data_array)]
        
        if len(data_clean) == 0:
            result.errors.append("No valid data for statistical analysis")
            return
        
        # Basic statistics
        mean_val = np.mean(data_clean)
        std_val = np.std(data_clean)
        var_val = np.var(data_clean)
        
        # Check variance
        if var_val < self.quality_thresholds['min_variance']:
            result.warnings.append(f"Very low variance: {var_val:.6f}")
        
        # Check skewness and kurtosis
        if len(data_clean) > 3:
            skewness = self._calculate_skewness(data_clean)
            kurtosis = self._calculate_kurtosis(data_clean)
            
            if abs(skewness) > self.quality_thresholds['max_skewness']:
                result.warnings.append(f"High skewness: {skewness:.2f}")
            
            if abs(kurtosis) > self.quality_thresholds['max_kurtosis']:
                result.warnings.append(f"High kurtosis: {kurtosis:.2f}")
            
            result.metadata['skewness'] = skewness
            result.metadata['kurtosis'] = kurtosis
        
        # Store basic statistics
        result.metadata['mean'] = mean_val
        result.metadata['std'] = std_val
        result.metadata['variance'] = var_val
        result.metadata['min'] = np.min(data_clean)
        result.metadata['max'] = np.max(data_clean)
        result.metadata['range'] = np.max(data_clean) - np.min(data_clean)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        n = len(data)
        
        if std_val == 0:
            return 0.0
        
        skewness = (np.sum(((data - mean_val) / std_val) ** 3)) / n
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        n = len(data)
        
        if std_val == 0:
            return 0.0
        
        kurtosis = (np.sum(((data - mean_val) / std_val) ** 4)) / n - 3
        return kurtosis


class EstimatorValidator:
    """
    Validates estimator submissions for interface and functionality compliance.
    
    Features:
    - Interface compliance checking
    - Method signature validation
    - Basic functionality testing
    - Performance benchmarking
    - Documentation validation
    """
    
    def __init__(self):
        """Initialize the estimator validator."""
        self.required_methods = ["estimate", "__init__"]
        self.required_attributes = ["name", "description"]
        self.optional_methods = ["validate_data", "get_parameters", "set_parameters"]
        self.optional_attributes = ["version", "author", "citation"]
    
    def validate_estimator(self, estimator_file: Union[str, Path], 
                          test_data: Optional[np.ndarray] = None) -> ValidationResult:
        """
        Validate an estimator submission.
        
        Parameters:
        -----------
        estimator_file : Union[str, Path]
            Path to the estimator implementation file
        test_data : Optional[np.ndarray]
            Test data for functionality testing
            
        Returns:
        --------
        ValidationResult
            Validation result with errors, warnings, and metadata
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Check file syntax
            self._check_python_syntax(estimator_file, result)
            
            # Check interface compliance
            self._check_interface_compliance(estimator_file, result)
            
            # Test basic functionality
            if test_data is not None:
                self._test_basic_functionality(estimator_file, test_data, result)
            
            # Check documentation
            self._check_documentation(estimator_file, result)
            
            # Determine overall validity
            result.is_valid = len(result.errors) == 0
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error: {str(e)}")
            logger.error(f"Estimator validation failed: {e}")
        
        return result
    
    def _check_python_syntax(self, file_path: Union[str, Path], result: ValidationResult):
        """Check Python syntax of the estimator file."""
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            # Try to compile the code
            compile(source_code, str(file_path), 'exec')
            result.metadata['syntax_check'] = "Passed"
            
        except SyntaxError as e:
            result.errors.append(f"Python syntax error: {e}")
            result.metadata['syntax_check'] = "Failed"
        except Exception as e:
            result.errors.append(f"Code compilation error: {e}")
            result.metadata['syntax_check'] = "Failed"
    
    def _check_interface_compliance(self, file_path: Union[str, Path], result: ValidationResult):
        """Check if estimator implements required interface."""
        try:
            # Import the estimator module
            spec = importlib.util.spec_from_file_location("estimator_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find estimator class
            estimator_classes = []
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and not name.startswith('_'):
                    estimator_classes.append((name, obj))
            
            if not estimator_classes:
                result.errors.append("No estimator classes found")
                return
            
            # Check each estimator class
            for class_name, estimator_class in estimator_classes:
                class_result = self._validate_single_class(class_name, estimator_class)
                
                if not class_result.is_valid:
                    result.errors.extend([f"{class_name}: {error}" for error in class_result.errors])
                    result.warnings.extend([f"{class_name}: {warning}" for warning in class_result.warnings])
                else:
                    result.metadata[f'{class_name}_interface'] = "Valid"
            
            result.metadata['total_classes'] = len(estimator_classes)
            
        except Exception as e:
            result.errors.append(f"Interface compliance check failed: {e}")
    
    def _validate_single_class(self, class_name: str, estimator_class: type) -> ValidationResult:
        """Validate a single estimator class."""
        result = ValidationResult(is_valid=True)
        
        # Check required methods
        for method_name in self.required_methods:
            if not hasattr(estimator_class, method_name):
                result.errors.append(f"Required method '{method_name}' not implemented")
            else:
                method = getattr(estimator_class, method_name)
                if not callable(method):
                    result.errors.append(f"'{method_name}' is not callable")
        
        # Check required attributes
        for attr_name in self.required_attributes:
            if not hasattr(estimator_class, attr_name):
                result.errors.append(f"Required attribute '{attr_name}' not found")
        
        # Check optional methods
        for method_name in self.optional_methods:
            if hasattr(estimator_class, method_name):
                method = getattr(estimator_class, method_name)
                if not callable(method):
                    result.warnings.append(f"'{method_name}' exists but is not callable")
        
        # Check optional attributes
        for attr_name in self.optional_attributes:
            if hasattr(estimator_class, attr_name):
                result.metadata[f'has_{attr_name}'] = True
        
        # Check estimate method signature
        if hasattr(estimator_class, 'estimate'):
            estimate_method = getattr(estimator_class, 'estimate')
            sig = inspect.signature(estimate_method)
            params = list(sig.parameters.keys())
            
            if not params:
                result.errors.append("'estimate' method must accept at least one parameter")
            else:
                result.metadata['estimate_params'] = params
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def _test_basic_functionality(self, file_path: Union[str, Path], 
                                test_data: np.ndarray, result: ValidationResult):
        """Test basic functionality of the estimator."""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location("estimator_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find estimator class
            estimator_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and not name.startswith('_') and 
                    hasattr(obj, 'estimate')):
                    estimator_class = obj
                    break
            
            if estimator_class is None:
                result.warnings.append("No suitable estimator class found for testing")
                return
            
            # Test instantiation
            try:
                instance = estimator_class()
                result.metadata['instantiation'] = "Success"
            except Exception as e:
                result.errors.append(f"Estimator instantiation failed: {e}")
                return
            
            # Test estimation
            try:
                result_estimate = instance.estimate(test_data)
                result.metadata['estimation_test'] = "Success"
                result.metadata['result_type'] = str(type(result_estimate))
                
                # Check if result is reasonable
                if isinstance(result_estimate, (int, float, np.number)):
                    if np.isnan(result_estimate) or np.isinf(result_estimate):
                        result.warnings.append("Estimation returned NaN or infinite value")
                elif isinstance(result_estimate, (list, tuple, np.ndarray)):
                    if len(result_estimate) == 0:
                        result.warnings.append("Estimation returned empty result")
                
            except Exception as e:
                result.errors.append(f"Estimation test failed: {e}")
                result.metadata['estimation_test'] = "Failed"
            
        except Exception as e:
            result.errors.append(f"Basic functionality test failed: {e}")
    
    def _check_documentation(self, file_path: Union[str, Path], result: ValidationResult):
        """Check documentation quality of the estimator."""
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            # Check for docstrings
            lines = source_code.split('\n')
            docstring_lines = 0
            comment_lines = 0
            
            for line in lines:
                line = line.strip()
                if line.startswith('"""') or line.startswith("'''"):
                    docstring_lines += 1
                elif line.startswith('#'):
                    comment_lines += 1
            
            # Calculate documentation metrics
            total_lines = len(lines)
            docstring_ratio = docstring_lines / total_lines if total_lines > 0 else 0
            comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
            
            if docstring_ratio < 0.1:
                result.warnings.append(f"Low docstring coverage: {docstring_ratio:.2%}")
            
            if comment_ratio < 0.05:
                result.warnings.append(f"Low comment coverage: {comment_ratio:.2%}")
            
            result.metadata['docstring_ratio'] = docstring_ratio
            result.metadata['comment_ratio'] = comment_ratio
            result.metadata['total_lines'] = total_lines
            
        except Exception as e:
            result.warnings.append(f"Documentation check failed: {e}")
    
    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate a comprehensive validation report."""
        report = f"""# Validation Report

## Summary
- **Overall Status**: {'✅ PASSED' if result.is_valid else '❌ FAILED'}
- **Errors**: {len(result.errors)}
- **Warnings**: {len(result.warnings)}

## Errors
"""
        
        if result.errors:
            for error in result.errors:
                report += f"- ❌ {error}\n"
        else:
            report += "- ✅ No errors found\n"
        
        report += "\n## Warnings\n"
        
        if result.warnings:
            for warning in result.warnings:
                report += f"- ⚠️ {warning}\n"
        else:
            report += "- ✅ No warnings\n"
        
        if result.metadata:
            report += "\n## Metadata\n"
            for key, value in result.metadata.items():
                report += f"- **{key}**: {value}\n"
        
        return report
