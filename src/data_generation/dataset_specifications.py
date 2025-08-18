"""
Dataset Specifications and Submission System

This module provides standardized dataset specifications and submission protocols
for benchmarking physics-based fractional machine learning models.
"""

import numpy as np
import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class DatasetFormat(Enum):
    """Supported dataset formats."""
    NUMPY = "numpy"
    PANDAS = "pandas"
    CSV = "csv"
    JSON = "json"
    HDF5 = "hdf5"
    MAT = "matlab"


class DatasetType(Enum):
    """Types of datasets."""
    SYNTHETIC = "synthetic"
    REAL_WORLD = "real_world"
    BENCHMARK = "benchmark"
    VALIDATION = "validation"


class DomainCategory(Enum):
    """Domain categories for datasets."""
    BIOMEDICAL = "biomedical"
    FINANCIAL = "financial"
    CLIMATE = "climate"
    HYDROLOGY = "hydrology"
    GEOPHYSICAL = "geophysical"
    NETWORK = "network"
    IMAGE = "image"
    AUDIO = "audio"
    GENERAL = "general"


@dataclass
class DatasetMetadata:
    """Metadata for dataset specification."""
    name: str
    description: str
    author: str
    contact: str
    creation_date: str
    version: str = "1.0.0"
    license: str = "MIT"
    citation: Optional[str] = None
    keywords: List[str] = None
    references: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.references is None:
            self.references = []
        if not self.creation_date:
            self.creation_date = datetime.now().isoformat()


@dataclass
class DatasetProperties:
    """Properties of the dataset."""
    n_points: int
    n_variables: int = 1
    sampling_frequency: Optional[float] = None
    time_unit: str = "points"
    spatial_dimensions: Optional[Tuple[int, ...]] = None
    missing_values: bool = False
    outliers: bool = False
    
    # Statistical properties
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # LRD properties (if known)
    hurst_exponent: Optional[float] = None
    alpha_stable_alpha: Optional[float] = None
    is_stationary: Optional[bool] = None
    has_seasonality: Optional[bool] = None
    has_trends: Optional[bool] = None


@dataclass
class ConfoundDescription:
    """Description of confounds present in the data."""
    non_stationarity: bool = False
    heavy_tails: bool = False
    baseline_drift: bool = False
    artifacts: bool = False
    seasonality: bool = False
    trend_changes: bool = False
    volatility_clustering: bool = False
    regime_changes: bool = False
    jumps: bool = False
    measurement_noise: bool = False
    missing_data: bool = False
    outliers: bool = False
    
    # Additional confound descriptions
    confound_details: Dict[str, str] = None
    
    def __post_init__(self):
        if self.confound_details is None:
            self.confound_details = {}


@dataclass
class BenchmarkProtocol:
    """Benchmark protocol specification."""
    name: str
    description: str
    estimators_to_test: List[str]
    performance_metrics: List[str]
    validation_methods: List[str]
    cross_validation_folds: int = 5
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Additional protocol parameters
    custom_parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_parameters is None:
            self.custom_parameters = {}


@dataclass
class DatasetSpecification:
    """Complete dataset specification."""
    metadata: DatasetMetadata
    properties: DatasetProperties
    confounds: ConfoundDescription
    benchmark_protocol: Optional[BenchmarkProtocol] = None
    
    # Data storage
    data_path: Optional[str] = None
    data_format: DatasetFormat = DatasetFormat.NUMPY
    data_hash: Optional[str] = None
    
    # Validation
    validation_status: str = "pending"
    validation_notes: List[str] = None
    
    def __post_init__(self):
        if self.validation_notes is None:
            self.validation_notes = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Convert to JSON string."""
        # Handle numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        spec_dict = self.to_dict()
        spec_dict = self._recursive_convert(spec_dict, convert_numpy)
        
        json_str = json.dumps(spec_dict, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Dataset specification saved to {filepath}")
        
        return json_str
    
    def _recursive_convert(self, obj, convert_func):
        """Recursively convert numpy types in nested structures."""
        if isinstance(obj, dict):
            return {k: self._recursive_convert(v, convert_func) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._recursive_convert(item, convert_func) for item in obj]
        else:
            return convert_func(obj)
    
    def validate_specification(self) -> bool:
        """Validate the dataset specification."""
        try:
            # Validate metadata
            if not self.metadata.name or not self.metadata.description:
                raise ValueError("Dataset must have name and description")
            
            # Validate properties
            if self.properties.n_points <= 0:
                raise ValueError("Number of points must be positive")
            if self.properties.n_variables <= 0:
                raise ValueError("Number of variables must be positive")
            
            # Validate confounds
            # (confounds are boolean, so no validation needed)
            
            # Validate benchmark protocol if present
            if self.benchmark_protocol:
                if not self.benchmark_protocol.estimators_to_test:
                    raise ValueError("Benchmark protocol must specify estimators to test")
                if not self.benchmark_protocol.performance_metrics:
                    raise ValueError("Benchmark protocol must specify performance metrics")
            
            logger.info("Dataset specification validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Dataset specification validation failed: {e}")
            return False
    
    def calculate_data_hash(self, data: np.ndarray) -> str:
        """Calculate hash of the data for integrity checking."""
        data_bytes = data.tobytes()
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        self.data_hash = data_hash
        return data_hash
    
    def verify_data_integrity(self, data: np.ndarray) -> bool:
        """Verify data integrity using stored hash."""
        if self.data_hash is None:
            logger.warning("No data hash stored for verification")
            return False
        
        calculated_hash = hashlib.sha256(data.tobytes()).hexdigest()
        return calculated_hash == self.data_hash


class DatasetSpecificationManager:
    """Manager for dataset specifications and submissions."""
    
    def __init__(self, base_directory: str = "dataset_specifications"):
        """
        Initialize the dataset specification manager.
        
        Parameters:
        -----------
        base_directory : str
            Base directory for storing dataset specifications
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_directory / "submitted").mkdir(exist_ok=True)
        (self.base_directory / "validated").mkdir(exist_ok=True)
        (self.base_directory / "benchmark").mkdir(exist_ok=True)
        (self.base_directory / "templates").mkdir(exist_ok=True)
        
        logger.info(f"Dataset specification manager initialized at {self.base_directory}")
    
    def create_template_specification(self, domain: DomainCategory, 
                                   dataset_type: DatasetType) -> DatasetSpecification:
        """Create a template dataset specification."""
        
        # Create metadata template
        metadata = DatasetMetadata(
            name=f"template_{domain.value}_{dataset_type.value}",
            description=f"Template dataset specification for {domain.value} {dataset_type.value} data",
            author="[Your Name]",
            contact="[your.email@example.com]",
            creation_date=datetime.now().isoformat(),
            keywords=[domain.value, dataset_type.value, "template"],
            references=[]
        )
        
        # Create properties template
        properties = DatasetProperties(
            n_points=1000,
            n_variables=1,
            sampling_frequency=1.0,
            time_unit="seconds",
            missing_values=False,
            outliers=False
        )
        
        # Create confounds template
        confounds = ConfoundDescription()
        
        # Create benchmark protocol template
        benchmark_protocol = BenchmarkProtocol(
            name=f"benchmark_{domain.value}_{dataset_type.value}",
            description=f"Standard benchmark protocol for {domain.value} {dataset_type.value} data",
            estimators_to_test=[
                "HighPerformanceDFAEstimator",
                "HighPerformanceMFDFAEstimator",
                "HighPerformanceRSEstimator",
                "HighPerformanceHiguchiEstimator",
                "HighPerformanceWhittleMLEEstimator",
                "HighPerformancePeriodogramEstimator",
                "HighPerformanceGPHEstimator",
                "HighPerformanceWaveletLeadersEstimator",
                "HighPerformanceWaveletWhittleEstimator",
                "HighPerformanceWaveletLogVarianceEstimator"
            ],
            performance_metrics=[
                "execution_time",
                "memory_usage",
                "accuracy",
                "reliability",
                "scaling_quality"
            ],
            validation_methods=[
                "cross_validation",
                "bootstrap",
                "holdout"
            ]
        )
        
        spec = DatasetSpecification(
            metadata=metadata,
            properties=properties,
            confounds=confounds,
            benchmark_protocol=benchmark_protocol
        )
        
        return spec
    
    def save_specification(self, spec: DatasetSpecification, 
                          filename: Optional[str] = None) -> str:
        """Save dataset specification to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{spec.metadata.name}_{timestamp}.json"
        
        filepath = self.base_directory / "submitted" / filename
        
        # Save specification
        spec.to_json(str(filepath))
        
        logger.info(f"Dataset specification saved to {filepath}")
        return str(filepath)
    
    def load_specification(self, filepath: str) -> DatasetSpecification:
        """Load dataset specification from file."""
        with open(filepath, 'r') as f:
            spec_dict = json.load(f)
        
        # Reconstruct the specification
        spec = self._reconstruct_specification(spec_dict)
        
        logger.info(f"Dataset specification loaded from {filepath}")
        return spec
    
    def _reconstruct_specification(self, spec_dict: Dict[str, Any]) -> DatasetSpecification:
        """Reconstruct DatasetSpecification from dictionary."""
        # Reconstruct nested objects
        metadata = DatasetMetadata(**spec_dict['metadata'])
        properties = DatasetProperties(**spec_dict['properties'])
        confounds = ConfoundDescription(**spec_dict['confounds'])
        
        benchmark_protocol = None
        if spec_dict.get('benchmark_protocol'):
            benchmark_protocol = BenchmarkProtocol(**spec_dict['benchmark_protocol'])
        
        spec = DatasetSpecification(
            metadata=metadata,
            properties=properties,
            confounds=confounds,
            benchmark_protocol=benchmark_protocol
        )
        
        # Restore additional attributes
        spec.data_path = spec_dict.get('data_path')
        spec.data_format = DatasetFormat(spec_dict.get('data_format', 'numpy'))
        spec.data_hash = spec_dict.get('data_hash')
        spec.validation_status = spec_dict.get('validation_status', 'pending')
        spec.validation_notes = spec_dict.get('validation_notes', [])
        
        return spec
    
    def submit_dataset(self, spec: DatasetSpecification, data: np.ndarray,
                      data_format: DatasetFormat = DatasetFormat.NUMPY) -> str:
        """Submit a dataset for validation and inclusion in benchmarks."""
        
        # Validate specification
        if not spec.validate_specification():
            raise ValueError("Dataset specification validation failed")
        
        # Calculate data hash
        spec.calculate_data_hash(data)
        
        # Save data
        data_filename = f"{spec.metadata.name}_data.npy"
        data_path = self.base_directory / "submitted" / data_filename
        np.save(data_path, data)
        spec.data_path = str(data_path)
        
        # Save specification
        spec_filename = f"{spec.metadata.name}_spec.json"
        spec_path = self.save_specification(spec, spec_filename)
        
        logger.info(f"Dataset submitted successfully: {spec.metadata.name}")
        logger.info(f"Data saved to: {data_path}")
        logger.info(f"Specification saved to: {spec_path}")
        
        return spec_path
    
    def validate_submitted_dataset(self, spec_filepath: str) -> bool:
        """Validate a submitted dataset."""
        try:
            # Load specification
            spec = self.load_specification(spec_filepath)
            
            # Load data
            if spec.data_path is None:
                raise ValueError("No data path specified")
            
            data = np.load(spec.data_path)
            
            # Verify data integrity
            if not spec.verify_data_integrity(data):
                raise ValueError("Data integrity check failed")
            
            # Validate data properties
            if len(data) != spec.properties.n_points:
                raise ValueError(f"Data length mismatch: expected {spec.properties.n_points}, got {len(data)}")
            
            # Update validation status
            spec.validation_status = "validated"
            spec.validation_notes.append(f"Validated on {datetime.now().isoformat()}")
            
            # Move to validated directory
            validated_path = self.base_directory / "validated" / f"{spec.metadata.name}_validated.json"
            spec.to_json(str(validated_path))
            
            logger.info(f"Dataset validation successful: {spec.metadata.name}")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False
    
    def list_datasets(self, status: str = "all") -> List[Dict[str, Any]]:
        """List available datasets."""
        datasets = []
        
        if status in ["all", "submitted"]:
            submitted_dir = self.base_directory / "submitted"
            for spec_file in submitted_dir.glob("*_spec.json"):
                try:
                    spec = self.load_specification(str(spec_file))
                    datasets.append({
                        'name': spec.metadata.name,
                        'status': 'submitted',
                        'domain': spec.metadata.keywords[0] if spec.metadata.keywords else 'unknown',
                        'n_points': spec.properties.n_points,
                        'filepath': str(spec_file)
                    })
                except Exception as e:
                    logger.warning(f"Could not load specification {spec_file}: {e}")
        
        if status in ["all", "validated"]:
            validated_dir = self.base_directory / "validated"
            for spec_file in validated_dir.glob("*_validated.json"):
                try:
                    spec = self.load_specification(str(spec_file))
                    datasets.append({
                        'name': spec.metadata.name,
                        'status': 'validated',
                        'domain': spec.metadata.keywords[0] if spec.metadata.keywords else 'unknown',
                        'n_points': spec.properties.n_points,
                        'filepath': str(spec_file)
                    })
                except Exception as e:
                    logger.warning(f"Could not load specification {spec_file}: {e}")
        
        return datasets
    
    def create_benchmark_suite(self, dataset_names: List[str]) -> Dict[str, Any]:
        """Create a benchmark suite from validated datasets."""
        benchmark_suite = {
            'name': f"benchmark_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'creation_date': datetime.now().isoformat(),
            'datasets': [],
            'protocols': {},
            'estimators': [
                "HighPerformanceDFAEstimator",
                "HighPerformanceMFDFAEstimator",
                "HighPerformanceRSEstimator",
                "HighPerformanceHiguchiEstimator",
                "HighPerformanceWhittleMLEEstimator",
                "HighPerformancePeriodogramEstimator",
                "HighPerformanceGPHEstimator",
                "HighPerformanceWaveletLeadersEstimator",
                "HighPerformanceWaveletWhittleEstimator",
                "HighPerformanceWaveletLogVarianceEstimator"
            ]
        }
        
        validated_dir = self.base_directory / "validated"
        
        for dataset_name in dataset_names:
            spec_file = validated_dir / f"{dataset_name}_validated.json"
            if spec_file.exists():
                try:
                    spec = self.load_specification(str(spec_file))
                    benchmark_suite['datasets'].append({
                        'name': spec.metadata.name,
                        'specification': spec.to_dict(),
                        'data_path': spec.data_path
                    })
                    
                    if spec.benchmark_protocol:
                        benchmark_suite['protocols'][spec.metadata.name] = spec.benchmark_protocol.to_dict()
                    
                except Exception as e:
                    logger.warning(f"Could not include dataset {dataset_name}: {e}")
        
        # Save benchmark suite
        suite_path = self.base_directory / "benchmark" / f"{benchmark_suite['name']}.json"
        with open(suite_path, 'w') as f:
            json.dump(benchmark_suite, f, indent=2, default=str)
        
        logger.info(f"Benchmark suite created: {suite_path}")
        return benchmark_suite


def create_standard_benchmark_protocols() -> Dict[str, BenchmarkProtocol]:
    """Create standard benchmark protocols for different domains."""
    protocols = {}
    
    # General LRD estimation protocol
    protocols['general_lrd'] = BenchmarkProtocol(
        name="General LRD Estimation",
        description="Standard protocol for long-range dependence estimation",
        estimators_to_test=[
            "HighPerformanceDFAEstimator",
            "HighPerformanceMFDFAEstimator",
            "HighPerformanceRSEstimator",
            "HighPerformanceHiguchiEstimator"
        ],
        performance_metrics=[
            "execution_time",
            "memory_usage",
            "hurst_accuracy",
            "reliability"
        ],
        validation_methods=[
            "cross_validation",
            "bootstrap"
        ]
    )
    
    # Spectral analysis protocol
    protocols['spectral_analysis'] = BenchmarkProtocol(
        name="Spectral Analysis",
        description="Protocol for spectral-based LRD estimation",
        estimators_to_test=[
            "HighPerformanceWhittleMLEEstimator",
            "HighPerformancePeriodogramEstimator",
            "HighPerformanceGPHEstimator"
        ],
        performance_metrics=[
            "execution_time",
            "memory_usage",
            "frequency_resolution",
            "estimation_accuracy"
        ],
        validation_methods=[
            "cross_validation",
            "holdout"
        ]
    )
    
    # Wavelet analysis protocol
    protocols['wavelet_analysis'] = BenchmarkProtocol(
        name="Wavelet Analysis",
        description="Protocol for wavelet-based LRD estimation",
        estimators_to_test=[
            "HighPerformanceWaveletLeadersEstimator",
            "HighPerformanceWaveletWhittleEstimator",
            "HighPerformanceWaveletLogVarianceEstimator"
        ],
        performance_metrics=[
            "execution_time",
            "memory_usage",
            "scale_resolution",
            "wavelet_accuracy"
        ],
        validation_methods=[
            "cross_validation",
            "bootstrap"
        ]
    )
    
    return protocols


if __name__ == "__main__":
    # Example usage
    manager = DatasetSpecificationManager()
    
    # Create a template specification
    template = manager.create_template_specification(
        DomainCategory.BIOMEDICAL, 
        DatasetType.REAL_WORLD
    )
    
    # Save template
    template_path = manager.save_specification(template)
    print(f"Template saved to: {template_path}")
    
    # List available datasets
    datasets = manager.list_datasets()
    print(f"Available datasets: {len(datasets)}")
    
    # Create standard protocols
    protocols = create_standard_benchmark_protocols()
    print(f"Created {len(protocols)} standard protocols")


