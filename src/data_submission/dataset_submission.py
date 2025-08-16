"""
Dataset Submission Manager

This module handles the submission, validation, and storage of datasets
for the Long-Range Dependence Analysis Framework.

Supports:
- Raw data uploads
- Processed data submissions  
- Synthetic data generation
- Data validation and quality checks
- Metadata management
- Storage organization
"""

import os
import shutil
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import zipfile
import tempfile

try:
    from ..data_generation.dataset_specifications import (
        DatasetSpecification, DatasetMetadata, DatasetProperties, 
        ConfoundDescription, BenchmarkProtocol, DatasetFormat, DomainCategory
    )
    
    # Import quality evaluation system
    from ..validation.synthetic_data_quality import (
        SyntheticDataQualityEvaluator, create_domain_specific_evaluator
    )
except ImportError:
    # Fallback to absolute imports for demo purposes
    from src.data_generation.dataset_specifications import (
        DatasetSpecification, DatasetMetadata, DatasetProperties, 
        ConfoundDescription, BenchmarkProtocol, DatasetFormat, DomainCategory
    )
    
    # Import quality evaluation system
    from src.validation.synthetic_data_quality import (
        SyntheticDataQualityEvaluator, create_domain_specific_evaluator
    )

logger = logging.getLogger(__name__)


@dataclass
class SubmissionMetadata:
    """Metadata for dataset submissions."""
    submission_id: str
    submitter_name: str
    submitter_email: str
    submission_date: str
    submission_type: str  # 'raw', 'processed', 'synthetic'
    dataset_name: str
    dataset_description: str
    dataset_version: str
    license: str
    citation: Optional[str] = None
    keywords: List[str] = None
    references: List[str] = None
    validation_status: str = "pending"
    validation_notes: List[str] = None
    file_paths: Dict[str, str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.references is None:
            self.references = []
        if self.validation_notes is None:
            self.validation_notes = []
        if self.file_paths is None:
            self.file_paths = {}


class DatasetSubmissionManager:
    """
    Manages dataset submissions including validation, storage, and organization.
    
    Features:
    - Multiple data format support (CSV, JSON, NPY, HDF5, etc.)
    - Automatic data validation and quality checks
    - Metadata extraction and management
    - Storage organization by data type
    - Version control and tracking
    """
    
    def __init__(self, base_data_dir: str = "data"):
        """
        Initialize the dataset submission manager.
        
        Parameters:
        -----------
        base_data_dir : str
            Base directory for data storage
        """
        self.base_data_dir = Path(base_data_dir)
        self.submissions_dir = self.base_data_dir / "submissions"
        self.metadata_dir = self.submissions_dir / "metadata"
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize submission counter
        self.submission_counter = self._load_submission_counter()
        
        logger.info(f"Dataset Submission Manager initialized at {self.base_data_dir.absolute()}")
    
    def _create_directories(self):
        """Create necessary directory structure."""
        directories = [
            self.submissions_dir,
            self.metadata_dir,
            self.base_data_dir / "synthetic",
            self.base_data_dir / "raw", 
            self.base_data_dir / "realistic"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def _load_submission_counter(self) -> int:
        """Load submission counter from file."""
        counter_file = self.submissions_dir / "submission_counter.txt"
        if counter_file.exists():
            try:
                with open(counter_file, 'r') as f:
                    return int(f.read().strip())
            except (ValueError, IOError):
                logger.warning("Could not load submission counter, starting from 0")
        return 0
    
    def _save_submission_counter(self):
        """Save submission counter to file."""
        counter_file = self.submissions_dir / "submission_counter.txt"
        with open(counter_file, 'w') as f:
            f.write(str(self.submission_counter))
    
    def _generate_submission_id(self) -> str:
        """Generate unique submission ID."""
        self.submission_counter += 1
        self._save_submission_counter()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sub_{self.submission_counter:06d}_{timestamp}"
    
    def submit_raw_dataset(self, 
                          file_path: Union[str, Path],
                          submitter_name: str,
                          submitter_email: str,
                          dataset_name: str,
                          dataset_description: str,
                          dataset_version: str = "1.0.0",
                          license: str = "MIT",
                          citation: Optional[str] = None,
                          keywords: Optional[List[str]] = None,
                          references: Optional[List[str]] = None) -> str:
        """
        Submit a raw dataset for processing and validation.
        
        Parameters:
        -----------
        file_path : Union[str, Path]
            Path to the raw dataset file
        submitter_name : str
            Name of the person submitting the dataset
        submitter_email : str
            Email of the submitter
        dataset_name : str
            Name of the dataset
        dataset_description : str
            Description of the dataset
        dataset_version : str
            Version of the dataset
        license : str
            License for the dataset
        citation : Optional[str]
            Citation information
        keywords : Optional[List[str]]
            Keywords describing the dataset
        references : Optional[List[str]]
            References related to the dataset
            
        Returns:
        --------
        str
            Submission ID for tracking
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Generate submission ID
        submission_id = self._generate_submission_id()
        
        # Create submission metadata
        metadata = SubmissionMetadata(
            submission_id=submission_id,
            submitter_name=submitter_name,
            submitter_email=submitter_email,
            submission_date=datetime.now().isoformat(),
            submission_type="raw",
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            dataset_version=dataset_version,
            license=license,
            citation=citation,
            keywords=keywords or [],
            references=references or []
        )
        
        # Copy file to raw data directory
        raw_data_dir = self.base_data_dir / "raw" / submission_id
        raw_data_dir.mkdir(exist_ok=True)
        
        # Handle different file types
        if file_path.suffix.lower() in ['.zip', '.tar.gz', '.tar']:
            # Extract compressed files
            self._extract_compressed_file(file_path, raw_data_dir)
            metadata.file_paths['extracted_dir'] = str(raw_data_dir)
        else:
            # Copy single file
            dest_file = raw_data_dir / file_path.name
            shutil.copy2(file_path, dest_file)
            metadata.file_paths['data_file'] = str(dest_file)
        
        # Save metadata
        self._save_submission_metadata(metadata)
        
        # Run initial validation
        self._validate_raw_submission(metadata)
        
        logger.info(f"Raw dataset submitted successfully: {submission_id}")
        return submission_id
    
    def submit_processed_dataset(self,
                                data: Union[np.ndarray, pd.DataFrame, Dict[str, Any]],
                                submitter_name: str,
                                submitter_email: str,
                                dataset_name: str,
                                dataset_description: str,
                                dataset_specification: DatasetSpecification,
                                dataset_version: str = "1.0.0",
                                license: str = "MIT",
                                citation: Optional[str] = None,
                                keywords: Optional[List[str]] = None,
                                references: Optional[List[str]] = None) -> str:
        """
        Submit a processed dataset with full specifications.
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.DataFrame, Dict[str, Any]]
            The processed dataset
        submitter_name : str
            Name of the person submitting the dataset
        submitter_email : str
            Email of the submitter
        dataset_name : str
            Name of the dataset
        dataset_description : str
            Description of the dataset
        dataset_specification : DatasetSpecification
            Full dataset specification
        dataset_version : str
            Version of the dataset
        license : str
            License for the dataset
        citation : Optional[str]
            Citation information
        keywords : Optional[List[str]]
            Keywords describing the dataset
        references : Optional[List[str]]
            References related to the dataset
            
        Returns:
        --------
        str
            Submission ID for tracking
        """
        # Generate submission ID
        submission_id = self._generate_submission_id()
        
        # Create submission metadata
        metadata = SubmissionMetadata(
            submission_id=submission_id,
            submitter_name=submitter_name,
            submitter_email=submitter_email,
            submission_date=datetime.now().isoformat(),
            submission_type="processed",
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            dataset_version=dataset_version,
            license=license,
            citation=citation,
            keywords=keywords or [],
            references=references or []
        )
        
        # Save data to realistic data directory
        realistic_data_dir = self.base_data_dir / "realistic" / submission_id
        realistic_data_dir.mkdir(exist_ok=True)
        
        # Save data in multiple formats
        data_file = realistic_data_dir / "data.npy"
        np.save(data_file, data)
        metadata.file_paths['data_file'] = str(data_file)
        
        # Save dataset specification
        spec_file = realistic_data_dir / "specification.json"
        dataset_specification.to_json(str(spec_file))
        metadata.file_paths['specification_file'] = str(spec_file)
        
        # Save metadata
        self._save_submission_metadata(metadata)
        
        # Run validation
        self._validate_processed_submission(metadata, dataset_specification)
        
        logger.info(f"Processed dataset submitted successfully: {submission_id}")
        return submission_id
    
    def submit_synthetic_dataset(self,
                                generator_name: str,
                                generator_parameters: Dict[str, Any],
                                submitter_name: str,
                                submitter_email: str,
                                dataset_name: str,
                                dataset_description: str,
                                dataset_specification: DatasetSpecification,
                                dataset_version: str = "1.0.0",
                                license: str = "MIT",
                                citation: Optional[str] = None,
                                keywords: Optional[List[str]] = None,
                                references: Optional[List[str]] = None) -> str:
        """
        Submit a synthetic dataset with generation parameters.
        
        Parameters:
        -----------
        generator_name : str
            Name of the synthetic data generator used
        generator_parameters : Dict[str, Any]
            Parameters used to generate the synthetic data
        submitter_name : str
            Name of the person submitting the dataset
        submitter_email : str
            Email of the submitter
        dataset_name : str
            Name of the dataset
        dataset_description : str
            Description of the dataset
        dataset_specification : DatasetSpecification
            Full dataset specification
        dataset_version : str
            Version of the dataset
        license : str
            License for the dataset
        citation : Optional[str]
            Citation information
        keywords : Optional[List[str]]
            Keywords describing the dataset
        references : Optional[List[str]]
            References related to the dataset
            
        Returns:
        --------
        str
            Submission ID for tracking
        """
        # Generate submission ID
        submission_id = self._generate_submission_id()
        
        # Create submission metadata
        metadata = SubmissionMetadata(
            submission_id=submission_id,
            submitter_name=submitter_name,
            submitter_email=submitter_email,
            submission_date=datetime.now().isoformat(),
            submission_type="synthetic",
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            dataset_version=dataset_version,
            license=license,
            citation=citation,
            keywords=keywords or [],
            references=references or []
        )
        
        # Save to synthetic data directory
        synthetic_data_dir = self.base_data_dir / "synthetic" / submission_id
        synthetic_data_dir.mkdir(exist_ok=True)
        
        # Save generator parameters
        params_file = synthetic_data_dir / "generator_parameters.json"
        with open(params_file, 'w') as f:
            json.dump(generator_parameters, f, indent=2, default=str)
        metadata.file_paths['generator_parameters'] = str(params_file)
        
        # Save dataset specification
        spec_file = synthetic_data_dir / "specification.json"
        dataset_specification.to_json(str(spec_file))
        metadata.file_paths['specification_file'] = str(spec_file)
        
        # Save metadata
        self._save_submission_metadata(metadata)
        
        # Run validation
        self._validate_synthetic_submission(metadata, dataset_specification)
        
        logger.info(f"Synthetic dataset submitted successfully: {submission_id}")
        return submission_id
    
    def _extract_compressed_file(self, file_path: Path, extract_dir: Path):
        """Extract compressed files to the specified directory."""
        if file_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            # Handle other compression formats
            import tarfile
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
    
    def _save_submission_metadata(self, metadata: SubmissionMetadata):
        """Save submission metadata to file."""
        metadata_file = self.metadata_dir / f"{metadata.submission_id}.json"
        
        # Convert to dictionary and handle numpy types
        metadata_dict = asdict(metadata)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
    
    def _validate_raw_submission(self, metadata: SubmissionMetadata):
        """Run initial validation on raw dataset submission."""
        try:
            # Basic file checks
            if 'extracted_dir' in metadata.file_paths:
                extracted_dir = Path(metadata.file_paths['extracted_dir'])
                if not extracted_dir.exists():
                    raise ValueError("Extracted directory not found")
                
                # Check for data files
                data_files = list(extracted_dir.rglob("*"))
                if not data_files:
                    raise ValueError("No data files found in extracted directory")
                
                metadata.validation_notes.append(f"Found {len(data_files)} files in extracted directory")
            else:
                data_file = Path(metadata.file_paths['data_file'])
                if not data_file.exists():
                    raise ValueError("Data file not found")
                
                # Check file size
                file_size = data_file.stat().st_size
                if file_size == 0:
                    raise ValueError("Data file is empty")
                
                metadata.validation_notes.append(f"Data file size: {file_size} bytes")
            
            metadata.validation_status = "validated"
            metadata.validation_notes.append("Raw dataset validation passed")
            
        except Exception as e:
            metadata.validation_status = "validation_failed"
            metadata.validation_notes.append(f"Validation error: {str(e)}")
            logger.error(f"Raw dataset validation failed: {e}")
        
        # Update metadata file
        self._save_submission_metadata(metadata)
    
    def _validate_processed_submission(self, metadata: SubmissionMetadata, 
                                     dataset_specification: DatasetSpecification):
        """Validate processed dataset submission."""
        try:
            # Check data file
            data_file = Path(metadata.file_paths['data_file'])
            if not data_file.exists():
                raise ValueError("Data file not found")
            
            # Load and validate data
            data = np.load(data_file)
            
            # Basic data checks
            if len(data) == 0:
                raise ValueError("Dataset is empty")
            
            if np.any(np.isnan(data)):
                raise ValueError("Dataset contains NaN values")
            
            if np.any(np.isinf(data)):
                raise ValueError("Dataset contains infinite values")
            
            # Check against specification
            if len(data) != dataset_specification.properties.n_points:
                metadata.validation_notes.append(
                    f"Warning: Data length ({len(data)}) doesn't match specification ({dataset_specification.properties.n_points})"
                )
            
            metadata.validation_status = "validated"
            metadata.validation_notes.append(f"Processed dataset validation passed: {len(data)} points")
            
        except Exception as e:
            metadata.validation_status = "validation_failed"
            metadata.validation_notes.append(f"Validation error: {str(e)}")
            logger.error(f"Processed dataset validation failed: {e}")
        
        # Update metadata file
        self._save_submission_metadata(metadata)
    
    def _validate_synthetic_submission(self, metadata: SubmissionMetadata,
                                     dataset_specification: DatasetSpecification):
        """Validate synthetic dataset submission with quality gates."""
        try:
            # Check specification file
            spec_file = Path(metadata.file_paths['specification_file'])
            if not spec_file.exists():
                raise ValueError("Specification file not found")
            
            # Check generator parameters
            params_file = Path(metadata.file_paths['generator_parameters'])
            if not params_file.exists():
                raise ValueError("Generator parameters file not found")
            
            # Validate specification
            if not hasattr(dataset_specification, 'validate'):
                raise ValueError("Invalid dataset specification")
            
            dataset_specification.validate()
            
            # QUALITY GATE: Run synthetic data quality evaluation
            quality_result = self._evaluate_synthetic_data_quality(metadata, dataset_specification)
            
            # Check if quality meets minimum threshold
            if quality_result.overall_score < 0.5:  # Configurable threshold
                raise ValueError(
                    f"Synthetic data quality too low: {quality_result.overall_score:.3f} "
                    f"(minimum required: 0.5). Quality level: {quality_result.quality_level}"
                )
            
            # Add quality information to metadata
            metadata.validation_notes.append(f"Quality evaluation passed: {quality_result.overall_score:.3f} ({quality_result.quality_level})")
            metadata.validation_notes.append(f"Best performing metrics: {', '.join([m.metric_name for m in quality_result.metrics if m.score > 0.8][:3])}")
            
            if quality_result.recommendations:
                metadata.validation_notes.append(f"Quality recommendations: {quality_result.recommendations[0]}")
            
            metadata.validation_status = "validated"
            metadata.validation_notes.append("Synthetic dataset validation passed with quality gates")
            
        except Exception as e:
            metadata.validation_status = "validation_failed"
            metadata.validation_notes.append(f"Validation error: {str(e)}")
            logger.error(f"Synthetic dataset validation failed: {e}")
        
        # Update metadata file
        self._save_submission_metadata(metadata)
    
    def _evaluate_synthetic_data_quality(self, metadata: SubmissionMetadata,
                                       dataset_specification: DatasetSpecification):
        """
        Evaluate synthetic data quality using our quality evaluation system.
        
        Parameters:
        -----------
        metadata : SubmissionMetadata
            Submission metadata containing file paths
        dataset_specification : DatasetSpecification
            Dataset specification with properties
            
        Returns:
        --------
        QualityEvaluationResult
            Quality evaluation results
        """
        try:
            # Load the generated synthetic data
            data_file = Path(metadata.file_paths.get('data_file'))
            if not data_file or not data_file.exists():
                # If no data file, we'll need to generate it for evaluation
                logger.warning("No data file found for quality evaluation, generating synthetic data")
                synthetic_data = self._generate_synthetic_data_for_evaluation(dataset_specification)
            else:
                synthetic_data = np.load(data_file)
            
            # Determine domain for domain-specific evaluation
            domain = dataset_specification.properties.domain_category.value.lower()
            
            # Create appropriate evaluator
            if domain in ['hydrology', 'financial', 'biomedical', 'climate']:
                evaluator = create_domain_specific_evaluator(domain)
            else:
                evaluator = SyntheticDataQualityEvaluator()
            
            # Load or create reference data for comparison
            reference_data = self._get_reference_data_for_domain(domain)
            reference_metadata = {"domain": domain, "source": "reference_dataset"}
            
            # Run quality evaluation
            quality_result = evaluator.evaluate_quality(
                synthetic_data=synthetic_data,
                reference_data=reference_data,
                reference_metadata=reference_metadata,
                domain=domain,
                normalize_for_comparison=True
            )
            
            # Save quality evaluation results
            self._save_quality_evaluation_results(metadata.submission_id, quality_result)
            
            return quality_result
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            # Return a default low-quality result to trigger rejection
            from ..validation.synthetic_data_quality import QualityEvaluationResult, QualityMetricResult, QualityMetricType
            
            # Create a minimal quality result indicating failure
            failed_metric = QualityMetricResult(
                metric_name="quality_evaluation_failed",
                metric_type=QualityMetricType.COMPOSITE,
                value=0.0,
                score=0.0,
                weight=1.0,
                description="Quality evaluation failed during validation",
                details={"error": str(e)}
            )
            
            failed_result = QualityEvaluationResult(
                synthetic_data=np.array([]),
                reference_data=np.array([]),
                reference_metadata={},
                metrics=[failed_metric],
                overall_score=0.0,
                quality_level="poor",
                recommendations=[f"Quality evaluation failed: {str(e)}"],
                evaluation_date=datetime.now().isoformat(),
                normalization_info={"method": "none", "applied": False}
            )
            
            return failed_result
    
    def _generate_synthetic_data_for_evaluation(self, dataset_specification: DatasetSpecification):
        """Generate synthetic data for quality evaluation if not available."""
        try:
            from ..data_generation.synthetic_data_generator import SyntheticDataGenerator
            
            generator = SyntheticDataGenerator()
            result = generator.generate_data(dataset_specification, [])
            return result['data']
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic data for evaluation: {e}")
            # Return a simple array for evaluation
            return np.random.normal(0, 1, dataset_specification.properties.n_points)
    
    def _get_reference_data_for_domain(self, domain: str):
        """Get reference data for quality evaluation based on domain."""
        try:
            # Try to load domain-specific reference data
            if domain == "hydrology":
                reference_file = Path("data/realistic/nile_river_flow.npy")
            elif domain == "financial":
                reference_file = Path("data/realistic/dow_jones_monthly.npy")
            elif domain == "biomedical":
                reference_file = Path("data/realistic/eeg_sample.npy")
            elif domain == "climate":
                reference_file = Path("data/realistic/daily_temperature.npy")
            else:
                # Default to a general reference dataset
                reference_file = Path("data/realistic/nile_river_flow.npy")
            
            if reference_file.exists():
                return np.load(reference_file)
            else:
                # Generate a simple reference dataset
                logger.warning(f"Reference data not found for domain {domain}, generating simple reference")
                return np.random.normal(0, 1, 1000)
                
        except Exception as e:
            logger.error(f"Failed to load reference data for domain {domain}: {e}")
            # Return a simple reference dataset
            return np.random.normal(0, 1, 1000)
    
    def _save_quality_evaluation_results(self, submission_id: str, quality_result):
        """Save quality evaluation results for the submission."""
        try:
            quality_dir = self.submissions_dir / "quality_evaluations"
            quality_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            quality_file = quality_dir / f"{submission_id}_quality.json"
            
            # Convert to serializable format
            quality_dict = {
                "submission_id": submission_id,
                "evaluation_date": quality_result.evaluation_date,
                "overall_score": quality_result.overall_score,
                "quality_level": quality_result.quality_level,
                "metrics": [
                    {
                        "metric_name": m.metric_name,
                        "metric_type": m.metric_type.value,
                        "score": m.score,
                        "weight": m.weight,
                        "description": m.description,
                        "details": m.details
                    }
                    for m in quality_result.metrics
                ],
                "recommendations": quality_result.recommendations,
                "normalization_info": quality_result.normalization_info
            }
            
            with open(quality_file, 'w') as f:
                json.dump(quality_dict, f, indent=2, default=str)
                
            logger.info(f"Quality evaluation results saved for submission {submission_id}")
            
        except Exception as e:
            logger.error(f"Failed to save quality evaluation results: {e}")
    
    def get_submission(self, submission_id: str) -> Optional[SubmissionMetadata]:
        """Retrieve submission metadata by ID."""
        metadata_file = self.metadata_dir / f"{submission_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            return SubmissionMetadata(**metadata_dict)
        except Exception as e:
            logger.error(f"Error loading submission {submission_id}: {e}")
            return None
    
    def list_submissions(self, submission_type: Optional[str] = None) -> List[SubmissionMetadata]:
        """List all submissions, optionally filtered by type."""
        submissions = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                metadata = SubmissionMetadata(**metadata_dict)
                
                if submission_type is None or metadata.submission_type == submission_type:
                    submissions.append(metadata)
                    
            except Exception as e:
                logger.error(f"Error loading submission from {metadata_file}: {e}")
        
        # Sort by submission date (newest first)
        submissions.sort(key=lambda x: x.submission_date, reverse=True)
        
        return submissions
    
    def delete_submission(self, submission_id: str) -> bool:
        """Delete a submission and all associated files."""
        try:
            # Get submission metadata
            metadata = self.get_submission(submission_id)
            if not metadata:
                return False
            
            # Delete data files
            for file_path in metadata.file_paths.values():
                if file_path and Path(file_path).exists():
                    if Path(file_path).is_file():
                        Path(file_path).unlink()
                    else:
                        shutil.rmtree(Path(file_path))
            
            # Delete metadata file
            metadata_file = self.metadata_dir / f"{submission_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            logger.info(f"Submission {submission_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting submission {submission_id}: {e}")
            return False
    
    def get_submission_statistics(self) -> Dict[str, Any]:
        """Get statistics about all submissions."""
        submissions = self.list_submissions()
        
        stats = {
            'total_submissions': len(submissions),
            'by_type': {},
            'by_status': {},
            'by_month': {},
            'total_submitters': len(set(s.submitter_email for s in submissions))
        }
        
        for submission in submissions:
            # Count by type
            submission_type = submission.submission_type
            stats['by_type'][submission_type] = stats['by_type'].get(submission_type, 0) + 1
            
            # Count by status
            status = submission.validation_status
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            # Count by month
            month = submission.submission_date[:7]  # YYYY-MM
            stats['by_month'][month] = stats['by_month'].get(month, 0) + 1
        
        return stats
