"""
Estimator Submission Manager

This module handles the submission, validation, and integration of new estimators
for the Long-Range Dependence Analysis Framework.

Supports:
- New estimator implementations
- Performance benchmarking
- Validation and testing
- Integration into the framework
- Documentation generation
"""

import os
import shutil
import json
import inspect
import importlib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Type
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import tempfile
import subprocess
import sys

logger = logging.getLogger(__name__)


@dataclass
class EstimatorSubmissionMetadata:
    """Metadata for estimator submissions."""
    submission_id: str
    submitter_name: str
    submitter_email: str
    submission_date: str
    estimator_name: str
    estimator_class: str
    estimator_description: str
    estimator_version: str
    license: str
    citation: Optional[str] = None
    keywords: List[str] = None
    references: List[str] = None
    dependencies: List[str] = None
    validation_status: str = "pending"
    validation_notes: List[str] = None
    performance_metrics: Dict[str, Any] = None
    file_paths: Dict[str, str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.references is None:
            self.references = []
        if self.dependencies is None:
            self.dependencies = []
        if self.validation_notes is None:
            self.validation_notes = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.file_paths is None:
            self.file_paths = {}


@dataclass
class EstimatorRequirements:
    """Requirements for estimator submissions."""
    base_class: str = "BaseEstimator"
    required_methods: List[str] = None
    required_attributes: List[str] = None
    optional_methods: List[str] = None
    optional_attributes: List[str] = None
    
    def __post_init__(self):
        if self.required_methods is None:
            self.required_methods = ["estimate", "__init__"]
        if self.required_attributes is None:
            self.required_attributes = ["name", "description"]
        if self.optional_methods is None:
            self.optional_methods = ["validate_data", "get_parameters", "set_parameters"]
        if self.optional_attributes is None:
            self.optional_attributes = ["version", "author", "citation"]


class EstimatorSubmissionManager:
    """
    Manages estimator submissions including validation, testing, and integration.
    
    Features:
    - Estimator code validation
    - Interface compliance checking
    - Performance benchmarking
    - Integration testing
    - Documentation generation
    """
    
    def __init__(self, base_data_dir: str = "data", estimators_dir: str = "src/estimators"):
        """
        Initialize the estimator submission manager.
        
        Parameters:
        -----------
        base_data_dir : str
            Base directory for data storage
        estimators_dir : str
            Directory for estimator implementations
        """
        self.base_data_dir = Path(base_data_dir)
        self.estimators_dir = Path(estimators_dir)
        self.submissions_dir = self.base_data_dir / "submissions"
        self.estimator_submissions_dir = self.submissions_dir / "estimators"
        self.metadata_dir = self.estimator_submissions_dir / "metadata"
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize submission counter
        self.submission_counter = self._load_submission_counter()
        
        # Load estimator requirements
        self.requirements = EstimatorRequirements()
        
        logger.info(f"Estimator Submission Manager initialized at {self.base_data_dir.absolute()}")
    
    def _create_directories(self):
        """Create necessary directory structure."""
        directories = [
            self.estimator_submissions_dir,
            self.metadata_dir,
            self.estimators_dir / "submitted",
            self.estimators_dir / "testing"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def _load_submission_counter(self) -> int:
        """Load submission counter from file."""
        counter_file = self.estimator_submissions_dir / "submission_counter.txt"
        if counter_file.exists():
            try:
                with open(counter_file, 'r') as f:
                    return int(f.read().strip())
            except (ValueError, IOError):
                logger.warning("Could not load submission counter, starting from 0")
        return 0
    
    def _save_submission_counter(self):
        """Save submission counter to file."""
        counter_file = self.estimator_submissions_dir / "submission_counter.txt"
        with open(counter_file, 'w') as f:
            f.write(str(self.submission_counter))
    
    def _generate_submission_id(self) -> str:
        """Generate unique submission ID."""
        self.submission_counter += 1
        self._save_submission_counter()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"est_{self.submission_counter:06d}_{timestamp}"
    
    def submit_estimator(self,
                         estimator_file: Union[str, Path],
                         submitter_name: str,
                         submitter_email: str,
                         estimator_name: str,
                         estimator_description: str,
                         estimator_version: str = "1.0.0",
                         license: str = "MIT",
                         citation: Optional[str] = None,
                         keywords: Optional[List[str]] = None,
                         references: Optional[List[str]] = None,
                         dependencies: Optional[List[str]] = None,
                         test_data: Optional[np.ndarray] = None) -> str:
        """
        Submit a new estimator for validation and integration.
        
        Parameters:
        -----------
        estimator_file : Union[str, Path]
            Path to the estimator implementation file
        submitter_name : str
            Name of the person submitting the estimator
        submitter_email : str
            Email of the submitter
        estimator_name : str
            Name of the estimator
        estimator_description : str
            Description of the estimator
        estimator_version : str
            Version of the estimator
        license : str
            License for the estimator
        citation : Optional[str]
            Citation information
        keywords : Optional[List[str]]
            Keywords describing the estimator
        references : Optional[List[str]]
            References related to the estimator
        dependencies : Optional[List[str]]
            External dependencies required
        test_data : Optional[np.ndarray]
            Test data for validation
            
        Returns:
        --------
        str
            Submission ID for tracking
        """
        estimator_file = Path(estimator_file)
        
        if not estimator_file.exists():
            raise FileNotFoundError(f"Estimator file not found: {estimator_file}")
        
        # Generate submission ID
        submission_id = self._generate_submission_id()
        
        # Create submission metadata
        metadata = EstimatorSubmissionMetadata(
            submission_id=submission_id,
            submitter_name=submitter_name,
            submitter_email=submitter_email,
            submission_date=datetime.now().isoformat(),
            estimator_name=estimator_name,
            estimator_class=estimator_name.replace(" ", "").replace("-", "_"),
            estimator_description=estimator_description,
            estimator_version=estimator_version,
            license=license,
            citation=citation,
            keywords=keywords or [],
            references=references or [],
            dependencies=dependencies or []
        )
        
        # Copy estimator file to testing directory
        testing_dir = self.estimators_dir / "testing" / submission_id
        testing_dir.mkdir(exist_ok=True)
        
        dest_file = testing_dir / estimator_file.name
        shutil.copy2(estimator_file, dest_file)
        metadata.file_paths['estimator_file'] = str(dest_file)
        
        # Save test data if provided
        if test_data is not None:
            test_data_file = testing_dir / "test_data.npy"
            np.save(test_data_file, test_data)
            metadata.file_paths['test_data'] = str(test_data_file)
        
        # Save metadata
        self._save_submission_metadata(metadata)
        
        # Run validation
        self._validate_estimator_submission(metadata)
        
        logger.info(f"Estimator submitted successfully: {submission_id}")
        return submission_id
    
    def _save_submission_metadata(self, metadata: EstimatorSubmissionMetadata):
        """Save submission metadata to file."""
        metadata_file = self.metadata_dir / f"{metadata.submission_id}.json"
        
        # Convert to dictionary and handle numpy types
        metadata_dict = asdict(metadata)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
    
    def _validate_estimator_submission(self, metadata: EstimatorSubmissionMetadata):
        """Validate estimator submission."""
        try:
            # Load and validate estimator code
            estimator_file = Path(metadata.file_paths['estimator_file'])
            
            # Check file syntax
            self._check_python_syntax(estimator_file)
            
            # Check estimator interface
            self._check_estimator_interface(metadata)
            
            # Run basic tests
            self._run_basic_tests(metadata)
            
            metadata.validation_status = "validated"
            metadata.validation_notes.append("Estimator validation passed")
            
        except Exception as e:
            metadata.validation_status = "validation_failed"
            metadata.validation_notes.append(f"Validation error: {str(e)}")
            logger.error(f"Estimator validation failed: {e}")
        
        # Update metadata file
        self._save_submission_metadata(metadata)
    
    def _check_python_syntax(self, file_path: Path):
        """Check Python syntax of the estimator file."""
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            # Try to compile the code
            compile(source_code, str(file_path), 'exec')
            
        except SyntaxError as e:
            raise ValueError(f"Python syntax error: {e}")
        except Exception as e:
            raise ValueError(f"Code compilation error: {e}")
    
    def _check_estimator_interface(self, metadata: EstimatorSubmissionMetadata):
        """Check if estimator implements required interface."""
        try:
            # Import the estimator module
            estimator_file = Path(metadata.file_paths['estimator_file'])
            
            # Create a temporary module name
            module_name = f"estimator_{metadata.submission_id}"
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, estimator_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find estimator class
            estimator_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    name.lower().replace("_", "").replace(" ", "") == metadata.estimator_class.lower().replace("_", "").replace(" ", "")):
                    estimator_class = obj
                    break
            
            if estimator_class is None:
                raise ValueError(f"Estimator class '{metadata.estimator_class}' not found")
            
            # Check required methods
            for method_name in self.requirements.required_methods:
                if not hasattr(estimator_class, method_name):
                    raise ValueError(f"Required method '{method_name}' not implemented")
                
                method = getattr(estimator_class, method_name)
                if not callable(method):
                    raise ValueError(f"'{method_name}' is not callable")
            
            # Check required attributes
            for attr_name in self.requirements.required_attributes:
                if not hasattr(estimator_class, attr_name):
                    raise ValueError(f"Required attribute '{attr_name}' not found")
            
            # Check if it's a proper estimator
            if not hasattr(estimator_class, 'estimate'):
                raise ValueError("Estimator must have an 'estimate' method")
            
            # Check estimate method signature
            estimate_method = getattr(estimator_class, 'estimate')
            sig = inspect.signature(estimate_method)
            
            # Should accept data as first parameter
            params = list(sig.parameters.keys())
            if not params:
                raise ValueError("'estimate' method must accept at least one parameter")
            
            metadata.validation_notes.append(f"Interface validation passed for class '{estimator_class.__name__}'")
            
        except Exception as e:
            raise ValueError(f"Interface validation failed: {e}")
    
    def _run_basic_tests(self, metadata: EstimatorSubmissionMetadata):
        """Run basic tests on the estimator."""
        try:
            estimator_file = Path(metadata.file_paths['estimator_file'])
            
            # Import the module
            module_name = f"estimator_{metadata.submission_id}"
            spec = importlib.util.spec_from_file_location(module_name, estimator_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find estimator class
            estimator_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    name.lower().replace("_", "").replace(" ", "") == metadata.estimator_class.lower().replace("_", "").replace(" ", "")):
                    estimator_class = obj
                    break
            
            if estimator_class is None:
                raise ValueError("Estimator class not found")
            
            # Test instantiation
            try:
                instance = estimator_class()
                metadata.validation_notes.append("Estimator instantiation successful")
            except Exception as e:
                raise ValueError(f"Estimator instantiation failed: {e}")
            
            # Test with sample data if available
            if 'test_data' in metadata.file_paths:
                test_data_file = Path(metadata.file_paths['test_data'])
                if test_data_file.exists():
                    test_data = np.load(test_data_file)
                    
                    try:
                        # Test estimation
                        result = instance.estimate(test_data)
                        metadata.validation_notes.append(f"Estimation test passed with result: {type(result)}")
                        
                        # Store basic performance info
                        if hasattr(instance, 'name'):
                            metadata.performance_metrics['estimator_name'] = instance.name
                        
                    except Exception as e:
                        metadata.validation_notes.append(f"Estimation test failed: {e}")
            
            metadata.validation_notes.append("Basic tests completed")
            
        except Exception as e:
            raise ValueError(f"Basic tests failed: {e}")
    
    def approve_estimator(self, submission_id: str) -> bool:
        """
        Approve an estimator submission and integrate it into the framework.
        
        Parameters:
        -----------
        submission_id : str
            ID of the submission to approve
            
        Returns:
        --------
        bool
            True if approval successful, False otherwise
        """
        try:
            # Get submission metadata
            metadata = self.get_submission(submission_id)
            if not metadata:
                logger.error(f"Submission {submission_id} not found")
                return False
            
            if metadata.validation_status != "validated":
                logger.error(f"Submission {submission_id} not validated")
                return False
            
            # Copy to submitted estimators directory
            submitted_dir = self.estimators_dir / "submitted"
            estimator_file = Path(metadata.file_paths['estimator_file'])
            
            # Create a clean filename
            clean_name = metadata.estimator_name.lower().replace(" ", "_").replace("-", "_")
            dest_file = submitted_dir / f"{clean_name}.py"
            
            # Copy file
            shutil.copy2(estimator_file, dest_file)
            
            # Update metadata
            metadata.file_paths['approved_file'] = str(dest_file)
            metadata.validation_status = "approved"
            metadata.validation_notes.append(f"Estimator approved and integrated at {dest_file}")
            
            # Save updated metadata
            self._save_submission_metadata(metadata)
            
            logger.info(f"Estimator {submission_id} approved and integrated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error approving estimator {submission_id}: {e}")
            return False
    
    def get_submission(self, submission_id: str) -> Optional[EstimatorSubmissionMetadata]:
        """Retrieve submission metadata by ID."""
        metadata_file = self.metadata_dir / f"{submission_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            return EstimatorSubmissionMetadata(**metadata_dict)
        except Exception as e:
            logger.error(f"Error loading submission {submission_id}: {e}")
            return None
    
    def list_submissions(self, status: Optional[str] = None) -> List[EstimatorSubmissionMetadata]:
        """List all submissions, optionally filtered by status."""
        submissions = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                metadata = EstimatorSubmissionMetadata(**metadata_dict)
                
                if status is None or metadata.validation_status == status:
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
            'by_status': {},
            'by_month': {},
            'total_submitters': len(set(s.submitter_email for s in submissions)),
            'approved_estimators': len([s for s in submissions if s.validation_status == 'approved'])
        }
        
        for submission in submissions:
            # Count by status
            status = submission.validation_status
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            # Count by month
            month = submission.submission_date[:7]  # YYYY-MM
            stats['by_month'][month] = stats['by_month'].get(month, 0) + 1
        
        return stats
    
    def generate_documentation(self, submission_id: str) -> str:
        """Generate documentation for an estimator submission."""
        try:
            metadata = self.get_submission(submission_id)
            if not metadata:
                return "Submission not found"
            
            # Load estimator code
            estimator_file = Path(metadata.file_paths['estimator_file'])
            
            with open(estimator_file, 'r') as f:
                source_code = f.read()
            
            # Generate basic documentation
            doc = f"""# {metadata.estimator_name}

## Overview
{metadata.estimator_description}

## Metadata
- **Submitter**: {metadata.submitter_name} ({metadata.submitter_email})
- **Version**: {metadata.estimator_version}
- **License**: {metadata.license}
- **Submission Date**: {metadata.submission_date}
- **Status**: {metadata.validation_status}

## Citation
{metadata.citation or 'Not provided'}

## Keywords
{', '.join(metadata.keywords) if metadata.keywords else 'None'}

## Dependencies
{', '.join(metadata.dependencies) if metadata.dependencies else 'None'}

## Code
```python
{source_code}
```

## Validation Notes
{chr(10).join(f'- {note}' for note in metadata.validation_notes) if metadata.validation_notes else 'None'}

## Performance Metrics
{json.dumps(metadata.performance_metrics, indent=2) if metadata.performance_metrics else 'None'}
"""
            
            # Save documentation
            doc_file = self.metadata_dir / f"{submission_id}_documentation.md"
            with open(doc_file, 'w') as f:
                f.write(doc)
            
            metadata.file_paths['documentation'] = str(doc_file)
            self._save_submission_metadata(metadata)
            
            return str(doc_file)
            
        except Exception as e:
            logger.error(f"Error generating documentation for {submission_id}: {e}")
            return f"Error generating documentation: {e}"
