"""
Benchmark Submission Manager

This module handles the submission, validation, and storage of benchmark results
for the Long-Range Dependence Analysis Framework.

Supports:
- Benchmark result submissions
- Performance metrics
- Comparison reports
- Leaderboard management
- Result validation
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSubmissionMetadata:
    """Metadata for benchmark submissions."""
    submission_id: str
    submitter_name: str
    submitter_email: str
    submission_date: str
    benchmark_name: str
    benchmark_description: str
    benchmark_version: str
    dataset_name: str
    estimators_tested: List[str]
    performance_metrics: Dict[str, Any]
    validation_status: str = "pending"
    validation_notes: List[str] = None
    file_paths: Dict[str, str] = None
    leaderboard_score: Optional[float] = None
    
    def __post_init__(self):
        if self.validation_notes is None:
            self.validation_notes = []
        if self.file_paths is None:
            self.file_paths = {}


@dataclass
class BenchmarkResult:
    """Individual benchmark result for an estimator."""
    estimator_name: str
    dataset_name: str
    hurst_exponent: float
    execution_time: float
    memory_usage: float
    accuracy_metrics: Dict[str, float]
    robustness_metrics: Dict[str, float]
    timestamp: str


class BenchmarkSubmissionManager:
    """
    Manages benchmark submissions including validation, storage, and leaderboard management.
    
    Features:
    - Benchmark result validation
    - Performance metrics analysis
    - Leaderboard management
    - Result comparison and visualization
    - Quality assessment
    """
    
    def __init__(self, base_data_dir: str = "data"):
        """
        Initialize the benchmark submission manager.
        
        Parameters:
        -----------
        base_data_dir : str
            Base directory for data storage
        """
        self.base_data_dir = Path(base_data_dir)
        self.submissions_dir = self.base_data_dir / "submissions"
        self.benchmark_submissions_dir = self.submissions_dir / "benchmarks"
        self.metadata_dir = self.benchmark_submissions_dir / "metadata"
        self.results_dir = self.benchmark_submissions_dir / "results"
        self.leaderboard_dir = self.benchmark_submissions_dir / "leaderboard"
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize submission counter
        self.submission_counter = self._load_submission_counter()
        
        logger.info(f"Benchmark Submission Manager initialized at {self.base_data_dir.absolute()}")
    
    def _create_directories(self):
        """Create necessary directory structure."""
        directories = [
            self.benchmark_submissions_dir,
            self.metadata_dir,
            self.results_dir,
            self.leaderboard_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def _load_submission_counter(self) -> int:
        """Load submission counter from file."""
        counter_file = self.benchmark_submissions_dir / "submission_counter.txt"
        if counter_file.exists():
            try:
                with open(counter_file, 'r') as f:
                    return int(f.read().strip())
            except (ValueError, IOError):
                logger.warning("Could not load submission counter, starting from 0")
        return 0
    
    def _save_submission_counter(self):
        """Save submission counter to file."""
        counter_file = self.benchmark_submissions_dir / "submission_counter.txt"
        with open(counter_file, 'w') as f:
            f.write(str(self.submission_counter))
    
    def _generate_submission_id(self) -> str:
        """Generate unique submission ID."""
        self.submission_counter += 1
        self._save_submission_counter()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"bench_{self.submission_counter:06d}_{timestamp}"
    
    def submit_benchmark_results(self,
                                benchmark_name: str,
                                benchmark_description: str,
                                dataset_name: str,
                                estimators_tested: List[str],
                                performance_metrics: Dict[str, Any],
                                submitter_name: str,
                                submitter_email: str,
                                benchmark_version: str = "1.0.0",
                                additional_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit benchmark results for validation and leaderboard consideration.
        
        Parameters:
        -----------
        benchmark_name : str
            Name of the benchmark
        benchmark_description : str
            Description of the benchmark
        dataset_name : str
            Name of the dataset used
        estimators_tested : List[str]
            List of estimator names tested
        performance_metrics : Dict[str, Any]
            Performance metrics for each estimator
        submitter_name : str
            Name of the person submitting the benchmark
        submitter_email : str
            Email of the submitter
        benchmark_version : str
            Version of the benchmark
        additional_metadata : Optional[Dict[str, Any]]
            Additional metadata for the benchmark
            
        Returns:
        --------
        str
            Submission ID for tracking
        """
        # Generate submission ID
        submission_id = self._generate_submission_id()
        
        # Create submission metadata
        metadata = BenchmarkSubmissionMetadata(
            submission_id=submission_id,
            submitter_name=submitter_name,
            submitter_email=submitter_email,
            submission_date=datetime.now().isoformat(),
            benchmark_name=benchmark_name,
            benchmark_description=benchmark_description,
            benchmark_version=benchmark_version,
            dataset_name=dataset_name,
            estimators_tested=estimators_tested,
            performance_metrics=performance_metrics
        )
        
        # Save benchmark results
        results_file = self.results_dir / f"{submission_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(performance_metrics, f, indent=2, default=str)
        metadata.file_paths['results_file'] = str(results_file)
        
        # Save metadata
        self._save_submission_metadata(metadata)
        
        # Run validation
        self._validate_benchmark_submission(metadata)
        
        # Calculate leaderboard score
        self._calculate_leaderboard_score(metadata)
        
        # Update leaderboard
        self._update_leaderboard(metadata)
        
        logger.info(f"Benchmark results submitted successfully: {submission_id}")
        return submission_id
    
    def _save_submission_metadata(self, metadata: BenchmarkSubmissionMetadata):
        """Save submission metadata to file."""
        metadata_file = self.metadata_dir / f"{metadata.submission_id}.json"
        
        # Convert to dictionary and handle numpy types
        metadata_dict = asdict(metadata)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
    
    def _validate_benchmark_submission(self, metadata: BenchmarkSubmissionMetadata):
        """Validate benchmark submission."""
        try:
            # Check required fields
            if not metadata.estimators_tested:
                raise ValueError("No estimators specified")
            
            if not metadata.performance_metrics:
                raise ValueError("No performance metrics provided")
            
            # Validate performance metrics structure
            for estimator in metadata.estimators_tested:
                if estimator not in metadata.performance_metrics:
                    raise ValueError(f"Performance metrics missing for estimator: {estimator}")
                
                estimator_metrics = metadata.performance_metrics[estimator]
                
                # Check for required metrics
                required_metrics = ['execution_time', 'memory_usage', 'accuracy']
                for metric in required_metrics:
                    if metric not in estimator_metrics:
                        raise ValueError(f"Required metric '{metric}' missing for estimator '{estimator}'")
                
                # Validate metric values
                if estimator_metrics['execution_time'] < 0:
                    raise ValueError(f"Execution time must be non-negative for estimator '{estimator}'")
                
                if estimator_metrics['memory_usage'] < 0:
                    raise ValueError(f"Memory usage must be non-negative for estimator '{estimator}'")
                
                if not 0 <= estimator_metrics['accuracy'] <= 1:
                    raise ValueError(f"Accuracy must be between 0 and 1 for estimator '{estimator}'")
            
            metadata.validation_status = "validated"
            metadata.validation_notes.append("Benchmark validation passed")
            
        except Exception as e:
            metadata.validation_status = "validation_failed"
            metadata.validation_notes.append(f"Validation error: {str(e)}")
            logger.error(f"Benchmark validation failed: {e}")
        
        # Update metadata file
        self._save_submission_metadata(metadata)
    
    def _calculate_leaderboard_score(self, metadata: BenchmarkSubmissionMetadata):
        """Calculate leaderboard score for the benchmark submission."""
        try:
            if metadata.validation_status != "validated":
                metadata.leaderboard_score = None
                return
            
            # Calculate composite score based on multiple factors
            scores = []
            
            for estimator in metadata.estimators_tested:
                metrics = metadata.performance_metrics[estimator]
                
                # Normalize metrics to 0-1 scale
                accuracy_score = metrics['accuracy']
                
                # Time efficiency score (lower is better, normalized to 0-1)
                time_score = 1.0 / (1.0 + metrics['execution_time'])
                
                # Memory efficiency score (lower is better, normalized to 0-1)
                memory_score = 1.0 / (1.0 + metrics['memory_usage'])
                
                # Composite score (weighted average)
                composite_score = 0.5 * accuracy_score + 0.25 * time_score + 0.25 * memory_score
                scores.append(composite_score)
            
            # Overall benchmark score is the average of estimator scores
            metadata.leaderboard_score = np.mean(scores)
            metadata.validation_notes.append(f"Leaderboard score calculated: {metadata.leaderboard_score:.4f}")
            
        except Exception as e:
            metadata.leaderboard_score = None
            metadata.validation_notes.append(f"Error calculating leaderboard score: {e}")
            logger.error(f"Error calculating leaderboard score: {e}")
    
    def _update_leaderboard(self, metadata: BenchmarkSubmissionMetadata):
        """Update the leaderboard with new benchmark results."""
        try:
            if metadata.leaderboard_score is None:
                return
            
            # Load existing leaderboard
            leaderboard_file = self.leaderboard_dir / "leaderboard.json"
            leaderboard = []
            
            if leaderboard_file.exists():
                with open(leaderboard_file, 'r') as f:
                    leaderboard = json.load(f)
            
            # Add new entry
            leaderboard_entry = {
                'submission_id': metadata.submission_id,
                'benchmark_name': metadata.benchmark_name,
                'dataset_name': metadata.dataset_name,
                'submitter_name': metadata.submitter_name,
                'submitter_email': metadata.submitter_email,
                'submission_date': metadata.submission_date,
                'leaderboard_score': metadata.leaderboard_score,
                'estimators_tested': metadata.estimators_tested,
                'validation_status': metadata.validation_status
            }
            
            leaderboard.append(leaderboard_entry)
            
            # Sort by score (descending)
            leaderboard.sort(key=lambda x: x['leaderboard_score'], reverse=True)
            
            # Keep top 100 entries
            leaderboard = leaderboard[:100]
            
            # Save updated leaderboard
            with open(leaderboard_file, 'w') as f:
                json.dump(leaderboard, f, indent=2, default=str)
            
            metadata.validation_notes.append("Leaderboard updated successfully")
            
        except Exception as e:
            metadata.validation_notes.append(f"Error updating leaderboard: {e}")
            logger.error(f"Error updating leaderboard: {e}")
    
    def get_leaderboard(self, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the current leaderboard."""
        leaderboard_file = self.leaderboard_dir / "leaderboard.json"
        
        if not leaderboard_file.exists():
            return []
        
        try:
            with open(leaderboard_file, 'r') as f:
                leaderboard = json.load(f)
            
            if top_n is not None:
                leaderboard = leaderboard[:top_n]
            
            return leaderboard
            
        except Exception as e:
            logger.error(f"Error loading leaderboard: {e}")
            return []
    
    def get_benchmark_comparison(self, submission_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple benchmark submissions."""
        try:
            comparisons = {}
            
            for submission_id in submission_ids:
                metadata = self.get_submission(submission_id)
                if metadata and metadata.validation_status == "validated":
                    comparisons[submission_id] = {
                        'benchmark_name': metadata.benchmark_name,
                        'dataset_name': metadata.dataset_name,
                        'submitter_name': metadata.submitter_name,
                        'leaderboard_score': metadata.leaderboard_score,
                        'performance_metrics': metadata.performance_metrics
                    }
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error generating benchmark comparison: {e}")
            return {}
    
    def generate_benchmark_report(self, submission_id: str) -> str:
        """Generate a comprehensive report for a benchmark submission."""
        try:
            metadata = self.get_submission(submission_id)
            if not metadata:
                return "Submission not found"
            
            # Generate report
            report = f"""# Benchmark Report: {metadata.benchmark_name}

## Overview
{metadata.benchmark_description}

## Metadata
- **Submitter**: {metadata.submitter_name} ({metadata.submitter_email})
- **Version**: {metadata.benchmark_version}
- **Dataset**: {metadata.dataset_name}
- **Submission Date**: {metadata.submission_date}
- **Status**: {metadata.validation_status}
- **Leaderboard Score**: {metadata.leaderboard_score:.4f if metadata.leaderboard_score else 'N/A'}

## Estimators Tested
{chr(10).join(f'- {estimator}' for estimator in metadata.estimators_tested)}

## Performance Metrics

"""
            
            # Add detailed performance metrics
            for estimator in metadata.estimators_tested:
                metrics = metadata.performance_metrics[estimator]
                report += f"### {estimator}\n"
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        report += f"- **{metric}**: {value:.6f}\n"
                    else:
                        report += f"- **{metric}**: {value}\n"
                report += "\n"
            
            # Add validation notes
            if metadata.validation_notes:
                report += "## Validation Notes\n"
                for note in metadata.validation_notes:
                    report += f"- {note}\n"
                report += "\n"
            
            # Save report
            report_file = self.metadata_dir / f"{submission_id}_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            metadata.file_paths['report'] = str(report_file)
            self._save_submission_metadata(metadata)
            
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error generating benchmark report for {submission_id}: {e}")
            return f"Error generating report: {e}"
    
    def get_submission(self, submission_id: str) -> Optional[BenchmarkSubmissionMetadata]:
        """Retrieve submission metadata by ID."""
        metadata_file = self.metadata_dir / f"{submission_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            return BenchmarkSubmissionMetadata(**metadata_dict)
        except Exception as e:
            logger.error(f"Error loading submission {submission_id}: {e}")
            return None
    
    def list_submissions(self, status: Optional[str] = None) -> List[BenchmarkSubmissionMetadata]:
        """List all submissions, optionally filtered by status."""
        submissions = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                metadata = BenchmarkSubmissionMetadata(**metadata_dict)
                
                if status is None or metadata.validation_status == status:
                    submissions.append(metadata)
                    
            except Exception as e:
                logger.error(f"Error loading submission from {metadata_file}: {e}")
        
        # Sort by submission date (newest first)
        submissions.sort(key=lambda x: x.submission_date, reverse=True)
        
        return submissions
    
    def get_submission_statistics(self) -> Dict[str, Any]:
        """Get statistics about all benchmark submissions."""
        submissions = self.list_submissions()
        
        stats = {
            'total_submissions': len(submissions),
            'by_status': {},
            'by_month': {},
            'total_submitters': len(set(s.submitter_email for s in submissions)),
            'total_benchmarks': len(set(s.benchmark_name for s in submissions)),
            'total_datasets': len(set(s.dataset_name for s in submissions)),
            'total_estimators': len(set(est for s in submissions for est in s.estimators_tested)),
            'average_leaderboard_score': 0.0
        }
        
        valid_scores = []
        
        for submission in submissions:
            # Count by status
            status = submission.validation_status
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            # Count by month
            month = submission.submission_date[:7]  # YYYY-MM
            stats['by_month'][month] = stats['by_month'].get(month, 0) + 1
            
            # Collect valid scores
            if submission.leaderboard_score is not None:
                valid_scores.append(submission.leaderboard_score)
        
        if valid_scores:
            stats['average_leaderboard_score'] = np.mean(valid_scores)
        
        return stats
    
    def export_leaderboard_csv(self, output_path: Optional[str] = None) -> str:
        """Export leaderboard to CSV format."""
        try:
            leaderboard = self.get_leaderboard()
            
            if not leaderboard:
                return "No leaderboard data available"
            
            # Convert to DataFrame
            df = pd.DataFrame(leaderboard)
            
            # Set output path
            if output_path is None:
                output_path = self.leaderboard_dir / f"leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            else:
                output_path = Path(output_path)
            
            # Export to CSV
            df.to_csv(output_path, index=False)
            
            logger.info(f"Leaderboard exported to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting leaderboard: {e}")
            return f"Error exporting leaderboard: {e}"
    
    def generate_leaderboard_visualization(self, output_path: Optional[str] = None) -> str:
        """Generate a visualization of the leaderboard."""
        try:
            leaderboard = self.get_leaderboard(top_n=20)  # Top 20 for visualization
            
            if not leaderboard:
                return "No leaderboard data available"
            
            # Extract data for plotting
            names = [entry['benchmark_name'][:20] + '...' if len(entry['benchmark_name']) > 20 
                    else entry['benchmark_name'] for entry in leaderboard]
            scores = [entry['leaderboard_score'] for entry in leaderboard]
            colors = ['gold' if i == 0 else 'silver' if i == 1 else 'bronze' if i == 2 else 'lightblue' 
                     for i in range(len(leaderboard))]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(scores)), scores, color=colors)
            
            # Customize plot
            plt.yticks(range(len(names)), names)
            plt.xlabel('Leaderboard Score')
            plt.title('Top 20 Benchmark Submissions')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, scores)):
                plt.text(score + 0.01, i, f'{score:.3f}', va='center')
            
            plt.tight_layout()
            
            # Set output path
            if output_path is None:
                output_path = self.leaderboard_dir / f"leaderboard_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            else:
                output_path = Path(output_path)
            
            # Save plot
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Leaderboard visualization saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating leaderboard visualization: {e}")
            return f"Error generating visualization: {e}"
