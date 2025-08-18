"""
Performance Leaderboard for Long-Range Dependence Estimators

This module provides tools for creating and managing performance
leaderboards to compare different LRD estimators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class PerformanceLeaderboard:
    """
    Performance leaderboard for comparing long-range dependence estimators.
    
    This class provides methods for creating, updating, and displaying
    performance rankings of different estimators.
    """
    
    def __init__(self):
        """Initialize the performance leaderboard."""
        self.results = []
        self.metrics = []
        
    def add_result(self, estimator_name: str, dataset_name: str, 
                  metrics: Dict[str, Any]) -> None:
        """
        Add a benchmark result to the leaderboard.
        
        Parameters
        ----------
        estimator_name : str
            Name of the estimator
        dataset_name : str
            Name of the dataset
        metrics : Dict[str, Any]
            Dictionary containing performance metrics
        """
        result = {
            'estimator': estimator_name,
            'dataset': dataset_name,
            'timestamp': pd.Timestamp.now(),
            **metrics
        }
        
        self.results.append(result)
        
        # Update available metrics
        for key in metrics.keys():
            if key not in self.metrics:
                self.metrics.append(key)
                
    def get_leaderboard(self, dataset_name: str = None, 
                       sort_by: str = 'rmse', 
                       ascending: bool = True) -> pd.DataFrame:
        """
        Get the current leaderboard.
        
        Parameters
        ----------
        dataset_name : str
            Filter by specific dataset (if None, include all)
        sort_by : str
            Metric to sort by
        ascending : bool
            Sort order (True for ascending, False for descending)
            
        Returns
        -------
        pd.DataFrame
            Sorted leaderboard
        """
        if not self.results:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Filter by dataset if specified
        if dataset_name is not None:
            df = df[df['dataset'] == dataset_name]
            
        # Sort by specified metric
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
            
        return df
        
    def get_top_performers(self, n: int = 5, dataset_name: str = None,
                          metric: str = 'rmse') -> pd.DataFrame:
        """
        Get top N performers for a specific metric.
        
        Parameters
        ----------
        n : int
            Number of top performers to return
        dataset_name : str
            Filter by specific dataset (if None, include all)
        metric : str
            Metric to rank by
            
        Returns
        -------
        pd.DataFrame
            Top N performers
        """
        leaderboard = self.get_leaderboard(dataset_name, sort_by=metric, ascending=True)
        
        if leaderboard.empty:
            return leaderboard
            
        return leaderboard.head(n)
        
    def compare_estimators(self, estimator_names: List[str], 
                          dataset_name: str = None) -> pd.DataFrame:
        """
        Compare specific estimators side by side.
        
        Parameters
        ----------
        estimator_names : List[str]
            List of estimator names to compare
        dataset_name : str
            Filter by specific dataset (if None, include all)
            
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        if not self.results:
            return pd.DataFrame()
            
        # Filter results
        filtered_results = []
        for result in self.results:
            if result['estimator'] in estimator_names:
                if dataset_name is None or result['dataset'] == dataset_name:
                    filtered_results.append(result)
                    
        if not filtered_results:
            return pd.DataFrame()
            
        # Convert to DataFrame and pivot
        df = pd.DataFrame(filtered_results)
        
        # Select key metrics for comparison
        key_metrics = ['rmse', 'mae', 'wall_time', 'memory_delta_mb', 'robustness_score']
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        if not available_metrics:
            return df
            
        # Create comparison table
        comparison = df.pivot(index='dataset', columns='estimator', values=available_metrics)
        
        return comparison
        
    def get_summary_statistics(self, dataset_name: str = None) -> Dict[str, Any]:
        """
        Get summary statistics across all estimators.
        
        Parameters
        ----------
        dataset_name : str
            Filter by specific dataset (if None, include all)
            
        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        if not self.results:
            return {}
            
        # Filter results
        filtered_results = []
        for result in self.results:
            if dataset_name is None or result['dataset'] == dataset_name:
                filtered_results.append(result)
                
        if not filtered_results:
            return {}
            
        df = pd.DataFrame(filtered_results)
        
        summary = {}
        
        # Calculate statistics for each metric
        for metric in self.metrics:
            if metric in df.columns and df[metric].dtype in ['float64', 'int64']:
                summary[f'{metric}_mean'] = df[metric].mean()
                summary[f'{metric}_std'] = df[metric].std()
                summary[f'{metric}_min'] = df[metric].min()
                summary[f'{metric}_max'] = df[metric].max()
                
        # Add counts
        summary['total_results'] = len(filtered_results)
        summary['unique_estimators'] = df['estimator'].nunique()
        summary['unique_datasets'] = df['dataset'].nunique()
        
        return summary
        
    def export_to_csv(self, filename: str, dataset_name: str = None) -> None:
        """
        Export leaderboard to CSV file.
        
        Parameters
        ----------
        filename : str
            Output filename
        dataset_name : str
            Filter by specific dataset (if None, export all)
        """
        leaderboard = self.get_leaderboard(dataset_name)
        
        if not leaderboard.empty:
            leaderboard.to_csv(filename, index=False)
            logger.info(f"Leaderboard exported to {filename}")
        else:
            logger.warning("No data to export")
            
    def export_to_excel(self, filename: str, dataset_name: str = None) -> None:
        """
        Export leaderboard to Excel file with multiple sheets.
        
        Parameters
        ----------
        filename : str
            Output filename
        dataset_name : str
            Filter by specific dataset (if None, export all)
        """
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main leaderboard
                leaderboard = self.get_leaderboard(dataset_name)
                if not leaderboard.empty:
                    leaderboard.to_excel(writer, sheet_name='Leaderboard', index=False)
                    
                # Summary statistics
                summary = self.get_summary_statistics(dataset_name)
                if summary:
                    summary_df = pd.DataFrame([summary])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                # Top performers
                top_performers = self.get_top_performers(n=10, dataset_name=dataset_name)
                if not top_performers.empty:
                    top_performers.to_excel(writer, sheet_name='Top_Performers', index=False)
                    
            logger.info(f"Leaderboard exported to {filename}")
        except ImportError as e:
            logger.error(f"Excel export failed (missing openpyxl dependency): {e}")
            logger.info("Please install 'openpyxl' for Excel export support")
            raise
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            raise
        
    def plot_performance_comparison(self, metric: str = 'rmse', 
                                  dataset_name: str = None) -> None:
        """
        Create a performance comparison plot.
        
        Parameters
        ----------
        metric : str
            Metric to plot
        dataset_name : str
            Filter by specific dataset (if None, include all)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            leaderboard = self.get_leaderboard(dataset_name)
            
            if leaderboard.empty or metric not in leaderboard.columns:
                logger.warning(f"No data available for plotting {metric}")
                return
                
            # Create plot
            plt.figure(figsize=(10, 6))
            
            if dataset_name is None:
                # Group by estimator and calculate mean
                grouped = leaderboard.groupby('estimator')[metric].mean().sort_values()
                grouped.plot(kind='bar')
                plt.title(f'Average {metric.upper()} by Estimator')
            else:
                # Single dataset comparison
                leaderboard = leaderboard.sort_values(by=metric, ascending=True)
                plt.bar(range(len(leaderboard)), leaderboard[metric])
                plt.xticks(range(len(leaderboard)), leaderboard['estimator'], rotation=45)
                plt.title(f'{metric.upper()} by Estimator - {dataset_name}')
                
            plt.ylabel(metric.upper())
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            logger.error(f"Plotting failed: {e}")
            
    def clear_results(self) -> None:
        """Clear all results from the leaderboard."""
        self.results = []
        self.metrics = []
        logger.info("Leaderboard cleared")
