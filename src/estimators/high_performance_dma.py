#!/usr/bin/env python3
"""
High-Performance DMA (Detrended Moving Average) Estimator

This module provides a high-performance implementation of the DMA estimator
for long-range dependence analysis, optimized for large datasets and
parallel processing.
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

from .temporal import DMAEstimator

logger = logging.getLogger(__name__)

# Suppress warnings for better performance
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class HighPerformanceDMAEstimator(DMAEstimator):
    """
    High-performance DMA estimator with parallel processing and memory optimization.
    
    This estimator extends the base DMAEstimator with:
    - Parallel processing for multiple window sizes
    - Memory-efficient data handling
    - Optimized moving average calculations
    - Progress tracking for long computations
    """
    
    def __init__(self, name: str = "HighPerformanceDMA", **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Performance optimization parameters
        self.n_jobs = kwargs.get('n_jobs', -1)  # -1 for all available cores
        self.chunk_size = kwargs.get('chunk_size', 1000)
        self.memory_limit = kwargs.get('memory_limit', 0.8)  # Use up to 80% of available RAM
        self.progress_interval = kwargs.get('progress_interval', 100)
        
        # Performance tracking
        self.parallel_processing_time = None
        self.memory_usage = None
        self.optimization_level = None
        
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """High-performance estimation with parallel processing."""
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss
        
        # Update parameters from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        if data is not None:
            self.fit(data, **kwargs)
            
        if self.data is None:
            raise ValueError("No data available. Call fit() first.")
            
        # Determine optimization level based on data size
        self._determine_optimization_level()
        
        # Calculate fluctuations with parallel processing
        self._calculate_fluctuations_parallel()
        
        # Fit scaling law
        self._fit_scaling_law()
        
        # Calculate confidence interval
        self._calculate_confidence_interval()
        
        # Record performance metrics
        self.execution_time = time.time() - start_time
        self.memory_usage = psutil.Process().memory_info().rss - initial_memory
        
        return self.get_results()
        
    def _determine_optimization_level(self):
        """Determine the level of optimization based on data size and available resources."""
        data_size = len(self.data)
        available_memory = psutil.virtual_memory().available
        n_cores = psutil.cpu_count()
        
        if data_size < 1000:
            self.optimization_level = "minimal"
            self.n_jobs = 1
        elif data_size < 10000:
            self.optimization_level = "moderate"
            self.n_jobs = min(2, n_cores)
        elif data_size < 100000:
            self.optimization_level = "high"
            self.n_jobs = min(4, n_cores)
        else:
            self.optimization_level = "maximum"
            self.n_jobs = min(8, n_cores)
            
        # Adjust based on available memory
        if available_memory < 1e9:  # Less than 1GB
            self.optimization_level = "memory_constrained"
            self.n_jobs = 1
            self.chunk_size = 500
            
        logger.info(f"Optimization level: {self.optimization_level}, Jobs: {self.n_jobs}")
        
    def _calculate_fluctuations_parallel(self):
        """Calculate fluctuations using parallel processing."""
        if self.n_jobs == 1:
            # Fall back to sequential processing
            super()._calculate_fluctuations()
            return
            
        start_time = time.time()
        
        # Prepare tasks for parallel processing
        tasks = []
        for i, window_size in enumerate(self.window_sizes_used):
            task = (i, window_size)
            tasks.append(task)
            
        # Process in parallel
        self.fluctuations = np.zeros(len(self.window_sizes_used))
        self.fluctuation_std = np.zeros(len(self.window_sizes_used))
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_window_parallel, task): task 
                for task in tasks
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result is not None:
                        i, fluctuation, fluctuation_std = result
                        self.fluctuations[i] = fluctuation
                        self.fluctuation_std[i] = fluctuation_std
                except Exception as e:
                    logger.warning(f"Task {task} failed: {e}")
                    i, window_size = task
                    self.fluctuations[i] = np.nan
                    self.fluctuation_std[i] = np.nan
                    
                completed += 1
                if completed % self.progress_interval == 0:
                    logger.info(f"Processed {completed}/{len(tasks)} windows")
                    
        self.parallel_processing_time = time.time() - start_time
        
    def _process_window_parallel(self, task: Tuple[int, int]) -> Optional[Tuple[int, float, float]]:
        """Process a single window for parallel execution."""
        try:
            i, window_size = task
            
            # Calculate fluctuations for this window size
            fluctuation_list = []
            n_starting_points = min(20, len(self.data) // window_size)
            
            if n_starting_points == 0:
                return (i, np.nan, np.nan)
                
            for start_idx in range(0, len(self.data) - window_size, 
                                 max(1, (len(self.data) - window_size) // n_starting_points)):
                fluctuation = self._calculate_fluctuation_for_window(window_size, start_idx)
                if not np.isnan(fluctuation):
                    fluctuation_list.append(fluctuation)
                    
            if fluctuation_list:
                mean_fluctuation = np.mean(fluctuation_list)
                std_fluctuation = np.std(fluctuation_list)
                return (i, mean_fluctuation, std_fluctuation)
            else:
                return (i, np.nan, np.nan)
                
        except Exception as e:
            logger.warning(f"Error processing window {window_size}: {e}")
            return (i, np.nan, np.nan)
            
    def _calculate_moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized moving average calculation."""
        if window <= 1:
            return np.zeros_like(data)
            
        # Use optimized convolution for moving average
        kernel = np.ones(window) / window
        
        # Handle edge cases efficiently
        if len(data) <= window:
            return np.full_like(data, np.mean(data))
            
        # Use numpy's convolve with optimized mode
        padded_data = np.pad(data, (window//2, window//2), mode='edge')
        moving_avg = np.convolve(padded_data, kernel, mode='valid')
        
        # Ensure correct length
        if len(moving_avg) > len(data):
            moving_avg = moving_avg[:len(data)]
        elif len(moving_avg) < len(data):
            moving_avg = np.pad(moving_avg, (0, len(data) - len(moving_avg)), mode='edge')
            
        return moving_avg
        
    def get_results(self) -> Dict[str, Any]:
        """Get estimation results with performance metrics."""
        results = super().get_results()
        
        # Add performance metrics
        results.update({
            'parallel_processing_time': self.parallel_processing_time,
            'memory_usage_bytes': self.memory_usage,
            'memory_usage_mb': self.memory_usage / (1024 * 1024) if self.memory_usage else None,
            'optimization_level': self.optimization_level,
            'n_jobs_used': self.n_jobs,
            'chunk_size': self.chunk_size
        })
        
        return results
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get detailed performance summary."""
        return {
            'estimator_name': self.name,
            'data_size': len(self.data) if self.data is not None else 0,
            'execution_time': self.execution_time,
            'parallel_processing_time': self.parallel_processing_time,
            'memory_usage_mb': self.memory_usage / (1024 * 1024) if self.memory_usage else None,
            'optimization_level': self.optimization_level,
            'n_jobs_used': self.n_jobs,
            'window_sizes_processed': len(self.window_sizes_used) if self.window_sizes_used is not None else 0,
            'successful_estimations': np.sum(~np.isnan(self.fluctuations)) if self.fluctuations is not None else 0
        }
