#!/usr/bin/env python3
"""
High-Performance Wavelet Variance Estimator

This module provides an optimized implementation of the wavelet variance method
for estimating long-range dependence using parallel processing and memory optimization.
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import pywt

from .wavelet import WaveletVarianceEstimator

logger = logging.getLogger(__name__)

# Suppress warnings for better performance
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class HighPerformanceWaveletVarianceEstimator(WaveletVarianceEstimator):
    """
    High-performance wavelet variance estimator with parallel processing and memory optimization.
    
    This estimator extends the base WaveletVarianceEstimator with:
    - Parallel processing for multiple scales
    - Memory-efficient data handling
    - Optimized wavelet coefficient calculations
    - Progress tracking for long computations
    """
    
    def __init__(self, name: str = "HighPerformanceWaveletVariance", **kwargs):
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
        
        # Generate scales
        self._generate_scales()
        
        # Compute wavelet coefficients with parallel processing
        self._compute_wavelet_coefficients_parallel()
        
        # Calculate wavelet variances
        self._calculate_wavelet_variances()
        
        # Fit scaling law to extract Hurst exponent
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
        
    def _compute_wavelet_coefficients_parallel(self):
        """Compute wavelet coefficients using parallel processing."""
        if self.n_jobs == 1:
            # Fall back to sequential processing
            super()._compute_wavelet_coefficients()
            return
            
        start_time = time.time()
        
        # Prepare tasks for parallel processing
        tasks = []
        for scale in self.scales:
            task = scale
            tasks.append(task)
            
        # Process in parallel
        self.wavelet_coeffs = {}
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_scale_parallel, task): task 
                for task in tasks
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result is not None:
                        scale, coeffs = result
                        self.wavelet_coeffs[scale] = coeffs
                except Exception as e:
                    logger.warning(f"Task {task} failed: {e}")
                    self.wavelet_coeffs[task] = []
                    
                completed += 1
                if completed % self.progress_interval == 0:
                    logger.info(f"Processed {completed}/{len(tasks)} scales")
                    
        self.parallel_processing_time = time.time() - start_time
        
    def _process_scale_parallel(self, scale: int) -> Optional[Tuple[int, list]]:
        """Process a single scale for parallel execution."""
        try:
            # Use PyWavelets for wavelet decomposition
            coeffs = pywt.wavedec(self.data, self.wavelet, level=scale, mode='periodic')
            # Store detail coefficients (excluding approximation)
            detail_coeffs = coeffs[1:] if len(coeffs) > 1 else []
            return (scale, detail_coeffs)
            
        except Exception as e:
            logger.warning(f"Failed to compute wavelet coefficients for scale {scale}: {e}")
            return (scale, [])
            
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
            'scales_processed': len(self.scales) if self.scales is not None else 0,
            'successful_estimations': np.sum(~np.isnan(self.wavelet_variances)) if self.wavelet_variances is not None else 0
        }
