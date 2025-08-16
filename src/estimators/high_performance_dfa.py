"""
High-Performance DFA Estimator using NUMBA and JAX

This module provides an optimized implementation of the Detrended
Fluctuation Analysis (DFA) estimator using NUMBA and JAX for
GPU acceleration and parallel computing.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from numba import jit, prange
import time

from .base import BaseEstimator
from ..utils.numba_utils import NumbaOptimizer
from ..utils.jax_utils import JAXOptimizer
from ..utils.memory_utils import MemoryManager
from ..utils.parallel_utils import ParallelProcessor

logger = logging.getLogger(__name__)


class HighPerformanceDFAEstimator(BaseEstimator):
    """
    High-performance DFA estimator using NUMBA and JAX.
    
    This estimator provides multiple optimization strategies:
    - NUMBA for CPU optimization and parallelization
    - JAX for GPU acceleration and automatic differentiation
    - Memory-efficient processing for large datasets
    - Parallel processing across multiple cores/GPUs
    """
    
    def __init__(self, name: str = "HighPerformanceDFA", 
                 optimization_backend: str = "auto",
                 use_gpu: bool = True,
                 memory_efficient: bool = True,
                 **kwargs):
        """
        Initialize high-performance DFA estimator.
        
        Parameters
        ----------
        name : str
            Name identifier for the estimator
        optimization_backend : str
            Optimization backend ('numba', 'jax', or 'auto')
        use_gpu : bool
            Whether to use GPU acceleration
        memory_efficient : bool
            Whether to use memory-efficient processing
        **kwargs
            Additional parameters including:
            - min_scale: Minimum scale for analysis (default: 4)
            - max_scale: Maximum scale for analysis (default: len(data)//4)
            - num_scales: Number of scales to analyze (default: 20)
            - polynomial_order: Order of polynomial for detrending (default: 1)
            - batch_size: Batch size for processing (default: 1000)
        """
        super().__init__(name=name, **kwargs)
        
        # DFA parameters
        self.min_scale = kwargs.get('min_scale', 4)
        self.max_scale = kwargs.get('max_scale', None)
        self.num_scales = kwargs.get('num_scales', 20)
        self.polynomial_order = kwargs.get('polynomial_order', 1)
        self.batch_size = kwargs.get('batch_size', 1000)
        
        # Optimization settings
        self.optimization_backend = optimization_backend
        self.use_gpu = use_gpu
        self.memory_efficient = memory_efficient
        
        # Initialize optimization frameworks
        self._initialize_optimization_frameworks()
        
        # Data storage
        self.data = None
        self.scales = None
        self.fluctuations = None
        
        # Performance tracking
        self.performance_metrics = {}
    
    def _initialize_optimization_frameworks(self):
        """Initialize optimization frameworks."""
        # NUMBA optimizer
        self.numba_optimizer = NumbaOptimizer(use_gpu=self.use_gpu)
        
        # JAX optimizer
        self.jax_optimizer = JAXOptimizer(device="auto" if self.use_gpu else "cpu")
        
        # Memory manager
        self.memory_manager = MemoryManager(enable_monitoring=True)
        
        # Parallel processor
        self.parallel_processor = ParallelProcessor(n_jobs=-1, backend="auto")
        
        # Select optimal backend
        if self.optimization_backend == "auto":
            self.optimization_backend = self._select_optimal_backend()
        
        logger.info(f"Initialized {self.optimization_backend} optimization backend")
    
    def _select_optimal_backend(self) -> str:
        """Select optimal optimization backend."""
        if self.use_gpu and len(self.parallel_processor.available_gpus) > 0:
            return "jax"
        else:
            return "numba"
    
    def fit(self, data: np.ndarray, **kwargs) -> 'HighPerformanceDFAEstimator':
        """
        Fit the high-performance DFA estimator to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        self : HighPerformanceDFAEstimator
            Fitted estimator instance
        """
        start_time = time.time()
        
        with self.memory_manager.memory_context("DFA fitting"):
            self.data = np.asarray(data, dtype=np.float64)
            self._validate_data()
            
            # Optimize memory layout
            if self.memory_efficient:
                self.data = self.memory_manager.optimize_memory_layout([self.data])[0]
            
            # Generate scales
            self._generate_scales()
            
            # Pre-allocate memory pools if needed
            if self.memory_efficient:
                self._setup_memory_pools()
        
        self.performance_metrics['fit_time'] = time.time() - start_time
        logger.info(f"DFA fitting completed in {self.performance_metrics['fit_time']:.3f} seconds")
        
        return self
    
    def estimate(self, data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Estimate long-range dependence using high-performance DFA.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Input time series data (if not provided, uses fitted data)
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing DFA estimation results
        """
        if data is not None:
            self.fit(data, **kwargs)
        
        if self.data is None:
            raise ValueError("No data provided. Call fit() first.")
        
        start_time = time.time()
        
        with self.memory_manager.memory_context("DFA estimation"):
            # Calculate fluctuations using selected backend
            if self.optimization_backend == "jax":
                self.fluctuations = self._calculate_fluctuations_jax()
            else:
                self.fluctuations = self._calculate_fluctuations_numba()
            
            # Fit power law to get Hurst exponent
            if self.optimization_backend == "jax":
                hurst_exponent, r_squared, std_error = self._fit_power_law_jax()
            else:
                hurst_exponent, r_squared, std_error = self._fit_power_law_numba()
            
            # Calculate alpha (long-range dependence parameter)
            alpha = 2 * hurst_exponent - 1
        
        estimation_time = time.time() - start_time
        self.performance_metrics['estimation_time'] = estimation_time
        
        results = {
            'hurst_exponent': hurst_exponent,
            'alpha': alpha,
            'r_squared': r_squared,
            'std_error': std_error,
            'scales': self.scales,
            'fluctuations': self.fluctuations,
            'method': 'HighPerformanceDFA',
            'optimization_backend': self.optimization_backend,
            'performance_metrics': self.performance_metrics
        }
        
        logger.info(f"DFA estimation completed in {estimation_time:.3f} seconds")
        return results
    
    def _validate_data(self):
        """Validate input data."""
        if len(self.data) < 100:
            logger.warning("Data length is small for reliable DFA estimation")
        
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("Data contains NaN or infinite values")
        
        # Check memory requirements
        estimated_memory = len(self.data) * 8 / 1024 / 1024  # MB
        if estimated_memory > 1000:  # 1GB threshold
            logger.info(f"Large dataset detected ({estimated_memory:.1f} MB). "
                       "Enabling memory-efficient processing.")
            self.memory_efficient = True
    
    def _generate_scales(self):
        """Generate scales for analysis."""
        if self.max_scale is None:
            self.max_scale = len(self.data) // 4
        
        if self.optimization_backend == "jax":
            # Use JAX for scale generation
            self.scales = self.jax_optimizer.fast_logspace(
                np.log10(self.min_scale),
                np.log10(self.max_scale),
                self.num_scales
            )
        else:
            # Use NUMBA for scale generation
            self.scales = self.numba_optimizer.fast_logspace(
                np.log10(self.min_scale),
                np.log10(self.max_scale),
                self.num_scales
            )
        
        # Ensure unique scales and convert to integers
        self.scales = np.unique(self.scales.astype(int))
    
    def _setup_memory_pools(self):
        """Setup memory pools for efficient processing."""
        max_scale = np.max(self.scales)
        pool_size = max_scale * 2  # Buffer for intermediate calculations
        
        self.memory_manager.create_memory_pool(
            "detrend_buffer", pool_size, np.float64
        )
        self.memory_manager.create_memory_pool(
            "fluctuation_buffer", len(self.scales), np.float64
        )
    
    def _calculate_fluctuations_numba(self) -> np.ndarray:
        """Calculate fluctuations using NUMBA optimization."""
        fluctuations = np.zeros(len(self.scales), dtype=np.float64)
        
        # Use NUMBA-optimized functions
        for i, scale in enumerate(self.scales):
            # Get memory from pool if available
            buffer = self.memory_manager.get_from_pool("detrend_buffer", scale)
            
            if buffer is not None:
                # Use pooled memory
                segment_fluctuations = self._process_scale_numba_pooled(scale, buffer)
                self.memory_manager.return_to_pool("detrend_buffer")
            else:
                # Allocate new memory
                segment_fluctuations = self._process_scale_numba(scale)
            
            fluctuations[i] = np.mean(segment_fluctuations) if segment_fluctuations else 0.0
        
        return fluctuations
    
    def _calculate_fluctuations_jax(self) -> np.ndarray:
        """Calculate fluctuations using JAX optimization."""
        # Convert data to JAX array
        data_jax = jnp.array(self.data)
        scales_jax = jnp.array(self.scales)
        
        # Vectorized processing using JAX
        def process_scale(scale):
            return self._process_scale_jax(data_jax, scale)
        
        # Use JAX vectorization
        fluctuations = jax.vmap(process_scale)(scales_jax)
        
        return np.array(fluctuations)
    
    def _process_scale_numba(self, scale: int) -> List[float]:
        """Process a single scale using NUMBA."""
        segment_fluctuations = []
        
        # Divide data into segments
        num_segments = len(self.data) // scale
        if num_segments == 0:
            return segment_fluctuations
        
        for i in range(num_segments):
            start_idx = i * scale
            end_idx = start_idx + scale
            segment = self.data[start_idx:end_idx]
            
            # Use NUMBA-optimized detrending
            detrended = self.numba_optimizer.fast_detrend(segment, self.polynomial_order)
            
            # Use NUMBA-optimized RMS calculation
            rms = self.numba_optimizer.fast_rms(detrended)
            segment_fluctuations.append(rms)
        
        return segment_fluctuations
    
    def _process_scale_numba_pooled(self, scale: int, buffer: np.ndarray) -> List[float]:
        """Process a single scale using NUMBA with pooled memory."""
        segment_fluctuations = []
        
        # Divide data into segments
        num_segments = len(self.data) // scale
        if num_segments == 0:
            return segment_fluctuations
        
        for i in range(num_segments):
            start_idx = i * scale
            end_idx = start_idx + scale
            segment = self.data[start_idx:end_idx]
            
            # Copy to buffer
            buffer[:scale] = segment
            
            # Use NUMBA-optimized detrending with buffer
            detrended = self.numba_optimizer.fast_detrend(buffer[:scale], self.polynomial_order)
            
            # Use NUMBA-optimized RMS calculation
            rms = self.numba_optimizer.fast_rms(detrended)
            segment_fluctuations.append(rms)
        
        return segment_fluctuations
    
    def _process_scale_jax(self, data: jnp.ndarray, scale: int) -> float:
        """Process a single scale using JAX."""
        # Divide data into segments
        num_segments = len(data) // scale
        if num_segments == 0:
            return 0.0
        
        # Process all segments at once using JAX
        segments = []
        for i in range(num_segments):
            start_idx = i * scale
            end_idx = start_idx + scale
            segment = data[start_idx:end_idx]
            segments.append(segment)
        
        if not segments:
            return 0.0
        
        # Stack segments for vectorized processing
        segments_array = jnp.stack(segments)
        
        # Vectorized detrending
        detrended = self.jax_optimizer.fast_detrend(segments_array, self.polynomial_order)
        
        # Vectorized RMS calculation
        rms_values = self.jax_optimizer.fast_rms(detrended, axis=1)
        
        return float(jnp.mean(rms_values))
    
    def _fit_power_law_numba(self) -> Tuple[float, float, float]:
        """Fit power law using NUMBA optimization."""
        if len(self.scales) != len(self.fluctuations):
            # Filter out scales where fluctuations couldn't be calculated
            valid_indices = np.arange(len(self.scales))[:len(self.fluctuations)]
            scales = self.scales[valid_indices]
        else:
            scales = self.scales
        
        # Use NUMBA-optimized linear regression
        return self.numba_optimizer.fast_linregress(
            np.log(scales), np.log(self.fluctuations)
        )
    
    def _fit_power_law_jax(self) -> Tuple[float, float, float]:
        """Fit power law using JAX optimization."""
        if len(self.scales) != len(self.fluctuations):
            # Filter out scales where fluctuations couldn't be calculated
            valid_indices = np.arange(len(self.scales))[:len(self.fluctuations)]
            scales = self.scales[valid_indices]
        else:
            scales = self.scales
        
        # Use JAX-optimized linear regression
        return self.jax_optimizer.fast_linregress(
            jnp.log(scales), jnp.log(self.fluctuations)
        )
    
    def batch_estimate(self, data_batch: List[np.ndarray], 
                      **kwargs) -> List[Dict[str, Any]]:
        """
        Estimate long-range dependence for multiple datasets in batch.
        
        Parameters
        ----------
        data_batch : List[np.ndarray]
            List of datasets to process
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        List[Dict[str, Any]]
            List of estimation results
        """
        start_time = time.time()
        
        with self.memory_manager.memory_context("Batch DFA estimation"):
            if self.optimization_backend == "jax":
                # Use JAX vectorization for batch processing
                results = self.jax_optimizer.parallel_estimation(
                    data_batch, 
                    lambda data: self.estimate(data, **kwargs),
                    batch_size=self.batch_size
                )
            else:
                # Use parallel processing for batch estimation
                results = self.parallel_processor.parallel_map(
                    lambda data: self.estimate(data, **kwargs),
                    data_batch
                )
        
        batch_time = time.time() - start_time
        self.performance_metrics['batch_estimation_time'] = batch_time
        self.performance_metrics['batch_size'] = len(data_batch)
        
        logger.info(f"Batch estimation completed in {batch_time:.3f} seconds "
                   f"for {len(data_batch)} datasets")
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns
        -------
        Dict[str, Any]
            Performance summary
        """
        memory_summary = self.memory_manager.get_memory_summary()
        parallel_summary = self.parallel_processor.get_performance_summary()
        
        summary = {
            'estimator_name': self.name,
            'optimization_backend': self.optimization_backend,
            'use_gpu': self.use_gpu,
            'memory_efficient': self.memory_efficient,
            'performance_metrics': self.performance_metrics,
            'memory_summary': memory_summary,
            'parallel_summary': parallel_summary,
            'data_size': len(self.data) if self.data is not None else 0,
            'scales_count': len(self.scales) if self.scales is not None else 0
        }
        
        return summary
    
    def cleanup(self):
        """Clean up resources and memory."""
        self.memory_manager.cleanup_memory(aggressive=True)
        self.memory_manager.clear_memory_pools()
        logger.info("High-performance DFA estimator cleanup completed")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()
