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
        
        # Performance optimization: caching
        self._scale_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
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
        # Store data
        self.data = np.asarray(data, dtype=np.float64)
        
        # Validate data
        self._validate_data()
        
        # Generate scales
        self._generate_scales()
        
        # Setup memory pools if memory efficient
        if self.memory_efficient:
            self._setup_memory_pools()
        
        logger.info(f"DFA fitting completed")
        return self
    
    def estimate(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Estimate long-range dependence using DFA.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series data
        **kwargs
            Additional estimation parameters
            
        Returns
        -------
        Dict[str, Any]
            Estimation results including Hurst exponent
        """
        start_time = time.time()
        
        # Store data
        self.data = np.asarray(data, dtype=np.float64)
        
        # Validate data
        self._validate_data()
        
        try:
            # Generate scales
            self._generate_scales()
            
            # Setup memory pools if memory efficient
            if self.memory_efficient:
                self._setup_memory_pools()
            
            # Calculate fluctuations
            if self.optimization_backend == "jax":
                self.fluctuations = self._calculate_fluctuations_jax()
            elif self.optimization_backend == "numba":
                self.fluctuations = self._calculate_fluctuations_numba()
            else:
                # Auto mode - try JAX first, then NUMBA, then numpy
                try:
                    self.fluctuations = self._calculate_fluctuations_jax()
                except Exception as e:
                    logger.warning(f"JAX failed, trying NUMBA: {e}")
                    try:
                        self.fluctuations = self._calculate_fluctuations_numba()
                    except Exception as e2:
                        logger.warning(f"NUMBA failed, using numpy fallback: {e2}")
                        self.fluctuations = self._calculate_fluctuations_numpy()
            
            # Fit power law
            if self.optimization_backend == "jax":
                slope, intercept, r_value = self._fit_power_law_jax()
            elif self.optimization_backend == "numba":
                slope, intercept, r_value = self._fit_power_law_numba()
            else:
                # Auto mode - try JAX first, then NUMBA, then numpy
                try:
                    slope, intercept, r_value = self._fit_power_law_jax()
                except Exception as e:
                    logger.warning(f"JAX power law fitting failed, trying NUMBA: {e}")
                    try:
                        slope, intercept, r_value = self._fit_power_law_numba()
                    except Exception as e2:
                        logger.warning(f"NUMBA power law fitting failed, using numpy fallback: {e2}")
                        slope, intercept, r_value = self._fit_power_law_numpy()
            
            # Calculate Hurst exponent
            hurst_exponent = slope / 2.0
            
            # Calculate standard error
            std_error = self._calculate_standard_error()
            
        except Exception as e:
            # Complete fallback to numpy implementation
            logger.error(f"All optimization methods failed, using complete numpy fallback: {e}")
            return self._estimate_numpy_fallback(data, **kwargs)
        
        estimation_time = time.time() - start_time
        self.performance_metrics['estimation_time'] = estimation_time
        
        # Store results
        self.hurst_exponent = hurst_exponent
        self.r_squared = r_value ** 2
        
        # Prepare results
        results = {
            'hurst_exponent': hurst_exponent,
            'r_squared': r_value ** 2,
            'slope': slope,
            'intercept': intercept,
            'std_error': std_error,
            'scales': self.scales,
            'fluctuations': self.fluctuations,
            'method': 'HighPerformanceDFA',
            'optimization_backend': self.optimization_backend,
            'performance_metrics': self.performance_metrics
        }
        
        logger.info(f"DFA estimation completed in {estimation_time:.3f} seconds")
        return results
    
    def _estimate_numpy_fallback(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Complete numpy fallback implementation."""
        start_time = time.time()
        
        # Store data
        self.data = np.asarray(data, dtype=np.float64)
        
        # Validate data
        self._validate_data()
        
        # Generate scales using numpy
        self._generate_scales()
        
        # Calculate fluctuations using numpy
        self.fluctuations = self._calculate_fluctuations_numpy()
        
        # Fit power law using numpy
        slope, intercept, r_value = self._fit_power_law_numpy()
        
        # Calculate Hurst exponent
        hurst_exponent = slope / 2.0
        
        # Calculate standard error
        std_error = self._calculate_standard_error()
        
        estimation_time = time.time() - start_time
        self.performance_metrics['estimation_time'] = estimation_time
        self.performance_metrics['fallback_used'] = True
        
        # Store results
        self.hurst_exponent = hurst_exponent
        self.r_squared = r_value ** 2
        
        # Prepare results
        results = {
            'hurst_exponent': hurst_exponent,
            'r_squared': r_value ** 2,
            'slope': slope,
            'intercept': intercept,
            'std_error': std_error,
            'scales': self.scales,
            'fluctuations': self.fluctuations,
            'method': 'HighPerformanceDFA',
            'optimization_backend': 'numpy_fallback',
            'performance_metrics': self.performance_metrics
        }
        
        logger.info(f"DFA estimation completed using numpy fallback in {estimation_time:.3f} seconds")
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
        """Generate scales for analysis with caching optimization."""
        if self.max_scale is None:
            self.max_scale = len(self.data) // 4
        
        # Create cache key
        cache_key = (self.min_scale, self.max_scale, self.num_scales)
        
        # Check cache first
        if cache_key in self._scale_cache:
            self.scales = self._scale_cache[cache_key]
            self._cache_hits += 1
            logger.debug(f"Scale cache hit! Using cached scales for {cache_key}")
            return
        
        # Cache miss - generate new scales
        self._cache_misses += 1
        
        try:
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
        except Exception as e:
            # JAX/NUMBA failed - fall back to numpy
            logger.warning(f"JAX/NUMBA scale generation failed, falling back to numpy: {e}")
            self.scales = self._generate_scales_numpy()
        
        # Ensure unique scales and convert to integers
        self.scales = np.unique(self.scales.astype(int))
        
        # Cache the result
        self._scale_cache[cache_key] = self.scales.copy()
        
        # Limit cache size to prevent memory issues
        if len(self._scale_cache) > 100:
            # Remove oldest entries
            oldest_key = next(iter(self._scale_cache))
            del self._scale_cache[oldest_key]
            logger.debug("Scale cache size limit reached, removed oldest entry")
    
    def _generate_scales_numpy(self) -> np.ndarray:
        """Generate scales using numpy."""
        return np.logspace(
            np.log10(self.min_scale),
            np.log10(self.max_scale),
            self.num_scales
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_size': len(self._scale_cache),
            'cache_efficiency': f"{hit_rate:.1%}"
        }
    
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
        try:
            # Convert data to JAX array
            data_jax = jnp.array(self.data)
            scales_jax = jnp.array(self.scales)
            
            # Vectorized processing using JAX
            def process_scale(scale):
                return self._process_scale_jax(data_jax, scale)
            
            # Use JAX vectorization
            fluctuations = jax.vmap(process_scale)(scales_jax)
            
            return np.array(fluctuations)
        except Exception as e:
            # JAX compilation failed - fall back to numpy
            logger.warning(f"JAX fluctuation calculation failed, falling back to numpy: {e}")
            return self._calculate_fluctuations_numpy()
    
    def _calculate_fluctuations_numpy(self) -> np.ndarray:
        """Calculate fluctuations using numpy optimization."""
        fluctuations = np.zeros(len(self.scales), dtype=np.float64)
        
        for i, scale in enumerate(self.scales):
            segment_fluctuations = self._process_scale_numpy(scale)
            # Fix: Check array length instead of truthiness
            if len(segment_fluctuations) > 0:
                fluctuations[i] = np.mean(segment_fluctuations)
            else:
                fluctuations[i] = 0.0
        
        return fluctuations
    
    def _calculate_fluctuations_numpy_vectorized(self) -> np.ndarray:
        """Calculate fluctuations using numpy optimization with vectorized operations."""
        fluctuations = np.zeros(len(self.scales), dtype=np.float64)
        
        # Vectorized processing for better performance
        for i, scale in enumerate(self.scales):
            segment_fluctuations = self._process_scale_numpy_vectorized(scale)
            # Fix: Check array length instead of truthiness
            if len(segment_fluctuations) > 0:
                fluctuations[i] = np.mean(segment_fluctuations)
            else:
                fluctuations[i] = 0.0
        
        return fluctuations
    
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
    
    def _process_scale_numpy(self, scale: int) -> List[float]:
        """Process a single scale using numpy."""
        segment_fluctuations = []
        
        # Divide data into segments
        num_segments = len(self.data) // scale
        if num_segments == 0:
            return segment_fluctuations
        
        for i in range(num_segments):
            start_idx = i * scale
            end_idx = start_idx + scale
            segment = self.data[start_idx:end_idx]
            
            # Use numpy-based detrending
            detrended = self._detrend_numpy(segment, self.polynomial_order)
            
            # Use numpy-based RMS calculation
            rms = np.sqrt(np.mean(detrended ** 2))
            segment_fluctuations.append(rms)
        
        return segment_fluctuations
    
    def _process_scale_numpy_vectorized(self, scale: int) -> np.ndarray:
        """Process a single scale using numpy with vectorized operations."""
        # Divide data into segments
        num_segments = len(self.data) // scale
        if num_segments == 0:
            return np.array([])
        
        # Pre-allocate array for better performance
        segment_fluctuations = np.zeros(num_segments, dtype=np.float64)
        
        # Vectorized segment extraction
        segment_indices = np.arange(num_segments)
        start_indices = segment_indices * scale
        end_indices = start_indices + scale
        
        # Process all segments at once where possible
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            segment = self.data[start:end]
            
            # Use vectorized detrending
            detrended = self._detrend_numpy_vectorized(segment, self.polynomial_order)
            
            # Use vectorized RMS calculation
            segment_fluctuations[i] = np.sqrt(np.mean(detrended ** 2))
        
        return segment_fluctuations
    
    def _detrend_numpy(self, data: np.ndarray, order: int) -> np.ndarray:
        """Detrend data using numpy polynomial fitting."""
        if order == 0:
            return data - np.mean(data)
        
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, order)
        trend = np.polyval(coeffs, x)
        return data - trend
    
    def _detrend_numpy_vectorized(self, data: np.ndarray, order: int) -> np.ndarray:
        """Detrend data using numpy with vectorized operations."""
        if order == 0:
            return data - np.mean(data)
        
        # Vectorized polynomial fitting
        x = np.arange(len(data), dtype=np.float64)
        coeffs = np.polyfit(x, data, order)
        trend = np.polyval(coeffs, x)
        return data - trend
    
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
        try:
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
        except Exception as e:
            # JAX compilation failed - fall back to numpy
            logger.warning(f"JAX power law fitting failed, falling back to numpy: {e}")
            return self._fit_power_law_numpy()
    
    def _fit_power_law_numpy(self) -> Tuple[float, float, float]:
        """Fit power law using numpy optimization with vectorized operations."""
        if len(self.scales) != len(self.fluctuations):
            # Filter out scales where fluctuations couldn't be calculated
            valid_indices = np.arange(len(self.scales))[:len(self.fluctuations)]
            scales = self.scales[valid_indices]
        else:
            scales = self.scales
        
        # Vectorized logarithmic transformation
        log_scales = np.log(scales.astype(np.float64))
        log_fluctuations = np.log(self.fluctuations.astype(np.float64))
        
        # Vectorized linear regression
        n = len(log_scales)
        if n < 2:
            return 0.0, 0.0, 0.0
        
        # Vectorized mean calculation
        x_mean = np.mean(log_scales)
        y_mean = np.mean(log_fluctuations)
        
        # Vectorized difference calculation
        dx = log_scales - x_mean
        dy = log_fluctuations - y_mean
        
        # Vectorized sum calculations
        sum_xy = np.sum(dx * dy)
        sum_xx = np.sum(dx * dx)
        
        # Calculate slope and intercept
        if sum_xx == 0.0:
            return 0.0, y_mean, 0.0
        
        slope = sum_xy / sum_xx
        intercept = y_mean - slope * x_mean
        
        # Vectorized R-squared calculation
        sum_yy = np.sum(dy * dy)
        r_squared = (sum_xy * sum_xy) / (sum_xx * sum_yy) if sum_yy > 0.0 else 0.0
        r_value = np.sqrt(r_squared) if r_squared >= 0.0 else 0.0
        
        return slope, intercept, r_value
    
    def _calculate_standard_error(self) -> float:
        """Calculate standard error of the Hurst exponent estimate."""
        if len(self.scales) < 3:
            return 0.0
        
        # Calculate residuals from power law fit
        log_scales = np.log(self.scales)
        log_fluctuations = np.log(self.fluctuations)
        
        # Simple linear regression for error calculation
        n = len(log_scales)
        x_mean = np.mean(log_scales)
        y_mean = np.mean(log_fluctuations)
        
        dx = log_scales - x_mean
        dy = log_fluctuations - y_mean
        sum_xy = np.sum(dx * dy)
        sum_xx = np.sum(dx * dx)
        
        if sum_xx == 0.0:
            return 0.0
        
        slope = sum_xy / sum_xx
        intercept = y_mean - slope * x_mean
        
        # Calculate residuals
        predicted = slope * log_scales + intercept
        residuals = log_fluctuations - predicted
        
        # Calculate standard error
        if n > 2:
            mse = np.sum(residuals ** 2) / (n - 2)
            std_error = np.sqrt(mse / sum_xx)
        else:
            std_error = 0.0
        
        return std_error
    
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
        cache_stats = self.get_cache_stats()
        
        summary = {
            'estimator_name': self.name,
            'optimization_backend': self.optimization_backend,
            'use_gpu': self.use_gpu,
            'memory_efficient': self.memory_efficient,
            'performance_metrics': self.performance_metrics,
            'memory_summary': memory_summary,
            'parallel_summary': parallel_summary,
            'cache_performance': cache_stats,
            'data_size': len(self.data) if self.data is not None else 0,
            'scales_count': len(self.scales) if self.scales is not None else 0,
            'optimization_features': {
                'vectorized_operations': True,
                'caching_enabled': True,
                'memory_pools': self.memory_efficient,
                'parallel_processing': True
            }
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
