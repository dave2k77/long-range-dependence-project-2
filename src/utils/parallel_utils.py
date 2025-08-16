"""
Parallel Processing Utilities for High-Performance Computing

This module provides tools for efficient parallel processing using
multiple cores, GPUs, and distributed computing.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import joblib
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import logging
import os
import psutil
from functools import partial
import time

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """
    Parallel processing utilities for efficient computation.
    
    This class provides tools for parallel processing using multiple
    cores, GPUs, and distributed computing strategies.
    """
    
    def __init__(self, n_jobs: int = -1, backend: str = "auto"):
        """
        Initialize parallel processor.
        
        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs (-1 for all available cores)
        backend : str
            Backend to use ('multiprocessing', 'threading', 'joblib', 'jax', or 'auto')
        """
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.backend = backend
        self.available_cores = mp.cpu_count()
        self.available_gpus = self._detect_gpus()
        
        # Auto-select backend
        if backend == "auto":
            self.backend = self._select_optimal_backend()
        
        logger.info(f"Parallel processor initialized with {self.n_jobs} jobs "
                   f"using {self.backend} backend")
        logger.info(f"Available: {self.available_cores} CPU cores, {len(self.available_gpus)} GPUs")
    
    def _detect_gpus(self) -> List[str]:
        """Detect available GPUs."""
        gpus = []
        
        # Check for CUDA GPUs
        try:
            import cupy as cp
            gpu_count = cp.cuda.runtime.getDeviceCount()
            for i in range(gpu_count):
                gpus.append(f"cuda:{i}")
        except ImportError:
            pass
        
        # Check for JAX GPUs
        try:
            jax_devices = jax.devices()
            gpu_devices = [d for d in jax_devices if "gpu" in str(d).lower()]
            gpus.extend([f"jax:{d}" for d in gpu_devices])
        except:
            pass
        
        return gpus
    
    def _select_optimal_backend(self) -> str:
        """Select optimal backend based on available resources."""
        if len(self.available_gpus) > 0 and self.n_jobs > 4:
            return "jax"  # GPU acceleration for large jobs
        elif self.n_jobs > 1:
            return "multiprocessing"  # Multi-core CPU
        else:
            return "threading"  # Single core with threading
    
    def parallel_map(self, func: Callable, data: List[Any], 
                    chunk_size: int = None, 
                    progress_bar: bool = True) -> List[Any]:
        """
        Parallel map operation using the selected backend.
        
        Parameters
        ----------
        func : Callable
            Function to apply to each element
        data : List[Any]
            Data to process
        chunk_size : int
            Chunk size for processing
        progress_bar : bool
            Whether to show progress bar
            
        Returns
        -------
        List[Any]
            Results of parallel processing
        """
        if self.backend == "jax":
            return self._jax_parallel_map(func, data, chunk_size)
        elif self.backend == "multiprocessing":
            return self._multiprocessing_map(func, data, chunk_size, progress_bar)
        elif self.backend == "threading":
            return self._threading_map(func, data, chunk_size, progress_bar)
        elif self.backend == "joblib":
            return self._joblib_map(func, data, chunk_size, progress_bar)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _jax_parallel_map(self, func: Callable, data: List[Any], 
                          chunk_size: int = None) -> List[Any]:
        """Parallel processing using JAX vectorization."""
        if chunk_size is None:
            chunk_size = min(32, len(data))
        
        # Convert data to JAX arrays if possible
        try:
            data_array = jnp.array(data)
            # Use JAX vectorization
            vectorized_func = jax.vmap(func)
            results = vectorized_func(data_array)
            return results.tolist()
        except Exception as e:
            logger.warning(f"JAX vectorization failed: {e}. Falling back to batching.")
            return self._jax_batch_processing(func, data, chunk_size)
    
    def _jax_batch_processing(self, func: Callable, data: List[Any], 
                             chunk_size: int) -> List[Any]:
        """Batch processing using JAX."""
        results = []
        
        for i in range(0, len(data), chunk_size):
            batch = data[i:i + chunk_size]
            try:
                batch_array = jnp.array(batch)
                batch_results = func(batch_array)
                if hasattr(batch_results, 'tolist'):
                    results.extend(batch_results.tolist())
                else:
                    results.extend(batch_results)
            except Exception as e:
                logger.warning(f"JAX batch processing failed: {e}")
                # Fallback to sequential processing for this batch
                for item in batch:
                    results.append(func(item))
        
        return results
    
    def _multiprocessing_map(self, func: Callable, data: List[Any], 
                            chunk_size: int = None, 
                            progress_bar: bool = True) -> List[Any]:
        """Parallel processing using multiprocessing."""
        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.n_jobs * 4))
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(func, item): item 
                for item in data
            }
            
            # Collect results
            results = [None] * len(data)
            completed = 0
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    idx = data.index(item)
                    results[idx] = result
                    completed += 1
                    
                    if progress_bar and completed % max(1, len(data) // 20) == 0:
                        logger.info(f"Progress: {completed}/{len(data)} ({100*completed/len(data):.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    idx = data.index(item)
                    results[idx] = None
        
        return results
    
    def _threading_map(self, func: Callable, data: List[Any], 
                       chunk_size: int = None, 
                       progress_bar: bool = True) -> List[Any]:
        """Parallel processing using threading."""
        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.n_jobs * 4))
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(func, item): item 
                for item in data
            }
            
            # Collect results
            results = [None] * len(data)
            completed = 0
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    idx = data.index(item)
                    results[idx] = result
                    completed += 1
                    
                    if progress_bar and completed % max(1, len(data) // 20) == 0:
                        logger.info(f"Progress: {completed}/{len(data)} ({100*completed/len(data):.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    idx = data.index(item)
                    results[idx] = None
        
        return results
    
    def _joblib_map(self, func: Callable, data: List[Any], 
                    chunk_size: int = None, 
                    progress_bar: bool = True) -> List[Any]:
        """Parallel processing using joblib."""
        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.n_jobs * 4))
        
        return joblib.Parallel(
            n_jobs=self.n_jobs,
            verbose=10 if progress_bar else 0,
            batch_size=chunk_size
        )(joblib.delayed(func)(item) for item in data)
    
    def parallel_apply(self, func: Callable, data: List[Any], 
                      batch_size: int = None,
                      strategy: str = "auto") -> List[Any]:
        """
        Apply function to data using optimal parallel strategy.
        
        Parameters
        ----------
        func : Callable
            Function to apply
        data : List[Any]
            Data to process
        batch_size : int
            Batch size for processing
        strategy : str
            Parallelization strategy ('auto', 'chunk', 'stream', 'pipeline')
            
        Returns
        -------
        List[Any]
            Results of parallel processing
        """
        if strategy == "auto":
            strategy = self._select_optimal_strategy(len(data))
        
        if strategy == "chunk":
            return self._chunk_processing(func, data, batch_size)
        elif strategy == "stream":
            return self._stream_processing(func, data, batch_size)
        elif strategy == "pipeline":
            return self._pipeline_processing(func, data, batch_size)
        else:
            return self.parallel_map(func, data, batch_size)
    
    def _select_optimal_strategy(self, data_size: int) -> str:
        """Select optimal parallelization strategy."""
        if data_size < 100:
            return "chunk"
        elif data_size < 10000:
            return "stream"
        else:
            return "pipeline"
    
    def _chunk_processing(self, func: Callable, data: List[Any], 
                         batch_size: int = None) -> List[Any]:
        """Process data in chunks."""
        if batch_size is None:
            batch_size = max(1, len(data) // self.n_jobs)
        
        results = []
        for i in range(0, len(data), batch_size):
            chunk = data[i:i + batch_size]
            chunk_results = self.parallel_map(func, chunk)
            results.extend(chunk_results)
        
        return results
    
    def _stream_processing(self, func: Callable, data: List[Any], 
                          batch_size: int = None) -> List[Any]:
        """Process data in streaming fashion."""
        if batch_size is None:
            batch_size = max(1, len(data) // (self.n_jobs * 2))
        
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit initial batch
            futures = []
            for i in range(0, min(batch_size, len(data))):
                future = executor.submit(func, data[i])
                futures.append((future, i))
            
            # Process remaining data
            next_item = batch_size
            while futures:
                # Wait for one future to complete
                done, not_done = futures[0][0], futures[1:]
                futures = not_done
                
                try:
                    result = done.result()
                    results.append(result)
                    
                    # Submit next item if available
                    if next_item < len(data):
                        future = executor.submit(func, data[next_item])
                        futures.append((future, next_item))
                        next_item += 1
                        
                except Exception as e:
                    logger.error(f"Error in stream processing: {e}")
                    results.append(None)
        
        return results
    
    def _pipeline_processing(self, func: Callable, data: List[Any], 
                            batch_size: int = None) -> List[Any]:
        """Process data using pipeline parallelism."""
        if batch_size is None:
            batch_size = max(1, len(data) // (self.n_jobs * 4))
        
        # Create pipeline stages
        stages = []
        for i in range(self.n_jobs):
            stage_data = data[i::self.n_jobs]
            if stage_data:
                stages.append(stage_data)
        
        # Process each stage in parallel
        stage_results = self.parallel_map(
            lambda stage: [func(item) for item in stage], 
            stages
        )
        
        # Interleave results
        results = []
        max_stage_len = max(len(stage) for stage in stages)
        
        for i in range(max_stage_len):
            for stage_idx, stage in enumerate(stages):
                if i < len(stage):
                    results.append(stage_results[stage_idx][i])
        
        return results
    
    def gpu_parallel_processing(self, func: Callable, data: List[Any], 
                               device: str = "auto") -> List[Any]:
        """
        GPU-accelerated parallel processing.
        
        Parameters
        ----------
        func : Callable
            Function to apply (must be JAX-compatible)
        data : List[Any]
            Data to process
        device : str
            GPU device to use
            
        Returns
        -------
        List[Any]
            Results of GPU processing
        """
        if not self.available_gpus:
            logger.warning("No GPUs available, falling back to CPU")
            return self.parallel_map(func, data)
        
        # Select GPU device
        if device == "auto":
            device = self.available_gpus[0]
        
        # Convert data to JAX arrays
        try:
            data_array = jnp.array(data)
            
            # Compile function for GPU
            gpu_func = jax.jit(func, device=device)
            
            # Process on GPU
            results = gpu_func(data_array)
            
            return results.tolist() if hasattr(results, 'tolist') else results
            
        except Exception as e:
            logger.error(f"GPU processing failed: {e}. Falling back to CPU.")
            return self.parallel_map(func, data)
    
    def distributed_processing(self, func: Callable, data: List[Any], 
                             nodes: List[str] = None) -> List[Any]:
        """
        Distributed processing across multiple nodes.
        
        Parameters
        ----------
        func : Callable
            Function to apply
        data : List[Any]
            Data to process
        nodes : List[str]
            List of node addresses
            
        Returns
        -------
        List[Any]
            Results of distributed processing
        """
        if not nodes:
            # Single node processing
            return self.parallel_map(func, data)
        
        # Simple distributed processing using SSH (basic implementation)
        logger.info(f"Distributed processing across {len(nodes)} nodes")
        
        # For now, fall back to local parallel processing
        # In a real implementation, you would use libraries like Dask, Ray, or custom MPI
        return self.parallel_map(func, data)
    
    def benchmark_parallel_strategies(self, func: Callable, data: List[Any], 
                                    strategies: List[str] = None) -> Dict[str, float]:
        """
        Benchmark different parallel processing strategies.
        
        Parameters
        ----------
        func : Callable
            Function to benchmark
        data : List[Any]
            Data to process
        strategies : List[str]
            List of strategies to benchmark
            
        Returns
        -------
        Dict[str, float]
            Execution times for each strategy
        """
        if strategies is None:
            strategies = ["chunk", "stream", "pipeline", "jax"]
        
        results = {}
        
        for strategy in strategies:
            try:
                start_time = time.time()
                
                if strategy == "jax":
                    self._jax_parallel_map(func, data)
                else:
                    self.parallel_apply(func, data, strategy=strategy)
                
                execution_time = time.time() - start_time
                results[strategy] = execution_time
                
                logger.info(f"Strategy '{strategy}': {execution_time:.3f} seconds")
                
            except Exception as e:
                logger.error(f"Strategy '{strategy}' failed: {e}")
                results[strategy] = float('inf')
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for parallel processing.
        
        Returns
        -------
        Dict[str, Any]
            Performance summary
        """
        return {
            'backend': self.backend,
            'n_jobs': self.n_jobs,
            'available_cores': self.available_cores,
            'available_gpus': self.available_gpus,
            'cpu_utilization': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'optimal_strategy': self._select_optimal_strategy(1000)
        }
