"""
Memory Management Utilities for High-Performance Computing

This module provides tools for efficient memory usage, including
memory pooling, garbage collection, and memory monitoring.
"""

import psutil
import gc
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import weakref
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Memory management utilities for efficient memory usage.
    
    This class provides tools for monitoring memory usage, managing
    memory pools, and optimizing memory allocation for large computations.
    """
    
    def __init__(self, enable_monitoring: bool = True):
        """
        Initialize memory manager.
        
        Parameters
        ----------
        enable_monitoring : bool
            Whether to enable memory monitoring
        """
        self.enable_monitoring = enable_monitoring
        self.memory_pools = {}
        self.memory_history = []
        self.peak_memory = 0.0
        
        if enable_monitoring:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Start memory monitoring."""
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        logger.info(f"Memory monitoring started. Initial usage: {self.initial_memory:.2f} MB")
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns
        -------
        float
            Memory usage in MB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    
    def get_memory_percent(self) -> float:
        """
        Get current memory usage as percentage of system memory.
        
        Returns
        -------
        float
            Memory usage as percentage
        """
        return psutil.virtual_memory().percent
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """
        Get system memory information.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing memory information
        """
        vm = psutil.virtual_memory()
        return {
            'total': vm.total / 1024 / 1024 / 1024,  # GB
            'available': vm.available / 1024 / 1024 / 1024,  # GB
            'used': vm.used / 1024 / 1024 / 1024,  # GB
            'percent': vm.percent
        }
    
    def monitor_memory(self, operation_name: str = "operation"):
        """
        Monitor memory usage for an operation.
        
        Parameters
        ----------
        operation_name : str
            Name of the operation being monitored
        """
        if not self.enable_monitoring:
            return
        
        current_memory = self.get_memory_usage()
        memory_change = current_memory - self.initial_memory
        
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        self.memory_history.append({
            'operation': operation_name,
            'memory_usage': current_memory,
            'memory_change': memory_change,
            'timestamp': psutil.cpu_times()
        })
        
        logger.info(f"{operation_name}: Memory usage: {current_memory:.2f} MB "
                   f"(Change: {memory_change:+.2f} MB)")
    
    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """
        Context manager for memory monitoring.
        
        Parameters
        ----------
        operation_name : str
            Name of the operation being monitored
        """
        if self.enable_monitoring:
            initial_memory = self.get_memory_usage()
            logger.info(f"Starting {operation_name}. Memory: {initial_memory:.2f} MB")
        
        try:
            yield
        finally:
            if self.enable_monitoring:
                final_memory = self.get_memory_usage()
                memory_change = final_memory - initial_memory
                logger.info(f"Completed {operation_name}. Memory: {final_memory:.2f} MB "
                           f"(Change: {memory_change:+.2f} MB)")
    
    def create_memory_pool(self, name: str, size: int, dtype: np.dtype = np.float64):
        """
        Create a memory pool for efficient memory allocation.
        
        Parameters
        ----------
        name : str
            Name of the memory pool
        size : int
            Size of the pool in elements
        dtype : np.dtype
            Data type for the pool
        """
        pool_size = size * np.dtype(dtype).itemsize / 1024 / 1024  # MB
        self.memory_pools[name] = {
            'data': np.zeros(size, dtype=dtype),
            'size': size,
            'dtype': dtype,
            'pool_size_mb': pool_size,
            'allocated': False
        }
        
        logger.info(f"Created memory pool '{name}' with size {pool_size:.2f} MB")
    
    def get_from_pool(self, name: str, size: int) -> Optional[np.ndarray]:
        """
        Get memory from a pool.
        
        Parameters
        ----------
        name : str
            Name of the memory pool
        size : int
            Size needed
        
        Returns
        -------
        Optional[np.ndarray]
            Memory slice from pool, or None if not available
        """
        if name not in self.memory_pools:
            return None
        
        pool = self.memory_pools[name]
        if pool['allocated'] or size > pool['size']:
            return None
        
        pool['allocated'] = True
        return pool['data'][:size]
    
    def return_to_pool(self, name: str):
        """
        Return memory to a pool.
        
        Parameters
        ----------
        name : str
            Name of the memory pool
        """
        if name in self.memory_pools:
            self.memory_pools[name]['allocated'] = False
    
    def clear_memory_pools(self):
        """Clear all memory pools."""
        for name in list(self.memory_pools.keys()):
            del self.memory_pools[name]
        gc.collect()
        logger.info("Cleared all memory pools")
    
    def optimize_memory_layout(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optimize memory layout for better cache performance.
        
        Parameters
        ----------
        arrays : List[np.ndarray]
            List of arrays to optimize
            
        Returns
        -------
        List[np.ndarray]
            List of optimized arrays
        """
        optimized = []
        
        for arr in arrays:
            # Ensure contiguous memory layout
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            
            # Align to cache line boundaries if possible
            if arr.size > 0:
                # Simple alignment strategy
                aligned_size = ((arr.size + 63) // 64) * 64
                if aligned_size != arr.size:
                    # Pad array to cache line boundary
                    padded = np.zeros(aligned_size, dtype=arr.dtype)
                    padded[:arr.size] = arr
                    arr = padded
            
            optimized.append(arr)
        
        return optimized
    
    def batch_processing_memory_optimization(self, data_size: int, 
                                          batch_size: int,
                                          dtype: np.dtype = np.float64) -> int:
        """
        Calculate optimal batch size for memory efficiency.
        
        Parameters
        ----------
        data_size : int
            Total size of data to process
        batch_size : int
            Initial batch size
        dtype : np.dtype
            Data type
        
        Returns
        -------
        int
            Optimized batch size
        """
        # Get available memory
        available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
        
        # Calculate memory per element
        element_size = np.dtype(dtype).itemsize / 1024 / 1024  # MB
        
        # Reserve some memory for other operations (20%)
        usable_memory = available_memory * 0.8
        
        # Calculate optimal batch size
        optimal_batch_size = int(usable_memory / (data_size * element_size))
        
        # Ensure batch size is reasonable
        optimal_batch_size = max(1, min(optimal_batch_size, batch_size))
        
        logger.info(f"Memory optimization: Available: {available_memory:.2f} MB, "
                   f"Optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def jax_memory_optimization(self, device: str = "auto"):
        """
        Optimize JAX memory usage.
        
        Parameters
        ----------
        device : str
            Device to optimize for
        """
        if device == "auto":
            devices = jax.devices()
            if any("gpu" in str(d).lower() for d in devices):
                device = "gpu"
            else:
                device = "cpu"
        
        if device == "gpu":
            # Set JAX memory fraction
            import os
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
            
            # Enable memory preallocation
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
            
            logger.info("JAX GPU memory optimization enabled")
        else:
            # CPU memory optimization
            jax.config.update('jax_platform_name', 'cpu')
            logger.info("JAX CPU memory optimization enabled")
    
    def cleanup_memory(self, aggressive: bool = False):
        """
        Clean up memory using garbage collection.
        
        Parameters
        ----------
        aggressive : bool
            Whether to use aggressive cleanup
        """
        if aggressive:
            # Multiple garbage collection passes
            for _ in range(3):
                gc.collect()
            
            # Clear JAX compilation cache if available
            try:
                jax.clear_caches()
            except:
                pass
        else:
            # Single garbage collection pass
            gc.collect()
        
        current_memory = self.get_memory_usage()
        logger.info(f"Memory cleanup completed. Current usage: {current_memory:.2f} MB")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage summary.
        
        Returns
        -------
        Dict[str, Any]
            Memory usage summary
        """
        current_memory = self.get_memory_usage()
        system_info = self.get_system_memory_info()
        
        summary = {
            'current_memory_mb': current_memory,
            'initial_memory_mb': self.initial_memory if self.enable_monitoring else 0.0,
            'peak_memory_mb': self.peak_memory if self.enable_monitoring else 0.0,
            'memory_change_mb': current_memory - self.initial_memory if self.enable_monitoring else 0.0,
            'system_memory': system_info,
            'memory_pools': {
                name: {
                    'size_mb': pool['pool_size_mb'],
                    'allocated': pool['allocated']
                }
                for name, pool in self.memory_pools.items()
            },
            'pool_count': len(self.memory_pools),
            'total_pool_memory_mb': sum(pool['pool_size_mb'] for pool in self.memory_pools.values())
        }
        
        return summary
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'memory_pools'):
            self.clear_memory_pools()
