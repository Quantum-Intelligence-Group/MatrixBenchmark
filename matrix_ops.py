"""
Matrix Operations Module

A focused Python module for matrix generation, multiplication, and performance calculations.
Provides optimized matrix operations using NumPy with proper seeding and performance metrics.
"""

import numpy as np
from typing import Tuple, Union, Optional
import time


def generate_matrix(
    size: int,
    seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
    low: float = 0.0,
    high: float = 1.0
) -> np.ndarray:
    """
    Generate a reproducible random matrix using NumPy.
    
    Args:
        size: Matrix dimension (creates size x size matrix)
        seed: Random seed for reproducibility (optional)
        dtype: Data type for matrix elements
        low: Lower bound for random values
        high: Upper bound for random values
        
    Returns:
        np.ndarray: Generated matrix of shape (size, size)
        
    Raises:
        ValueError: If size is not positive or low >= high
    """
    if size <= 0:
        raise ValueError("Matrix size must be positive")
    if low >= high:
        raise ValueError("Low bound must be less than high bound")
    
    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random matrix
    matrix = np.random.uniform(low, high, size=(size, size)).astype(dtype)
    
    return matrix


def generate_matrix_pair(
    size: int,
    seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
    low: float = 0.0,
    high: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a pair of reproducible random matrices for multiplication.
    
    Args:
        size: Matrix dimension (creates size x size matrices)
        seed: Random seed for reproducibility (optional)
        dtype: Data type for matrix elements
        low: Lower bound for random values
        high: Upper bound for random values
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Pair of matrices A and B
        
    Raises:
        ValueError: If size is not positive or low >= high
    """
    if size <= 0:
        raise ValueError("Matrix size must be positive")
    if low >= high:
        raise ValueError("Low bound must be less than high bound")
    
    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate two matrices with different random states
    matrix_a = np.random.uniform(low, high, size=(size, size)).astype(dtype)
    matrix_b = np.random.uniform(low, high, size=(size, size)).astype(dtype)
    
    return matrix_a, matrix_b


def multiply_matrices(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    validate: bool = True
) -> np.ndarray:
    """
    Perform optimized matrix multiplication using NumPy.
    
    Args:
        matrix_a: First matrix (m x n)
        matrix_b: Second matrix (n x p)
        validate: Whether to validate matrix dimensions
        
    Returns:
        np.ndarray: Result of matrix multiplication (m x p)
        
    Raises:
        ValueError: If matrices have incompatible dimensions
        TypeError: If inputs are not numpy arrays
    """
    if not isinstance(matrix_a, np.ndarray) or not isinstance(matrix_b, np.ndarray):
        raise TypeError("Both inputs must be numpy arrays")
    
    if validate:
        if matrix_a.ndim != 2 or matrix_b.ndim != 2:
            raise ValueError("Both matrices must be 2-dimensional")
        if matrix_a.shape[1] != matrix_b.shape[0]:
            raise ValueError(
                f"Incompatible matrix dimensions: {matrix_a.shape} and {matrix_b.shape}"
            )
    
    # Use np.matmul for optimized matrix multiplication
    result = np.matmul(matrix_a, matrix_b)
    
    return result


def timed_matrix_multiply(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    validate: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Perform matrix multiplication with timing measurement.
    
    Args:
        matrix_a: First matrix
        matrix_b: Second matrix
        validate: Whether to validate matrix dimensions
        
    Returns:
        Tuple[np.ndarray, float]: (result_matrix, execution_time_seconds)
        
    Raises:
        ValueError: If matrices have incompatible dimensions
        TypeError: If inputs are not numpy arrays
    """
    start_time = time.perf_counter()
    result = multiply_matrices(matrix_a, matrix_b, validate)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    return result, execution_time


def calculate_theoretical_ops(
    matrix_size: int,
    operation: str = "matmul"
) -> int:
    """
    Calculate theoretical number of operations for matrix operations.
    
    Args:
        matrix_size: Size of square matrix (n x n)
        operation: Type of operation ("matmul" for matrix multiplication)
        
    Returns:
        int: Number of theoretical operations
        
    Raises:
        ValueError: If matrix_size is not positive or operation is unknown
    """
    if matrix_size <= 0:
        raise ValueError("Matrix size must be positive")
    
    if operation == "matmul":
        # For matrix multiplication: 2 * n^3 operations (n^3 multiplications + n^3 additions)
        return 2 * (matrix_size ** 3)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def calculate_gflops(
    matrix_size: int,
    execution_time: float,
    operation: str = "matmul"
) -> float:
    """
    Calculate GFLOPS (Giga Floating Point Operations Per Second) from timing data.
    
    Args:
        matrix_size: Size of square matrix (n x n)
        execution_time: Execution time in seconds
        operation: Type of operation ("matmul" for matrix multiplication)
        
    Returns:
        float: GFLOPS performance metric
        
    Raises:
        ValueError: If matrix_size is not positive, execution_time is not positive,
                   or operation is unknown
    """
    if matrix_size <= 0:
        raise ValueError("Matrix size must be positive")
    if execution_time <= 0:
        raise ValueError("Execution time must be positive")
    
    # Calculate theoretical operations
    ops = calculate_theoretical_ops(matrix_size, operation)
    
    # Convert to GFLOPS (operations per second / 1e9)
    gflops = ops / (execution_time * 1e9)
    
    return gflops


def benchmark_matrix_multiply(
    size: int,
    seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
    num_runs: int = 1
) -> dict:
    """
    Comprehensive benchmark for matrix multiplication.
    
    Args:
        size: Matrix dimension (creates size x size matrices)
        seed: Random seed for reproducibility
        dtype: Data type for matrix elements
        num_runs: Number of benchmark runs for averaging
        
    Returns:
        dict: Benchmark results including timing, GFLOPS, and statistics
        
    Raises:
        ValueError: If size is not positive or num_runs is not positive
    """
    if size <= 0:
        raise ValueError("Matrix size must be positive")
    if num_runs <= 0:
        raise ValueError("Number of runs must be positive")
    
    # Generate matrices
    matrix_a, matrix_b = generate_matrix_pair(size, seed, dtype)
    
    # Perform multiple runs
    times = []
    for _ in range(num_runs):
        _, exec_time = timed_matrix_multiply(matrix_a, matrix_b, validate=False)
        times.append(exec_time)
    
    # Calculate statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    # Calculate performance metrics
    avg_gflops = calculate_gflops(size, avg_time)
    max_gflops = calculate_gflops(size, min_time)  # Best performance
    
    # Calculate theoretical operations
    theoretical_ops = calculate_theoretical_ops(size)
    
    return {
        "matrix_size": size,
        "data_type": str(dtype),
        "num_runs": num_runs,
        "times": {
            "average": avg_time,
            "minimum": min_time,
            "maximum": max_time,
            "std_dev": std_time
        },
        "performance": {
            "avg_gflops": avg_gflops,
            "max_gflops": max_gflops
        },
        "theoretical_ops": theoretical_ops,
        "matrix_memory_mb": (size * size * np.dtype(dtype).itemsize) / (1024 * 1024)
    }