#!/usr/bin/env python3
"""
Example Integration Script

Demonstrates how to use the timing module with the matrix_ops module
for comprehensive benchmarking.
"""

# import numpy as np  # Not directly used
from matrix_ops import generate_matrix_pair, multiply_matrices, benchmark_matrix_multiply
from timing import (
    timer, quick_benchmark, compare_functions, measure_execution_time,
    run_benchmark_iterations, calculate_statistics, format_benchmark_results,
    detect_thermal_throttling
)


def matrix_multiply_wrapper(size=1000, seed=42):
    """Wrapper function for matrix multiplication benchmarking."""
    matrix_a, matrix_b = generate_matrix_pair(size, seed=seed)
    result = multiply_matrices(matrix_a, matrix_b)
    return result


def demonstrate_timing_features():
    """Demonstrate various timing module features."""
    
    print("=" * 60)
    print("TIMING MODULE INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # 1. Basic timing with context manager
    print("\n1. Basic Timing with Context Manager")
    print("-" * 40)
    
    with timer("Matrix multiplication (500x500)", monitor_resources=True) as result:
        matrix_a, matrix_b = generate_matrix_pair(500, seed=42)
        multiply_matrices(matrix_a, matrix_b)
    
    print(f"Memory usage: {result.memory_usage_mb:.2f} MB")
    print(f"Peak memory: {result.peak_memory_mb:.2f} MB")
    print(f"CPU usage: {result.cpu_percent:.1f}%")
    
    # 2. Quick benchmark comparison
    print("\n2. Quick Benchmark Comparison")
    print("-" * 40)
    
    def small_matrix_mult():
        return matrix_multiply_wrapper(300, seed=42)
    
    def medium_matrix_mult():
        return matrix_multiply_wrapper(500, seed=42)
    
    # Compare different matrix sizes
    _ = compare_functions(
        [small_matrix_mult, medium_matrix_mult],
        iterations=5,
        warmup_runs=1,
        labels=["300x300", "500x500"]
    )
    
    # 3. Detailed benchmark with statistical analysis
    print("\n3. Detailed Benchmark with Statistics")
    print("-" * 40)
    
    results = run_benchmark_iterations(
        matrix_multiply_wrapper,
        400,  # matrix size
        iterations=10,
        warmup_runs=2,
        progress_callback=lambda current, total: print(f"  Run {current}/{total}")
    )
    
    times = [r.execution_time for r in results]
    stats = calculate_statistics(times)
    
    print(format_benchmark_results(results, stats, "Matrix Multiplication 400x400"))
    
    # 4. Thermal throttling detection
    print("\n4. Thermal Throttling Detection")
    print("-" * 40)
    
    throttling = detect_thermal_throttling(
        cpu_freq_samples=5,
        sample_interval=0.5,
        threshold_percent=3.0
    )
    
    print(f"Throttling detected: {throttling['throttling_detected']}")
    if throttling.get('initial_frequency_mhz'):
        print(f"Initial CPU freq: {throttling['initial_frequency_mhz']:.0f} MHz")
        print(f"Min CPU freq: {throttling['min_frequency_mhz']:.0f} MHz")
        print(f"Frequency drop: {throttling['frequency_drop_percent']:.1f}%")
    
    # 5. Integration with existing benchmark function
    print("\n5. Integration with Existing Benchmark Function")
    print("-" * 40)
    
    # Use the existing benchmark function from matrix_ops
    existing_results = benchmark_matrix_multiply(
        size=400,
        seed=42,
        num_runs=5
    )
    
    print(f"Existing benchmark avg time: {existing_results['times']['average']:.6f} seconds")
    print(f"Existing benchmark GFLOPS: {existing_results['performance']['avg_gflops']:.2f}")
    
    # Compare with our timing module
    _, our_stats = quick_benchmark(
        matrix_multiply_wrapper,
        400, 42,
        iterations=5,
        print_results=False
    )
    
    print(f"Our timing module avg time: {our_stats.mean:.6f} seconds")
    print(f"Difference: {abs(our_stats.mean - existing_results['times']['average']):.6f} seconds")
    
    # 6. Memory usage analysis
    print("\n6. Memory Usage Analysis")
    print("-" * 40)
    
    memory_results = []
    for size in [200, 400, 600, 800]:
        result = measure_execution_time(
            matrix_multiply_wrapper,
            size, 42,
            monitor_resources=True
        )
        memory_results.append((size, result.memory_usage_mb, result.peak_memory_mb))
    
    print("Matrix Size | Memory Delta | Peak Memory")
    print("-" * 40)
    for size, mem_delta, peak_mem in memory_results:
        print(f"{size:10d} | {mem_delta:11.2f} | {peak_mem:10.2f}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_timing_features()