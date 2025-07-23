#!/usr/bin/env python3
"""
Demo script showing how to use the Matrix Benchmark Orchestrator

This script demonstrates various ways to use the benchmark orchestrator
for different benchmarking scenarios.
"""

import os
import sys
from benchmark import (
    MatrixBenchmarkOrchestrator, BenchmarkConfig, create_default_config,
    load_benchmark_results, compare_benchmark_results
)


def demo_basic_benchmark():
    """Demo: Basic benchmark with default settings."""
    print("="*80)
    print("DEMO 1: Basic Benchmark")
    print("="*80)
    
    # Create a quick configuration
    config = BenchmarkConfig(
        matrix_sizes=[300, 500],
        num_runs=3,
        warmup_runs=1,
        seeds=[42],
        dtype="float64",
        monitor_resources=True,
        detect_throttling=False,  # Skip for demo
        save_results=True,
        output_dir="demo_results",
        output_prefix="demo_basic"
    )
    
    # Run benchmark
    orchestrator = MatrixBenchmarkOrchestrator(config)
    results = orchestrator.run_comprehensive_benchmark()
    
    # Display results
    orchestrator.display_results(results)
    
    return results


def demo_custom_benchmark():
    """Demo: Custom benchmark with specific parameters."""
    print("="*80)
    print("DEMO 2: Custom Benchmark Configuration")
    print("="*80)
    
    # Create custom configuration
    config = BenchmarkConfig(
        matrix_sizes=[400, 600, 800],
        num_runs=2,
        warmup_runs=1,
        seeds=[42, 123],
        dtype="float32",  # Use float32 for faster computation
        monitor_resources=True,
        detect_throttling=False,
        save_results=True,
        output_dir="demo_results",
        output_prefix="demo_custom"
    )
    
    # Run benchmark
    orchestrator = MatrixBenchmarkOrchestrator(config)
    results = orchestrator.run_comprehensive_benchmark()
    
    # Show custom analysis
    print("\nCustom Analysis:")
    print("-" * 40)
    
    # Find best performing matrix size
    best_size = results.summary_statistics["overall"]["peak_performance_size"]
    best_gflops = results.summary_statistics["overall"]["max_gflops"]
    
    print(f"Best performing matrix size: {best_size}")
    print(f"Peak performance: {best_gflops:.2f} GFLOPS")
    
    # Show performance scaling
    by_size = results.summary_statistics["by_size"]
    print("\nPerformance Scaling:")
    prev_gflops = None
    for size in sorted(by_size.keys()):
        gflops = by_size[size]["avg_gflops"]
        if prev_gflops is not None:
            scaling = gflops / prev_gflops
            print(f"  {size}: {gflops:.2f} GFLOPS (scaling: {scaling:.2f}x)")
        else:
            print(f"  {size}: {gflops:.2f} GFLOPS (baseline)")
        prev_gflops = gflops
    
    return results


def demo_programmatic_usage():
    """Demo: Using the benchmark orchestrator programmatically."""
    print("="*80)
    print("DEMO 3: Programmatic Usage")
    print("="*80)
    
    # Start with default config and modify
    config = create_default_config()
    config.matrix_sizes = [200, 400]
    config.num_runs = 2
    config.seeds = [42]
    config.save_results = True
    config.output_dir = "demo_results"
    config.output_prefix = "demo_programmatic"
    
    # Create orchestrator
    orchestrator = MatrixBenchmarkOrchestrator(config)
    
    # Access system profile before running
    system_profile = orchestrator.system_profiler.get_complete_profile()
    
    print("System Profile:")
    print(f"  Hardware: {system_profile.hardware.chip_type}")
    print(f"  Memory: {system_profile.hardware.memory_total_gb:.1f} GB")
    print(f"  CPU Score: {system_profile.performance.cpu_benchmark_score:.2f} GFLOPS")
    print(f"  Memory Bandwidth: {system_profile.performance.memory_bandwidth_measured:.2f} GB/s")
    
    # Run benchmark
    results = orchestrator.run_comprehensive_benchmark()
    
    # Programmatic analysis
    print("\nProgrammatic Analysis:")
    print("-" * 40)
    
    # Calculate efficiency metrics
    for result in results.benchmark_results:
        memory_efficiency = result.gflops / system_profile.performance.memory_bandwidth_measured
        print(f"Size {result.matrix_size}: {result.gflops:.2f} GFLOPS, "
              f"Memory Efficiency: {memory_efficiency:.4f}")
    
    return results


def demo_load_and_compare():
    """Demo: Loading and comparing benchmark results."""
    print("="*80)
    print("DEMO 4: Loading and Comparing Results")
    print("="*80)
    
    # Check if we have results to compare
    demo_dir = "demo_results"
    if not os.path.exists(demo_dir):
        print("No demo results directory found. Run other demos first.")
        return
    
    # Find result files
    result_files = [f for f in os.listdir(demo_dir) if f.endswith('.json')]
    
    if len(result_files) < 2:
        print(f"Need at least 2 result files, found {len(result_files)}")
        return
    
    # Load two results
    file1 = os.path.join(demo_dir, result_files[0])
    file2 = os.path.join(demo_dir, result_files[1])
    
    print(f"Comparing: {result_files[0]} vs {result_files[1]}")
    
    results1 = load_benchmark_results(file1)
    results2 = load_benchmark_results(file2)
    
    print("\nComparison Results:")
    print("-" * 40)
    
    # Basic comparison
    if (results1.get('summary_statistics', {}).get('overall') and 
        results2.get('summary_statistics', {}).get('overall')):
        
        gflops1 = results1['summary_statistics']['overall']['avg_gflops']
        gflops2 = results2['summary_statistics']['overall']['avg_gflops']
        
        print(f"Result 1 average: {gflops1:.2f} GFLOPS")
        print(f"Result 2 average: {gflops2:.2f} GFLOPS")
        
        if gflops1 > gflops2:
            speedup = gflops1 / gflops2
            print(f"Result 1 is {speedup:.2f}x faster")
        else:
            speedup = gflops2 / gflops1
            print(f"Result 2 is {speedup:.2f}x faster")


def main():
    """Run all demos."""
    print("Matrix Benchmark Orchestrator Demo")
    print("="*80)
    
    try:
        # Demo 1: Basic benchmark
        demo_basic_benchmark()
        
        # Demo 2: Custom configuration
        demo_custom_benchmark()
        
        # Demo 3: Programmatic usage
        demo_programmatic_usage()
        
        # Demo 4: Load and compare (if possible)
        demo_load_and_compare()
        
        print("\n" + "="*80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nTo run benchmarks manually:")
        print("python benchmark.py --help")
        print("python benchmark.py --quick")
        print("python benchmark.py --sizes 1000 2000 --runs 5")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()