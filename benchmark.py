#!/usr/bin/env python3
"""
Matrix Benchmark Orchestrator

A comprehensive benchmarking system that coordinates matrix operations, timing, 
and system profiling to provide reproducible cross-machine performance comparison.
This is the main entry point for the benchmarking suite.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# Import our modules
from matrix_ops import (
    generate_matrix_pair, multiply_matrices, timed_matrix_multiply,
    calculate_gflops, benchmark_matrix_multiply
)
from timing import (
    run_benchmark_iterations, calculate_statistics, format_benchmark_results,
    save_benchmark_results, load_benchmark_results, detect_thermal_throttling,
    BenchmarkStats, TimingResult
)
from system_info import MacSystemProfiler, SystemProfile, create_system_profile


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    matrix_sizes: List[int]
    num_runs: int
    warmup_runs: int
    seeds: List[int]
    dtype: str
    monitor_resources: bool
    detect_throttling: bool
    save_results: bool
    save_matrices: bool
    output_dir: str
    output_prefix: str


@dataclass
class BenchmarkResult:
    """Single benchmark result combining all metrics."""
    matrix_size: int
    seed: int
    run_id: str
    timing_stats: BenchmarkStats
    timing_results: List[TimingResult]
    gflops: float
    theoretical_ops: int
    matrix_memory_mb: float
    timestamp: float
    matrix_files: Optional[Dict[str, str]] = None  # Paths to saved matrix files


@dataclass
class ComprehensiveResults:
    """Complete benchmark results with system profile."""
    benchmark_id: str
    system_profile: SystemProfile
    benchmark_config: BenchmarkConfig
    benchmark_results: List[BenchmarkResult]
    thermal_state: Dict[str, Any]
    summary_statistics: Dict[str, Any]
    timestamp: float
    duration_seconds: float


class MatrixBenchmarkOrchestrator:
    """Main orchestrator for matrix benchmarking operations."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark orchestrator."""
        self.config = config
        self.system_profiler = MacSystemProfiler()
        self.logger = self._setup_logging()
        self.benchmark_id = self._generate_benchmark_id()
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the benchmark."""
        logger = logging.getLogger('matrix_benchmark')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _generate_benchmark_id(self) -> str:
        """Generate a unique benchmark ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.output_prefix}_{timestamp}"
    
    def _matrix_multiply_benchmark_func(self, size: int, seed: int) -> np.ndarray:
        """Benchmark function wrapper for matrix multiplication."""
        matrix_a, matrix_b = generate_matrix_pair(size, seed=seed)
        result = multiply_matrices(matrix_a, matrix_b, validate=False)
        return result
    
    def _save_matrices(self, run_id: str, matrix_a: np.ndarray, matrix_b: np.ndarray, result: np.ndarray) -> Dict[str, str]:
        """Save matrices to separate .npy files using benchmark naming convention."""
        matrix_files = {}
        
        try:
            # Create matrix files with same naming pattern as benchmark results
            matrix_a_path = os.path.join(self.config.output_dir, f"{run_id}_matrix_a.npy")
            matrix_b_path = os.path.join(self.config.output_dir, f"{run_id}_matrix_b.npy")
            result_path = os.path.join(self.config.output_dir, f"{run_id}_result.npy")
            
            # Save matrices
            np.save(matrix_a_path, matrix_a)
            np.save(matrix_b_path, matrix_b)
            np.save(result_path, result)
            
            matrix_files = {
                "matrix_a": matrix_a_path,
                "matrix_b": matrix_b_path,
                "result": result_path
            }
            
            self.logger.debug(f"Matrices saved for {run_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save matrices for {run_id}: {e}")
        
        return matrix_files
    
    def _run_single_benchmark(self, size: int, seed: int) -> BenchmarkResult:
        """Run a single benchmark for given matrix size and seed."""
        self.logger.info(f"Running benchmark: size={size}, seed={seed}")
        
        # Create run ID
        run_id = f"{self.benchmark_id}_size{size}_seed{seed}"
        
        # Generate matrices once for consistency (same matrices used across all runs)
        matrix_a, matrix_b = generate_matrix_pair(size, seed=seed)
        
        # Save matrices if requested
        matrix_files = None
        if self.config.save_matrices:
            # Calculate result matrix for saving
            result_matrix = multiply_matrices(matrix_a, matrix_b, validate=False)
            matrix_files = self._save_matrices(run_id, matrix_a, matrix_b, result_matrix)
        
        # Progress callback
        def progress_callback(current: int, total: int):
            if current % max(1, total // 4) == 0:
                self.logger.info(f"  Progress: {current}/{total} ({100*current/total:.0f}%)")
        
        # Modified benchmark function to use pre-generated matrices
        def matrix_multiply_with_pregenerated(size: int, seed: int) -> np.ndarray:
            return multiply_matrices(matrix_a, matrix_b, validate=False)
        
        # Run benchmark iterations
        timing_results = run_benchmark_iterations(
            matrix_multiply_with_pregenerated,
            size, seed,
            iterations=self.config.num_runs,
            warmup_runs=self.config.warmup_runs,
            monitor_resources=self.config.monitor_resources,
            progress_callback=progress_callback
        )
        
        # Calculate statistics
        times = [r.execution_time for r in timing_results]
        timing_stats = calculate_statistics(times)
        
        # Calculate performance metrics
        avg_time = timing_stats.mean
        gflops = calculate_gflops(size, avg_time)
        theoretical_ops = 2 * (size ** 3)  # Matrix multiplication operations
        
        # Calculate memory usage
        dtype_map = {'float64': np.float64, 'float32': np.float32}
        dtype = dtype_map.get(self.config.dtype, np.float64)
        matrix_memory_mb = (size * size * np.dtype(dtype).itemsize) / (1024 * 1024)
        
        return BenchmarkResult(
            matrix_size=size,
            seed=seed,
            run_id=run_id,
            timing_stats=timing_stats,
            timing_results=timing_results,
            gflops=gflops,
            theoretical_ops=theoretical_ops,
            matrix_memory_mb=matrix_memory_mb,
            timestamp=time.time(),
            matrix_files=matrix_files
        )
    
    def run_comprehensive_benchmark(self) -> ComprehensiveResults:
        """Run comprehensive benchmark suite."""
        start_time = time.time()
        self.logger.info(f"Starting comprehensive benchmark: {self.benchmark_id}")
        
        # Get system profile
        self.logger.info("Collecting system profile...")
        system_profile = self.system_profiler.get_complete_profile()
        
        # Log system information
        self.logger.info(f"System: {system_profile.hardware.chip_type}")
        self.logger.info(f"Cores: {system_profile.hardware.cpu_cores_performance}P + {system_profile.hardware.cpu_cores_efficiency}E")
        self.logger.info(f"Memory: {system_profile.hardware.memory_total_gb:.1f} GB")
        self.logger.info(f"BLAS: {system_profile.environment.blas_library}")
        
        # Check thermal state
        thermal_state = {}
        if self.config.detect_throttling:
            self.logger.info("Checking thermal throttling...")
            thermal_state = detect_thermal_throttling(
                cpu_freq_samples=10,
                sample_interval=1.0,
                threshold_percent=5.0
            )
            if thermal_state.get('throttling_detected'):
                self.logger.warning("Thermal throttling detected!")
        
        # Run benchmarks
        benchmark_results = []
        total_benchmarks = len(self.config.matrix_sizes) * len(self.config.seeds)
        current_benchmark = 0
        
        for size in self.config.matrix_sizes:
            for seed in self.config.seeds:
                current_benchmark += 1
                self.logger.info(f"Benchmark {current_benchmark}/{total_benchmarks}")
                
                try:
                    result = self._run_single_benchmark(size, seed)
                    benchmark_results.append(result)
                    
                    # Log immediate results
                    self.logger.info(f"  Completed: {result.gflops:.2f} GFLOPS, "
                                   f"{result.timing_stats.mean:.6f}s avg")
                    
                except Exception as e:
                    self.logger.error(f"Benchmark failed for size={size}, seed={seed}: {e}")
                    self.logger.debug(traceback.format_exc())
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(benchmark_results)
        
        # Create comprehensive results
        end_time = time.time()
        results = ComprehensiveResults(
            benchmark_id=self.benchmark_id,
            system_profile=system_profile,
            benchmark_config=self.config,
            benchmark_results=benchmark_results,
            thermal_state=thermal_state,
            summary_statistics=summary_stats,
            timestamp=start_time,
            duration_seconds=end_time - start_time
        )
        
        self.logger.info(f"Benchmark completed in {results.duration_seconds:.1f} seconds")
        
        # Save results if requested
        if self.config.save_results:
            self._save_comprehensive_results(results)
        
        return results
    
    def _calculate_summary_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics across all benchmark results."""
        if not results:
            return {}
        
        # Group by matrix size
        size_groups = {}
        for result in results:
            size = result.matrix_size
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(result)
        
        summary = {
            "total_benchmarks": len(results),
            "matrix_sizes": list(size_groups.keys()),
            "by_size": {},
            "overall": {}
        }
        
        # Calculate statistics by size
        all_gflops = []
        all_times = []
        
        for size, size_results in size_groups.items():
            gflops_values = [r.gflops for r in size_results]
            time_values = [r.timing_stats.mean for r in size_results]
            
            summary["by_size"][size] = {
                "avg_gflops": np.mean(gflops_values),
                "max_gflops": np.max(gflops_values),
                "min_gflops": np.min(gflops_values),
                "std_gflops": np.std(gflops_values),
                "avg_time": np.mean(time_values),
                "min_time": np.min(time_values),
                "max_time": np.max(time_values),
                "sample_count": len(size_results)
            }
            
            all_gflops.extend(gflops_values)
            all_times.extend(time_values)
        
        # Overall statistics
        if all_gflops:
            # Find the size with the highest max GFLOPS
            best_size = max(size_groups.keys(), key=lambda s: summary["by_size"][s]["max_gflops"])
            
            summary["overall"] = {
                "avg_gflops": np.mean(all_gflops),
                "max_gflops": np.max(all_gflops),
                "peak_performance_size": int(best_size),
                "efficiency_score": np.mean(all_gflops) / np.max(all_gflops) if all_gflops else 0
            }
        
        return summary
    
    def _save_comprehensive_results(self, results: ComprehensiveResults) -> None:
        """Save comprehensive results to JSON file."""
        filename = os.path.join(self.config.output_dir, f"{self.benchmark_id}.json")
        
        # Convert to serializable format
        data = {
            "benchmark_id": results.benchmark_id,
            "system_profile": asdict(results.system_profile),
            "benchmark_config": asdict(results.benchmark_config),
            "benchmark_results": [
                {
                    "matrix_size": r.matrix_size,
                    "seed": r.seed,
                    "run_id": r.run_id,
                    "timing_stats": asdict(r.timing_stats),
                    "timing_results": [asdict(tr) for tr in r.timing_results],
                    "gflops": r.gflops,
                    "theoretical_ops": r.theoretical_ops,
                    "matrix_memory_mb": r.matrix_memory_mb,
                    "timestamp": r.timestamp,
                    "matrix_files": r.matrix_files
                }
                for r in results.benchmark_results
            ],
            "thermal_state": results.thermal_state,
            "summary_statistics": results.summary_statistics,
            "timestamp": results.timestamp,
            "duration_seconds": results.duration_seconds
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to: {filename}")
            
            # Also save a human-readable summary
            summary_filename = os.path.join(self.config.output_dir, f"{self.benchmark_id}_summary.txt")
            self._save_human_readable_summary(results, summary_filename)
            self.logger.info(f"Summary saved to: {summary_filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def _save_human_readable_summary(self, results: ComprehensiveResults, filename: str) -> None:
        """Save human-readable summary of benchmark results."""
        with open(filename, 'w') as f:
            f.write(f"Matrix Benchmark Results Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Benchmark ID: {results.benchmark_id}\n")
            f.write(f"Timestamp: {datetime.fromtimestamp(results.timestamp)}\n")
            f.write(f"Duration: {results.duration_seconds:.1f} seconds\n\n")
            
            # System Information
            f.write(f"System Information:\n")
            f.write(f"  Hardware: {results.system_profile.hardware.chip_type}\n")
            f.write(f"  CPU Cores: {results.system_profile.hardware.cpu_cores_performance}P + {results.system_profile.hardware.cpu_cores_efficiency}E\n")
            f.write(f"  Memory: {results.system_profile.hardware.memory_total_gb:.1f} GB\n")
            f.write(f"  BLAS Library: {results.system_profile.environment.blas_library}\n")
            f.write(f"  Thermal State: {results.system_profile.environment.thermal_state}\n")
            f.write(f"  Power State: {results.system_profile.environment.power_state}\n\n")
            
            # Benchmark Configuration
            f.write(f"Benchmark Configuration:\n")
            f.write(f"  Matrix Sizes: {results.benchmark_config.matrix_sizes}\n")
            f.write(f"  Runs per Size: {results.benchmark_config.num_runs}\n")
            f.write(f"  Warmup Runs: {results.benchmark_config.warmup_runs}\n")
            f.write(f"  Seeds: {results.benchmark_config.seeds}\n")
            f.write(f"  Data Type: {results.benchmark_config.dtype}\n\n")
            
            # Performance Summary
            f.write(f"Performance Summary:\n")
            overall = results.summary_statistics.get("overall", {})
            if overall:
                f.write(f"  Overall Avg GFLOPS: {overall['avg_gflops']:.2f}\n")
                f.write(f"  Peak GFLOPS: {overall['max_gflops']:.2f}\n")
                f.write(f"  Peak Performance Size: {overall['peak_performance_size']}\n")
                f.write(f"  Efficiency Score: {overall['efficiency_score']:.3f}\n\n")
            
            # Results by Size
            f.write(f"Results by Matrix Size:\n")
            f.write(f"{'Size':<8} {'Avg GFLOPS':<12} {'Peak GFLOPS':<13} {'Avg Time (s)':<12} {'Samples':<8}\n")
            f.write(f"{'-'*60}\n")
            
            by_size = results.summary_statistics.get("by_size", {})
            for size in sorted(by_size.keys()):
                stats = by_size[size]
                f.write(f"{size:<8} {stats['avg_gflops']:<12.2f} {stats['max_gflops']:<13.2f} "
                       f"{stats['avg_time']:<12.6f} {stats['sample_count']:<8}\n")
            
            # Thermal Information
            if results.thermal_state:
                f.write(f"\nThermal Analysis:\n")
                f.write(f"  Throttling Detected: {results.thermal_state.get('throttling_detected', 'Unknown')}\n")
                if results.thermal_state.get('initial_frequency_mhz'):
                    f.write(f"  Initial CPU Freq: {results.thermal_state['initial_frequency_mhz']:.0f} MHz\n")
                    f.write(f"  Min CPU Freq: {results.thermal_state['min_frequency_mhz']:.0f} MHz\n")
                    f.write(f"  Frequency Drop: {results.thermal_state['frequency_drop_percent']:.1f}%\n")
    
    def display_results(self, results: ComprehensiveResults) -> None:
        """Display benchmark results to console."""
        print("\n" + "="*80)
        print("MATRIX BENCHMARK RESULTS")
        print("="*80)
        
        # System Information
        hw = results.system_profile.hardware
        env = results.system_profile.environment
        
        print(f"\nSystem Information:")
        print(f"  Hardware: {hw.chip_type}")
        print(f"  CPU Cores: {hw.cpu_cores_performance}P + {hw.cpu_cores_efficiency}E")
        print(f"  Memory: {hw.memory_total_gb:.1f} GB")
        print(f"  BLAS Library: {env.blas_library}")
        print(f"  Thermal State: {env.thermal_state}")
        
        # Performance Summary
        overall = results.summary_statistics.get("overall", {})
        if overall:
            print(f"\nOverall Performance:")
            print(f"  Average GFLOPS: {overall['avg_gflops']:.2f}")
            print(f"  Peak GFLOPS: {overall['max_gflops']:.2f}")
            print(f"  Peak Performance Size: {overall['peak_performance_size']}")
            print(f"  Efficiency Score: {overall['efficiency_score']:.3f}")
        
        # Detailed Results
        print(f"\nDetailed Results:")
        print(f"{'Size':<8} {'Avg GFLOPS':<12} {'Peak GFLOPS':<13} {'Avg Time':<12} {'Samples':<8}")
        print("-" * 60)
        
        by_size = results.summary_statistics.get("by_size", {})
        for size in sorted(by_size.keys()):
            stats = by_size[size]
            print(f"{size:<8} {stats['avg_gflops']:<12.2f} {stats['max_gflops']:<13.2f} "
                  f"{stats['avg_time']:<12.6f} {stats['sample_count']:<8}")
        
        # Thermal Information
        if results.thermal_state and results.thermal_state.get('throttling_detected'):
            print(f"\n⚠️  Thermal throttling detected during benchmarks!")
            print(f"   Frequency drop: {results.thermal_state['frequency_drop_percent']:.1f}%")
        
        print("\n" + "="*80)


def load_benchmark_results(filename: str) -> ComprehensiveResults:
    """Load comprehensive benchmark results from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Reconstruct objects (simplified version)
    # Note: Full reconstruction would require rebuilding all dataclasses
    return data


def compare_benchmark_results(results1: ComprehensiveResults, results2: ComprehensiveResults) -> Dict[str, Any]:
    """Compare two benchmark results."""
    comparison = {
        "system_comparison": {},
        "performance_comparison": {},
        "recommendations": []
    }
    
    # Compare systems
    hw1 = results1.system_profile.hardware
    hw2 = results2.system_profile.hardware
    
    comparison["system_comparison"] = {
        "chip_types": (hw1.chip_type, hw2.chip_type),
        "memory": (hw1.memory_total_gb, hw2.memory_total_gb),
        "cores": ((hw1.cpu_cores_performance, hw1.cpu_cores_efficiency),
                 (hw2.cpu_cores_performance, hw2.cpu_cores_efficiency))
    }
    
    # Compare performance
    overall1 = results1.summary_statistics.get("overall", {})
    overall2 = results2.summary_statistics.get("overall", {})
    
    if overall1 and overall2:
        avg_speedup = overall1["avg_gflops"] / overall2["avg_gflops"]
        peak_speedup = overall1["max_gflops"] / overall2["max_gflops"]
        
        comparison["performance_comparison"] = {
            "avg_gflops": (overall1["avg_gflops"], overall2["avg_gflops"]),
            "peak_gflops": (overall1["max_gflops"], overall2["max_gflops"]),
            "avg_speedup": avg_speedup,
            "peak_speedup": peak_speedup
        }
    
    return comparison


def create_default_config() -> BenchmarkConfig:
    """Create default benchmark configuration."""
    return BenchmarkConfig(
        matrix_sizes=[500, 1000, 1500, 2000],
        num_runs=5,
        warmup_runs=1,
        seeds=[42, 123, 456],
        dtype="float64",
        monitor_resources=True,
        detect_throttling=True,
        save_results=True,
        save_matrices=False,
        output_dir="benchmark_results",
        output_prefix="matrix_benchmark"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Matrix Benchmark Orchestrator - Comprehensive cross-machine benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                          # Run with defaults
  python benchmark.py --sizes 1000 2000       # Custom matrix sizes
  python benchmark.py --runs 10 --seeds 42    # More runs, single seed
  python benchmark.py --quick                  # Quick test run
  python benchmark.py --load results.json     # Load and display results
        """
    )
    
    parser.add_argument(
        "--sizes", 
        type=int, 
        nargs="+",
        default=[500, 1000, 1500, 2000],
        help="Matrix sizes to benchmark (default: 500 1000 1500 2000)"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per matrix size (default: 5)"
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)"
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for reproducibility (default: 42 123 456)"
    )
    
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Data type for matrices (default: float64)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    
    parser.add_argument(
        "--output-prefix",
        default="matrix_benchmark",
        help="Prefix for output files (default: matrix_benchmark)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    
    parser.add_argument(
        "--no-thermal",
        action="store_true",
        help="Skip thermal throttling detection"
    )
    
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Skip resource monitoring"
    )
    
    parser.add_argument(
        "--save-matrices",
        action="store_true",
        help="Save actual matrices to .npy files"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run (small sizes, fewer runs)"
    )
    
    parser.add_argument(
        "--load",
        type=str,
        help="Load and display results from JSON file"
    )
    
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("FILE1", "FILE2"),
        help="Compare two benchmark result files"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle load operation
    if args.load:
        try:
            results = load_benchmark_results(args.load)
            print(f"Loaded benchmark results from: {args.load}")
            print(f"Benchmark ID: {results['benchmark_id']}")
            print(f"System: {results['system_profile']['hardware']['chip_type']}")
            print(f"Duration: {results['duration_seconds']:.1f} seconds")
            
            if results['summary_statistics'].get('overall'):
                overall = results['summary_statistics']['overall']
                print(f"Average GFLOPS: {overall['avg_gflops']:.2f}")
                print(f"Peak GFLOPS: {overall['max_gflops']:.2f}")
            
        except Exception as e:
            print(f"Error loading results: {e}", file=sys.stderr)
            return 1
        
        return 0
    
    # Handle comparison operation
    if args.compare:
        try:
            results1 = load_benchmark_results(args.compare[0])
            results2 = load_benchmark_results(args.compare[1])
            
            print(f"Comparing benchmark results:")
            print(f"  File 1: {args.compare[0]}")
            print(f"  File 2: {args.compare[1]}")
            print()
            
            # Basic comparison output
            print("System Comparison:")
            hw1 = results1['system_profile']['hardware']
            hw2 = results2['system_profile']['hardware']
            print(f"  System 1: {hw1['chip_type']}, {hw1['memory_total_gb']:.1f} GB")
            print(f"  System 2: {hw2['chip_type']}, {hw2['memory_total_gb']:.1f} GB")
            
            if (results1['summary_statistics'].get('overall') and 
                results2['summary_statistics'].get('overall')):
                overall1 = results1['summary_statistics']['overall']
                overall2 = results2['summary_statistics']['overall']
                
                speedup = overall1['avg_gflops'] / overall2['avg_gflops']
                print(f"\nPerformance Comparison:")
                print(f"  System 1 avg: {overall1['avg_gflops']:.2f} GFLOPS")
                print(f"  System 2 avg: {overall2['avg_gflops']:.2f} GFLOPS")
                print(f"  Speedup: {speedup:.2f}x")
                
        except Exception as e:
            print(f"Error comparing results: {e}", file=sys.stderr)
            return 1
        
        return 0
    
    # Create benchmark configuration
    if args.quick:
        config = BenchmarkConfig(
            matrix_sizes=[500, 1000],
            num_runs=3,
            warmup_runs=1,
            seeds=[42],
            dtype=args.dtype,
            monitor_resources=not args.no_monitor,
            detect_throttling=not args.no_thermal,
            save_results=not args.no_save,
            save_matrices=args.save_matrices,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix
        )
    else:
        config = BenchmarkConfig(
            matrix_sizes=args.sizes,
            num_runs=args.runs,
            warmup_runs=args.warmup,
            seeds=args.seeds,
            dtype=args.dtype,
            monitor_resources=not args.no_monitor,
            detect_throttling=not args.no_thermal,
            save_results=not args.no_save,
            save_matrices=args.save_matrices,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix
        )
    
    # Set up logging level
    if args.verbose:
        logging.getLogger('matrix_benchmark').setLevel(logging.DEBUG)
    
    # Run benchmark
    try:
        orchestrator = MatrixBenchmarkOrchestrator(config)
        results = orchestrator.run_comprehensive_benchmark()
        orchestrator.display_results(results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Benchmark failed: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())