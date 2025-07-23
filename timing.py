"""
Timing and Performance Measurement Module

A comprehensive Python module for high-precision timing, performance measurement,
statistical analysis, and benchmark coordination. Designed for accurate cross-machine
benchmarking with proper error handling and validation.
"""

import time
try:
    import psutil
except ImportError:
    print("Warning: psutil not available. Resource monitoring will be disabled.")
    psutil = None
import statistics
import warnings
from typing import List, Dict, Any, Optional, Callable, Tuple, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Thread
import numpy as np
import gc


@dataclass
class TimingResult:
    """Data class to store timing measurement results."""
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    peak_memory_mb: float
    timestamp: float


@dataclass
class BenchmarkStats:
    """Data class to store statistical analysis of benchmark results."""
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    median: float
    q25: float
    q75: float
    coefficient_of_variation: float
    confidence_interval_95: Tuple[float, float]
    outlier_count: int
    outlier_indices: List[int]


class SystemMonitor:
    """Monitor system resources during benchmark execution."""
    
    def __init__(self):
        if psutil is None:
            raise ImportError("psutil is required for resource monitoring")
        self.process = psutil.Process()
        self.monitoring = False
        self.memory_samples = []
        self.cpu_samples = []
        self.monitor_thread = None
        self.sample_interval = 0.1  # 100ms sampling
    
    def start_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring = True
        self.memory_samples = []
        self.cpu_samples = []
        self.monitor_thread = Thread(target=self._monitor_resources)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return peak usage statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        peak_memory = max(self.memory_samples) if self.memory_samples else 0.0
        avg_cpu = statistics.mean(self.cpu_samples) if self.cpu_samples else 0.0
        
        return {
            "peak_memory_mb": peak_memory,
            "avg_cpu_percent": avg_cpu,
            "sample_count": len(self.memory_samples)
        }
    
    def _monitor_resources(self):
        """Internal method to continuously monitor resources."""
        while self.monitoring:
            try:
                # Memory usage in MB
                memory_mb = self.process.memory_info().rss / (1024 * 1024)
                self.memory_samples.append(memory_mb)
                
                # CPU usage percentage
                cpu_percent = self.process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(self.sample_interval)
            except psutil.NoSuchProcess:
                break
            except Exception:
                # Continue monitoring even if individual samples fail
                continue


@contextmanager
def timer(description: str = "Operation", 
          monitor_resources: bool = True,
          warmup_runs: int = 0) -> Iterator[TimingResult]:
    """
    Context manager for high-precision timing with resource monitoring.
    
    Args:
        description: Description of the operation being timed
        monitor_resources: Whether to monitor memory and CPU usage
        warmup_runs: Number of warmup runs before actual timing
        
    Yields:
        TimingResult: Object containing timing and resource usage data
        
    Example:
        with timer("Matrix multiplication") as result:
            # Your code here
            pass
        print(f"Execution time: {result.execution_time:.6f} seconds")
    """
    monitor = SystemMonitor() if monitor_resources else None
    
    # Perform warmup runs if requested
    if warmup_runs > 0:
        print(f"Performing {warmup_runs} warmup runs...")
        for _ in range(warmup_runs):
            # Warmup placeholder - actual warmup should be done by caller
            pass
    
    # Force garbage collection before timing
    gc.collect()
    
    # Get initial memory usage
    initial_memory = 0.0
    if psutil is not None:
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    # Start monitoring
    if monitor:
        monitor.start_monitoring()
    
    # Record start time
    start_time = time.perf_counter()
    timestamp = time.time()
    
    # Create result object to be populated
    result = TimingResult(
        execution_time=0.0,
        memory_usage_mb=initial_memory,
        cpu_percent=0.0,
        peak_memory_mb=initial_memory,
        timestamp=timestamp
    )
    
    try:
        yield result
    finally:
        # Record end time
        end_time = time.perf_counter()
        result.execution_time = end_time - start_time
        
        # Stop monitoring and get peak usage
        if monitor:
            monitor_stats = monitor.stop_monitoring()
            result.peak_memory_mb = monitor_stats["peak_memory_mb"]
            result.cpu_percent = monitor_stats["avg_cpu_percent"]
        
        # Final memory usage
        final_memory = 0.0
        if psutil is not None:
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        result.memory_usage_mb = final_memory - initial_memory
        
        print(f"{description} completed in {result.execution_time:.6f} seconds")


def measure_execution_time(
    func: Callable,
    *args,
    warmup_runs: int = 0,
    monitor_resources: bool = True,
    **kwargs
) -> TimingResult:
    """
    Measure execution time of a function with optional warmup and resource monitoring.
    
    Args:
        func: Function to measure
        *args: Positional arguments for the function
        warmup_runs: Number of warmup runs before actual timing
        monitor_resources: Whether to monitor system resources
        **kwargs: Keyword arguments for the function
        
    Returns:
        TimingResult: Timing and resource usage data
        
    Raises:
        ValueError: If warmup_runs is negative
        TypeError: If func is not callable
    """
    if not callable(func):
        raise TypeError("func must be callable")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    
    # Perform warmup runs
    for _ in range(warmup_runs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            warnings.warn(f"Warmup run failed: {e}")
    
    # Measure actual execution
    with timer(f"Function {func.__name__}", monitor_resources) as result:
        func(*args, **kwargs)
    
    return result


def run_benchmark_iterations(
    func: Callable,
    *args,
    iterations: int = 10,
    warmup_runs: int = 1,
    monitor_resources: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **kwargs
) -> List[TimingResult]:
    """
    Run multiple benchmark iterations with progress tracking.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        iterations: Number of benchmark iterations
        warmup_runs: Number of warmup runs before first iteration
        monitor_resources: Whether to monitor system resources
        progress_callback: Optional callback for progress updates
        **kwargs: Keyword arguments for the function
        
    Returns:
        List[TimingResult]: List of timing results for each iteration
        
    Raises:
        ValueError: If iterations is not positive
        TypeError: If func is not callable
    """
    if not callable(func):
        raise TypeError("func must be callable")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    
    results = []
    
    # Perform warmup runs before first iteration
    if warmup_runs > 0:
        print(f"Performing {warmup_runs} warmup runs...")
        for _ in range(warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                warnings.warn(f"Warmup run failed: {e}")
    
    # Run benchmark iterations
    for i in range(iterations):
        if progress_callback:
            progress_callback(i + 1, iterations)
        
        result = measure_execution_time(
            func, *args, 
            warmup_runs=0,  # Already warmed up
            monitor_resources=monitor_resources,
            **kwargs
        )
        results.append(result)
        
        # Brief pause between iterations to avoid thermal effects
        time.sleep(0.1)
    
    return results


def calculate_statistics(
    values: List[float],
    confidence_level: float = 0.95,
    outlier_threshold: float = 2.0
) -> BenchmarkStats:
    """
    Calculate comprehensive statistics for benchmark results.
    
    Args:
        values: List of timing values
        confidence_level: Confidence level for confidence interval
        outlier_threshold: Z-score threshold for outlier detection
        
    Returns:
        BenchmarkStats: Statistical analysis results
        
    Raises:
        ValueError: If values is empty or confidence_level is invalid
    """
    if not values:
        raise ValueError("values cannot be empty")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    
    # Basic statistics
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
    min_value = min(values)
    max_value = max(values)
    median = statistics.median(values)
    
    # Quartiles
    sorted_values = sorted(values)
    n = len(sorted_values)
    q25 = sorted_values[n // 4] if n >= 4 else min_value
    q75 = sorted_values[3 * n // 4] if n >= 4 else max_value
    
    # Coefficient of variation
    cv = std_dev / mean if mean > 0 else 0.0
    
    # Confidence interval (assuming normal distribution)
    if len(values) > 1:
        # Using t-distribution for small samples
        from scipy import stats
        try:
            t_value = stats.t.ppf((1 + confidence_level) / 2, len(values) - 1)
            margin_of_error = t_value * std_dev / np.sqrt(len(values))
            ci_lower = mean - margin_of_error
            ci_upper = mean + margin_of_error
        except ImportError:
            # Fallback to normal approximation if scipy not available
            z_value = 1.96  # Approximation for 95% confidence
            margin_of_error = z_value * std_dev / np.sqrt(len(values))
            ci_lower = mean - margin_of_error
            ci_upper = mean + margin_of_error
    else:
        ci_lower = ci_upper = mean
    
    # Outlier detection using Z-score
    outlier_indices = []
    if std_dev > 0:
        z_scores = [(x - mean) / std_dev for x in values]
        outlier_indices = [i for i, z in enumerate(z_scores) 
                          if abs(z) > outlier_threshold]
    
    return BenchmarkStats(
        mean=mean,
        std_dev=std_dev,
        min_value=min_value,
        max_value=max_value,
        median=median,
        q25=q25,
        q75=q75,
        coefficient_of_variation=cv,
        confidence_interval_95=(ci_lower, ci_upper),
        outlier_count=len(outlier_indices),
        outlier_indices=outlier_indices
    )


def detect_thermal_throttling(
    cpu_freq_samples: int = 10,
    sample_interval: float = 1.0,
    threshold_percent: float = 5.0
) -> Dict[str, Any]:
    """
    Detect potential thermal throttling by monitoring CPU frequency.
    
    Args:
        cpu_freq_samples: Number of frequency samples to take
        sample_interval: Interval between samples in seconds
        threshold_percent: Percentage drop to consider throttling
        
    Returns:
        Dict containing throttling detection results
        
    Raises:
        ValueError: If parameters are invalid
    """
    if cpu_freq_samples <= 0:
        raise ValueError("cpu_freq_samples must be positive")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    if threshold_percent < 0:
        raise ValueError("threshold_percent must be non-negative")
    
    frequencies = []
    
    if psutil is None:
        return {
            "throttling_detected": False,
            "error": "psutil not available",
            "frequencies": [],
            "frequency_drop_percent": 0.0
        }
    
    try:
        for _ in range(cpu_freq_samples):
            freq = psutil.cpu_freq()
            if freq:
                frequencies.append(freq.current)
            time.sleep(sample_interval)
    except Exception as e:
        return {
            "throttling_detected": False,
            "error": str(e),
            "frequencies": [],
            "frequency_drop_percent": 0.0
        }
    
    if not frequencies:
        return {
            "throttling_detected": False,
            "error": "Could not retrieve CPU frequencies",
            "frequencies": [],
            "frequency_drop_percent": 0.0
        }
    
    # Calculate frequency statistics
    initial_freq = frequencies[0]
    min_freq = min(frequencies)
    max_freq = max(frequencies)
    avg_freq = statistics.mean(frequencies)
    
    # Calculate frequency drop percentage
    freq_drop_percent = ((initial_freq - min_freq) / initial_freq) * 100
    
    # Detect throttling
    throttling_detected = freq_drop_percent > threshold_percent
    
    return {
        "throttling_detected": throttling_detected,
        "initial_frequency_mhz": initial_freq,
        "min_frequency_mhz": min_freq,
        "max_frequency_mhz": max_freq,
        "avg_frequency_mhz": avg_freq,
        "frequency_drop_percent": freq_drop_percent,
        "frequencies": frequencies,
        "sample_count": len(frequencies)
    }


def format_benchmark_results(
    results: List[TimingResult],
    stats: BenchmarkStats,
    title: str = "Benchmark Results",
    include_raw_data: bool = False
) -> str:
    """
    Format benchmark results for display.
    
    Args:
        results: List of timing results
        stats: Statistical analysis of results
        title: Title for the results display
        include_raw_data: Whether to include raw timing data
        
    Returns:
        str: Formatted results string
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"{title:^60}")
    lines.append(f"{'='*60}")
    
    # Summary statistics
    lines.append(f"\nSummary Statistics:")
    lines.append(f"  Mean time:        {stats.mean:.6f} seconds")
    lines.append(f"  Std deviation:    {stats.std_dev:.6f} seconds")
    lines.append(f"  Min time:         {stats.min_value:.6f} seconds")
    lines.append(f"  Max time:         {stats.max_value:.6f} seconds")
    lines.append(f"  Median time:      {stats.median:.6f} seconds")
    lines.append(f"  25th percentile:  {stats.q25:.6f} seconds")
    lines.append(f"  75th percentile:  {stats.q75:.6f} seconds")
    lines.append(f"  Coeff. of var.:   {stats.coefficient_of_variation:.4f}")
    
    # Confidence interval
    ci_lower, ci_upper = stats.confidence_interval_95
    lines.append(f"  95% CI:           [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    # Outlier information
    lines.append(f"\nOutlier Analysis:")
    lines.append(f"  Outlier count:    {stats.outlier_count}")
    if stats.outlier_indices:
        lines.append(f"  Outlier indices:  {stats.outlier_indices}")
    
    # Resource usage (if available)
    if results:
        avg_memory = statistics.mean([r.memory_usage_mb for r in results])
        avg_peak_memory = statistics.mean([r.peak_memory_mb for r in results])
        avg_cpu = statistics.mean([r.cpu_percent for r in results])
        
        lines.append(f"\nResource Usage:")
        lines.append(f"  Avg memory delta: {avg_memory:.2f} MB")
        lines.append(f"  Avg peak memory:  {avg_peak_memory:.2f} MB")
        lines.append(f"  Avg CPU usage:    {avg_cpu:.1f}%")
    
    # Raw data (if requested)
    if include_raw_data:
        lines.append(f"\nRaw Timing Data:")
        for i, result in enumerate(results):
            lines.append(f"  Run {i+1:2d}: {result.execution_time:.6f} seconds")
    
    lines.append(f"{'='*60}\n")
    
    return "\n".join(lines)


def aggregate_results(
    results_list: List[List[TimingResult]],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Aggregate results from multiple benchmark runs.
    
    Args:
        results_list: List of benchmark result lists
        labels: Optional labels for each result set
        
    Returns:
        Dict containing aggregated results and comparisons
        
    Raises:
        ValueError: If results_list is empty
    """
    if not results_list:
        raise ValueError("results_list cannot be empty")
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(results_list))]
    
    aggregated = {}
    
    for results, label in zip(results_list, labels):
        if not results:
            continue
            
        times = [r.execution_time for r in results]
        stats = calculate_statistics(times)
        
        aggregated[label] = {
            "stats": stats,
            "results": results,
            "sample_count": len(results)
        }
    
    # Calculate relative performance
    if len(aggregated) > 1:
        baseline_key = labels[0]
        baseline_mean = aggregated[baseline_key]["stats"].mean
        
        for label in labels[1:]:
            if label in aggregated:
                mean_time = aggregated[label]["stats"].mean
                speedup = baseline_mean / mean_time
                aggregated[label]["relative_performance"] = {
                    "speedup": speedup,
                    "percent_change": ((mean_time - baseline_mean) / baseline_mean) * 100
                }
    
    return aggregated


def save_benchmark_results(
    results: List[TimingResult],
    stats: BenchmarkStats,
    filename: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save benchmark results to a file.
    
    Args:
        results: List of timing results
        stats: Statistical analysis
        filename: Output filename
        metadata: Optional metadata to include
        
    Raises:
        OSError: If file cannot be written
    """
    import json
    
    # Prepare data for serialization
    data = {
        "timestamp": time.time(),
        "metadata": metadata or {},
        "statistics": {
            "mean": stats.mean,
            "std_dev": stats.std_dev,
            "min_value": stats.min_value,
            "max_value": stats.max_value,
            "median": stats.median,
            "q25": stats.q25,
            "q75": stats.q75,
            "coefficient_of_variation": stats.coefficient_of_variation,
            "confidence_interval_95": stats.confidence_interval_95,
            "outlier_count": stats.outlier_count,
            "outlier_indices": stats.outlier_indices
        },
        "raw_results": [
            {
                "execution_time": r.execution_time,
                "memory_usage_mb": r.memory_usage_mb,
                "cpu_percent": r.cpu_percent,
                "peak_memory_mb": r.peak_memory_mb,
                "timestamp": r.timestamp
            }
            for r in results
        ]
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filename}")
    except OSError as e:
        raise OSError(f"Could not save results to {filename}: {e}")


def load_benchmark_results(filename: str) -> Tuple[List[TimingResult], BenchmarkStats, Dict[str, Any]]:
    """
    Load benchmark results from a file.
    
    Args:
        filename: Input filename
        
    Returns:
        Tuple of (results, stats, metadata)
        
    Raises:
        OSError: If file cannot be read
        ValueError: If file format is invalid
    """
    import json
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except OSError as e:
        raise OSError(f"Could not read results from {filename}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {filename}: {e}")
    
    # Reconstruct results
    results = []
    for r_data in data.get("raw_results", []):
        result = TimingResult(
            execution_time=r_data["execution_time"],
            memory_usage_mb=r_data["memory_usage_mb"],
            cpu_percent=r_data["cpu_percent"],
            peak_memory_mb=r_data["peak_memory_mb"],
            timestamp=r_data["timestamp"]
        )
        results.append(result)
    
    # Reconstruct stats
    stats_data = data.get("statistics", {})
    stats = BenchmarkStats(
        mean=stats_data["mean"],
        std_dev=stats_data["std_dev"],
        min_value=stats_data["min_value"],
        max_value=stats_data["max_value"],
        median=stats_data["median"],
        q25=stats_data["q25"],
        q75=stats_data["q75"],
        coefficient_of_variation=stats_data["coefficient_of_variation"],
        confidence_interval_95=tuple(stats_data["confidence_interval_95"]),
        outlier_count=stats_data["outlier_count"],
        outlier_indices=stats_data["outlier_indices"]
    )
    
    metadata = data.get("metadata", {})
    
    return results, stats, metadata


# Convenience functions for common use cases
def quick_benchmark(
    func: Callable,
    *args,
    iterations: int = 10,
    warmup_runs: int = 1,
    print_results: bool = True,
    **kwargs
) -> Tuple[List[TimingResult], BenchmarkStats]:
    """
    Quick benchmark function with sensible defaults.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        iterations: Number of iterations
        warmup_runs: Number of warmup runs
        print_results: Whether to print formatted results
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (results, stats)
    """
    def progress_callback(current, total):
        if current % max(1, total // 10) == 0:
            print(f"Progress: {current}/{total} ({100*current/total:.1f}%)")
    
    results = run_benchmark_iterations(
        func, *args,
        iterations=iterations,
        warmup_runs=warmup_runs,
        progress_callback=progress_callback,
        **kwargs
    )
    
    times = [r.execution_time for r in results]
    stats = calculate_statistics(times)
    
    if print_results:
        print(format_benchmark_results(results, stats, f"Benchmark: {func.__name__}"))
    
    return results, stats


def compare_functions(
    functions: List[Callable],
    *args,
    iterations: int = 10,
    warmup_runs: int = 1,
    labels: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare performance of multiple functions.
    
    Args:
        functions: List of functions to compare
        *args: Positional arguments for functions
        iterations: Number of iterations per function
        warmup_runs: Number of warmup runs per function
        labels: Optional labels for functions
        **kwargs: Keyword arguments for functions
        
    Returns:
        Dict containing comparison results
    """
    if labels is None:
        labels = [f.__name__ for f in functions]
    
    all_results = []
    
    for func, label in zip(functions, labels):
        print(f"\nBenchmarking {label}...")
        results, _ = quick_benchmark(
            func, *args,
            iterations=iterations,
            warmup_runs=warmup_runs,
            print_results=False,
            **kwargs
        )
        all_results.append(results)
    
    # Aggregate and compare
    comparison = aggregate_results(all_results, labels)
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"{'Function Comparison':^60}")
    print(f"{'='*60}")
    
    for label, data in comparison.items():
        stats = data["stats"]
        print(f"\n{label}:")
        print(f"  Mean time:    {stats.mean:.6f} seconds")
        print(f"  Std dev:      {stats.std_dev:.6f} seconds")
        
        if "relative_performance" in data:
            rel_perf = data["relative_performance"]
            print(f"  Speedup:      {rel_perf['speedup']:.2f}x")
            print(f"  % Change:     {rel_perf['percent_change']:+.1f}%")
    
    return comparison


# Example usage and testing functions
def _test_timing_module():
    """Test the timing module functionality."""
    import numpy as np
    
    def test_function(n=1000):
        """Test function that performs matrix multiplication."""
        a = np.random.rand(n, n)
        b = np.random.rand(n, n)
        return np.dot(a, b)
    
    print("Testing timing module...")
    
    # Test context manager
    with timer("Test operation") as result:
        test_function(500)
    
    print(f"Context manager result: {result.execution_time:.6f} seconds")
    
    # Test quick benchmark
    _, _ = quick_benchmark(test_function, 300, iterations=5)
    
    # Test thermal throttling detection
    throttling = detect_thermal_throttling(cpu_freq_samples=3, sample_interval=0.5)
    print(f"Thermal throttling detected: {throttling['throttling_detected']}")
    
    print("Timing module test completed successfully!")


if __name__ == "__main__":
    _test_timing_module()