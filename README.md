# Matrix Benchmark Suite for Mac Hardware

A comprehensive matrix benchmarking system specifically designed for Mac hardware profiling and cross-machine performance comparison. This suite provides reproducible matrix multiplication benchmarks with detailed system profiling, timing analysis, and statistical reporting.

## Overview

The Matrix Benchmark Suite consists of several coordinated modules:

- **`benchmark.py`** - Main orchestrator and command-line interface
- **`matrix_ops.py`** - Matrix operations and performance calculations
- **`timing.py`** - High-precision timing and statistical analysis
- **`system_info.py`** - Comprehensive Mac system profiling
- **`analysis.py`** - Results analysis, visualization, and reporting

## Quick Start

### Basic Usage

Run a quick benchmark test:
```bash
python benchmark.py --quick
```

Run with default settings:
```bash
python benchmark.py
```

### Custom Benchmark

Specify matrix sizes and run parameters:
```bash
python benchmark.py --sizes 1000 2000 3000 --runs 10 --seeds 42 123 456
```

### Load and View Results

```bash
python benchmark.py --load benchmark_results/matrix_benchmark_20231215_143022.json
```

### Compare Results

```bash
python benchmark.py --compare file1.json file2.json
```

### Analyze Results

```bash
# Interactive analysis
python -m analysis --interactive

# Generate reports
python -m analysis --report-html my_report.html --plot-scaling
python -m analysis --export-csv benchmark_data.csv
```

## Command-Line Options

```
python benchmark.py [OPTIONS]

Options:
  --sizes SIZE [SIZE ...]       Matrix sizes to benchmark (default: 500 1000 1500 2000)
  --runs RUNS                   Number of runs per size (default: 5)
  --warmup WARMUP               Number of warmup runs (default: 1)
  --seeds SEED [SEED ...]       Random seeds for reproducibility (default: 42 123 456)
  --dtype {float32,float64}     Data type for matrices (default: float64)
  --output-dir DIR              Output directory (default: benchmark_results)
  --output-prefix PREFIX        Output file prefix (default: matrix_benchmark)
  --no-save                     Don't save results to file
  --no-thermal                  Skip thermal throttling detection
  --no-monitor                  Skip resource monitoring
  --quick                       Quick test run (small sizes, fewer runs)
  --load FILE                   Load and display results from JSON file
  --compare FILE1 FILE2         Compare two benchmark result files
  --verbose, -v                 Verbose output
```

## Example Workflows

### Performance Testing
```bash
# Quick performance check
python benchmark.py --quick

# Comprehensive benchmark
python benchmark.py --sizes 500 1000 1500 2000 2500 --runs 10

# High-precision benchmark with multiple seeds
python benchmark.py --sizes 1000 2000 --runs 20 --seeds 42 123 456 789 999
```

### Cross-Machine Comparison
```bash
# Machine 1
python benchmark.py --output-prefix machine1 --sizes 1000 2000 3000

# Machine 2  
python benchmark.py --output-prefix machine2 --sizes 1000 2000 3000

# Compare results
python benchmark.py --compare benchmark_results/machine1_*.json benchmark_results/machine2_*.json
```

### Development Testing
```bash
# Fast iteration during development
python benchmark.py --quick --no-thermal --no-monitor

# Test specific scenarios
python benchmark.py --sizes 1000 --runs 3 --seeds 42 --dtype float32
```

## Programmatic Usage

The benchmark orchestrator can be used programmatically in Python:

```python
from benchmark import MatrixBenchmarkOrchestrator, BenchmarkConfig

# Create custom configuration
config = BenchmarkConfig(
    matrix_sizes=[1000, 2000],
    num_runs=5,
    warmup_runs=1,
    seeds=[42, 123],
    dtype="float64",
    monitor_resources=True,
    detect_throttling=True,
    save_results=True,
    output_dir="my_results",
    output_prefix="my_benchmark"
)

# Run benchmark
orchestrator = MatrixBenchmarkOrchestrator(config)
results = orchestrator.run_comprehensive_benchmark()

# Display results
orchestrator.display_results(results)

# Access detailed data
for result in results.benchmark_results:
    print(f"Size {result.matrix_size}: {result.gflops:.2f} GFLOPS")
```

## Output Files

The benchmark generates several output files:

### JSON Results File
Comprehensive machine-readable results including:
- Complete system profile (hardware, environment, performance baselines)
- Detailed timing results for each benchmark run
- Statistical analysis (mean, std dev, confidence intervals, outliers)
- Performance metrics (GFLOPS, efficiency scores)
- Configuration and metadata

### Human-Readable Summary
Text summary including:
- System information and configuration
- Performance summary and statistics
- Results breakdown by matrix size
- Thermal analysis (if enabled)

## System Profiling

The benchmark automatically collects comprehensive system information:

### Hardware Profile
- Mac model and Apple Silicon chip type (M1/M2/M3, Pro/Max/Ultra)
- CPU core counts (performance + efficiency cores)
- Memory capacity and estimated bandwidth
- GPU information and cache sizes
- Hardware capability score

### Environment Profile
- Operating system version and Python environment
- NumPy version and BLAS library detection
- CPU frequency and thermal state
- Power state (battery/AC) and system load
- Background process count

### Performance Baseline
- CPU benchmark score from matrix operations
- Measured memory bandwidth
- Cache performance characteristics
- Thermal throttling detection
- Sustained performance ratio

## Benchmark Features

### Reproducible Results
- Deterministic random seed control
- Multiple runs with statistical analysis
- Comprehensive system profiling for comparison context
- Standardized matrix operations and timing methodology

### Statistical Analysis
- Mean, median, standard deviation calculations
- Confidence intervals and outlier detection
- Coefficient of variation for stability assessment
- Performance scaling analysis across matrix sizes

### Cross-Machine Comparison
- Normalized hardware capability scoring
- System compatibility assessment
- Performance difference quantification
- Recommendation generation for benchmark validity

### Thermal Management
- Thermal throttling detection during benchmarks
- CPU frequency monitoring
- Performance degradation analysis
- Recommendations for thermal considerations

## Best Practices

### For Accurate Results
1. Close unnecessary applications
2. Ensure stable power supply (AC power recommended)
3. Allow system to cool down between major benchmark runs
4. Use multiple seeds for statistical validity
5. Run benchmarks multiple times and compare results

### For Cross-Machine Comparison
1. Use identical benchmark configurations
2. Ensure similar thermal conditions
3. Document any environmental differences
4. Compare results from similar time periods
5. Account for different BLAS library implementations

### For Development
1. Use `--quick` flag for rapid iteration
2. Disable resource monitoring for faster runs with `--no-monitor`
3. Skip thermal detection during development with `--no-thermal`
4. Use smaller matrix sizes for initial testing

## Results Analysis

The analysis module (`analysis.py`) provides comprehensive tools for analyzing benchmark results:

### Features
- **Data Visualization**: Performance scaling plots, timing distributions, system comparisons
- **Statistical Analysis**: Outlier detection, significance testing, performance ratios
- **Report Generation**: HTML reports with embedded charts, Markdown summaries, CSV exports
- **Interactive Analysis**: Menu-driven exploration of results

### Quick Start
```bash
# Interactive analysis mode
python -m analysis --interactive

# Generate comprehensive HTML report
python -m analysis --report-html report.html --plot-scaling --plot-comparison

# Export data for further analysis
python -m analysis --export-csv data.csv
```

### Analysis Capabilities
- Cross-machine performance comparison
- Statistical significance testing
- Performance scaling analysis
- Outlier detection and investigation
- Professional report generation

See `ANALYSIS_GUIDE.md` for detailed documentation.

## Dependencies

### Core Dependencies
- Python 3.7+
- NumPy
- psutil (for system monitoring)
- scipy (for statistical analysis)
- macOS (for system profiling features)

### Analysis Dependencies (Optional)
- matplotlib (for plotting)
- seaborn (for statistical visualization)
- pandas (for data manipulation)
- jinja2 (for HTML report generation)

Install all dependencies:
```bash
pip install -r requirements.txt
```

Or install core only:
```bash
pip install numpy psutil scipy
```

## Module Architecture

The benchmark suite is designed as a modular system:

### Core Modules
- **matrix_ops.py**: Matrix generation and multiplication operations
- **timing.py**: Precision timing and statistical analysis
- **system_info.py**: Mac hardware and system profiling
- **benchmark.py**: Main orchestrator coordinating all modules
- **analysis.py**: Results analysis, visualization, and reporting

### Integration
Each module can be used independently or together through the main orchestrator. The orchestrator provides:
- Configuration management
- Cross-module coordination  
- Result aggregation and analysis
- Command-line interface
- File I/O operations

## Performance Expectations

Typical performance ranges for different Mac hardware:

### M1 MacBook Air/Pro
- Small matrices (500x500): 15-25 GFLOPS
- Medium matrices (1000x1000): 30-50 GFLOPS  
- Large matrices (2000x2000): 40-70 GFLOPS

### M1 Pro/Max
- Small matrices (500x500): 20-35 GFLOPS
- Medium matrices (1000x1000): 50-80 GFLOPS
- Large matrices (2000x2000): 70-120 GFLOPS

### M2/M3 Series
- Generally 10-20% improvement over equivalent M1 hardware
- Better sustained performance under thermal load

*Note: Actual performance depends on many factors including thermal conditions, power settings, memory configuration, and BLAS library implementation.*

## Troubleshooting

### Common Issues

**Import errors**: Ensure all dependencies are installed
```bash
pip install numpy psutil
```

**Permission errors**: Ensure write access to output directory
```bash
chmod 755 benchmark_results/
```

**Thermal throttling warnings**: Allow system to cool down or run with `--no-thermal`

**Resource monitoring errors**: Install or update psutil
```bash
pip install --upgrade psutil
```

### Performance Issues

**Lower than expected performance**:
- Check thermal throttling warnings
- Verify power management settings
- Close other applications
- Check BLAS library implementation

**Inconsistent results**:
- Increase number of runs (`--runs`)
- Use multiple seeds for statistical analysis  
- Check for background system activity
- Ensure stable thermal conditions

## Contributing

This benchmark suite is designed for extensibility. Key areas for contribution:

1. **Additional matrix operations** (eigenvalue decomposition, SVD, etc.)
2. **More system profiling metrics** (GPU utilization, memory hierarchy details)
3. **Enhanced statistical analysis** (distribution fitting, performance modeling)
4. **Cross-platform support** (Linux, Windows system profiling)
5. **Advanced visualization** (interactive dashboards, real-time monitoring)
6. **Analysis features** (machine learning models, performance prediction)

## License

This project is released under the MIT License. See LICENSE file for details.