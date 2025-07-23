# Matrix Benchmark Analysis Module Guide

The `analysis.py` module provides comprehensive analysis capabilities for matrix benchmark results, including data visualization, comparative analysis, report generation, and interactive analysis functions.

## Features

### 1. Data Visualization
- **Performance scaling plots** (GFLOPS vs matrix size)
- **Cross-machine comparison charts**
- **Statistical distribution plots** (timing histograms)
- **System performance profiles**

### 2. Comparative Analysis
- Compare results from different machines
- Performance ratio calculations
- Efficiency scoring and ranking
- Statistical significance testing

### 3. Report Generation
- **HTML reports** with embedded charts
- **Markdown summaries** for documentation
- **CSV exports** for further analysis
- **Executive summary** generation

### 4. Interactive Analysis
- Functions to load and analyze benchmark results
- Filtering by matrix size, system type, etc.
- Performance trend analysis
- Outlier detection and investigation

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `matplotlib` (visualization)
- `seaborn` (statistical plotting)
- `pandas` (data manipulation, optional)
- `scipy` (statistical analysis)
- `jinja2` (HTML report generation)
- `numpy` (numerical computing)

## Usage

### Command Line Interface

#### Basic Usage

```bash
# Show analysis summary
python -m analysis --results-dir demo_results

# Interactive mode
python -m analysis --interactive

# Generate specific reports
python -m analysis --report-html my_report.html
python -m analysis --report-md my_report.md
python -m analysis --export-csv my_data.csv

# Generate plots
python -m analysis --plot-scaling
python -m analysis --plot-distributions
python -m analysis --plot-comparison

# Combine multiple operations
python -m analysis --plot-scaling --plot-comparison --report-html full_report.html
```

#### Command Line Options

- `--results-dir, -d`: Directory containing benchmark results (default: `benchmark_results`)
- `--pattern, -p`: File pattern for result files (default: `*.json`)
- `--interactive, -i`: Start interactive analysis session
- `--plot-scaling`: Generate performance scaling plot
- `--plot-distributions`: Generate timing distribution plots
- `--plot-comparison`: Generate system comparison plots
- `--export-csv FILE`: Export data to CSV file
- `--report-md FILE`: Generate Markdown report
- `--report-html FILE`: Generate HTML report
- `--output-dir, -o`: Output directory for generated files

### Python API

#### Basic Analysis

```python
from analysis import BenchmarkAnalyzer

# Initialize analyzer
analyzer = BenchmarkAnalyzer("demo_results")

# Load results
results = analyzer.load_results("*.json")

# Get system information
systems = analyzer.get_system_summary()
print(f"Found {len(systems)} systems")

# Extract performance data
data = analyzer.extract_performance_data()
print(f"Performance range: {min(data['gflops']):.2f} - {max(data['gflops']):.2f} GFLOPS")
```

#### Generate Reports

```python
# Generate Markdown report
analyzer.generate_markdown_report("my_report.md")

# Generate HTML report with embedded plots
analyzer.generate_html_report("my_report.html", include_plots=True)

# Export to CSV
analyzer.export_to_csv("benchmark_data.csv")
```

#### Create Visualizations

```python
# Performance scaling plot
analyzer.plot_performance_scaling(save_path="scaling.png")

# Timing distributions
analyzer.plot_timing_distributions(save_path="distributions.png")

# System comparison
analyzer.plot_system_comparison(save_path="comparison.png")
```

#### Advanced Analysis

```python
# Calculate performance ratios between systems
ratios = analyzer.calculate_performance_ratios()
print("Performance ranking:", ratios['ranking'])

# Detect outliers
outliers = analyzer.detect_outliers(method='iqr', threshold=1.5)
print(f"Found {outliers['count']} outliers")

# Statistical significance testing
stats = analyzer.statistical_significance_test()
if 'tests' in stats:
    for test in stats['tests']:
        print(f"{test['system1']} vs {test['system2']}: p-value = {test['t_test']['p_value']:.4f}")
```

## Interactive Mode

The interactive mode provides a menu-driven interface for exploring benchmark results:

```bash
python -m analysis --interactive
```

Interactive options include:
1. Show system summary
2. Show performance statistics
3. Compare systems
4. Detect outliers
5. Generate scaling plot
6. Generate timing distributions
7. Generate system comparison
8. Export to CSV
9. Generate Markdown report
10. Generate HTML report

## Analysis Features

### System Information

The analyzer automatically extracts system information from benchmark results:

- **Chip type** (e.g., M1 Pro, Intel Core i7)
- **Memory configuration**
- **CPU core counts** (performance vs efficiency cores)
- **Hardware scores**
- **Results count per system**

### Performance Metrics

Key performance metrics analyzed:

- **GFLOPS** (Giga Floating Point Operations Per Second)
- **Execution times** (mean, std dev, min, max)
- **Memory usage** during benchmarks
- **Performance scaling** across matrix sizes
- **Efficiency scores** (performance per resource unit)

### Statistical Analysis

- **Descriptive statistics** (mean, median, std dev, quartiles)
- **Outlier detection** using IQR or Z-score methods
- **Statistical significance testing** (t-tests, Mann-Whitney U, ANOVA)
- **Confidence intervals** for performance estimates
- **Coefficient of variation** for performance consistency

### Visualization Types

#### Performance Scaling Plot
- Scatter plot of GFLOPS vs matrix size
- Trend lines for each system
- Performance distribution comparison

#### Timing Distribution Plots
- Histograms of execution times
- GFLOPS distribution
- Memory usage distribution
- Performance vs matrix size scatter plot

#### System Comparison Plots
- Average performance comparison (bar charts)
- Performance vs matrix size by system
- Execution time comparison
- Memory usage comparison
- Performance efficiency box plots

## Report Formats

### Markdown Reports

Comprehensive text-based reports including:
- Executive summary with key findings
- System information and specifications
- Performance analysis and statistics
- Matrix size analysis
- Outlier detection results
- Recommendations

### HTML Reports

Rich HTML reports with:
- Professional styling and layout
- Table of contents
- Embedded charts and plots
- Interactive elements
- Printer-friendly format

### CSV Exports

Raw data exports for further analysis:
- Matrix size, GFLOPS, execution times
- Memory usage, system identifiers
- Benchmark IDs and timestamps
- Compatible with Excel, R, Python pandas

## Examples

### Example 1: Basic Analysis Workflow

```python
from analysis import BenchmarkAnalyzer

# Initialize and load results
analyzer = BenchmarkAnalyzer("demo_results")
results = analyzer.load_results()

# Quick overview
data = analyzer.extract_performance_data()
print(f"Average performance: {sum(data['gflops'])/len(data['gflops']):.2f} GFLOPS")

# Generate comprehensive report
analyzer.generate_markdown_report("basic_report.md")
```

### Example 2: Multi-System Comparison

```python
# Compare multiple systems
analyzer = BenchmarkAnalyzer("benchmark_results")
results = analyzer.load_results()

# Get performance ratios
ratios = analyzer.calculate_performance_ratios()
for system in ratios['ranking']:
    perf = ratios['system_performance'][system]
    print(f"{system}: {perf['mean']:.2f} GFLOPS")

# Generate comparison plots
analyzer.plot_system_comparison("comparison.png")
```

### Example 3: Statistical Analysis

```python
# Perform statistical tests
stats = analyzer.statistical_significance_test()
if 'tests' in stats:
    print("Statistical significance tests:")
    for test in stats['tests']:
        significant = "Yes" if test['t_test']['significant'] else "No"
        print(f"  {test['system1']} vs {test['system2']}: {significant}")
```

## Error Handling

The analysis module includes comprehensive error handling:

- **Missing dependencies**: Graceful degradation when optional packages aren't available
- **Invalid data**: Robust handling of malformed or missing benchmark results
- **Empty datasets**: Appropriate messages when no data is available
- **File I/O errors**: Clear error messages for file access issues

## Performance Tips

- **Large datasets**: Use filtering options to analyze subsets of data
- **Memory usage**: The analyzer caches results to improve performance
- **Plotting**: Disable plot display with `show_plot=False` for batch processing
- **Parallel processing**: Multiple result files are processed efficiently

## Troubleshooting

### Common Issues

1. **"No results found"**
   - Check the results directory path
   - Ensure JSON files are present
   - Verify file permissions

2. **"Plotting not available"**
   - Install matplotlib: `pip install matplotlib seaborn`
   - Check for backend issues on headless systems

3. **"HTML report generation failed"**
   - Install jinja2: `pip install jinja2`
   - Check available disk space

4. **"Statistical analysis not available"**
   - Install scipy: `pip install scipy`
   - Ensure multiple systems for comparison

### Debug Mode

Enable verbose output for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

analyzer = BenchmarkAnalyzer("demo_results")
results = analyzer.load_results()
```

## Integration with Benchmark System

The analysis module is designed to work seamlessly with the benchmark system:

1. **Benchmark Generation**: Run `benchmark.py` or `demo_benchmark.py` to generate results
2. **Result Storage**: Results are automatically saved in JSON format
3. **Analysis**: Use the analysis module to process and visualize results
4. **Reporting**: Generate professional reports for documentation

## Future Enhancements

Planned features for future versions:

- **Interactive web dashboard** using Plotly/Dash
- **Real-time monitoring** during benchmark execution
- **Machine learning models** for performance prediction
- **Database integration** for large-scale result storage
- **Advanced statistical models** for performance analysis

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review the example scripts
3. Use the interactive mode for exploration
4. Examine the source code for detailed implementation

The analysis module is designed to be extensible and customizable for specific analysis needs.