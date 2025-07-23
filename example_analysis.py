#!/usr/bin/env python3
"""
Example usage of the MatrixBenchmark Analysis Module

This script demonstrates various ways to use the analysis module for
different analysis scenarios.
"""

from analysis import BenchmarkAnalyzer
import os


def example_basic_analysis():
    """Example: Basic analysis workflow."""
    print("=== Example: Basic Analysis Workflow ===")
    
    # Initialize analyzer with specific results directory
    analyzer = BenchmarkAnalyzer("demo_results")
    
    # Load all JSON results
    results = analyzer.load_results("*.json")
    print(f"Loaded {len(results)} result files")
    
    # Get quick overview
    data = analyzer.extract_performance_data()
    if data['gflops']:
        print(f"Performance range: {min(data['gflops']):.2f} - {max(data['gflops']):.2f} GFLOPS")
    
    # Generate comprehensive report
    analyzer.generate_markdown_report(output_path="basic_analysis_report.md")
    print("Basic analysis report generated: basic_analysis_report.md\n")


def example_multi_system_comparison():
    """Example: Comparing multiple systems."""
    print("=== Example: Multi-System Comparison ===")
    
    analyzer = BenchmarkAnalyzer("benchmark_results")
    
    # Load results from multiple directories if available
    results = analyzer.load_results("*.json")
    
    if not results:
        print("No results found in benchmark_results directory")
        return
    
    # Get system information
    systems = analyzer.get_system_summary()
    print(f"Found {len(systems)} different systems:")
    
    for system_id, info in systems.items():
        print(f"  - {system_id}: {info['chip_type']}, {info['memory_gb']} GB")
    
    # Calculate performance ratios
    ratios = analyzer.calculate_performance_ratios()
    
    if 'ranking' in ratios and len(ratios['ranking']) > 1:
        print("\nPerformance ranking:")
        for i, system in enumerate(ratios['ranking'], 1):
            perf = ratios['system_performance'][system]
            print(f"  {i}. {system}: {perf['mean']:.2f} GFLOPS")
        
        # Show performance advantages
        print("\nPerformance ratios:")
        for ratio_key, ratio_data in ratios['ratios'].items():
            if ratio_data['ratio'] > 1:
                advantage = ratio_data['performance_advantage']
                print(f"  {ratio_data['system1']} is {advantage:.1f}x faster than {ratio_data['system2']}")
    
    # Generate comparison plots
    try:
        analyzer.plot_system_comparison(save_path="multi_system_comparison.png", show_plot=False)
        print("System comparison plots saved: multi_system_comparison.png")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print()


def example_performance_analysis():
    """Example: Detailed performance analysis."""
    print("=== Example: Performance Analysis ===")
    
    analyzer = BenchmarkAnalyzer("demo_results")
    results = analyzer.load_results("*.json")
    
    if not results:
        print("No results found")
        return
    
    # Extract performance data
    data = analyzer.extract_performance_data()
    
    # Analyze performance by matrix size
    print("Performance by matrix size:")
    size_performance = {}
    for i, size in enumerate(data['matrix_sizes']):
        if size not in size_performance:
            size_performance[size] = []
        size_performance[size].append(data['gflops'][i])
    
    import numpy as np
    
    for size in sorted(size_performance.keys()):
        gflops_list = size_performance[size]
        avg_gflops = np.mean(gflops_list)
        std_gflops = np.std(gflops_list)
        print(f"  Size {size}: {avg_gflops:.2f} ± {std_gflops:.2f} GFLOPS ({len(gflops_list)} samples)")
    
    # Detect outliers
    outliers = analyzer.detect_outliers()
    
    if outliers.get('count', 0) > 0:
        print(f"\nOutlier detection found {outliers['count']} outliers:")
        for detail in outliers['details']:
            print(f"  - {detail['system_id']} (Size: {detail['matrix_size']}): {detail['gflops']:.2f} GFLOPS")
    else:
        print("\nNo outliers detected in the data")
    
    # Generate detailed analysis plots
    try:
        analyzer.plot_performance_scaling(save_path="performance_analysis.png", show_plot=False)
        analyzer.plot_timing_distributions(save_path="timing_analysis.png", show_plot=False)
        print("Performance analysis plots saved")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print()


def example_export_and_reports():
    """Example: Exporting data and generating reports."""
    print("=== Example: Export and Report Generation ===")
    
    analyzer = BenchmarkAnalyzer("demo_results")
    results = analyzer.load_results("*.json")
    
    if not results:
        print("No results found")
        return
    
    # Export to CSV for further analysis
    csv_path = analyzer.export_to_csv("detailed_analysis.csv")
    print(f"Data exported to CSV: {csv_path}")
    
    # Generate Markdown report
    md_path = analyzer.generate_markdown_report("detailed_report.md")
    print(f"Markdown report generated: {md_path}")
    
    # Generate HTML report with plots
    try:
        html_path = analyzer.generate_html_report("detailed_report.html", include_plots=True)
        print(f"HTML report generated: {html_path}")
    except Exception as e:
        print(f"HTML report generation failed: {e}")
    
    print()


def example_custom_analysis():
    """Example: Custom analysis using extracted data."""
    print("=== Example: Custom Analysis ===")
    
    analyzer = BenchmarkAnalyzer("demo_results")
    results = analyzer.load_results("*.json")
    
    if not results:
        print("No results found")
        return
    
    # Extract raw data for custom analysis
    data = analyzer.extract_performance_data()
    
    if not data['gflops']:
        print("No performance data available")
        return
    
    import numpy as np
    
    # Custom analysis: Performance efficiency
    print("Custom Performance Efficiency Analysis:")
    
    efficiency_scores = []
    for i in range(len(data['matrix_sizes'])):
        size = data['matrix_sizes'][i]
        gflops = data['gflops'][i]
        exec_time = data['execution_times'][i]
        memory = data['memory_usage'][i]
        
        # Calculate efficiency score (performance per resource unit)
        if exec_time > 0 and memory > 0:
            efficiency = gflops / (exec_time * memory)
            efficiency_scores.append(efficiency)
            print(f"  Size {size}: {gflops:.2f} GFLOPS, Efficiency: {efficiency:.2f}")
    
    if efficiency_scores:
        avg_efficiency = np.mean(efficiency_scores)
        print(f"\nAverage efficiency score: {avg_efficiency:.2f}")
    
    # Custom analysis: Performance scaling
    print("\nCustom Scaling Analysis:")
    
    # Group by matrix size and calculate theoretical vs actual scaling
    size_groups = {}
    for i, size in enumerate(data['matrix_sizes']):
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(data['gflops'][i])
    
    sizes = sorted(size_groups.keys())
    if len(sizes) > 1:
        base_size = sizes[0]
        base_performance = np.mean(size_groups[base_size])
        
        print(f"  Base size {base_size}: {base_performance:.2f} GFLOPS")
        
        for size in sizes[1:]:
            actual_performance = np.mean(size_groups[size])
            theoretical_ratio = (size / base_size) ** 3  # O(n^3) scaling
            actual_ratio = actual_performance / base_performance
            efficiency = actual_ratio / theoretical_ratio
            
            print(f"  Size {size}: {actual_performance:.2f} GFLOPS")
            print(f"    Theoretical scaling: {theoretical_ratio:.2f}x")
            print(f"    Actual scaling: {actual_ratio:.2f}x")
            print(f"    Scaling efficiency: {efficiency:.2f}")
    
    print()


def example_statistical_analysis():
    """Example: Statistical analysis of results."""
    print("=== Example: Statistical Analysis ===")
    
    analyzer = BenchmarkAnalyzer("demo_results")
    results = analyzer.load_results("*.json")
    
    if not results:
        print("No results found")
        return
    
    # Perform statistical significance testing
    stats_results = analyzer.statistical_significance_test()
    
    if 'error' in stats_results:
        print(f"Statistical analysis not available: {stats_results['error']}")
        return
    
    # Display test results
    if stats_results.get('tests'):
        print("Statistical Significance Tests:")
        for test in stats_results['tests']:
            print(f"\n  Comparing {test['system1']} vs {test['system2']}:")
            print(f"    Sample sizes: {test['sample1_size']} vs {test['sample2_size']}")
            print(f"    Means: {test['sample1_mean']:.2f} vs {test['sample2_mean']:.2f} GFLOPS")
            
            t_test = test['t_test']
            print(f"    T-test p-value: {t_test['p_value']:.4f}")
            print(f"    Significant difference: {'Yes' if t_test['significant'] else 'No'}")
            
            mw_test = test['mann_whitney']
            print(f"    Mann-Whitney p-value: {mw_test['p_value']:.4f}")
            print(f"    Non-parametric significant: {'Yes' if mw_test['significant'] else 'No'}")
    
    # Overall ANOVA if available
    if stats_results.get('overall_anova'):
        anova = stats_results['overall_anova']
        print(f"\n  Overall ANOVA:")
        print(f"    F-statistic: {anova['f_statistic']:.4f}")
        print(f"    P-value: {anova['p_value']:.4f}")
        print(f"    Significant difference between systems: {'Yes' if anova['significant'] else 'No'}")
    
    print()


def main():
    """Run all examples."""
    print("MatrixBenchmark Analysis Module Examples\n")
    
    # Check if we have any results to analyze
    demo_results_exist = os.path.exists("demo_results") and any(
        f.endswith('.json') for f in os.listdir("demo_results")
    )
    
    benchmark_results_exist = os.path.exists("benchmark_results") and any(
        f.endswith('.json') for f in os.listdir("benchmark_results")
    )
    
    if not demo_results_exist and not benchmark_results_exist:
        print("No benchmark results found. Please run some benchmarks first:")
        print("  python demo_benchmark.py")
        print("  python benchmark.py")
        return
    
    # Run examples
    try:
        example_basic_analysis()
        example_multi_system_comparison()
        example_performance_analysis()
        example_export_and_reports()
        example_custom_analysis()
        example_statistical_analysis()
        
        print("=== All Examples Complete ===")
        print("\nGenerated files:")
        example_files = [
            "basic_analysis_report.md",
            "multi_system_comparison.png",
            "performance_analysis.png",
            "timing_analysis.png",
            "detailed_analysis.csv",
            "detailed_report.md",
            "detailed_report.html"
        ]
        
        for file_path in example_files:
            if os.path.exists(file_path):
                print(f"  ✓ {file_path}")
        
        print("\nTo explore more features, try:")
        print("  python -m analysis --interactive")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("This might be due to missing dependencies. Install them with:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()