#!/usr/bin/env python3
"""
Test script for the MatrixBenchmark Analysis Module

This script demonstrates the key features of the analysis module using
existing benchmark results.
"""

import os
import sys
from analysis import BenchmarkAnalyzer

def main():
    """Test the analysis module with existing results."""
    print("=== Testing MatrixBenchmark Analysis Module ===\n")
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer("demo_results")
    
    # Load results
    print("1. Loading benchmark results...")
    results = analyzer.load_results("*.json")
    
    if not results:
        print("No results found. Please run some benchmarks first.")
        return
    
    print(f"   Loaded {len(results)} result files\n")
    
    # Test system summary
    print("2. Analyzing system information...")
    systems = analyzer.get_system_summary()
    
    for system_id, info in systems.items():
        print(f"   System: {system_id}")
        print(f"     Chip: {info['chip_type']}")
        print(f"     Memory: {info['memory_gb']} GB")
        print(f"     CPU Cores: {info['cpu_cores_total']}")
        print(f"     Results: {info['results_count']}")
    print()
    
    # Test performance data extraction
    print("3. Extracting performance data...")
    data = analyzer.extract_performance_data()
    
    if data['gflops']:
        print(f"   Total benchmarks: {len(data['matrix_sizes'])}")
        print(f"   Matrix sizes: {sorted(list(set(data['matrix_sizes'])))}")
        print(f"   Performance range: {min(data['gflops']):.2f} - {max(data['gflops']):.2f} GFLOPS")
        print(f"   Average performance: {sum(data['gflops'])/len(data['gflops']):.2f} GFLOPS")
    print()
    
    # Test performance ratios
    print("4. Calculating performance ratios...")
    ratios = analyzer.calculate_performance_ratios()
    
    if 'ranking' in ratios:
        print("   Performance ranking:")
        for i, system in enumerate(ratios['ranking'], 1):
            perf = ratios['system_performance'][system]
            print(f"     {i}. {system}: {perf['mean']:.2f} GFLOPS (±{perf['std']:.2f})")
    print()
    
    # Test outlier detection
    print("5. Detecting outliers...")
    outliers = analyzer.detect_outliers()
    
    print(f"   Found {outliers.get('count', 0)} outliers ({outliers.get('percentage', 0):.1f}%)")
    for detail in outliers.get('details', []):
        print(f"     {detail['system_id']} (Size: {detail['matrix_size']}): {detail['gflops']:.2f} GFLOPS")
    print()
    
    # Test CSV export
    print("6. Exporting to CSV...")
    csv_path = analyzer.export_to_csv(output_path="test_analysis_export.csv")
    if csv_path and os.path.exists(csv_path):
        print(f"   CSV exported to: {csv_path}")
    print()
    
    # Test Markdown report
    print("7. Generating Markdown report...")
    md_path = analyzer.generate_markdown_report(output_path="test_analysis_report.md")
    if md_path and os.path.exists(md_path):
        print(f"   Markdown report saved to: {md_path}")
    print()
    
    # Test HTML report
    print("8. Generating HTML report...")
    try:
        html_path = analyzer.generate_html_report(output_path="test_analysis_report.html")
        if html_path and os.path.exists(html_path):
            print(f"   HTML report saved to: {html_path}")
    except Exception as e:
        print(f"   HTML report generation failed: {e}")
    print()
    
    # Test plotting (only if matplotlib is available)
    print("9. Testing plotting capabilities...")
    try:
        import matplotlib.pyplot as plt
        
        print("   Generating performance scaling plot...")
        fig1 = analyzer.plot_performance_scaling(save_path="test_performance_scaling.png", show_plot=False)
        if fig1:
            print("     Performance scaling plot saved")
            plt.close(fig1)
        
        print("   Generating timing distribution plots...")
        fig2 = analyzer.plot_timing_distributions(save_path="test_timing_distributions.png", show_plot=False)
        if fig2:
            print("     Timing distribution plots saved")
            plt.close(fig2)
        
        print("   Generating system comparison plots...")
        fig3 = analyzer.plot_system_comparison(save_path="test_system_comparison.png", show_plot=False)
        if fig3:
            print("     System comparison plots saved")
            plt.close(fig3)
            
    except ImportError:
        print("   Plotting not available (matplotlib not installed)")
    except Exception as e:
        print(f"   Plotting failed: {e}")
    print()
    
    print("=== Analysis Module Test Complete ===")
    print("\nGenerated files:")
    test_files = [
        "test_analysis_export.csv",
        "test_analysis_report.md", 
        "test_analysis_report.html",
        "test_performance_scaling.png",
        "test_timing_distributions.png",
        "test_system_comparison.png"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (not created)")
    
    print("\nTo run interactive analysis, use:")
    print("  python -m analysis --interactive")
    print("\nTo generate specific reports, use:")
    print("  python -m analysis --report-html my_report.html")
    print("  python -m analysis --plot-scaling --plot-comparison")

if __name__ == "__main__":
    main()