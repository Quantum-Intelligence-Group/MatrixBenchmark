#!/usr/bin/env python3
"""
Matrix Benchmark Results Analysis Module

This module provides comprehensive analysis capabilities for matrix benchmark results,
including data visualization, comparative analysis, report generation, and interactive
analysis functions.

Features:
- Performance scaling plots (GFLOPS vs matrix size)
- Cross-machine comparison charts
- Statistical distribution plots (timing histograms)
- System performance profiles
- HTML and Markdown report generation
- CSV exports for further analysis
- Interactive analysis functions

Requirements:
- matplotlib
- seaborn
- pandas
- numpy
- jinja2 (for HTML report generation)
"""

import json
import os
import csv
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

# Core data manipulation
import numpy as np

# Plotting and visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("matplotlib/seaborn not available. Plotting functions will be disabled.")

# Data analysis
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("pandas not available. Some analysis functions will be limited.")

# Statistical analysis
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Statistical significance testing will be disabled.")

# HTML report generation
try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    warnings.warn("jinja2 not available. HTML report generation will be disabled.")


class BenchmarkAnalyzer:
    """Main class for analyzing matrix benchmark results."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        """
        Initialize the analyzer with a results directory.
        
        Args:
            results_dir: Directory containing benchmark result JSON files
        """
        self.results_dir = Path(results_dir)
        self.results_cache = {}
        self.loaded_results = []
        
        # Configure plotting style
        if PLOTTING_AVAILABLE:
            self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Configure matplotlib and seaborn plotting style."""
        plt.style.use('default')
        if 'seaborn' in plt.style.available:
            plt.style.use('seaborn-v0_8')
        
        # Set up color palette
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.size': 10,
            'lines.linewidth': 2,
            'lines.markersize': 6
        })
    
    def load_results(self, file_patterns: Union[str, List[str]] = "*.json", 
                    reload: bool = False) -> List[Dict]:
        """
        Load benchmark results from JSON files.
        
        Args:
            file_patterns: Pattern(s) to match result files
            reload: Whether to reload cached results
            
        Returns:
            List of loaded benchmark results
        """
        if not reload and self.loaded_results:
            return self.loaded_results
            
        if isinstance(file_patterns, str):
            file_patterns = [file_patterns]
            
        self.loaded_results = []
        
        for pattern in file_patterns:
            for file_path in self.results_dir.glob(pattern):
                if file_path.suffix.lower() == '.json':
                    try:
                        with open(file_path, 'r') as f:
                            result = json.load(f)
                            result['_file_path'] = str(file_path)
                            result['_file_name'] = file_path.name
                            self.loaded_results.append(result)
                    except Exception as e:
                        warnings.warn(f"Failed to load {file_path}: {e}")
        
        print(f"Loaded {len(self.loaded_results)} benchmark result files")
        return self.loaded_results
    
    def get_system_summary(self, results: Optional[List[Dict]] = None) -> Dict:
        """
        Extract system information summary from results.
        
        Args:
            results: List of benchmark results (uses loaded results if None)
            
        Returns:
            Dictionary with system summaries
        """
        if results is None:
            results = self.loaded_results
            
        systems = {}
        
        for result in results:
            if 'system_profile' not in result:
                continue
                
            system_profile = result['system_profile']
            hardware = system_profile.get('hardware', {})
            
            # Create system identifier
            chip_type = hardware.get('chip_type', 'Unknown')
            model = hardware.get('model', 'Unknown')
            memory_gb = hardware.get('memory_total_gb', 0)
            
            system_id = f"{chip_type}_{memory_gb}GB"
            
            if system_id not in systems:
                systems[system_id] = {
                    'chip_type': chip_type,
                    'model': model,
                    'memory_gb': memory_gb,
                    'cpu_cores_total': hardware.get('cpu_cores_total', 0),
                    'cpu_cores_performance': hardware.get('cpu_cores_performance', 0),
                    'cpu_cores_efficiency': hardware.get('cpu_cores_efficiency', 0),
                    'hardware_score': hardware.get('hardware_score', 0),
                    'files': [],
                    'results_count': 0
                }
            
            systems[system_id]['files'].append(result['_file_name'])
            systems[system_id]['results_count'] += len(result.get('benchmark_results', []))
        
        return systems
    
    def extract_performance_data(self, results: Optional[List[Dict]] = None) -> Dict:
        """
        Extract performance data suitable for analysis and plotting.
        
        Args:
            results: List of benchmark results (uses loaded results if None)
            
        Returns:
            Dictionary with organized performance data
        """
        if results is None:
            results = self.loaded_results
            
        data = {
            'matrix_sizes': [],
            'gflops': [],
            'execution_times': [],
            'memory_usage': [],
            'system_ids': [],
            'benchmark_ids': [],
            'file_names': [],
            'timestamps': []
        }
        
        for result in results:
            if 'benchmark_results' not in result:
                continue
                
            # Extract system identifier
            system_profile = result.get('system_profile', {})
            hardware = system_profile.get('hardware', {})
            chip_type = hardware.get('chip_type', 'Unknown')
            memory_gb = hardware.get('memory_total_gb', 0)
            system_id = f"{chip_type}_{memory_gb}GB"
            
            benchmark_id = result.get('benchmark_id', 'Unknown')
            file_name = result.get('_file_name', 'Unknown')
            
            # Extract benchmark results
            for bench_result in result['benchmark_results']:
                data['matrix_sizes'].append(bench_result.get('matrix_size', 0))
                data['gflops'].append(bench_result.get('gflops', 0))
                data['system_ids'].append(system_id)
                data['benchmark_ids'].append(benchmark_id)
                data['file_names'].append(file_name)
                data['timestamps'].append(bench_result.get('timestamp', 0))
                
                # Extract timing statistics
                timing_stats = bench_result.get('timing_stats', {})
                data['execution_times'].append(timing_stats.get('mean', 0))
                
                # Extract memory usage (average from timing results)
                timing_results = bench_result.get('timing_results', [])
                if timing_results:
                    avg_memory = np.mean([tr.get('memory_usage_mb', 0) for tr in timing_results])
                    data['memory_usage'].append(avg_memory)
                else:
                    data['memory_usage'].append(0)
        
        return data
    
    def plot_performance_scaling(self, results: Optional[List[Dict]] = None,
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> Optional[plt.Figure]:
        """
        Create performance scaling plots (GFLOPS vs matrix size).
        
        Args:
            results: List of benchmark results (uses loaded results if None)
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Figure object if plotting is available
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Please install matplotlib and seaborn.")
            return None
            
        data = self.extract_performance_data(results)
        
        if not data['matrix_sizes']:
            print("No performance data found to plot.")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Convert to numpy arrays for easier manipulation
        matrix_sizes = np.array(data['matrix_sizes'])
        gflops = np.array(data['gflops'])
        system_ids = np.array(data['system_ids'])
        
        # Plot 1: GFLOPS vs Matrix Size (scatter plot with trend lines)
        unique_systems = np.unique(system_ids)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_systems)))
        
        for i, system in enumerate(unique_systems):
            mask = system_ids == system
            sys_sizes = matrix_sizes[mask]
            sys_gflops = gflops[mask]
            
            ax1.scatter(sys_sizes, sys_gflops, 
                       label=system, color=colors[i], alpha=0.7, s=50)
            
            # Add trend line if we have enough points
            if len(sys_sizes) > 1:
                z = np.polyfit(sys_sizes, sys_gflops, 1)
                p = np.poly1d(z)
                ax1.plot(sys_sizes, p(sys_sizes), 
                        color=colors[i], linestyle='--', alpha=0.8)
        
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Performance (GFLOPS)')
        ax1.set_title('Performance Scaling by Matrix Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Performance comparison (box plot)
        if len(unique_systems) > 1:
            gflops_by_system = [gflops[system_ids == system] for system in unique_systems]
            ax2.boxplot(gflops_by_system, labels=unique_systems)
            ax2.set_ylabel('Performance (GFLOPS)')
            ax2.set_title('Performance Distribution by System')
            ax2.tick_params(axis='x', rotation=45)
        else:
            # Single system - show performance by matrix size
            ax2.bar(range(len(matrix_sizes)), gflops, 
                   tick_label=[f'{size}' for size in matrix_sizes])
            ax2.set_xlabel('Matrix Size')
            ax2.set_ylabel('Performance (GFLOPS)')
            ax2.set_title('Performance by Matrix Size')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance scaling plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_timing_distributions(self, results: Optional[List[Dict]] = None,
                                save_path: Optional[str] = None,
                                show_plot: bool = True) -> Optional[plt.Figure]:
        """
        Create timing distribution plots (histograms).
        
        Args:
            results: List of benchmark results (uses loaded results if None)
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Figure object if plotting is available
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Please install matplotlib and seaborn.")
            return None
            
        data = self.extract_performance_data(results)
        
        if not data['execution_times']:
            print("No timing data found to plot.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Execution time distribution
        axes[0].hist(data['execution_times'], bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Execution Time (seconds)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Execution Times')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: GFLOPS distribution
        axes[1].hist(data['gflops'], bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_xlabel('Performance (GFLOPS)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Performance (GFLOPS)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Memory usage distribution
        axes[2].hist(data['memory_usage'], bins=20, alpha=0.7, edgecolor='black', color='green')
        axes[2].set_xlabel('Memory Usage (MB)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Distribution of Memory Usage')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Performance vs Matrix Size (scatter)
        scatter = axes[3].scatter(data['matrix_sizes'], data['gflops'], 
                                alpha=0.6, c=data['execution_times'], cmap='viridis')
        axes[3].set_xlabel('Matrix Size')
        axes[3].set_ylabel('Performance (GFLOPS)')
        axes[3].set_title('Performance vs Matrix Size\n(Color = Execution Time)')
        axes[3].grid(True, alpha=0.3)
        
        # Add colorbar for scatter plot
        cbar = plt.colorbar(scatter, ax=axes[3])
        cbar.set_label('Execution Time (s)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Timing distribution plots saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_system_comparison(self, results: Optional[List[Dict]] = None,
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> Optional[plt.Figure]:
        """
        Create system comparison charts.
        
        Args:
            results: List of benchmark results (uses loaded results if None)
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Figure object if plotting is available
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Please install matplotlib and seaborn.")
            return None
            
        data = self.extract_performance_data(results)
        
        if not data['system_ids']:
            print("No system data found to plot.")
            return None
        
        # Group data by system
        systems = {}
        for i, system_id in enumerate(data['system_ids']):
            if system_id not in systems:
                systems[system_id] = {
                    'gflops': [],
                    'execution_times': [],
                    'matrix_sizes': [],
                    'memory_usage': []
                }
            
            systems[system_id]['gflops'].append(data['gflops'][i])
            systems[system_id]['execution_times'].append(data['execution_times'][i])
            systems[system_id]['matrix_sizes'].append(data['matrix_sizes'][i])
            systems[system_id]['memory_usage'].append(data['memory_usage'][i])
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Average performance comparison
        ax1 = fig.add_subplot(gs[0, 0])
        system_names = list(systems.keys())
        avg_gflops = [np.mean(systems[sys]['gflops']) for sys in system_names]
        std_gflops = [np.std(systems[sys]['gflops']) for sys in system_names]
        
        bars = ax1.bar(range(len(system_names)), avg_gflops, 
                      yerr=std_gflops, capsize=5, alpha=0.7)
        ax1.set_xlabel('System')
        ax1.set_ylabel('Average Performance (GFLOPS)')
        ax1.set_title('Average Performance by System')
        ax1.set_xticks(range(len(system_names)))
        ax1.set_xticklabels(system_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, avg_gflops)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Plot 2: Performance vs Matrix Size by System
        ax2 = fig.add_subplot(gs[0, 1])
        colors = plt.cm.Set3(np.linspace(0, 1, len(system_names)))
        
        for i, system in enumerate(system_names):
            sys_data = systems[system]
            ax2.scatter(sys_data['matrix_sizes'], sys_data['gflops'],
                       label=system, color=colors[i], alpha=0.7, s=50)
        
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Performance (GFLOPS)')
        ax2.set_title('Performance vs Matrix Size by System')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Execution time comparison
        ax3 = fig.add_subplot(gs[1, 0])
        avg_times = [np.mean(systems[sys]['execution_times']) for sys in system_names]
        std_times = [np.std(systems[sys]['execution_times']) for sys in system_names]
        
        bars = ax3.bar(range(len(system_names)), avg_times, 
                      yerr=std_times, capsize=5, alpha=0.7, color='orange')
        ax3.set_xlabel('System')
        ax3.set_ylabel('Average Execution Time (seconds)')
        ax3.set_title('Average Execution Time by System')
        ax3.set_xticks(range(len(system_names)))
        ax3.set_xticklabels(system_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Memory usage comparison
        ax4 = fig.add_subplot(gs[1, 1])
        avg_memory = [np.mean(systems[sys]['memory_usage']) for sys in system_names]
        std_memory = [np.std(systems[sys]['memory_usage']) for sys in system_names]
        
        bars = ax4.bar(range(len(system_names)), avg_memory, 
                      yerr=std_memory, capsize=5, alpha=0.7, color='green')
        ax4.set_xlabel('System')
        ax4.set_ylabel('Average Memory Usage (MB)')
        ax4.set_title('Average Memory Usage by System')
        ax4.set_xticks(range(len(system_names)))
        ax4.set_xticklabels(system_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Performance efficiency (spanning bottom row)
        ax5 = fig.add_subplot(gs[2, :])
        
        # Calculate efficiency metrics
        efficiency_data = []
        for system in system_names:
            sys_data = systems[system]
            # Efficiency = GFLOPS / (Memory Usage * Execution Time)
            efficiency = []
            for gflops, memory, time in zip(sys_data['gflops'], 
                                          sys_data['memory_usage'], 
                                          sys_data['execution_times']):
                if memory > 0 and time > 0:
                    efficiency.append(gflops / (memory * time))
                else:
                    efficiency.append(0)
            efficiency_data.append(efficiency)
        
        # Create box plot for efficiency
        ax5.boxplot(efficiency_data, labels=system_names)
        ax5.set_xlabel('System')
        ax5.set_ylabel('Efficiency (GFLOPS / (MB * s))')
        ax5.set_title('Performance Efficiency by System')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"System comparison plots saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def calculate_performance_ratios(self, results: Optional[List[Dict]] = None) -> Dict:
        """
        Calculate performance ratios between different systems.
        
        Args:
            results: List of benchmark results (uses loaded results if None)
            
        Returns:
            Dictionary with performance ratio calculations
        """
        data = self.extract_performance_data(results)
        
        if not data['system_ids']:
            return {}
        
        # Group data by system
        systems = {}
        for i, system_id in enumerate(data['system_ids']):
            if system_id not in systems:
                systems[system_id] = []
            systems[system_id].append(data['gflops'][i])
        
        # Calculate average performance for each system
        avg_performance = {}
        for system, gflops_list in systems.items():
            avg_performance[system] = {
                'mean': np.mean(gflops_list),
                'std': np.std(gflops_list),
                'max': np.max(gflops_list),
                'min': np.min(gflops_list),
                'count': len(gflops_list)
            }
        
        # Calculate ratios
        ratios = {}
        system_names = list(avg_performance.keys())
        
        for i, system1 in enumerate(system_names):
            for j, system2 in enumerate(system_names):
                if i != j:
                    ratio_key = f"{system1}_vs_{system2}"
                    ratio = avg_performance[system1]['mean'] / avg_performance[system2]['mean']
                    ratios[ratio_key] = {
                        'ratio': ratio,
                        'system1': system1,
                        'system2': system2,
                        'system1_mean': avg_performance[system1]['mean'],
                        'system2_mean': avg_performance[system2]['mean'],
                        'faster_system': system1 if ratio > 1 else system2,
                        'performance_advantage': max(ratio, 1/ratio)
                    }
        
        return {
            'system_performance': avg_performance,
            'ratios': ratios,
            'ranking': sorted(system_names, 
                            key=lambda x: avg_performance[x]['mean'], 
                            reverse=True)
        }
    
    def detect_outliers(self, results: Optional[List[Dict]] = None,
                       method: str = 'iqr', threshold: float = 1.5) -> Dict:
        """
        Detect outliers in performance data.
        
        Args:
            results: List of benchmark results (uses loaded results if None)
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier information
        """
        data = self.extract_performance_data(results)
        
        if not data['gflops']:
            return {}
        
        gflops = np.array(data['gflops'])
        outliers = {'indices': [], 'values': [], 'method': method, 'threshold': threshold}
        
        if method == 'iqr':
            Q1 = np.percentile(gflops, 25)
            Q3 = np.percentile(gflops, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (gflops < lower_bound) | (gflops > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(gflops) if SCIPY_AVAILABLE else 
                             (gflops - np.mean(gflops)) / np.std(gflops))
            outlier_mask = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        outliers['indices'] = np.where(outlier_mask)[0].tolist()
        outliers['values'] = gflops[outlier_mask].tolist()
        outliers['count'] = len(outliers['indices'])
        outliers['percentage'] = (outliers['count'] / len(gflops)) * 100
        
        # Add detailed information about outliers
        outliers['details'] = []
        for idx in outliers['indices']:
            outliers['details'].append({
                'index': idx,
                'gflops': data['gflops'][idx],
                'matrix_size': data['matrix_sizes'][idx],
                'system_id': data['system_ids'][idx],
                'execution_time': data['execution_times'][idx],
                'file_name': data['file_names'][idx]
            })
        
        return outliers
    
    def statistical_significance_test(self, results: Optional[List[Dict]] = None,
                                    alpha: float = 0.05) -> Dict:
        """
        Perform statistical significance testing between systems.
        
        Args:
            results: List of benchmark results (uses loaded results if None)
            alpha: Significance level for tests
            
        Returns:
            Dictionary with statistical test results
        """
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available for statistical testing"}
        
        data = self.extract_performance_data(results)
        
        if not data['system_ids']:
            return {}
        
        # Group data by system
        systems = {}
        for i, system_id in enumerate(data['system_ids']):
            if system_id not in systems:
                systems[system_id] = []
            systems[system_id].append(data['gflops'][i])
        
        system_names = list(systems.keys())
        
        if len(system_names) < 2:
            return {"error": "Need at least 2 systems for comparison"}
        
        # Perform pairwise t-tests
        results_dict = {
            'alpha': alpha,
            'tests': [],
            'overall_anova': None
        }
        
        # Pairwise t-tests
        for i, system1 in enumerate(system_names):
            for j, system2 in enumerate(system_names):
                if i < j:  # Avoid duplicate tests
                    sample1 = systems[system1]
                    sample2 = systems[system2]
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(sample1, sample2)
                    
                    # Perform Mann-Whitney U test (non-parametric)
                    u_stat, u_p_value = stats.mannwhitneyu(sample1, sample2, 
                                                          alternative='two-sided')
                    
                    test_result = {
                        'system1': system1,
                        'system2': system2,
                        'sample1_mean': np.mean(sample1),
                        'sample2_mean': np.mean(sample2),
                        'sample1_std': np.std(sample1),
                        'sample2_std': np.std(sample2),
                        'sample1_size': len(sample1),
                        'sample2_size': len(sample2),
                        't_test': {
                            'statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < alpha
                        },
                        'mann_whitney': {
                            'statistic': u_stat,
                            'p_value': u_p_value,
                            'significant': u_p_value < alpha
                        }
                    }
                    
                    results_dict['tests'].append(test_result)
        
        # Overall ANOVA test if more than 2 systems
        if len(system_names) > 2:
            sample_groups = [systems[system] for system in system_names]
            f_stat, anova_p = stats.f_oneway(*sample_groups)
            
            results_dict['overall_anova'] = {
                'f_statistic': f_stat,
                'p_value': anova_p,
                'significant': anova_p < alpha,
                'systems': system_names
            }
        
        return results_dict
    
    def export_to_csv(self, results: Optional[List[Dict]] = None,
                     output_path: str = "benchmark_analysis.csv") -> str:
        """
        Export analysis data to CSV format.
        
        Args:
            results: List of benchmark results (uses loaded results if None)
            output_path: Path for the output CSV file
            
        Returns:
            Path to the created CSV file
        """
        data = self.extract_performance_data(results)
        
        if not data['matrix_sizes']:
            print("No data to export.")
            return ""
        
        # Create DataFrame-like structure
        rows = []
        for i in range(len(data['matrix_sizes'])):
            row = {
                'matrix_size': data['matrix_sizes'][i],
                'gflops': data['gflops'][i],
                'execution_time': data['execution_times'][i],
                'memory_usage_mb': data['memory_usage'][i],
                'system_id': data['system_ids'][i],
                'benchmark_id': data['benchmark_ids'][i],
                'file_name': data['file_names'][i],
                'timestamp': data['timestamps'][i]
            }
            rows.append(row)
        
        # Write to CSV
        with open(output_path, 'w', newline='') as csvfile:
            if rows:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        print(f"Data exported to {output_path}")
        return output_path
    
    def generate_markdown_report(self, results: Optional[List[Dict]] = None,
                               output_path: str = "benchmark_report.md") -> str:
        """
        Generate a Markdown report with analysis results.
        
        Args:
            results: List of benchmark results (uses loaded results if None)
            output_path: Path for the output Markdown file
            
        Returns:
            Path to the created Markdown file
        """
        if results is None:
            results = self.loaded_results
        
        # Gather analysis data
        data = self.extract_performance_data(results)
        systems = self.get_system_summary(results)
        ratios = self.calculate_performance_ratios(results)
        outliers = self.detect_outliers(results)
        
        # Generate report content
        report_content = f"""# Matrix Benchmark Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report analyzes matrix multiplication benchmark results across {len(systems)} system(s) with {len(data['matrix_sizes'])} total benchmark runs.

### Key Findings

- **Total Benchmarks**: {len(data['matrix_sizes'])}
- **Matrix Sizes Tested**: {sorted(list(set(data['matrix_sizes'])))}
- **Systems Analyzed**: {len(systems)}
- **Performance Range**: {min(data['gflops']):.2f} - {max(data['gflops']):.2f} GFLOPS
- **Outliers Detected**: {outliers.get('count', 0)} ({outliers.get('percentage', 0):.1f}%)

## System Information

"""
        
        for system_id, system_info in systems.items():
            report_content += f"""### {system_id}

- **Chip Type**: {system_info['chip_type']}
- **Model**: {system_info['model']}
- **Memory**: {system_info['memory_gb']} GB
- **CPU Cores**: {system_info['cpu_cores_total']} ({system_info['cpu_cores_performance']} performance, {system_info['cpu_cores_efficiency']} efficiency)
- **Hardware Score**: {system_info['hardware_score']}
- **Results Count**: {system_info['results_count']}

"""
        
        # Performance Analysis
        report_content += """## Performance Analysis

### Overall Statistics

"""
        
        if data['gflops']:
            report_content += f"""- **Average Performance**: {np.mean(data['gflops']):.2f} GFLOPS
- **Peak Performance**: {np.max(data['gflops']):.2f} GFLOPS
- **Minimum Performance**: {np.min(data['gflops']):.2f} GFLOPS
- **Standard Deviation**: {np.std(data['gflops']):.2f} GFLOPS
- **Coefficient of Variation**: {(np.std(data['gflops']) / np.mean(data['gflops']) * 100):.1f}%

"""
        
        # System Comparisons
        if 'system_performance' in ratios and len(ratios['system_performance']) > 1:
            report_content += """### System Performance Comparison

| System | Avg GFLOPS | Std Dev | Max GFLOPS | Min GFLOPS | Samples |
|--------|------------|---------|------------|------------|---------|
"""
            
            for system, perf in ratios['system_performance'].items():
                report_content += f"| {system} | {perf['mean']:.2f} | {perf['std']:.2f} | {perf['max']:.2f} | {perf['min']:.2f} | {perf['count']} |\n"
            
            report_content += "\n### Performance Rankings\n\n"
            
            for i, system in enumerate(ratios['ranking'], 1):
                perf = ratios['system_performance'][system]
                report_content += f"{i}. **{system}**: {perf['mean']:.2f} GFLOPS (±{perf['std']:.2f})\n"
        
        # Outlier Analysis
        if outliers.get('count', 0) > 0:
            report_content += f"""
### Outlier Analysis

Detected {outliers['count']} outliers using {outliers['method']} method (threshold: {outliers['threshold']}):

"""
            
            for detail in outliers.get('details', []):
                report_content += f"- **{detail['system_id']}** (Matrix Size: {detail['matrix_size']}): {detail['gflops']:.2f} GFLOPS\n"
        
        # Matrix Size Analysis
        report_content += """
## Matrix Size Analysis

Performance scaling across different matrix sizes:

"""
        
        # Group by matrix size
        size_performance = {}
        for i, size in enumerate(data['matrix_sizes']):
            if size not in size_performance:
                size_performance[size] = []
            size_performance[size].append(data['gflops'][i])
        
        report_content += "| Matrix Size | Avg GFLOPS | Std Dev | Max GFLOPS | Min GFLOPS | Samples |\n"
        report_content += "|-------------|------------|---------|------------|------------|----------|\n"
        
        for size in sorted(size_performance.keys()):
            gflops_list = size_performance[size]
            report_content += f"| {size} | {np.mean(gflops_list):.2f} | {np.std(gflops_list):.2f} | {np.max(gflops_list):.2f} | {np.min(gflops_list):.2f} | {len(gflops_list)} |\n"
        
        # Recommendations
        report_content += """
## Recommendations

Based on the analysis:

1. **Optimal Matrix Size**: The matrix size with the highest average performance should be used for peak throughput.
2. **System Selection**: Choose systems with consistently high performance and low variance.
3. **Performance Monitoring**: Watch for outliers that might indicate thermal throttling or system issues.
4. **Benchmarking Strategy**: Consider testing larger matrix sizes to find the performance plateau.

## Technical Details

- **Analysis Method**: Statistical analysis of matrix multiplication benchmarks
- **Outlier Detection**: {outliers.get('method', 'N/A')} method with threshold {outliers.get('threshold', 'N/A')}
- **Data Source**: {len(results)} benchmark result files

---

*Report generated by MatrixBenchmark Analysis Module*
"""
        
        # Write report
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        print(f"Markdown report saved to {output_path}")
        return output_path
    
    def generate_html_report(self, results: Optional[List[Dict]] = None,
                           output_path: str = "benchmark_report.html",
                           include_plots: bool = True) -> str:
        """
        Generate an HTML report with embedded charts.
        
        Args:
            results: List of benchmark results (uses loaded results if None)
            output_path: Path for the output HTML file
            include_plots: Whether to include embedded plot images
            
        Returns:
            Path to the created HTML file
        """
        if not JINJA2_AVAILABLE:
            print("jinja2 not available. Cannot generate HTML report.")
            return ""
        
        # First generate the markdown report content
        md_path = "temp_report.md"
        self.generate_markdown_report(results, md_path)
        
        # Read the markdown content
        with open(md_path, 'r') as f:
            md_content = f.read()
        
        # Clean up temp file
        os.remove(md_path)
        
        # Generate plots if requested
        plot_images = {}
        if include_plots and PLOTTING_AVAILABLE:
            # Create temporary plot files
            plot_files = {
                'performance_scaling': 'temp_performance_scaling.png',
                'timing_distributions': 'temp_timing_distributions.png',
                'system_comparison': 'temp_system_comparison.png'
            }
            
            try:
                self.plot_performance_scaling(results, plot_files['performance_scaling'], show_plot=False)
                self.plot_timing_distributions(results, plot_files['timing_distributions'], show_plot=False)
                self.plot_system_comparison(results, plot_files['system_comparison'], show_plot=False)
                
                # Convert plots to base64 for embedding
                import base64
                for plot_name, file_path in plot_files.items():
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                            plot_images[plot_name] = f"data:image/png;base64,{img_data}"
                        os.remove(file_path)  # Clean up
                        
            except Exception as e:
                print(f"Error generating plots for HTML report: {e}")
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Matrix Benchmark Analysis Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        .plot-container {
            text-align: center;
            margin: 30px 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .summary-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px 15px;
            background-color: #e9ecef;
            border-radius: 5px;
            font-weight: bold;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .toc {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        .toc li {
            margin: 5px 0;
        }
        .toc a {
            text-decoration: none;
            color: #007bff;
        }
        .toc a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="summary-box">
        <h1>Matrix Benchmark Analysis Report</h1>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
    </div>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#executive-summary">Executive Summary</a></li>
            <li><a href="#system-information">System Information</a></li>
            <li><a href="#performance-analysis">Performance Analysis</a></li>
            <li><a href="#visualizations">Visualizations</a></li>
            <li><a href="#recommendations">Recommendations</a></li>
        </ul>
    </div>
    
    <div id="content">
        {{ content }}
    </div>
    
    {% if plot_images %}
    <div id="visualizations">
        <h2>Visualizations</h2>
        
        {% if plot_images.performance_scaling %}
        <div class="plot-container">
            <h3>Performance Scaling Analysis</h3>
            <img src="{{ plot_images.performance_scaling }}" alt="Performance Scaling Plot">
        </div>
        {% endif %}
        
        {% if plot_images.timing_distributions %}
        <div class="plot-container">
            <h3>Timing Distributions</h3>
            <img src="{{ plot_images.timing_distributions }}" alt="Timing Distributions Plot">
        </div>
        {% endif %}
        
        {% if plot_images.system_comparison %}
        <div class="plot-container">
            <h3>System Comparison</h3>
            <img src="{{ plot_images.system_comparison }}" alt="System Comparison Plot">
        </div>
        {% endif %}
    </div>
    {% endif %}
    
    <hr>
    <footer>
        <p><em>Report generated by MatrixBenchmark Analysis Module</em></p>
    </footer>
</body>
</html>
"""
        
        # Convert markdown to HTML (basic conversion)
        html_content = md_content.replace('\n', '<br>\n')
        html_content = html_content.replace('# ', '<h1>').replace('<br>\n', '</h1>\n', 1)
        html_content = html_content.replace('## ', '<h2>').replace('<br>\n', '</h2>\n')
        html_content = html_content.replace('### ', '<h3>').replace('<br>\n', '</h3>\n')
        html_content = html_content.replace('**', '<strong>').replace('**', '</strong>')
        html_content = html_content.replace('*', '<em>').replace('*', '</em>')
        
        # Handle tables (basic)
        lines = html_content.split('\n')
        processed_lines = []
        in_table = False
        
        for line in lines:
            if '|' in line and not in_table:
                in_table = True
                processed_lines.append('<table>')
                processed_lines.append('<tr>')
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                for cell in cells:
                    processed_lines.append(f'<th>{cell}</th>')
                processed_lines.append('</tr>')
            elif '|' in line and in_table:
                if '---' in line:
                    continue  # Skip separator line
                processed_lines.append('<tr>')
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                for cell in cells:
                    processed_lines.append(f'<td>{cell}</td>')
                processed_lines.append('</tr>')
            elif in_table and '|' not in line:
                processed_lines.append('</table>')
                in_table = False
                processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        if in_table:
            processed_lines.append('</table>')
        
        html_content = '\n'.join(processed_lines)
        
        # Create template and render
        template = Template(html_template)
        final_html = template.render(
            content=html_content,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            plot_images=plot_images
        )
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(final_html)
        
        print(f"HTML report saved to {output_path}")
        return output_path
    
    def interactive_analysis(self, results: Optional[List[Dict]] = None):
        """
        Start an interactive analysis session.
        
        Args:
            results: List of benchmark results (uses loaded results if None)
        """
        if results is None:
            results = self.loaded_results
        
        print("=== Interactive Matrix Benchmark Analysis ===")
        print(f"Loaded {len(results)} result files")
        
        data = self.extract_performance_data(results)
        systems = self.get_system_summary(results)
        
        print(f"\nSystems found: {list(systems.keys())}")
        print(f"Total benchmarks: {len(data['matrix_sizes'])}")
        
        while True:
            print("\n--- Analysis Options ---")
            print("1. Show system summary")
            print("2. Show performance statistics")
            print("3. Compare systems")
            print("4. Detect outliers")
            print("5. Generate scaling plot")
            print("6. Generate timing distributions")
            print("7. Generate system comparison")
            print("8. Export to CSV")
            print("9. Generate Markdown report")
            print("10. Generate HTML report")
            print("0. Exit")
            
            try:
                choice = input("\nEnter your choice (0-10): ").strip()
                
                if choice == '0':
                    print("Goodbye!")
                    break
                elif choice == '1':
                    print("\n=== System Summary ===")
                    for system_id, info in systems.items():
                        print(f"\n{system_id}:")
                        print(f"  Chip: {info['chip_type']}")
                        print(f"  Memory: {info['memory_gb']} GB")
                        print(f"  CPU Cores: {info['cpu_cores_total']}")
                        print(f"  Results: {info['results_count']}")
                        
                elif choice == '2':
                    print("\n=== Performance Statistics ===")
                    if data['gflops']:
                        print(f"Average GFLOPS: {np.mean(data['gflops']):.2f}")
                        print(f"Peak GFLOPS: {np.max(data['gflops']):.2f}")
                        print(f"Min GFLOPS: {np.min(data['gflops']):.2f}")
                        print(f"Std Dev: {np.std(data['gflops']):.2f}")
                        print(f"Matrix sizes: {sorted(list(set(data['matrix_sizes'])))}")
                        
                elif choice == '3':
                    ratios = self.calculate_performance_ratios(results)
                    print("\n=== System Comparison ===")
                    if 'ranking' in ratios:
                        print("Performance Ranking:")
                        for i, system in enumerate(ratios['ranking'], 1):
                            perf = ratios['system_performance'][system]
                            print(f"{i}. {system}: {perf['mean']:.2f} GFLOPS")
                            
                elif choice == '4':
                    outliers = self.detect_outliers(results)
                    print(f"\n=== Outlier Detection ===")
                    print(f"Found {outliers.get('count', 0)} outliers ({outliers.get('percentage', 0):.1f}%)")
                    for detail in outliers.get('details', []):
                        print(f"  {detail['system_id']} (Size: {detail['matrix_size']}): {detail['gflops']:.2f} GFLOPS")
                        
                elif choice == '5':
                    if PLOTTING_AVAILABLE:
                        self.plot_performance_scaling(results)
                    else:
                        print("Plotting not available. Please install matplotlib and seaborn.")
                        
                elif choice == '6':
                    if PLOTTING_AVAILABLE:
                        self.plot_timing_distributions(results)
                    else:
                        print("Plotting not available. Please install matplotlib and seaborn.")
                        
                elif choice == '7':
                    if PLOTTING_AVAILABLE:
                        self.plot_system_comparison(results)
                    else:
                        print("Plotting not available. Please install matplotlib and seaborn.")
                        
                elif choice == '8':
                    filename = input("Enter CSV filename (or press Enter for default): ").strip()
                    if not filename:
                        filename = "benchmark_analysis.csv"
                    self.export_to_csv(results, filename)
                    
                elif choice == '9':
                    filename = input("Enter Markdown filename (or press Enter for default): ").strip()
                    if not filename:
                        filename = "benchmark_report.md"
                    self.generate_markdown_report(results, filename)
                    
                elif choice == '10':
                    filename = input("Enter HTML filename (or press Enter for default): ").strip()
                    if not filename:
                        filename = "benchmark_report.html"
                    self.generate_html_report(results, filename)
                    
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function for running analysis from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Matrix Benchmark Results Analysis')
    parser.add_argument('--results-dir', '-d', default='benchmark_results',
                       help='Directory containing benchmark results')
    parser.add_argument('--pattern', '-p', default='*.json',
                       help='File pattern for result files')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Start interactive analysis session')
    parser.add_argument('--plot-scaling', action='store_true',
                       help='Generate performance scaling plot')
    parser.add_argument('--plot-distributions', action='store_true',
                       help='Generate timing distribution plots')
    parser.add_argument('--plot-comparison', action='store_true',
                       help='Generate system comparison plots')
    parser.add_argument('--export-csv', metavar='FILE',
                       help='Export data to CSV file')
    parser.add_argument('--report-md', metavar='FILE',
                       help='Generate Markdown report')
    parser.add_argument('--report-html', metavar='FILE',
                       help='Generate HTML report')
    parser.add_argument('--output-dir', '-o', default='.',
                       help='Output directory for generated files')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = BenchmarkAnalyzer(args.results_dir)
    
    # Load results
    results = analyzer.load_results(args.pattern)
    
    if not results:
        print("No benchmark results found.")
        return
    
    # Change to output directory
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    # Execute requested operations
    if args.interactive:
        analyzer.interactive_analysis(results)
    else:
        if args.plot_scaling:
            analyzer.plot_performance_scaling(results, 'performance_scaling.png')
        
        if args.plot_distributions:
            analyzer.plot_timing_distributions(results, 'timing_distributions.png')
        
        if args.plot_comparison:
            analyzer.plot_system_comparison(results, 'system_comparison.png')
        
        if args.export_csv:
            analyzer.export_to_csv(results, args.export_csv)
        
        if args.report_md:
            analyzer.generate_markdown_report(results, args.report_md)
        
        if args.report_html:
            analyzer.generate_html_report(results, args.report_html)
        
        # If no specific actions requested, show summary
        if not any([args.plot_scaling, args.plot_distributions, args.plot_comparison,
                   args.export_csv, args.report_md, args.report_html]):
            print("\n=== Benchmark Analysis Summary ===")
            
            data = analyzer.extract_performance_data(results)
            systems = analyzer.get_system_summary(results)
            
            print(f"Results loaded: {len(results)} files")
            print(f"Total benchmarks: {len(data['matrix_sizes'])}")
            print(f"Systems: {list(systems.keys())}")
            
            if data['gflops']:
                print(f"Performance range: {min(data['gflops']):.2f} - {max(data['gflops']):.2f} GFLOPS")
                print(f"Average performance: {np.mean(data['gflops']):.2f} GFLOPS")
            
            print("\nUse --interactive flag for detailed analysis or specify specific operations.")


if __name__ == "__main__":
    main()