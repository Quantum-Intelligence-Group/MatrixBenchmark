"""
System Information Module for Mac Hardware Profiling

A comprehensive Python module for detecting and profiling Mac hardware,
specifically optimized for M3 Ultra benchmarking and cross-machine comparison.
Provides detailed system information including hardware specifications,
performance baselines, and environment profiling.
"""

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, List, Tuple, Union
import psutil
import numpy as np


@dataclass
class HardwareProfile:
    """Data class for Mac hardware specifications."""
    model: str
    chip_type: str
    cpu_cores_performance: int
    cpu_cores_efficiency: int
    cpu_cores_total: int
    memory_total_gb: float
    memory_bandwidth_gbps: Optional[float]
    gpu_info: Dict[str, Any]
    cache_sizes: Dict[str, int]
    hardware_score: float


@dataclass
class SystemEnvironment:
    """Data class for system environment information."""
    os_version: str
    python_version: str
    numpy_version: str
    blas_library: str
    cpu_frequency: Dict[str, float]
    thermal_state: str
    power_state: str
    background_processes: int
    system_load: List[float]


@dataclass
class PerformanceBaseline:
    """Data class for performance baseline measurements."""
    cpu_benchmark_score: float
    memory_bandwidth_measured: float
    cache_performance: Dict[str, float]
    thermal_throttling_detected: bool
    sustained_performance_ratio: float
    timestamp: float


@dataclass
class SystemProfile:
    """Complete system profile combining all components."""
    hardware: HardwareProfile
    environment: SystemEnvironment
    performance: PerformanceBaseline
    profile_timestamp: float
    profile_version: str = "1.0"


class MacSystemProfiler:
    """Comprehensive Mac system profiler for M3 Ultra benchmarking."""
    
    def __init__(self):
        """Initialize the system profiler."""
        self.profile_version = "1.0"
        self._cached_system_info = None
        self._cached_hardware_info = None
        
    def _run_system_profiler(self, data_type: str) -> Dict[str, Any]:
        """Run system_profiler command and return parsed output."""
        try:
            cmd = ["system_profiler", "-json", data_type]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {}
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _run_sysctl(self, key: str) -> Optional[str]:
        """Run sysctl command to get system information."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", key], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
    
    def _get_cpu_brand(self) -> str:
        """Get CPU brand string."""
        brand = self._run_sysctl("machdep.cpu.brand_string")
        if brand:
            return brand
        # Fallback to platform info
        return platform.processor() or "Unknown"
    
    def _detect_apple_silicon_details(self) -> Tuple[str, str, int, int]:
        """Detect Apple Silicon specific details."""
        # Get CPU brand to determine chip type
        cpu_brand = self._get_cpu_brand()
        
        # Detect chip type from CPU brand
        chip_type = "Unknown"
        if "Apple M3" in cpu_brand:
            if "Ultra" in cpu_brand:
                chip_type = "M3 Ultra"
            elif "Max" in cpu_brand:
                chip_type = "M3 Max"
            elif "Pro" in cpu_brand:
                chip_type = "M3 Pro"
            else:
                chip_type = "M3"
        elif "Apple M2" in cpu_brand:
            if "Ultra" in cpu_brand:
                chip_type = "M2 Ultra"
            elif "Max" in cpu_brand:
                chip_type = "M2 Max"
            elif "Pro" in cpu_brand:
                chip_type = "M2 Pro"
            else:
                chip_type = "M2"
        elif "Apple M1" in cpu_brand:
            if "Ultra" in cpu_brand:
                chip_type = "M1 Ultra"
            elif "Max" in cpu_brand:
                chip_type = "M1 Max"
            elif "Pro" in cpu_brand:
                chip_type = "M1 Pro"
            else:
                chip_type = "M1"
        
        # Get performance and efficiency core counts
        perf_cores = self._run_sysctl("hw.perflevel0.logicalcpu")
        eff_cores = self._run_sysctl("hw.perflevel1.logicalcpu")
        
        perf_cores = int(perf_cores) if perf_cores else 0
        eff_cores = int(eff_cores) if eff_cores else 0
        
        # If we can't get separate core counts, estimate based on chip type
        if perf_cores == 0 and eff_cores == 0:
            total_cores = psutil.cpu_count(logical=False)
            if chip_type == "M3 Ultra":
                perf_cores = 16
                eff_cores = 8
            elif chip_type == "M3 Max":
                perf_cores = 12
                eff_cores = 4
            elif chip_type == "M3 Pro":
                perf_cores = 6
                eff_cores = 6
            elif chip_type == "M3":
                perf_cores = 4
                eff_cores = 4
            else:
                # Fallback estimation
                perf_cores = max(1, total_cores // 2)
                eff_cores = total_cores - perf_cores
        
        return cpu_brand, chip_type, perf_cores, eff_cores
    
    def _get_memory_info(self) -> Tuple[float, Optional[float]]:
        """Get memory information including bandwidth estimation."""
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024**3)
        
        # Estimate memory bandwidth based on chip type
        _, chip_type, _, _ = self._detect_apple_silicon_details()
        
        bandwidth_estimates = {
            "M3 Ultra": 800.0,  # GB/s
            "M3 Max": 400.0,
            "M3 Pro": 150.0,
            "M3": 100.0,
            "M2 Ultra": 800.0,
            "M2 Max": 400.0,
            "M2 Pro": 200.0,
            "M2": 100.0,
            "M1 Ultra": 800.0,
            "M1 Max": 400.0,
            "M1 Pro": 200.0,
            "M1": 68.0
        }
        
        estimated_bandwidth = bandwidth_estimates.get(chip_type)
        
        return total_memory_gb, estimated_bandwidth
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        gpu_info = {"integrated": True, "discrete": False, "details": {}}
        
        # Try to get GPU info from system_profiler
        display_info = self._run_system_profiler("SPDisplaysDataType")
        
        if display_info and "SPDisplaysDataType" in display_info:
            displays = display_info["SPDisplaysDataType"]
            if displays:
                gpu_info["details"] = displays[0]
                # Check if it's integrated or discrete
                chipset = displays[0].get("sppci_model", "").lower()
                if "intel" in chipset or "amd" in chipset or "nvidia" in chipset:
                    gpu_info["discrete"] = True
                    gpu_info["integrated"] = False
        
        return gpu_info
    
    def _get_cache_sizes(self) -> Dict[str, int]:
        """Get CPU cache sizes."""
        cache_info = {}
        
        # Get cache sizes from sysctl
        cache_mappings = {
            "l1_instruction": "hw.l1icachesize",
            "l1_data": "hw.l1dcachesize",
            "l2_cache": "hw.l2cachesize",
            "l3_cache": "hw.l3cachesize"
        }
        
        for cache_name, sysctl_key in cache_mappings.items():
            cache_size = self._run_sysctl(sysctl_key)
            if cache_size:
                try:
                    cache_info[cache_name] = int(cache_size)
                except ValueError:
                    cache_info[cache_name] = 0
            else:
                cache_info[cache_name] = 0
        
        return cache_info
    
    def _calculate_hardware_score(self, hardware: HardwareProfile) -> float:
        """Calculate a normalized hardware capability score."""
        # Base score components
        cpu_score = (hardware.cpu_cores_performance * 1.5 + 
                    hardware.cpu_cores_efficiency * 1.0) * 100
        memory_score = hardware.memory_total_gb * 10
        bandwidth_score = (hardware.memory_bandwidth_gbps or 0) * 0.5
        
        # Cache score
        cache_score = sum(hardware.cache_sizes.values()) / 1024  # Convert to KB
        
        # Combined score
        total_score = cpu_score + memory_score + bandwidth_score + cache_score
        
        return round(total_score, 2)
    
    def get_hardware_profile(self) -> HardwareProfile:
        """Get comprehensive hardware profile."""
        if self._cached_hardware_info is None:
            # Get Mac model
            model = platform.machine()
            
            # Get Apple Silicon details
            cpu_brand, chip_type, perf_cores, eff_cores = self._detect_apple_silicon_details()
            total_cores = perf_cores + eff_cores
            
            # Get memory information
            memory_total, memory_bandwidth = self._get_memory_info()
            
            # Get GPU information
            gpu_info = self._get_gpu_info()
            
            # Get cache sizes
            cache_sizes = self._get_cache_sizes()
            
            # Create hardware profile
            hardware = HardwareProfile(
                model=model,
                chip_type=chip_type,
                cpu_cores_performance=perf_cores,
                cpu_cores_efficiency=eff_cores,
                cpu_cores_total=total_cores,
                memory_total_gb=memory_total,
                memory_bandwidth_gbps=memory_bandwidth,
                gpu_info=gpu_info,
                cache_sizes=cache_sizes,
                hardware_score=0.0  # Will be calculated below
            )
            
            # Calculate hardware score
            hardware.hardware_score = self._calculate_hardware_score(hardware)
            
            self._cached_hardware_info = hardware
        
        return self._cached_hardware_info
    
    def _get_blas_library(self) -> str:
        """Detect BLAS library used by NumPy."""
        try:
            # Try to get BLAS configuration
            blas_config = np.show_config()
            if blas_config:
                config_str = str(blas_config)
                if "accelerate" in config_str.lower():
                    return "Accelerate"
                elif "openblas" in config_str.lower():
                    return "OpenBLAS"
                elif "mkl" in config_str.lower():
                    return "Intel MKL"
                elif "blas" in config_str.lower():
                    return "Generic BLAS"
            
            # Alternative method using numpy.distutils
            try:
                from numpy.distutils.system_info import get_info
                blas_info = get_info('blas_opt')
                if blas_info:
                    libraries = blas_info.get('libraries', [])
                    if any('accelerate' in lib.lower() for lib in libraries):
                        return "Accelerate"
                    elif any('openblas' in lib.lower() for lib in libraries):
                        return "OpenBLAS"
                    elif any('mkl' in lib.lower() for lib in libraries):
                        return "Intel MKL"
            except ImportError:
                pass
            
            return "Unknown"
        except Exception:
            return "Unknown"
    
    def _get_cpu_frequency(self) -> Dict[str, float]:
        """Get CPU frequency information."""
        freq_info = {}
        
        try:
            # Get current frequency
            freq = psutil.cpu_freq()
            if freq:
                freq_info["current"] = freq.current / 1000  # Convert to GHz
                freq_info["min"] = freq.min / 1000 if freq.min else 0
                freq_info["max"] = freq.max / 1000 if freq.max else 0
            else:
                freq_info["current"] = 0.0
                freq_info["min"] = 0.0
                freq_info["max"] = 0.0
        except Exception:
            freq_info["current"] = 0.0
            freq_info["min"] = 0.0
            freq_info["max"] = 0.0
        
        return freq_info
    
    def _get_thermal_state(self) -> str:
        """Get thermal state information."""
        try:
            # Try to get thermal state from system
            thermal_state = self._run_sysctl("machdep.xcpm.cpu_thermal_level")
            if thermal_state:
                level = int(thermal_state)
                if level == 0:
                    return "Normal"
                elif level <= 2:
                    return "Warm"
                elif level <= 4:
                    return "Hot"
                else:
                    return "Critical"
            return "Unknown"
        except Exception:
            return "Unknown"
    
    def _get_power_state(self) -> str:
        """Get power management state."""
        try:
            # Check if running on battery or AC power
            battery = psutil.sensors_battery()
            if battery:
                if battery.power_plugged:
                    return "AC Power"
                else:
                    return f"Battery ({battery.percent}%)"
            return "Unknown"
        except Exception:
            return "Unknown"
    
    def get_system_environment(self) -> SystemEnvironment:
        """Get system environment profile."""
        return SystemEnvironment(
            os_version=platform.platform(),
            python_version=sys.version,
            numpy_version=np.__version__,
            blas_library=self._get_blas_library(),
            cpu_frequency=self._get_cpu_frequency(),
            thermal_state=self._get_thermal_state(),
            power_state=self._get_power_state(),
            background_processes=len(psutil.pids()),
            system_load=list(os.getloadavg())
        )
    
    def _benchmark_cpu(self) -> float:
        """Simple CPU benchmark for baseline performance."""
        # Simple matrix operation benchmark
        start_time = time.perf_counter()
        
        # Create a moderately sized matrix operation
        size = 1000
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        
        # Perform matrix multiplication
        c = np.dot(a, b)
        
        # Simple computation to use the result
        result = np.sum(c)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Calculate a simple performance score (operations per second)
        ops = 2 * size**3  # Approximate operations for matrix multiply
        score = ops / execution_time / 1e9  # GFLOPS
        
        return score
    
    def _measure_memory_bandwidth(self) -> float:
        """Measure actual memory bandwidth."""
        # Create large arrays for memory bandwidth test
        size = 50_000_000  # 50M elements
        
        try:
            # Allocate arrays
            a = np.random.rand(size).astype(np.float64)
            b = np.random.rand(size).astype(np.float64)
            
            # Measure bandwidth with vector operations
            start_time = time.perf_counter()
            
            # Perform memory-bound operations
            for _ in range(5):
                c = a + b
                d = a * b
                result = np.sum(c + d)
            
            end_time = time.perf_counter()
            
            # Calculate bandwidth (bytes transferred / time)
            bytes_per_op = size * 8 * 4  # 4 operations, 8 bytes per double
            total_bytes = bytes_per_op * 5  # 5 iterations
            bandwidth = total_bytes / (end_time - start_time) / 1e9  # GB/s
            
            return bandwidth
            
        except MemoryError:
            # Fallback to smaller test if memory is insufficient
            return 0.0
    
    def _measure_cache_performance(self) -> Dict[str, float]:
        """Measure cache performance characteristics."""
        cache_perf = {}
        
        try:
            # L1 cache test (small array)
            size_l1 = 1024  # 1K elements
            a = np.random.rand(size_l1)
            
            start_time = time.perf_counter()
            for _ in range(10000):
                result = np.sum(a)
            end_time = time.perf_counter()
            
            cache_perf["l1_performance"] = 10000 / (end_time - start_time)
            
            # L2 cache test (medium array)
            size_l2 = 100_000  # 100K elements
            b = np.random.rand(size_l2)
            
            start_time = time.perf_counter()
            for _ in range(1000):
                result = np.sum(b)
            end_time = time.perf_counter()
            
            cache_perf["l2_performance"] = 1000 / (end_time - start_time)
            
            # L3/Memory test (large array)
            size_l3 = 10_000_000  # 10M elements
            c = np.random.rand(size_l3)
            
            start_time = time.perf_counter()
            for _ in range(10):
                result = np.sum(c)
            end_time = time.perf_counter()
            
            cache_perf["memory_performance"] = 10 / (end_time - start_time)
            
        except Exception:
            cache_perf = {"l1_performance": 0.0, "l2_performance": 0.0, "memory_performance": 0.0}
        
        return cache_perf
    
    def _detect_thermal_throttling(self) -> bool:
        """Detect if thermal throttling is occurring."""
        # Run a sustained CPU test and monitor performance
        try:
            initial_score = self._benchmark_cpu()
            
            # Wait a moment and run again
            time.sleep(2)
            
            # Run multiple iterations to heat up the CPU
            scores = []
            for _ in range(3):
                score = self._benchmark_cpu()
                scores.append(score)
                time.sleep(1)
            
            # Check if performance degraded significantly
            final_score = np.mean(scores)
            degradation = (initial_score - final_score) / initial_score
            
            # If performance dropped more than 10%, likely throttling
            return degradation > 0.1
            
        except Exception:
            return False
    
    def get_performance_baseline(self) -> PerformanceBaseline:
        """Get performance baseline measurements."""
        cpu_score = self._benchmark_cpu()
        memory_bandwidth = self._measure_memory_bandwidth()
        cache_performance = self._measure_cache_performance()
        throttling = self._detect_thermal_throttling()
        
        # Calculate sustained performance ratio
        sustained_performance = 1.0 - (0.1 if throttling else 0.0)
        
        return PerformanceBaseline(
            cpu_benchmark_score=cpu_score,
            memory_bandwidth_measured=memory_bandwidth,
            cache_performance=cache_performance,
            thermal_throttling_detected=throttling,
            sustained_performance_ratio=sustained_performance,
            timestamp=time.time()
        )
    
    def get_complete_profile(self) -> SystemProfile:
        """Get complete system profile."""
        hardware = self.get_hardware_profile()
        environment = self.get_system_environment()
        performance = self.get_performance_baseline()
        
        return SystemProfile(
            hardware=hardware,
            environment=environment,
            performance=performance,
            profile_timestamp=time.time(),
            profile_version=self.profile_version
        )
    
    def profile_to_json(self, profile: SystemProfile) -> str:
        """Convert system profile to JSON string."""
        return json.dumps(asdict(profile), indent=2, default=str)
    
    def profile_from_json(self, json_str: str) -> SystemProfile:
        """Create system profile from JSON string."""
        data = json.loads(json_str)
        
        # Reconstruct nested dataclasses
        hardware = HardwareProfile(**data["hardware"])
        environment = SystemEnvironment(**data["environment"])
        performance = PerformanceBaseline(**data["performance"])
        
        return SystemProfile(
            hardware=hardware,
            environment=environment,
            performance=performance,
            profile_timestamp=data["profile_timestamp"],
            profile_version=data["profile_version"]
        )
    
    def save_profile(self, profile: SystemProfile, filename: str) -> None:
        """Save system profile to file."""
        with open(filename, 'w') as f:
            f.write(self.profile_to_json(profile))
    
    def load_profile(self, filename: str) -> SystemProfile:
        """Load system profile from file."""
        with open(filename, 'r') as f:
            return self.profile_from_json(f.read())
    
    def compare_profiles(self, profile1: SystemProfile, profile2: SystemProfile) -> Dict[str, Any]:
        """Compare two system profiles."""
        comparison = {
            "hardware_differences": {},
            "performance_differences": {},
            "compatibility_score": 0.0,
            "recommendations": []
        }
        
        # Compare hardware
        hw1, hw2 = profile1.hardware, profile2.hardware
        comparison["hardware_differences"] = {
            "chip_type": (hw1.chip_type, hw2.chip_type),
            "cpu_cores_total": (hw1.cpu_cores_total, hw2.cpu_cores_total),
            "memory_total_gb": (hw1.memory_total_gb, hw2.memory_total_gb),
            "hardware_score": (hw1.hardware_score, hw2.hardware_score)
        }
        
        # Compare performance
        perf1, perf2 = profile1.performance, profile2.performance
        comparison["performance_differences"] = {
            "cpu_benchmark_score": (perf1.cpu_benchmark_score, perf2.cpu_benchmark_score),
            "memory_bandwidth_measured": (perf1.memory_bandwidth_measured, perf2.memory_bandwidth_measured),
            "thermal_throttling": (perf1.thermal_throttling_detected, perf2.thermal_throttling_detected)
        }
        
        # Calculate compatibility score
        hw_score_diff = abs(hw1.hardware_score - hw2.hardware_score) / max(hw1.hardware_score, hw2.hardware_score)
        perf_score_diff = abs(perf1.cpu_benchmark_score - perf2.cpu_benchmark_score) / max(perf1.cpu_benchmark_score, perf2.cpu_benchmark_score)
        
        compatibility_score = max(0, 1.0 - (hw_score_diff + perf_score_diff) / 2)
        comparison["compatibility_score"] = compatibility_score
        
        # Generate recommendations
        if hw1.chip_type != hw2.chip_type:
            comparison["recommendations"].append(f"Different chip types: {hw1.chip_type} vs {hw2.chip_type}")
        
        if abs(hw1.memory_total_gb - hw2.memory_total_gb) > 8:
            comparison["recommendations"].append("Significant memory difference detected")
        
        if perf1.thermal_throttling_detected != perf2.thermal_throttling_detected:
            comparison["recommendations"].append("Thermal throttling behavior differs")
        
        return comparison


def create_system_profile() -> SystemProfile:
    """Convenience function to create a complete system profile."""
    profiler = MacSystemProfiler()
    return profiler.get_complete_profile()


def save_system_profile(filename: str = "system_profile.json") -> str:
    """Convenience function to save current system profile."""
    profiler = MacSystemProfiler()
    profile = profiler.get_complete_profile()
    profiler.save_profile(profile, filename)
    return filename


def load_and_compare_profiles(profile1_file: str, profile2_file: str) -> Dict[str, Any]:
    """Convenience function to load and compare two system profiles."""
    profiler = MacSystemProfiler()
    profile1 = profiler.load_profile(profile1_file)
    profile2 = profiler.load_profile(profile2_file)
    return profiler.compare_profiles(profile1, profile2)


if __name__ == "__main__":
    # Example usage
    print("Mac System Profiler - M3 Ultra Optimized")
    print("=" * 50)
    
    profiler = MacSystemProfiler()
    
    # Get complete profile
    profile = profiler.get_complete_profile()
    
    # Print summary
    print(f"Hardware: {profile.hardware.chip_type}")
    print(f"CPU Cores: {profile.hardware.cpu_cores_performance}P + {profile.hardware.cpu_cores_efficiency}E")
    print(f"Memory: {profile.hardware.memory_total_gb:.1f} GB")
    print(f"Hardware Score: {profile.hardware.hardware_score}")
    print(f"CPU Benchmark: {profile.performance.cpu_benchmark_score:.2f} GFLOPS")
    print(f"Memory Bandwidth: {profile.performance.memory_bandwidth_measured:.2f} GB/s")
    print(f"BLAS Library: {profile.environment.blas_library}")
    print(f"Thermal State: {profile.environment.thermal_state}")
    print(f"Power State: {profile.environment.power_state}")
    
    # Save profile
    filename = save_system_profile()
    print(f"\nProfile saved to: {filename}")