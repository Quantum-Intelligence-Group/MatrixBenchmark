#!/usr/bin/env python3
"""
Example usage of the system_info module for Mac system profiling.
Demonstrates how to use the system profiler for M3 Ultra benchmarking.
"""

import time
from system_info import MacSystemProfiler, create_system_profile


def main():
    """Main example demonstrating system profiling capabilities."""
    print("Mac System Profiling Example")
    print("=" * 50)
    
    # Create a system profiler
    profiler = MacSystemProfiler()
    
    # Get a quick hardware overview
    print("\n1. Hardware Overview:")
    hardware = profiler.get_hardware_profile()
    print(f"   System: {hardware.chip_type}")
    print(f"   CPU: {hardware.cpu_cores_performance}P + {hardware.cpu_cores_efficiency}E cores")
    print(f"   Memory: {hardware.memory_total_gb:.1f} GB @ {hardware.memory_bandwidth_gbps} GB/s")
    print(f"   Hardware Score: {hardware.hardware_score:.0f}")
    
    # Get environment information
    print("\n2. Environment Information:")
    env = profiler.get_system_environment()
    print(f"   OS: {env.os_version.split('-')[0]}")
    print(f"   Python: {env.python_version.split()[0]}")
    print(f"   NumPy: {env.numpy_version}")
    print(f"   BLAS: {env.blas_library}")
    print(f"   CPU Frequency: {env.cpu_frequency.get('current', 0):.1f} GHz")
    print(f"   Thermal State: {env.thermal_state}")
    print(f"   Power State: {env.power_state}")
    
    # Run performance benchmarks
    print("\n3. Performance Benchmarks:")
    print("   Running performance tests...")
    perf = profiler.get_performance_baseline()
    print(f"   CPU Performance: {perf.cpu_benchmark_score:.1f} GFLOPS")
    print(f"   Memory Bandwidth: {perf.memory_bandwidth_measured:.1f} GB/s")
    print(f"   Cache Performance: L1={perf.cache_performance.get('l1_performance', 0):.0f} ops/s")
    print(f"   Thermal Throttling: {'Yes' if perf.thermal_throttling_detected else 'No'}")
    print(f"   Sustained Performance: {perf.sustained_performance_ratio:.1%}")
    
    # Create and save complete profile
    print("\n4. Complete Profile:")
    profile = create_system_profile()
    
    # Save the profile
    filename = f"system_profile_{int(time.time())}.json"
    profiler.save_profile(profile, filename)
    print(f"   Profile saved to: {filename}")
    
    # Demonstrate profile loading and comparison
    print("\n5. Profile Management:")
    loaded_profile = profiler.load_profile(filename)
    print(f"   Loaded profile: {loaded_profile.hardware.chip_type}")
    
    # Compare with itself (should be 100% compatible)
    comparison = profiler.compare_profiles(profile, loaded_profile)
    print(f"   Self-comparison score: {comparison['compatibility_score']:.1%}")
    
    # Show JSON serialization
    print("\n6. JSON Serialization:")
    json_data = profiler.profile_to_json(profile)
    print(f"   JSON size: {len(json_data):,} characters")
    
    # Summary for benchmarking
    print("\n" + "=" * 50)
    print("BENCHMARKING SUMMARY")
    print("=" * 50)
    print(f"System: {profile.hardware.chip_type}")
    print(f"Hardware Score: {profile.hardware.hardware_score:.0f}")
    print(f"CPU Performance: {profile.performance.cpu_benchmark_score:.1f} GFLOPS")
    print(f"Memory Bandwidth: {profile.performance.memory_bandwidth_measured:.1f} GB/s")
    print(f"Ready for M3 Ultra benchmarking: {'Yes' if 'M3' in profile.hardware.chip_type else 'Compatible'}")
    
    return profile


if __name__ == "__main__":
    try:
        profile = main()
        print(f"\n✅ System profiling completed successfully!")
        print(f"Profile timestamp: {time.ctime(profile.profile_timestamp)}")
    except Exception as e:
        print(f"\n❌ Error during profiling: {e}")
        import traceback
        traceback.print_exc()