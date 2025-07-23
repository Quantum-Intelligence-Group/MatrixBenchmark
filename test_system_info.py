#!/usr/bin/env python3
"""
Test script for system_info.py module
Demonstrates the functionality of the Mac system profiler.
"""

import time
from system_info import MacSystemProfiler, create_system_profile, save_system_profile


def test_system_profiler():
    """Test the MacSystemProfiler functionality."""
    print("Testing Mac System Profiler")
    print("=" * 50)
    
    profiler = MacSystemProfiler()
    
    # Test hardware profile
    print("\n1. Testing Hardware Profile...")
    hardware = profiler.get_hardware_profile()
    print(f"   Model: {hardware.model}")
    print(f"   Chip: {hardware.chip_type}")
    print(f"   CPU Cores: {hardware.cpu_cores_performance}P + {hardware.cpu_cores_efficiency}E = {hardware.cpu_cores_total}")
    print(f"   Memory: {hardware.memory_total_gb:.1f} GB")
    print(f"   Est. Memory Bandwidth: {hardware.memory_bandwidth_gbps} GB/s")
    print(f"   Hardware Score: {hardware.hardware_score}")
    print(f"   Cache L1D: {hardware.cache_sizes.get('l1_data', 0)} bytes")
    print(f"   Cache L2: {hardware.cache_sizes.get('l2_cache', 0)} bytes")
    
    # Test system environment
    print("\n2. Testing System Environment...")
    environment = profiler.get_system_environment()
    print(f"   OS: {environment.os_version}")
    print(f"   Python: {environment.python_version.split()[0]}")
    print(f"   NumPy: {environment.numpy_version}")
    print(f"   BLAS: {environment.blas_library}")
    print(f"   CPU Freq: {environment.cpu_frequency.get('current', 0):.2f} GHz")
    print(f"   Thermal State: {environment.thermal_state}")
    print(f"   Power State: {environment.power_state}")
    print(f"   Background Processes: {environment.background_processes}")
    print(f"   System Load: {environment.system_load}")
    
    # Test performance baseline
    print("\n3. Testing Performance Baseline...")
    print("   Running benchmarks (this may take a moment)...")
    performance = profiler.get_performance_baseline()
    print(f"   CPU Benchmark: {performance.cpu_benchmark_score:.2f} GFLOPS")
    print(f"   Memory Bandwidth: {performance.memory_bandwidth_measured:.2f} GB/s")
    print(f"   L1 Cache Performance: {performance.cache_performance.get('l1_performance', 0):.2f} ops/sec")
    print(f"   L2 Cache Performance: {performance.cache_performance.get('l2_performance', 0):.2f} ops/sec")
    print(f"   Memory Performance: {performance.cache_performance.get('memory_performance', 0):.2f} ops/sec")
    print(f"   Thermal Throttling: {performance.thermal_throttling_detected}")
    print(f"   Sustained Performance: {performance.sustained_performance_ratio:.2%}")
    
    # Test complete profile
    print("\n4. Testing Complete Profile...")
    profile = profiler.get_complete_profile()
    print(f"   Profile Version: {profile.profile_version}")
    print(f"   Timestamp: {time.ctime(profile.profile_timestamp)}")
    
    # Test JSON serialization
    print("\n5. Testing JSON Serialization...")
    json_str = profiler.profile_to_json(profile)
    print(f"   JSON Length: {len(json_str)} characters")
    
    # Test deserialization
    restored_profile = profiler.profile_from_json(json_str)
    print(f"   Restored Profile Chip: {restored_profile.hardware.chip_type}")
    
    # Test file operations
    print("\n6. Testing File Operations...")
    filename = "test_profile.json"
    profiler.save_profile(profile, filename)
    print(f"   Profile saved to: {filename}")
    
    loaded_profile = profiler.load_profile(filename)
    print(f"   Loaded Profile Chip: {loaded_profile.hardware.chip_type}")
    
    # Test comparison
    print("\n7. Testing Profile Comparison...")
    comparison = profiler.compare_profiles(profile, loaded_profile)
    print(f"   Compatibility Score: {comparison['compatibility_score']:.2%}")
    print(f"   Recommendations: {len(comparison['recommendations'])}")
    
    print("\n✅ All tests completed successfully!")
    
    return profile


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\nDemonstrating Convenience Functions")
    print("=" * 50)
    
    # Create and save profile using convenience functions
    print("\n1. Creating system profile...")
    profile = create_system_profile()
    print(f"   Created profile for: {profile.hardware.chip_type}")
    
    print("\n2. Saving system profile...")
    filename = save_system_profile("demo_profile.json")
    print(f"   Saved to: {filename}")
    
    print("\n✅ Convenience functions work correctly!")


if __name__ == "__main__":
    try:
        # Run main test
        profile = test_system_profiler()
        
        # Run convenience function demo
        demo_convenience_functions()
        
        print(f"\n🎉 System Info Module Test Complete!")
        print(f"Your system: {profile.hardware.chip_type} with {profile.hardware.cpu_cores_total} cores")
        print(f"Performance: {profile.performance.cpu_benchmark_score:.2f} GFLOPS")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()