#!/usr/bin/env python3
"""
Configuration Management for Matrix Benchmark

Provides JSON-based configuration system with schema validation, preset configurations,
and support for command-line overrides. Handles configuration file loading, saving,
validation, and inheritance.
"""

import json
import os
import sys
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import jsonschema
from jsonschema import validate, ValidationError


# Configuration schema for validation
CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "benchmark": {
            "type": "object",
            "properties": {
                "matrix_sizes": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "minItems": 1,
                    "description": "List of matrix sizes to benchmark"
                },
                "num_runs": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of runs per matrix size"
                },
                "warmup_runs": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of warmup runs"
                },
                "seeds": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "description": "Random seeds for reproducibility"
                },
                "dtype": {
                    "type": "string",
                    "enum": ["float32", "float64"],
                    "description": "Data type for matrices"
                }
            },
            "required": ["matrix_sizes", "num_runs", "warmup_runs", "seeds", "dtype"],
            "additionalProperties": False
        },
        "monitoring": {
            "type": "object",
            "properties": {
                "monitor_resources": {
                    "type": "boolean",
                    "description": "Enable resource monitoring during benchmarks"
                },
                "detect_throttling": {
                    "type": "boolean",
                    "description": "Enable thermal throttling detection"
                }
            },
            "required": ["monitor_resources", "detect_throttling"],
            "additionalProperties": False
        },
        "output": {
            "type": "object",
            "properties": {
                "save_results": {
                    "type": "boolean",
                    "description": "Save results to file"
                },
                "output_dir": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Output directory for results"
                },
                "output_prefix": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Prefix for output files"
                }
            },
            "required": ["save_results", "output_dir", "output_prefix"],
            "additionalProperties": False
        },
        "metadata": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Configuration name"
                },
                "description": {
                    "type": "string",
                    "description": "Configuration description"
                },
                "version": {
                    "type": "string",
                    "description": "Configuration version"
                },
                "inherits_from": {
                    "type": "string",
                    "description": "Base configuration to inherit from"
                }
            },
            "additionalProperties": True
        }
    },
    "required": ["benchmark", "monitoring", "output"],
    "additionalProperties": False
}


@dataclass
class BenchmarkConfigExtended:
    """Extended configuration class that supports JSON serialization and inheritance."""
    # Benchmark parameters
    matrix_sizes: List[int]
    num_runs: int
    warmup_runs: int
    seeds: List[int]
    dtype: str
    
    # Monitoring parameters
    monitor_resources: bool
    detect_throttling: bool
    
    # Output parameters
    save_results: bool
    output_dir: str
    output_prefix: str
    
    # Metadata (optional)
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = "1.0"
    inherits_from: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        config_dict = {
            "benchmark": {
                "matrix_sizes": self.matrix_sizes,
                "num_runs": self.num_runs,
                "warmup_runs": self.warmup_runs,
                "seeds": self.seeds,
                "dtype": self.dtype
            },
            "monitoring": {
                "monitor_resources": self.monitor_resources,
                "detect_throttling": self.detect_throttling
            },
            "output": {
                "save_results": self.save_results,
                "output_dir": self.output_dir,
                "output_prefix": self.output_prefix
            }
        }
        
        # Add metadata if present
        metadata = {}
        if self.name:
            metadata["name"] = self.name
        if self.description:
            metadata["description"] = self.description
        if self.version:
            metadata["version"] = self.version
        if self.inherits_from:
            metadata["inherits_from"] = self.inherits_from
            
        if metadata:
            config_dict["metadata"] = metadata
            
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BenchmarkConfigExtended':
        """Create instance from dictionary."""
        benchmark = config_dict["benchmark"]
        monitoring = config_dict["monitoring"]
        output = config_dict["output"]
        metadata = config_dict.get("metadata", {})
        
        return cls(
            matrix_sizes=benchmark["matrix_sizes"],
            num_runs=benchmark["num_runs"],
            warmup_runs=benchmark["warmup_runs"],
            seeds=benchmark["seeds"],
            dtype=benchmark["dtype"],
            monitor_resources=monitoring["monitor_resources"],
            detect_throttling=monitoring["detect_throttling"],
            save_results=output["save_results"],
            output_dir=output["output_dir"],
            output_prefix=output["output_prefix"],
            name=metadata.get("name"),
            description=metadata.get("description"),
            version=metadata.get("version", "1.0"),
            inherits_from=metadata.get("inherits_from")
        )
    
    def to_original_config(self):
        """Convert to original BenchmarkConfig format for compatibility."""
        # Import here to avoid circular imports
        from benchmark import BenchmarkConfig
        
        return BenchmarkConfig(
            matrix_sizes=self.matrix_sizes,
            num_runs=self.num_runs,
            warmup_runs=self.warmup_runs,
            seeds=self.seeds,
            dtype=self.dtype,
            monitor_resources=self.monitor_resources,
            detect_throttling=self.detect_throttling,
            save_results=self.save_results,
            output_dir=self.output_dir,
            output_prefix=self.output_prefix
        )


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class ConfigManager:
    """Manager for loading, saving, and validating benchmark configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Cache for loaded configurations to support inheritance
        self._config_cache: Dict[str, BenchmarkConfigExtended] = {}
    
    def validate_config(self, config_dict: Dict[str, Any]) -> None:
        """Validate configuration against schema."""
        try:
            validate(instance=config_dict, schema=CONFIG_SCHEMA)
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e.message}")
    
    def load_config(self, config_path: Union[str, Path]) -> BenchmarkConfigExtended:
        """Load configuration from JSON file with inheritance support."""
        config_path = Path(config_path)
        
        # Check cache first
        cache_key = str(config_path.absolute())
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # Load configuration file
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        
        # Validate configuration
        self.validate_config(config_dict)
        
        # Handle inheritance
        if "metadata" in config_dict and "inherits_from" in config_dict["metadata"]:
            base_config_name = config_dict["metadata"]["inherits_from"]
            base_config = self._load_base_config(base_config_name)
            config_dict = self._merge_configs(base_config.to_dict(), config_dict)
        
        # Create configuration object
        config = BenchmarkConfigExtended.from_dict(config_dict)
        
        # Cache the result
        self._config_cache[cache_key] = config
        
        return config
    
    def _load_base_config(self, base_config_name: str) -> BenchmarkConfigExtended:
        """Load base configuration for inheritance."""
        # Try different paths for base configuration
        possible_paths = [
            self.config_dir / f"{base_config_name}.json",
            Path(base_config_name),
            Path(f"{base_config_name}.json")
        ]
        
        for path in possible_paths:
            if path.exists():
                return self.load_config(path)
        
        raise ConfigurationError(f"Base configuration not found: {base_config_name}")
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge base configuration with override configuration."""
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                result[key] = self._merge_configs(result[key], value)
            else:
                # Override value
                result[key] = value
        
        return result
    
    def save_config(self, config: BenchmarkConfigExtended, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.to_dict()
        
        # Validate before saving
        self.validate_config(config_dict)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
        except IOError as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def list_configs(self) -> List[str]:
        """List available configuration files."""
        config_files = []
        for config_file in self.config_dir.glob("*.json"):
            config_files.append(config_file.stem)
        return sorted(config_files)
    
    def create_preset_configs(self) -> None:
        """Create preset configuration files."""
        presets = {
            "default": self._create_default_config(),
            "quick": self._create_quick_config(),
            "standard": self._create_standard_config(),
            "comprehensive": self._create_comprehensive_config(),
            "comparison": self._create_comparison_config()
        }
        
        for name, config in presets.items():
            config_path = self.config_dir / f"{name}.json"
            self.save_config(config, config_path)
            print(f"Created preset configuration: {config_path}")
    
    def _create_default_config(self) -> BenchmarkConfigExtended:
        """Create default configuration."""
        return BenchmarkConfigExtended(
            matrix_sizes=[500, 1000, 1500, 2000],
            num_runs=5,
            warmup_runs=1,
            seeds=[42, 123, 456],
            dtype="float64",
            monitor_resources=True,
            detect_throttling=True,
            save_results=True,
            output_dir="benchmark_results",
            output_prefix="matrix_benchmark",
            name="Default Configuration",
            description="Balanced configuration for general benchmarking",
            version="1.0"
        )
    
    def _create_quick_config(self) -> BenchmarkConfigExtended:
        """Create quick test configuration."""
        return BenchmarkConfigExtended(
            matrix_sizes=[500, 1000],
            num_runs=3,
            warmup_runs=1,
            seeds=[42],
            dtype="float64",
            monitor_resources=False,
            detect_throttling=False,
            save_results=True,
            output_dir="benchmark_results",
            output_prefix="quick_test",
            name="Quick Test",
            description="Fast configuration for quick testing and development",
            version="1.0"
        )
    
    def _create_standard_config(self) -> BenchmarkConfigExtended:
        """Create standard configuration."""
        return BenchmarkConfigExtended(
            matrix_sizes=[500, 1000, 1500, 2000, 2500],
            num_runs=5,
            warmup_runs=2,
            seeds=[42, 123, 456],
            dtype="float64",
            monitor_resources=True,
            detect_throttling=True,
            save_results=True,
            output_dir="benchmark_results",
            output_prefix="standard_benchmark",
            name="Standard Benchmark",
            description="Standard configuration for regular performance testing",
            version="1.0"
        )
    
    def _create_comprehensive_config(self) -> BenchmarkConfigExtended:
        """Create comprehensive configuration."""
        return BenchmarkConfigExtended(
            matrix_sizes=[250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000],
            num_runs=10,
            warmup_runs=3,
            seeds=[42, 123, 456, 789, 999],
            dtype="float64",
            monitor_resources=True,
            detect_throttling=True,
            save_results=True,
            output_dir="benchmark_results",
            output_prefix="comprehensive_test",
            name="Comprehensive Test",
            description="Thorough configuration for detailed performance analysis",
            version="1.0"
        )
    
    def _create_comparison_config(self) -> BenchmarkConfigExtended:
        """Create configuration optimized for cross-machine comparison."""
        return BenchmarkConfigExtended(
            matrix_sizes=[1000, 2000, 3000],
            num_runs=7,
            warmup_runs=2,
            seeds=[42, 123, 456],  # Fixed seeds for reproducibility
            dtype="float64",
            monitor_resources=True,
            detect_throttling=True,
            save_results=True,
            output_dir="benchmark_results",
            output_prefix="comparison_benchmark",
            name="Cross-Machine Comparison",
            description="Configuration optimized for reproducible cross-machine comparisons",
            version="1.0"
        )
    
    def apply_cli_overrides(self, config: BenchmarkConfigExtended, overrides: Dict[str, Any]) -> BenchmarkConfigExtended:
        """Apply command-line overrides to configuration."""
        # Create a copy to avoid modifying the original
        config_dict = config.to_dict()
        
        # Map CLI arguments to configuration paths
        cli_mapping = {
            "sizes": ("benchmark", "matrix_sizes"),
            "runs": ("benchmark", "num_runs"),
            "warmup": ("benchmark", "warmup_runs"),
            "seeds": ("benchmark", "seeds"),
            "dtype": ("benchmark", "dtype"),
            "no_monitor": ("monitoring", "monitor_resources"),  # inverted
            "no_thermal": ("monitoring", "detect_throttling"),  # inverted
            "no_save": ("output", "save_results"),  # inverted
            "output_dir": ("output", "output_dir"),
            "output_prefix": ("output", "output_prefix")
        }
        
        for cli_key, value in overrides.items():
            if cli_key in cli_mapping and value is not None:
                section, config_key = cli_mapping[cli_key]
                
                # Handle inverted boolean flags
                if cli_key.startswith("no_"):
                    value = not value
                
                config_dict[section][config_key] = value
        
        # Validate the modified configuration
        self.validate_config(config_dict)
        
        return BenchmarkConfigExtended.from_dict(config_dict)
    
    def get_config_info(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a configuration file without fully loading it."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            return {"error": f"Configuration file not found: {config_path}"}
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            metadata = config_dict.get("metadata", {})
            benchmark = config_dict.get("benchmark", {})
            
            return {
                "name": metadata.get("name", config_path.stem),
                "description": metadata.get("description", "No description"),
                "version": metadata.get("version", "Unknown"),
                "inherits_from": metadata.get("inherits_from"),
                "matrix_sizes": benchmark.get("matrix_sizes", []),
                "num_runs": benchmark.get("num_runs", 0),
                "file_path": str(config_path)
            }
        
        except (json.JSONDecodeError, KeyError) as e:
            return {"error": f"Invalid configuration file: {e}"}


def create_default_configs():
    """Utility function to create default configuration files."""
    config_manager = ConfigManager()
    config_manager.create_preset_configs()
    print(f"Created preset configurations in: {config_manager.config_dir}")


if __name__ == "__main__":
    # Create default configurations when run as script
    create_default_configs()