"""
Configuration management for SfM experiments.

This module provides:
- YAML configuration loading
- Configuration validation
- Default configuration values
- Type-safe configuration access

Dependencies:
- pyyaml: https://pyyaml.org/

Sample Input: load_config(Path("configs/hilti.yaml"))
Expected Output: Dict with validated configuration parameters
"""

from pathlib import Path
import os
import re
from typing import Any, Dict, List, Optional
import yaml

from loguru import logger

_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")

# BUGFIX: Whitelist of allowed environment variables to prevent information disclosure
# Only allow specific, safe environment variables that are project-related
ALLOWED_ENV_VARS = {
    # System variables (safe)
    'HOME', 'USER', 'PATH', 'TEMP', 'TMP', 'TMPDIR',
    # Project-specific variables (add your custom ones here)
    'LAMAR_BASE_DIR', 'HILTI_BASE_DIR', 'SFM_DATA_DIR',
    'SFM_OUTPUT_DIR', 'SFM_CACHE_DIR',
}


def _get_nested_value(data: Dict[str, Any], path: str) -> Optional[Any]:
    """Resolve dot-separated paths inside a nested dict."""
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _resolve_placeholders(value: Any, root: Dict[str, Any]) -> Any:
    """Recursively substitute ${...} placeholders from config or env vars."""
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(v, root) for v in value]
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            key = match.group(1)

            # BUGFIX: Only allow whitelisted environment variables (security)
            # This prevents information disclosure of sensitive env vars like API keys
            if key in ALLOWED_ENV_VARS:
                env_val = os.environ.get(key)
                if env_val is not None:
                    return env_val

            # Fall back to config-internal lookups
            nested_val = _get_nested_value(root, key)
            if nested_val is not None:
                return str(nested_val)

            # Key not found - leave placeholder as-is for visibility
            return match.group(0)

        # BUGFIX: Apply repeatedly with cycle detection
        new_value = value
        seen_values = {value}  # Track values to detect cycles
        max_iterations = 10  # Safety limit

        for iteration in range(max_iterations):
            updated = _PLACEHOLDER_PATTERN.sub(_replace, new_value)
            if updated == new_value:
                # No more substitutions possible - done
                break
            if updated in seen_values:
                # Cycle detected!
                raise ValueError(
                    f"Circular reference detected in config placeholders. "
                    f"Value '{value}' creates a cycle after {iteration + 1} iterations. "
                    f"Check for circular dependencies in your configuration."
                )
            seen_values.add(updated)
            new_value = updated

        return new_value
    return value


class Config:
    """
    Configuration container with attribute access.

    Allows accessing configuration values using dot notation:
    >>> config = Config({"dataset": {"name": "hilti"}})
    >>> print(config.dataset.name)  # "hilti"
    """

    def __init__(self, data: Dict[str, Any]):
        """Initialize configuration from dictionary."""
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        """String representation of config."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Config({attrs})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: Path) -> Config:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid

    Example:
        >>> config = load_config(Path("configs/hilti.yaml"))
        >>> print(config.dataset.name)
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty configuration file: {config_path}")

    data = _resolve_placeholders(data, data)

    config = Config(data)
    logger.info(f"Configuration loaded successfully")

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Configuration to override base values

    Returns:
        Merged configuration dictionary

    Example:
        >>> base = {"a": 1, "b": {"c": 2}}
        >>> override = {"b": {"c": 3, "d": 4}}
        >>> result = merge_configs(base, override)
        >>> # result = {"a": 1, "b": {"c": 3, "d": 4}}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Config, output_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        output_path: Output file path

    Example:
        >>> save_config(config, Path("output/experiment_config.yaml"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)

    logger.info(f"Configuration saved to: {output_path}")


def get_default_hilti_config() -> Dict[str, Any]:
    """
    Get default Hilti dataset configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "dataset": {
            "name": "hilti",
            "site": 1,
            "base_path": "datasets/hilti",
            "rosbags_dir": "datasets/hilti/rosbags",
            "frames_dir": "datasets/hilti/frames",
            "ground_truth_dir": "datasets/hilti/ground_truth",
            "camera_topic": "/camera/image_raw",
            "target_fps": 0.25,
            "jpeg_quality": 95,
            "sessions": ["sequence_01", "sequence_02", "sequence_03"],
        },
        "experiment": {
            "visit_counts": [1, 2, 3],
            "output_base": "results",
        },
    }


def get_default_colmap_config() -> Dict[str, Any]:
    """
    Get default COLMAP pipeline configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "colmap": {
            "camera_model": "PINHOLE",
            "features": {"max_num_features": 8192, "upright": False},
            "matching": {"method": "exhaustive", "cross_check": True},
            "reconstruction": {
                "multiple_models": True,
                "max_num_models": 50,
                "max_model_overlap": 20,
                "min_model_size": 10,
                "min_num_matches": 15,
            },
            "bundle_adjustment": {
                "refine_focal_length": True,
                "refine_principal_point": False,
                "refine_extra_params": True,
            },
            "num_threads": -1,
        },
        "metrics": {
            "completeness_threshold": 0.10,
            "alignment_method": "sim3",
        },
    }


def get_default_visualization_config() -> Dict[str, Any]:
    """
    Get default visualization configuration.

    Returns:
        Default visualization configuration dictionary

    Example:
        >>> viz_config = get_default_visualization_config()
        >>> print(viz_config['visualization']['enabled'])  # False
        >>> print(viz_config['visualization']['num_samples'])  # 10
    """
    return {
        "visualization": {
            "enabled": False,  # Disabled by default (no overhead)
            "num_samples": 10,  # Number of frames to visualize per pipeline stage
            "max_matches_display": 200,  # Maximum matches to show in match visualizations
            "image_scale": 0.5,  # Scale factor for images (0.5 = 50% size)
            "max_points_display": 10000,  # Maximum 3D points in trajectory visualization
            "enable_features": True,  # Generate feature extraction visualizations
            "enable_matching": True,  # Generate feature matching visualizations
            "enable_reconstruction": True,  # Generate reconstruction visualizations
        }
    }


# Validation
if __name__ == "__main__":
    """
    Validation of config.py functionality.
    Tests YAML loading, Config object access, and merging.
    """
    import sys
    import tempfile

    all_validation_failures = []
    total_tests = 0

    # Test 1: Load valid YAML config
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "test.yaml"

            # Create test config
            test_data = {
                "dataset": {"name": "test_dataset", "value": 42},
                "experiment": {"count": 3},
            }

            with open(config_file, "w") as f:
                yaml.dump(test_data, f)

            # Load config
            config = load_config(config_file)

            # Verify values
            if not hasattr(config, "dataset"):
                all_validation_failures.append("Config loading: Missing 'dataset' attribute")
            elif config.dataset.name != "test_dataset":
                all_validation_failures.append(
                    f"Config loading: Expected name='test_dataset', got '{config.dataset.name}'"
                )
            elif config.dataset.value != 42:
                all_validation_failures.append(
                    f"Config loading: Expected value=42, got {config.dataset.value}"
                )
    except Exception as e:
        all_validation_failures.append(f"Config loading: Exception raised: {e}")

    # Test 2: Config to_dict conversion
    total_tests += 1
    try:
        test_dict = {"a": 1, "b": {"c": 2, "d": 3}}
        config = Config(test_dict)
        result_dict = config.to_dict()

        if result_dict != test_dict:
            all_validation_failures.append(
                f"Config to_dict: Expected {test_dict}, got {result_dict}"
            )
    except Exception as e:
        all_validation_failures.append(f"Config to_dict: Exception raised: {e}")

    # Test 3: Merge configs
    total_tests += 1
    try:
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10, "e": 4}, "f": 5}
        result = merge_configs(base, override)

        expected = {"a": 1, "b": {"c": 10, "d": 3, "e": 4}, "f": 5}

        if result != expected:
            all_validation_failures.append(f"Config merge: Expected {expected}, got {result}")
    except Exception as e:
        all_validation_failures.append(f"Config merge: Exception raised: {e}")

    # Test 4: Save and reload config
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.yaml"

            original_data = {"test": {"value": 123, "name": "test"}}
            original_config = Config(original_data)

            # Save
            save_config(original_config, output_file)

            if not output_file.exists():
                all_validation_failures.append("Config save: Output file was not created")
            else:
                # Reload
                reloaded_config = load_config(output_file)
                reloaded_dict = reloaded_config.to_dict()

                if reloaded_dict != original_data:
                    all_validation_failures.append(
                        f"Config save/reload: Expected {original_data}, got {reloaded_dict}"
                    )
    except Exception as e:
        all_validation_failures.append(f"Config save/reload: Exception raised: {e}")

    # Test 5: Default configurations
    total_tests += 1
    try:
        hilti_config = get_default_hilti_config()
        colmap_config = get_default_colmap_config()
        viz_config = get_default_visualization_config()

        if "dataset" not in hilti_config:
            all_validation_failures.append("Default config: Hilti config missing 'dataset' key")
        if "colmap" not in colmap_config:
            all_validation_failures.append("Default config: COLMAP config missing 'colmap' key")
        if "visualization" not in viz_config:
            all_validation_failures.append("Default config: Visualization config missing 'visualization' key")
        elif viz_config["visualization"]["num_samples"] != 10:
            all_validation_failures.append(
                f"Default config: Visualization num_samples should be 10, got {viz_config['visualization']['num_samples']}"
            )
    except Exception as e:
        all_validation_failures.append(f"Default config: Exception raised: {e}")

    # Test 6: Error handling for missing file
    total_tests += 1
    try:
        try:
            load_config(Path("/nonexistent/config.yaml"))
            all_validation_failures.append(
                "Error handling: Expected FileNotFoundError for missing file"
            )
        except FileNotFoundError:
            # Expected behavior
            pass
    except Exception as e:
        all_validation_failures.append(f"Error handling: Unexpected exception: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Configuration management validated and ready for use")
        sys.exit(0)
