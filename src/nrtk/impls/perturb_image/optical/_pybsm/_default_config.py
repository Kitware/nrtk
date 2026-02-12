"""Utility functions for PyBSM sensor and scenario configuration management.

This module provides helper functions for loading and managing default PyBSM sensor
and scenario configurations from JSON configuration files.

Functions:
    load_default_config: Load default sensor and scenario configurations from config files.

Example usage:
    from nrtk.impls.perturb_image.optical._pybsm import load_default_config

    # Load Blackfly configuration
    sensor, scenario = load_default_config(preset="blackfly")
"""

from __future__ import annotations

__all__ = ["load_default_config"]

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_default_config(*, preset: str, data_dir: str | None = None) -> dict[str, Any]:  # noqa: C901 - config file resolution with multiple fallback paths
    """Load default sensor and scenario configurations for a given configuration type.

    This function loads pre-defined sensor and scenario configurations from JSON files
    located in the data directory. The configuration files should contain 'sensor' and
    'scenario' keys with the appropriate PyBSM configuration parameters.

    Currently supported presets:
    - "blackfly": Blackfly camera configuration for low-flying UAV imaging systems
    - "sample": Sample configuration for testing

    Args:
        preset:
            String identifier for the configuration type to load. Must be one of the supported configuration types.
        data_dir:
            Optional path to the directory containing configuration files. If None, the function will search in default
            locations. Can be a string path or pathlib.Path object.

    Returns:
        A dictionary containing sensor and scenario parameters

    Raises:
        ValueError:
            If the configuration type is not supported or the configuration file is missing required keys.
        FileNotFoundError:
            If the configuration file cannot be found.
        json.JSONDecodeError:
            If the configuration file contains invalid JSON.
        IOError:
            If there's an error reading the configuration file.
    """
    # Define supported configuration types
    supported_configs = {
        "blackfly": "blackfly.json",
        "sample": "sample.json",
    }

    # Validate configuration type
    if preset not in supported_configs:
        available_configs = ", ".join(supported_configs.keys())
        raise ValueError(
            f"Unsupported configuration type: '{preset}'. Available configurations: {available_configs}",
        )

    # Determine configuration file path
    config_filename = supported_configs[preset]

    # If data_dir is provided, use it as the primary location
    if data_dir is not None:
        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise ValueError(f"Specified data directory does not exist: {data_dir}")

        config_path = data_path / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file '{config_filename}' not found in specified data directory: {data_dir}",
            )
    else:
        # Try different potential locations for the config file
        potential_paths = [
            Path("./data") / config_filename,  # Current notebook directory
            Path(__file__).parent / "_data" / config_filename,  # Relative to this module
        ]

        config_path = None
        for path in potential_paths:
            if path.exists():
                config_path = path
                break

        if config_path is None:
            searched_paths = [str(p) for p in potential_paths]
            raise FileNotFoundError(
                f"Configuration file '{config_filename}' not found. Searched paths: {searched_paths}",
            )

    # Load and parse configuration file
    try:
        with open(config_path, encoding="utf-8") as file:
            config_data = json.load(file)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in configuration file '{config_path}': {e}",
            e.doc,
            e.pos,
        ) from e
    except OSError as e:
        raise OSError(
            f"Error reading configuration file '{config_path}': {e}",
        ) from e

    # Validate required keys exist
    required_keys = ["sensor", "scenario"]
    missing_keys = [key for key in required_keys if key not in config_data]
    if missing_keys:
        raise ValueError(
            f"Configuration file '{config_path}' is missing required keys: {missing_keys}",
        )

    # Extract sensor and scenario configurations
    sensor_config = config_data["sensor"]
    if "opt_trans_wavelengths" in sensor_config:
        sensor_config["opt_trans_wavelengths"] = np.array(sensor_config["opt_trans_wavelengths"])
    if "optics_transmission" in sensor_config:
        sensor_config["optics_transmission"] = np.array(sensor_config["optics_transmission"])
    if "qe_wavelengths" in sensor_config:
        sensor_config["qe_wavelengths"] = np.array(sensor_config["qe_wavelengths"])
    if "qe" in sensor_config:
        sensor_config["qe"] = np.array(sensor_config["qe"])

    scenario_config = config_data["scenario"]

    # Create and return sensor and scenario objects
    return sensor_config | scenario_config
