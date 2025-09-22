"""Utility functions for PyBSM sensor and scenario configuration management.

This module provides helper functions for loading and managing default PyBSM sensor
and scenario configurations from JSON configuration files.

Functions:
    default_sensor_scenario: Load default sensor and scenario configurations from config files.

Example usage:
    from nrtk.impls.perturb_image.pybsm.utils import default_sensor_scenario

    # Load UAV configuration
    sensor, scenario = default_sensor_scenario("uav")

    # Use in perturbation factory
    perturber_factory = CustomPybsmPerturbImageFactory(
        sensor=sensor,
        scenario=scenario,
        theta_keys=["ground_range"],
        thetas=[ground_range_values]
    )
"""

from __future__ import annotations

import json
from pathlib import Path

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor


def default_sensor_scenario(config_type: str, data_dir: str | None = None) -> tuple[PybsmSensor, PybsmScenario]:  # noqa C901
    """Load default sensor and scenario configurations for a given configuration type.

    This function loads pre-defined sensor and scenario configurations from JSON files
    located in the data directory. The configuration files should contain 'sensor' and
    'scenario' keys with the appropriate PyBSM configuration parameters.

    Currently supported configuration types:
    - "uav": Unmanned Aerial Vehicle configuration with typical parameters for
             drone-based imaging systems

    Future configuration types may include:
    - "satellite": Satellite-based imaging system configuration
    - "uav_high": High-altitude UAV configuration
    - "ground": Ground-based imaging system configuration

    Args:
        config_type: String identifier for the configuration type to load.
                    Must be one of the supported configuration types.
        data_dir: Optional path to the directory containing configuration files.
                 If None, the function will search in default locations.
                 Can be a string path or pathlib.Path object.

    Returns:
        A tuple containing:
            - PybsmSensor: Configured sensor object with loaded parameters
            - PybsmScenario: Configured scenario object with loaded parameters

    Raises:
        ValueError: If the configuration type is not supported or the
                   configuration file is missing required keys.
        FileNotFoundError: If the configuration file cannot be found.
        json.JSONDecodeError: If the configuration file contains invalid JSON.
        IOError: If there's an error reading the configuration file.
    """
    # Define supported configuration types
    supported_configs = {
        "uav": "uav_default_config_nrtk.json",
        # "satellite": "satellite_default_config_nrtk.json",
        # "uav_high": "uav_high_default_config_nrtk.json",
        # "ground": "ground_default_config_nrtk.json",
    }

    # Validate configuration type
    if config_type not in supported_configs:
        available_configs = ", ".join(supported_configs.keys())
        raise ValueError(
            f"Unsupported configuration type: '{config_type}'. Available configurations: {available_configs}",
        )

    # Determine configuration file path
    config_filename = supported_configs[config_type]

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
            Path(__file__).parent / "data" / config_filename,  # Relative to this module
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
    scenario_config = config_data["scenario"]

    # Create and return sensor and scenario objects
    try:
        sensor = PybsmSensor.from_config(sensor_config)
        scenario = PybsmScenario.from_config(scenario_config)
    except Exception as e:
        raise RuntimeError(
            f"Error creating sensor or scenario objects from configuration: {e}",
        ) from e
    return sensor, scenario
