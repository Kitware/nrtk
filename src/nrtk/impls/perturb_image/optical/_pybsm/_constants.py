from typing import Any

__all__ = []

import numpy as np

DEFAULT_PYBSM_PARAMS: dict[str, Any] = {
    # Sensor parameters
    "sensor_name": "Sensor",
    "D": 275e-3,
    "f": 4,
    "p_x": 0.008e-3,
    "p_y": None,
    "opt_trans_wavelengths": np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6,
    "optics_transmission": None,
    "eta": 0.0,
    "w_x": None,
    "w_y": None,
    "int_time": 1.0,
    "n_tdi": 1.0,
    "dark_current": 0.0,
    "read_noise": 0.0,
    "max_n": int(100.0e6),
    "bit_depth": 100.0,
    "max_well_fill": 1.0,
    "s_x": 0.0,
    "s_y": 0.0,
    "qe_wavelengths": None,
    "qe": None,
    # Scenario parameters
    "scenario_name": "Scenario",
    "ihaze": 1,
    "altitude": 9000,
    "ground_range": 0,
    "aircraft_speed": 0.0,
    "target_reflectance": 0.15,
    "target_temperature": 295.0,
    "background_reflectance": 0.07,
    "background_temperature": 293.0,
    "ha_wind_speed": 21.0,
    "cn2_at_1m": 1.7e-14,
    "interp": True,
    "reflectance_range": np.array([0.05, 0.5]),
}
