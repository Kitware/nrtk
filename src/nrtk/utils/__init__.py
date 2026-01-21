"""Package for nrtk utils needed for carrying out pertubations."""

from nrtk.utils._pybsm import default_sensor_scenario

default_sensor_scenario.__module__ = __name__

__all__ = ["default_sensor_scenario"]
