"""Implements JitterOTFPerturber which applies jitter perturbations using pyBSM with sensor and scenario configs.

Classes:
    JitterOTFPerturber: Applies OTF-based jitter perturbations to images using pyBSM.

Dependencies:
    - pyBSM for OTF and radiance calculations.
    - nrtk.interfaces.perturb_image.PerturbImage for base functionality.

Example usage:
    sensor = PybsmSensor(...)
    scenario = PybsmScenario(...)
    perturber = JitterOTFPerturber(sensor=sensor, scenario=scenario)
    perturbed_image = perturber.perturb(image)
"""

from __future__ import annotations

__all__ = ["JitterOTFPerturber"]

from typing import Any

import numpy as np
from smqtk_core.configuration import (
    to_config_dict,
)
from typing_extensions import override

from nrtk.impls.perturb_image.pybsm.pybsm_otf_perturber import PybsmOTFPerturber
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.utils._exceptions import PyBSMImportError
from nrtk.utils._import_guard import import_guard

# Import checks
pybsm_available: bool = import_guard("pybsm", PyBSMImportError, ["simulation"])

from pybsm.simulation import ImageSimulator, JitterSimulator  # noqa: E402


class JitterOTFPerturber(PybsmOTFPerturber):
    """Implements image perturbation using jitter and Optical Transfer Function (OTF).

    This class applies realistic perturbations to images based on sensor and scenario configurations,
    leveraging Optical Transfer Function (OTF) modeling through the pyBSM library. Perturbations include
    jitter effects that simulate real-world distortions in optical imaging systems.

    See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.

    Attributes:
        sensor (PybsmSensor | None):
            The sensor configuration used to define perturbation parameters.
        scenario (PybsmScenario | None):
            Scenario configuration providing environmental context for perturbations.
        additional_params (dict):
            Additional configuration options for customizing perturbations.

    Methods:
        perturb(image):
            Applies the jitter-based OTF perturbation to the provided image.
        get_config():
            Returns the configuration for the current instance.
        from_config(config_dict):
            Instantiates from a configuration dictionary.
    """

    def __init__(
        self,
        sensor: PybsmSensor | None = None,
        scenario: PybsmScenario | None = None,
        s_x: float | None = None,
        s_y: float | None = None,
        interp: bool = True,
    ) -> None:
        """Initializes the JitterOTFPerturber.

        Args:
            sensor:
                pyBSM sensor object.
            scenario:
                pyBSM scenario object
            s_x:
                root-mean-squared jitter amplitudes in the x direction (rad).
            s_y:
                root-mean-squared jitter amplitudes in the y direction (rad).
            interp:
                a boolean determining whether load_database_atmosphere is used with or without interpolation.

            If both sensor and scenario parameters are not present, then default values
            will be used for their parameters

            If neither s_x, s_y, sensor or scenario parameters are provided, the values
            of s_x and s_y will be the default of 0.0 as that results in a nadir view.

            If sensor and scenario parameters are provided, but not s_x and s_y, the
            values of s_x and s_y will come from the sensor and scenario objects.

            If s_x and s_y are ever provided by the user, those values will be used
            in the otf calculation.

        Raises:
            :raises ImportError: If pyBSM is not found
        """
        # Initialize base class
        super().__init__(sensor=sensor, scenario=scenario, interp=interp)

        # Store perturber-specific overrides
        self._override_s_x = s_x
        self._override_s_y = s_y

        self._simulator = self._create_simulator()

    @override
    def _create_simulator(self) -> ImageSimulator:
        """Create JitterSimulator with explicit parameters."""
        # If using default sensor/scenario, make adjustments from base class
        if self._use_default_psf:
            self.sensor.D = 0.003
            self.sensor.opt_trans_wavelengths = np.array([0.50e-6, 0.66e-6])

        # Override values if provided
        if self._override_s_x:
            self.sensor.s_x = self._override_s_x
        if self._override_s_y:
            self.sensor.s_y = self._override_s_y

        pybsm_sensor = self.sensor.create_sensor()
        pybsm_scenario = self.scenario.create_scenario()

        return JitterSimulator(
            sensor=pybsm_sensor,
            scenario=pybsm_scenario,
        )

    @override
    def get_config(self) -> dict[str, Any]:
        """Get current configuration including perturber-specific parameters."""
        cfg = super().get_config()
        cfg["sensor"] = to_config_dict(self._sensor) if self._sensor else None
        cfg["scenario"] = to_config_dict(self._scenario) if self._scenario else None
        cfg["s_x"] = self.s_x
        cfg["s_y"] = self.s_y
        cfg["interp"] = self._interp

        return cfg
