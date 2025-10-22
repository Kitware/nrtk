"""Implements DefocusOTFPerturber for optical defocus simulation via OTF using pybsm and OpenCV.

Classes:
    DefocusOTFPerturber: Simulates defocus effects in images using OTF and PSF calculations.

Dependencies:
    - pybsm: Required for radiance calculations, OTF/PSF handling, and atmosphere loading.
    - numpy: For numerical computations.
"""

from __future__ import annotations

__all__ = ["DefocusOTFPerturber"]

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

from pybsm.simulation import DefocusSimulator, ImageSimulator  # noqa: E402


class DefocusOTFPerturber(PybsmOTFPerturber):
    """Implements image perturbation using defocus and Optical Transfer Function (OTF).

    DefocusOTFPerturber applies optical defocus perturbations to input images based on
    specified sensor and scenario configurations. The perturbation uses the Optical
    Transfer Function (OTF) and Point Spread Function (PSF) for simulation.

    See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.

    Attributes:
        sensor (PybsmSensor | None):
            The sensor configuration for the simulation.
        scenario (PybsmScenario | None):
            The scenario configuration, such as altitude and ground range.
        w_x (float | None):
            Defocus parameter in the x-direction.
        w_y (float | None):
            Defocus parameter in the y-direction.
        interp (bool):
            Whether to interpolate atmosphere data.
        mtf_wavelengths (np.ndarray):
            Array of wavelengths used for Modulation Transfer Function (MTF).
        D (float):
            Lens diameter in meters.
        slant_range (float):
            Slant range in meters, calculated from altitude and ground range.
        ifov (float):
            Instantaneous Field of View (IFOV).

    Methods:
        perturb:
            Applies the defocus effect to the input image.
        __call__:
            Alias for the perturb method.
        get_default_config:
            Provides the default configuration for the perturber.
        from_config:
            Instantiates the perturber from a configuration dictionary.
        get_config:
            Retrieves the current configuration of the perturber instance.
    """

    def __init__(
        self,
        sensor: PybsmSensor | None = None,
        scenario: PybsmScenario | None = None,
        w_x: float | None = None,
        w_y: float | None = None,
        interp: bool = True,
    ) -> None:
        """Initializes a DefocusOTFPerturber instance with the specified parameters.

        Args:
            sensor:
                Sensor configuration for the simulation.
            scenario:
                Scenario configuration (altitude, ground range, etc.).
            w_x:
                the 1/e blur spot radii in the x direction. Defaults to the sensor's value if provided.
            w_y:
                the 1/e blur spot radii in the y direction. Defaults to the sensor's value if provided.
            interp:
                Whether to interpolate atmosphere data. Defaults to True.

            If a value is provided for w_x and/or w_y those values will be used in the otf calculation.

            If both sensor and scenario parameters are provided, but not w_x and/or w_y, the
            value(s) of w_x and/or w_y will come from the sensor and scenario objects.

            If either sensor or scenario parameters are absent, default values will be used for both
            sensor and scenario parameters (except for w_x/w_y as defined below).

            If any of w_x or w_y are absent and sensor/scenario objects are also absent,
            the absent value(s) will default to 0.0 for both.

        Raises:
            :raises ImportError: If pyBSM is not found, install via `pip install nrtk[pybsm]`.
        """
        # Initialize base class
        super().__init__(sensor=sensor, scenario=scenario, interp=interp)

        # Store perturber-specific overrides
        self._override_w_x = w_x
        self._override_w_y = w_y

        self._simulator = self._create_simulator()

    @override
    def _create_simulator(self) -> ImageSimulator:
        """Create DefocusSimulator with explicit parameters."""
        # If using default sensor/scenario, make adjustments from base class
        if self._use_default_psf:
            self.sensor.D = 0.003
            self.sensor.opt_trans_wavelengths = np.array([0.50e-6, 0.66e-6])

        # Override values if provided
        if self._override_w_x:
            self.sensor.w_x = self._override_w_x
        if self._override_w_y:
            self.sensor.w_y = self._override_w_y

        pybsm_sensor = self.sensor.create_sensor()
        pybsm_scenario = self.scenario.create_scenario()
        return DefocusSimulator(
            sensor=pybsm_sensor,
            scenario=pybsm_scenario,
        )

    @override
    def get_config(self) -> dict[str, Any]:
        """Get current configuration including perturber-specific parameters."""
        cfg = super().get_config()
        cfg["sensor"] = to_config_dict(self.sensor) if self.sensor else None
        cfg["scenario"] = to_config_dict(self.scenario) if self.scenario else None
        cfg["w_x"] = self.w_x
        cfg["w_y"] = self.w_y
        cfg["interp"] = self._interp

        return cfg
