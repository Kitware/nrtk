"""Implements DetectorOTFPerturber which applies detector perturbations using sensor and scenario settings.

Classes:
    DetectorOTFPerturber: Applies OTF-based perturbations to images using specified
    sensor and scenario configurations.

Dependencies:
    - pyBSM for radiance and OTF-related functionalities.
    - nrtk.interfaces.perturb_image.PerturbImage as the base interface for image perturbation.

Example usage:
    sensor = PybsmSensor(...)
    scenario = PybsmScenario(...)
    perturber = DetectorOTFPerturber(sensor=sensor, scenario=scenario)
    perturbed_image, boxes = perturber.perturb(image, boxes)

Notes:
    - The boxes returned from `perturb` are identical to the boxes passed in.
"""

from __future__ import annotations

__all__ = ["DetectorOTFPerturber"]

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

from pybsm.simulation import DetectorSimulator, ImageSimulator  # noqa: E402


class DetectorOTFPerturber(PybsmOTFPerturber):
    """Implements OTF-based image perturbation using detector specifications and atmospheric conditions.

    The `DetectorOTFPerturber` class uses sensor and scenario configurations to apply realistic
    perturbations to images. This includes adjusting for detector width, focal length, and atmospheric
    conditions using pyBSM functionalities.

    See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.

    Attributes:
        sensor (PybsmSensor | None):
            The sensor configuration used to define perturbation parameters.
        scenario (PybsmScenario | None):
            Scenario configuration providing environmental context.
        w_x (float | None):
            Detector width in the x direction (meters).
        w_y (float | None):
            Detector width in the y direction (meters).
        f (float | None):
            Focal length of the detector (meters).
        interp (bool):
            Indicates whether atmospheric database should use interpolation.
    """

    def __init__(
        self,
        sensor: PybsmSensor | None = None,
        scenario: PybsmScenario | None = None,
        w_x: float | None = None,
        w_y: float | None = None,
        f: float | None = None,
        interp: bool = True,
    ) -> None:
        """Initializes the DetectorOTFPerturber.

        Args:
            sensor:
                pyBSM sensor object.
            scenario:
                pyBSM scenario object.
            w_x:
                Detector width in the x direction (m).
            w_y:
                Detector width in the y direction (m).
            f:
                Focal length (m).
            interp:
                a boolean determining whether load_database_atmosphere is used with or without interpolation.

            If a value is provided for w_x, w_y and/or f that value(s) will be used in
            the otf calculation.

            If both sensor and scenario parameters are provided, but not w_x, w_y and/or f, the
            value(s) of w_x, w_y and/or f will come from the sensor and scenario objects.

            If either sensor or scenario parameters are absent, default values
            will be used for both sensor and scenario parameters (except for w_x/w_y/f, as defined
            below).

            If any of w_x, w_y, or f are absent and sensor/scenario objects are also absent,
            the absent value(s) will default to 4um for w_x/w_y and 50mm for f.

        Raises:
            :raises ImportError: If pyBSM is not found, install via `pip install nrtk[pybsm]`.
        """
        # Initialize base class
        super().__init__(sensor=sensor, scenario=scenario, interp=interp)

        # Store perturber-specific overrides
        self._override_w_x = w_x
        self._override_w_y = w_y
        self._override_f = f

        self._simulator = self._create_simulator()

    @override
    def _create_simulator(self) -> ImageSimulator:
        """Create DetectorSimulator with explicit parameters."""
        # If using default sensor/scenario, make adjustments from base class
        if self._use_default_psf:
            self.sensor.D = 0.003
            self.sensor.opt_trans_wavelengths = np.array([0.50e-6, 0.66e-6])
            self.sensor.w_x = 4e-6
            self.sensor.w_y = 4e-6
            self.sensor.f = 50e-3

        # Override values if provided
        if self._override_w_x:
            self.sensor.w_x = self._override_w_x
        if self._override_w_y:
            self.sensor.w_y = self._override_w_y
        if self._override_f:
            self.sensor.f = self._override_f

        pybsm_sensor = self.sensor.create_sensor()
        pybsm_scenario = self.scenario.create_scenario()

        return DetectorSimulator(
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
        cfg["f"] = self.f
        cfg["interp"] = self.interp

        return cfg
