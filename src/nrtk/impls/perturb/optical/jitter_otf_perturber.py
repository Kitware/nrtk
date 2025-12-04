"""Implements JitterOTFPerturber which applies jitter perturbations using pyBSM with sensor and scenario configs.

Classes:
    JitterOTFPerturber: Applies OTF-based jitter perturbations to images using pyBSM.

Dependencies:
    - pyBSM for OTF-related functionalities.
    - nrtk.impls.perturb.optical.pybsm_otf_perturber.PybsmOTFPerturber for base functionality.

Example usage:
    >>> if not JitterOTFPerturber.is_usable():
    ...     import pytest
    ...
    ...     pytest.skip("JitterOTFPerturber is not usable")
    >>> s_x = 0.01
    >>> perturber = JitterOTFPerturber(s_x=s_x)
    >>> image = np.ones((256, 256, 3))
    >>> img_gsd = 3.19 / 160
    >>> perturbed_image, _ = perturber.perturb(image=image, img_gsd=img_gsd)
"""

from __future__ import annotations

from typing import Any

__all__ = ["JitterOTFPerturber"]


import numpy as np
from typing_extensions import override

from nrtk.impls.perturb.optical.pybsm_otf_perturber import PybsmOTFPerturber
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
    """

    def __init__(
        self,
        s_x: float | None = None,
        s_y: float | None = None,
        interp: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the JitterOTFPerturber.

        Args:
            s_x:
                root-mean-squared jitter amplitudes in the x direction (rad).
            s_y:
                root-mean-squared jitter amplitudes in the y direction (rad).
            interp:
                a boolean determining whether load_database_atmosphere is used with or without interpolation.
            kwargs:
                sensor and/or scenario values to modify

            If both sensor and scenario parameters are not present, then default values
            will be used for their parameters

            If neither s_x, s_y, sensor or scenario parameters are provided, the values
            of s_x and s_y will be the default of 0.0 as that results in a nadir view.

            If sensor and scenario parameters are provided, but not s_x and s_y, the
            values of s_x and s_y will come from the sensor and scenario objects.

            If s_x and s_y are ever provided by the user, those values will be used
            in the otf calculation.

        Raises:
            ImportError: If pyBSM is not found
        """
        super().__init__(interp=interp, **kwargs)
        self._use_default_psf = not kwargs

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

        return JitterSimulator(
            sensor=self.sensor,
            scenario=self.scenario,
        )

    @override
    def get_config(self) -> dict[str, Any]:
        """Get current configuration including perturber-specific parameters."""
        cfg = {}
        cfg["s_x"] = self.s_x
        cfg["s_y"] = self.s_y
        cfg["interp"] = self.interp

        return cfg
