"""Implements DefocusOTFPerturber for optical defocus simulation via OTF using pybsm.

Classes:
    DefocusOTFPerturber: Simulates defocus effects in images using OTF and PSF calculations.

Dependencies:
    - pyBSM for OTF-related functionalities.
    - nrtk.impls.perturb.optical.pybsm_otf_perturber.PybsmOTFPerturber for base functionality.

Example usage:
    >>> if not DefocusOTFPerturber.is_usable():
    ...     import pytest
    ...
    ...     pytest.skip("DefocusOTFPerturber is not usable")
    >>> w_x = 4.0e-6
    >>> w_y = 4.5e-6
    >>> perturber = DefocusOTFPerturber(w_x=w_x, w_y=w_y)
    >>> image = np.ones((256, 256, 3))
    >>> img_gsd = 3.19 / 160
    >>> perturbed_image, _ = perturber.perturb(image=image, img_gsd=img_gsd)

"""

from __future__ import annotations

__all__ = ["DefocusOTFPerturber"]

from typing import Any

import numpy as np
from typing_extensions import override

from nrtk.impls.perturb.optical.pybsm_otf_perturber import PybsmOTFPerturber
from nrtk.utils._exceptions import PyBSMImportError
from nrtk.utils._import_guard import import_guard

# Import checks
pybsm_available: bool = import_guard(module_name="pybsm", exception=PyBSMImportError, submodules=["simulation"])

from pybsm.simulation import DefocusSimulator, ImageSimulator  # noqa: E402


class DefocusOTFPerturber(PybsmOTFPerturber):
    """Implements image perturbation using defocus and Optical Transfer Function (OTF).

    DefocusOTFPerturber applies optical defocus perturbations to input images based on
    specified sensor and scenario configurations. The perturbation uses the Optical
    Transfer Function (OTF) and Point Spread Function (PSF) for simulation.

    See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.
    """

    def __init__(
        self,
        *,
        w_x: float | None = None,
        w_y: float | None = None,
        interp: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes a DefocusOTFPerturber instance with the specified parameters.

        Args:
            w_x:
                the 1/e blur spot radii in the x direction. Defaults to the sensor's value if provided.
            w_y:
                the 1/e blur spot radii in the y direction. Defaults to the sensor's value if provided.
            interp:
                Whether to interpolate atmosphere data. Defaults to True.
            kwargs:
                sensor and/or scenario values to modify

            If a value is provided for w_x and/or w_y those values will be used in the otf calculation.

            If both sensor and scenario parameters are provided, but not w_x and/or w_y, the
            value(s) of w_x and/or w_y will come from the sensor and scenario objects.

            If either sensor or scenario parameters are absent, default values will be used for both
            sensor and scenario parameters (except for w_x/w_y as defined below).

            If any of w_x or w_y are absent and sensor/scenario objects are also absent,
            the absent value(s) will default to 0.0 for both.

        Raises:
            ImportError: If pyBSM is not found, install via `pip install nrtk[pybsm]`.
        """
        # Initialize base class (which handles kwargs application to sensor/scenario)
        super().__init__(interp=interp, **kwargs)
        self._use_default_psf = not kwargs

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

        return DefocusSimulator(
            sensor=self.sensor,
            scenario=self.scenario,
        )

    @override
    def get_config(self) -> dict[str, Any]:
        """Get current configuration including perturber-specific parameters."""
        cfg = {}
        cfg["w_x"] = self.w_x
        cfg["w_y"] = self.w_y
        cfg["interp"] = self.interp

        return cfg
