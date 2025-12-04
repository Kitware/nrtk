"""Implements CircularApertureOTFPerturber for circular aperture OTF perturbations with sensor and scenario configs.

Classes:
    CircularApertureOTFPerturber: Implements OTF-based perturbations using a circular aperture
    model, allowing for detailed wavelength and aperture-based image modifications.

Dependencies:
    - pyBSM for OTF-related functionalities.
    - nrtk.impls.perturb.optical.pybsm_otf_perturber.PybsmOTFPerturber for base functionality.

Example usage:
    >>> if not CircularApertureOTFPerturber.is_usable():
    ...     import pytest
    ...
    ...     pytest.skip("CircularApertureOTFPerturber is not usable")
    >>> D = 0.004
    >>> eta = 0.1
    >>> perturber = CircularApertureOTFPerturber(D=D, eta=eta)
    >>> image = np.ones((256, 256, 3))
    >>> img_gsd = 3.19 / 160
    >>> perturbed_image, _ = perturber.perturb(image=image, img_gsd=img_gsd)

Notes:
    - The boxes returned from `perturb` are identical to the boxes passed in.
"""

from __future__ import annotations

__all__ = ["CircularApertureOTFPerturber"]

from collections.abc import Sequence
from typing import Any

import numpy as np
from typing_extensions import override

from nrtk.impls.perturb.optical.pybsm_otf_perturber import PybsmOTFPerturber
from nrtk.utils._exceptions import PyBSMImportError
from nrtk.utils._import_guard import import_guard

# Import checks
pybsm_available: bool = import_guard("pybsm", PyBSMImportError, ["simulation"])

from pybsm.simulation import CircularApertureSimulator, ImageSimulator  # noqa: E402


class CircularApertureOTFPerturber(PybsmOTFPerturber):
    """Applies OTF-based image perturbation using a circular aperture model with sensor and scenario configurations.

    The `CircularApertureOTFPerturber` class uses a circular aperture model to simulate
    image perturbations, allowing for wavelength-specific and sensor-specific modifications
    based on the sensor and scenario configurations.

    See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.
    """

    def __init__(  # noqa: C901
        self,
        mtf_wavelengths: Sequence[float] | None = None,
        mtf_weights: Sequence[float] | None = None,
        D: float | None = None,  # noqa N802
        eta: float | None = None,
        interp: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the CircularApertureOTFPerturber.

        Args:
            mtf_wavelengths:
                a numpy array of wavelengths (m)
            mtf_weights:
                a numpy array of weights for each wavelength contribution (arb)
            D:
                a float representing effective aperture diameter (m)
            eta:
                a float representing relative linear obscuration (arb)
            interp:
                a boolean determining whether load_database_atmosphere is used with or without
                interpolation.
            **kwargs:
                sensor and/or scenario values to modify

            If both sensor and scenario parameters are absent, then default values
            will be used for their parameters

            If none of mtf_wavelengths, mtf_weights, sensor or scenario parameters are provided, the values
            of mtf_wavelengths and mtf_weights will default to [0.50e-6, 0.66e-6] and [1.0, 1.0] respectively

            If sensor and scenario parameters are provided, but not mtf_wavelengths and mtf_weights, the
            values of mtf_wavelengths and mtf_weights will come from the sensor and scenario objects.

            If mtf_wavelengths and mtf_weights are provided by the user, those values will be used
            in the otf calculation.

        Raises:
            ImportError: If pyBSM is not found, install via `pip install nrtk[pybsm]`.
            ValueError: If mtf_wavelengths and mtf_weights are not equal length
            ValueError: If mtf_wavelengths is empty or mtf_weights is empty
        """
        if mtf_wavelengths is not None and len(mtf_wavelengths) == 0:
            raise ValueError("mtf_wavelengths is empty")

        if mtf_weights is not None and len(mtf_weights) == 0:
            raise ValueError("mtf_weights is empty")

        if mtf_wavelengths is not None and mtf_weights is not None and len(mtf_wavelengths) != len(mtf_weights):
            raise ValueError("mtf_wavelengths and mtf_weights are not the same length")

        # Initialize base class (which handles kwargs application to sensor/scenario)
        super().__init__(interp=interp, **kwargs)
        self._use_default_psf = not kwargs

        # Store perturber-specific overrides
        if mtf_wavelengths is not None:
            self._override_mtf_wavelengths = np.asarray(mtf_wavelengths)
        elif self._use_default_psf:
            self._override_mtf_wavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
        else:
            self._override_mtf_wavelengths = None

        if mtf_weights is not None:
            self._override_mtf_weights = np.asarray(mtf_weights)
        elif self._use_default_psf and self._override_mtf_wavelengths is not None:
            self._override_mtf_weights = np.ones(len(self._override_mtf_wavelengths))
        else:
            self._override_mtf_weights = None

        self._override_D: float | None = D
        self._override_eta: float | None = eta

        self._simulator = self._create_simulator()

    @override
    def _create_simulator(self) -> ImageSimulator:
        """Create CircularApertureSimulator with explicit parameters."""
        # If using default sensor/scenario, make adjustments from base class
        if self._use_default_psf:
            self.sensor.D = 0.003
            self.sensor.eta = 0.0

        if self._override_D is not None:
            self.sensor.D = self._override_D
        if self._override_eta is not None:
            self.sensor.eta = self._override_eta

        return CircularApertureSimulator(
            sensor=self.sensor,
            scenario=self.scenario,
            mtf_wavelengths=self._override_mtf_wavelengths,
            mtf_weights=self._override_mtf_weights,
        )

    @override
    def get_config(self) -> dict[str, Any]:
        """Get current configuration including perturber-specific parameters."""
        cfg = {}
        cfg["mtf_wavelengths"] = self.mtf_wavelengths
        cfg["mtf_weights"] = self.mtf_weights
        cfg["D"] = self.D
        cfg["eta"] = self.eta
        cfg["interp"] = self.interp

        return cfg
