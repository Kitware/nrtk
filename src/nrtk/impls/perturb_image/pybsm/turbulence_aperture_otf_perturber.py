"""Implements TurbulenceApertureOTFPerturber for turbulence aperture-based OTF image perturbations using pyBSM.

Classes:
    TurbulenceApertureOTFPerturber: Applies OTF-based perturbations with turbulence and aperture
    effects to images, utilizing pyBSM functionalities.

Dependencies:
    - pyBSM for radiance and OTF-related calculations.
    - nrtk.interfaces.perturb_image.PerturbImage as the base interface for image perturbation.

Example usage:
    sensor = PybsmSensor(...)
    scenario = PybsmScenario(...)
    perturber = TurbulenceApertureOTFPerturber(sensor=sensor, scenario=scenario)
    perturbed_image, boxes = perturber.perturb(image, boxes)

Notes:
    - The boxes returned from `perturb` are identical to the boxes passed in.
"""

from __future__ import annotations

__all__ = ["TurbulenceApertureOTFPerturber"]

from collections.abc import Sequence
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

from pybsm.simulation import ImageSimulator, TurbulenceApertureSimulator  # noqa: E402


class TurbulenceApertureOTFPerturber(PybsmOTFPerturber):
    """Implements OTF-based image perturbation with turbulence and aperture effects.

    The `TurbulenceApertureOTFPerturber` class simulates image degradation due to atmospheric
    turbulence and optical aperture effects, using pyBSM sensor and scenario configurations.
    It supports adjustable wavelengths, weights, and other environmental parameters for
    realistic perturbations.

    See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.

    Attributes:
        sensor (PybsmSensor | None):
            Sensor configuration for the perturbation.
        scenario (PybsmScenario | None):
            Scenario settings applied during perturbation.
        mtf_wavelengths (Sequence[float]):
            Wavelengths used in MTF calculations.
        mtf_weights (Sequence[float]):
            Weights associated with each wavelength.
        altitude (float):
            Altitude of the imaging platform.
        slant_range (float):
            Line-of-sight distance between platform and target.
        D (float):
            Effective aperture diameter.
        ha_wind_speed (float):
            High-altitude wind speed affecting turbulence profile.
        cn2_at_1m (float):
            Refractive index structure parameter at ground level.
        int_time (float):
            Integration time for imaging.
        n_tdi (float):
            Number of time-delay integration stages.
        aircraft_speed (float):
            Apparent atmospheric velocity.
        interp (bool):
            Indicates whether to use interpolated atmospheric data.
    """

    def __init__(  # noqa: C901
        self,
        sensor: PybsmSensor | None = None,
        scenario: PybsmScenario | None = None,
        mtf_wavelengths: Sequence[float] | None = None,
        mtf_weights: Sequence[float] | None = None,
        altitude: float | None = None,
        slant_range: float | None = None,
        D: float | None = None,  # noqa: N803
        ha_wind_speed: float | None = None,
        cn2_at_1m: float | None = None,
        int_time: float | None = None,
        n_tdi: float | None = None,
        aircraft_speed: float | None = None,
        interp: bool = True,
    ) -> None:
        """Initializes the TurbulenceApertureOTFPerturber.

        Args:
            sensor:
                pyBSM sensor object
            scenario:
                pyBSM scenario object
            mtf_wavelengths:
                a sequence of wavelengths (m)
            mtf_weights:
                a sequence of weights for each wavelength contribution (arb)
            altitude:
                height of the aircraft above the ground (m)
            slant_range:
                line-of-sight range between the aircraft and target (target is assumed
                to be on the ground) (m)
            D:
                effective aperture diameter (m)
            ha_wind_speed:
                the high altitude windspeed (m/s); used to calculate the turbulence profile
            cn2_at_1m:
                the refractive index structure parameter "near the ground" (e.g. at
                h = 1 m); used to calculate the turbulence profile
            int_time:
                dwell (i.e. integration) time (seconds)
            n_tdi:
                the number of time-delay integration stages (relevant only when TDI cameras
                are used. For CMOS cameras, the value can be assumed to be 1.0)
            aircraft_speed:
                apparent atmospheric velocity (m/s); this can just be the windspeed
                at the sensor position if the sensor is stationary
            interp:
                a boolean determining whether load_database_atmosphere is used with or without
                interpolation

            If both sensor and scenario parameters are absent, then default values will be used for
            their parameters.

            If any of the individual or sensor/scenario parameters are absent, the following values
            will be set as defaults for the absent values:

            mtf_wavelengths = [0.50e-6, 0.66e-6]
            mtf_weights = [1.0, 1.0]
            altitude = 250
            slant_range = 250  # slant_range = altitude
            D = 40e-3 #m
            ha_wind_speed = 0
            cn2_at_1m = 1.7e-14
            int_time = 30e-3 #s
            n_tdi = 1.0
            aircraft_speed = 0

            If sensor and scenario parameters are provided, but not the other individial parameters,
            the values will come from the sensor and scenario objects.

            If individial parameter values are provided by the user, those values will be used
            in the otf calculation.

        Raises:
            :raises ImportError: If pyBSM is not found, install via `pip install nrtk[pybsm]`.
            :raises ImportError: If pyBSM is not found, install via `pip install nrtk[pybsm]`.
            :raises ValueError: If mtf_wavelengths and mtf_weights are not equal length
            :raises ValueError: If mtf_wavelengths is empty or mtf_weights is empty
            :raises ValueError: If cn2at1m <= 0.0
        """
        if mtf_wavelengths is not None and len(mtf_wavelengths) == 0:
            raise ValueError("mtf_wavelengths is empty")

        if mtf_weights is not None and len(mtf_weights) == 0:
            raise ValueError("mtf_weights is empty")

        if mtf_wavelengths is not None and mtf_weights is not None and len(mtf_wavelengths) != len(mtf_weights):
            raise ValueError("mtf_wavelengths and mtf_weights are not the same length")

        if cn2_at_1m is not None and cn2_at_1m <= 0.0:
            raise ValueError("Turbulence effect cannot be applied at ground level")

        # Initialize base class
        super().__init__(sensor=sensor, scenario=scenario, interp=interp)

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

        self._override_cn2_at_1m: float | None = cn2_at_1m
        self._override_altitude: float | None = altitude
        self._override_slant_range: float | None = slant_range
        self._override_D: float | None = D
        self._override_ha_wind_speed: float | None = ha_wind_speed
        self._override_int_time: float | None = int_time
        self._override_n_tdi: float | None = n_tdi
        self._override_aircraft_speed: float | None = aircraft_speed

        self._simulator = self._create_simulator()

    @override
    def _create_simulator(self) -> ImageSimulator:  # noqa C901
        """Create TurbulenceApertureOTFPerturber with explicit parameters."""
        if self._use_default_psf:
            self.sensor.D = 40e-3
            self.sensor.int_time = 30e-3
            self.sensor.n_tdi = 1.0
            self.scenario.ha_wind_speed = 0.0
            self.scenario.cn2_at_1m = 1.7e-14
            self.scenario.aircraft_speed = 0.0
            override_altitude = 250.0
        else:
            override_altitude = None

        # Override values if provided
        if self._override_D is not None:
            self.sensor.D = self._override_D
        if self._override_int_time is not None:
            self.sensor.int_time = self._override_int_time
        if self._override_n_tdi is not None:
            self.sensor.n_tdi = self._override_n_tdi
        if self._override_altitude is not None:
            override_altitude = self._override_altitude
        if self._override_ha_wind_speed is not None:
            self.scenario.ha_wind_speed = self._override_ha_wind_speed
        if self._override_cn2_at_1m is not None:
            self.scenario.cn2_at_1m = self._override_cn2_at_1m
        if self._override_aircraft_speed is not None:
            self.scenario.aircraft_speed = self._override_aircraft_speed

        self.altitude: float = override_altitude if override_altitude else self.scenario.altitude
        slant_range = self._override_slant_range if self._override_slant_range else self.altitude

        pybsm_sensor = self.sensor.create_sensor()
        pybsm_scenario = self.scenario.create_scenario()
        return TurbulenceApertureSimulator(
            sensor=pybsm_sensor,
            scenario=pybsm_scenario,
            mtf_wavelengths=self._override_mtf_wavelengths,
            mtf_weights=self._override_mtf_weights,
            slant_range=slant_range,
            altitude=override_altitude,
        )

    @override
    def get_config(self) -> dict[str, Any]:
        """Get current configuration including perturber-specific parameters."""
        cfg = super().get_config()
        cfg["sensor"] = to_config_dict(self.sensor) if self.sensor else None
        cfg["scenario"] = to_config_dict(self.scenario) if self.scenario else None
        cfg["mtf_wavelengths"] = self.mtf_wavelengths
        cfg["mtf_weights"] = self.mtf_weights
        cfg["altitude"] = self.altitude
        cfg["slant_range"] = self.slant_range
        cfg["D"] = self.D
        cfg["ha_wind_speed"] = self.ha_wind_speed
        cfg["cn2_at_1m"] = self.cn2_at_1m
        cfg["int_time"] = self.int_time
        cfg["n_tdi"] = self.n_tdi
        cfg["aircraft_speed"] = self.aircraft_speed
        cfg["interp"] = self.interp

        return cfg
