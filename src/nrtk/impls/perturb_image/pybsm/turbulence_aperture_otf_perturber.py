from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Type, TypeVar

try:
    import cv2

    cv2_available = True
except ImportError:
    cv2_available = False
import numpy as np
import pybsm.radiance as radiance
from pybsm.otf.functional import otf_to_psf, polychromatic_turbulence_OTF, resample_2D
from pybsm.utils import load_database_atmosphere, load_database_atmosphere_no_interp
from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage

C = TypeVar("C", bound="TurbulenceApertureOTFPerturber")


class TurbulenceApertureOTFPerturber(PerturbImage):
    def __init__(
        self,
        sensor: Optional[PybsmSensor] = None,
        scenario: Optional[PybsmScenario] = None,
        mtf_wavelengths: Optional[Sequence[float]] = None,
        mtf_weights: Optional[Sequence[float]] = None,
        altitude: Optional[float] = None,
        slant_range: Optional[float] = None,
        D: Optional[float] = None,  # noqa: N803
        ha_wind_speed: Optional[float] = None,
        cn2_at_1m: Optional[float] = None,
        int_time: Optional[float] = None,
        n_tdi: Optional[float] = None,
        aircraft_speed: Optional[float] = None,
        interp: Optional[float] = True,
    ) -> None:
        """Initializes the TurbulenceApertureOTFPerturber.

        :param sensor: pyBSM sensor object
        :param scenario: pyBSM scenario object
        :param mtf_wavelengths: a sequence of wavelengths (m)
        :param mtf_weights: a sequence of weights for each wavelength contribution (arb)
        :param altitude: height of the aircraft above the ground (m)
        :param slant_range: line-of-sight range between the aircraft and target (target is assumed
            to be on the ground) (m)
        :param D: effective aperture diameter (m)
        :param ha_wind_speed: the high altitude windspeed (m/s); used to calculate the turbulence
            profile
        :param cn2_at_1m: the refractive index structure parameter "near the ground" (e.g. at
            h = 1 m); used to calculate the turbulence profile
        :param int_time: dwell (i.e. integration) time (seconds)
        :param n_tdi: the number of time-delay integration stages (relevant only when TDI cameras
            are used. For CMOS cameras, the value can be assumed to be 1.0)
        :param aircraft_speed: apparent atmospheric velocity (m/s); this can just be the windspeed
            at the sensor position if the sensor is stationary

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

        :raises: ValueError if mtf_wavelengths and mtf_weights are not equal length
        :raises: ValueError if mtf_wavelengths is empty or mtf_weights is empty
        :raises: ValueError if cn2at1m <= 0.0
        """
        if not self.is_usable():
            raise ImportError("OpenCV not found. Please install 'nrtk[graphics]' or 'nrtk[headless]'.")

        if sensor and scenario:
            if interp:
                atm = load_database_atmosphere(scenario.altitude, scenario.ground_range, scenario.ihaze)
            else:
                atm = load_database_atmosphere_no_interp(scenario.altitude, scenario.ground_range, scenario.ihaze)
            _, _, spectral_weights = radiance.reflectance_to_photoelectrons(
                atm, sensor.create_sensor(), sensor.int_time
            )

            wavelengths = spectral_weights[0]
            weights = spectral_weights[1]

            # cut down the wavelength range to only the regions of interests
            pos_weights = np.where(weights > 0.0)
            self.mtf_wavelengths = (
                np.asarray(mtf_wavelengths) if mtf_wavelengths is not None else wavelengths[pos_weights]
            )
            self.mtf_weights = np.asarray(mtf_weights) if mtf_weights is not None else weights[pos_weights]

            # Sensor paramaters
            self.D = D if D is not None else sensor.D  # noqa: N806
            self.int_time = int_time if int_time is not None else sensor.int_time
            self.n_tdi = n_tdi if n_tdi is not None else sensor.n_tdi

            # Scenario Parameters
            self.altitude = altitude if altitude is not None else scenario.altitude
            self.ha_wind_speed = ha_wind_speed if ha_wind_speed is not None else scenario.ha_wind_speed
            self.cn2_at_1m = cn2_at_1m if cn2_at_1m is not None else scenario.cn2_at_1m
            self.aircraft_speed = aircraft_speed if aircraft_speed is not None else scenario.aircraft_speed
        else:
            self.mtf_wavelengths = (
                np.asarray(mtf_wavelengths)
                if mtf_wavelengths is not None
                else np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
            )
            self.mtf_weights = (
                np.asarray(mtf_weights) if mtf_weights is not None else np.ones(len(self.mtf_wavelengths))
            )

            # Sensor paramaters
            self.D = D if D is not None else 40e-3  # noqa: N806
            self.int_time = int_time if int_time is not None else 30e-3
            self.n_tdi = n_tdi if n_tdi is not None else 1.0

            # Scenario Parameters
            self.altitude = altitude if altitude is not None else 250
            self.ha_wind_speed = ha_wind_speed if ha_wind_speed is not None else 0
            self.cn2_at_1m = cn2_at_1m if cn2_at_1m is not None else 1.7e-14
            self.aircraft_speed = aircraft_speed if aircraft_speed is not None else 0

        # Assume visible spectrum of light
        self.slant_range = slant_range if slant_range is not None else self.altitude

        if self.mtf_wavelengths.size == 0:
            raise ValueError("mtf_wavelengths is empty")

        if self.mtf_weights.size == 0:
            raise ValueError("mtf_weights is empty")

        if self.mtf_wavelengths.size != self.mtf_weights.size:
            raise ValueError("mtf_wavelengths and mtf_weights are not the same length")

        if self.cn2_at_1m is not None and self.cn2_at_1m <= 0.0:
            raise ValueError("Turbulence effect cannot be applied at ground level")

        # Assume if nothing else cuts us off first, diffraction will set the
        # limit for spatial frequency that the imaging system is able
        # to resolve is (1/rad).
        cutoff_frequency = self.D / np.min(self.mtf_wavelengths)  # noqa: N806
        u_rng = np.linspace(-1.0, 1.0, 1501) * cutoff_frequency
        v_rng = np.linspace(1.0, -1.0, 1501) * cutoff_frequency

        # meshgrid of spatial frequencies out to the optics cutoff
        uu, vv = np.meshgrid(u_rng, v_rng)
        self.sensor = sensor
        self.scenario = scenario
        self.interp = interp

        self.df = (abs(u_rng[1] - u_rng[0]) + abs(v_rng[0] - v_rng[1])) / 2
        self.turbulence_otf, _ = polychromatic_turbulence_OTF(
            uu,
            vv,
            self.mtf_wavelengths,
            self.mtf_weights,
            self.altitude,
            self.slant_range,
            self.D,
            self.ha_wind_speed,
            self.cn2_at_1m,
            (self.int_time * self.n_tdi if self.int_time is not None and self.n_tdi is not None else 1.0),
            self.aircraft_speed,
        )

    def perturb(self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """:raises: ValueError if 'img_gsd' not present in additional_params"""
        if additional_params is None:
            additional_params = dict()
        if self.sensor and self.scenario:
            if "img_gsd" not in additional_params:
                raise ValueError("'img_gsd' must be present in image metadata for this perturber")
            ref_gsd = additional_params["img_gsd"]
            psf = otf_to_psf(
                self.turbulence_otf,
                self.df,
                2 * np.arctan(ref_gsd / 2 / self.slant_range),
            )

            # filter the image
            blur_img = cv2.filter2D(image, -1, psf)

            # resample the image to the camera's ifov
            sim_img = resample_2D(blur_img, ref_gsd / self.slant_range, ref_gsd / self.altitude)

        else:
            # Default is to set dxout param to same value as dxin
            psf = otf_to_psf(
                self.turbulence_otf,
                self.df,
                1 / (self.turbulence_otf.shape[0] * self.df),
            )

            sim_img = cv2.filter2D(image, -1, psf)

        return sim_img.astype(np.uint8)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        return cfg

    @classmethod
    def from_config(cls: Type[C], config_dict: Dict, merge_default: bool = True) -> C:
        config_dict = dict(config_dict)
        sensor = config_dict.get("sensor", None)
        if sensor is not None:
            config_dict["sensor"] = from_config_dict(sensor, [PybsmSensor])
        scenario = config_dict.get("scenario", None)
        if scenario is not None:
            config_dict["scenario"] = from_config_dict(scenario, [PybsmScenario])

        return super().from_config(config_dict, merge_default=merge_default)

    def get_config(self) -> Dict[str, Any]:
        sensor = to_config_dict(self.sensor) if self.sensor else None
        scenario = to_config_dict(self.scenario) if self.scenario else None

        config = {
            "sensor": sensor,
            "scenario": scenario,
            "mtf_wavelengths": self.mtf_wavelengths,
            "mtf_weights": self.mtf_weights,
            "altitude": self.altitude,
            "slant_range": self.slant_range,
            "D": self.D,  # noqa: N803
            "ha_wind_speed": self.ha_wind_speed,
            "cn2_at_1m": self.cn2_at_1m,
            "int_time": self.int_time,
            "n_tdi": self.n_tdi,
            "aircraft_speed": self.aircraft_speed,
            "interp": self.interp,
        }

        return config

    @classmethod
    def is_usable(cls) -> bool:
        # Requires opencv to be installed
        return cv2_available
