from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar

try:
    import cv2

    cv2_available = True
except ImportError:
    cv2_available = False
import numpy as np
import pybsm.radiance as radiance
from pybsm.otf.functional import jitter_OTF, otf_to_psf, resample_2D
from pybsm.utils import load_database_atmosphere, load_database_atmosphere_no_interp
from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage

C = TypeVar("C", bound="JitterOTFPerturber")


class JitterOTFPerturber(PerturbImage):
    def __init__(
        self,
        sensor: Optional[PybsmSensor] = None,
        scenario: Optional[PybsmScenario] = None,
        s_x: Optional[float] = None,
        s_y: Optional[float] = None,
        interp: Optional[bool] = True,
    ) -> None:
        """Initializes the JitterOTFPerturber.

        :param name: string representation of object
        :param sensor: pyBSM sensor object.
        :param scenario: pyBSM scenario object
        :param s_x: root-mean-squared jitter amplitudes in the x direction (rad).
        :param s_y: root-mean-squared jitter amplitudes in the y direction (rad).
        :param interp: a boolean determinings whether load_database_atmosphere is used with or without
                interpoloation

        If both sensor and scenario parameters are not present, then default values
        will be used for their parameters

        If neither s_x, s_y, sensor or scenario parameters are provided, the values
        of s_x and s_y will be the default of 0.0 as that results in a nadir view.

        If sensor and scenario parameters are provided, but not s_x and s_y, the
        values of s_x and s_y will come from the sensor and scenario objects.

        If s_x and s_y are ever provided by the user, those values will be used
        in the otf caluclattion
        """
        if not self.is_usable():
            raise ImportError("OpenCV not found. Please install 'nrtk[graphics]' or 'nrtk[headless]'.")

        if sensor and scenario:
            if interp:
                atm = load_database_atmosphere(scenario.altitude, scenario.ground_range, scenario.ihaze)
            else:
                atm = load_database_atmosphere_no_interp(scenario.altitude, scenario.ground_range, scenario.ihaze)
            (
                _,
                _,
                spectral_weights,
            ) = radiance.reflectance_to_photoelectrons(atm, sensor.create_sensor(), sensor.int_time)

            wavelengths = spectral_weights[0]
            weights = spectral_weights[1]

            # cut down the wavelength range to only the regions of interests
            mtf_wavelengths = wavelengths[weights > 0.0]

            D = sensor.D  # noqa: N806
            self.s_x = s_x if s_x is not None else sensor.s_x
            self.s_y = s_y if s_y is not None else sensor.s_y

            self.slant_range = np.sqrt(scenario.altitude**2 + scenario.ground_range**2)
            self.ifov = (sensor.p_x + sensor.p_y) / 2 / sensor.f
        else:
            self.s_x = s_x if s_x is not None else 0.0
            self.s_y = s_y if s_y is not None else 0.0
            # Assume visible spectrum of light
            self.ifov = -1
            self.slant_range = -1
            mtf_wavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
            # Default value for lens diameter
            D = 0.003  # noqa: N806

        # Assume if nothing else cuts us off first, diffraction will set the
        # limit for spatial frequency that the imaging system is able
        # to resolve is (1/rad).
        cutoff_frequency = D / np.min(mtf_wavelengths)
        u_rng = np.linspace(-1.0, 1.0, 1501) * cutoff_frequency
        v_rng = np.linspace(1.0, -1.0, 1501) * cutoff_frequency

        # meshgrid of spatial frequencies out to the optics cutoff
        uu, vv = np.meshgrid(u_rng, v_rng)
        self.sensor = sensor
        self.scenario = scenario
        self.interp = interp

        self.df = (abs(u_rng[1] - u_rng[0]) + abs(v_rng[0] - v_rng[1])) / 2
        self.jit_OTF = jitter_OTF(uu, vv, self.s_x, self.s_y)

    def perturb(self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """:raises: ValueError if 'img_gsd' not present in additional_params"""
        if additional_params is None:
            additional_params = dict()
        if self.ifov >= 0 and self.slant_range >= 0:
            if "img_gsd" not in additional_params:
                raise ValueError(
                    "'img_gsd' must be present in image metadata\
                                  for this perturber"
                )
            ref_gsd = additional_params["img_gsd"]
            psf = otf_to_psf(self.jit_OTF, self.df, 2 * np.arctan(ref_gsd / 2 / self.slant_range))

            # filter the image
            blur_img = cv2.filter2D(image, -1, psf)

            # resample the image to the camera's ifov
            sim_img = resample_2D(blur_img, ref_gsd / self.slant_range, self.ifov)

        else:
            # Default is to set dxout param to same value as dxin
            psf = otf_to_psf(self.jit_OTF, self.df, 1 / (self.jit_OTF.shape[0] * self.df))

            sim_img = cv2.filter2D(image, -1, psf)

        return sim_img.astype(np.uint8)

    def __call__(self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Alias for :meth:`.NIIRS.apply`."""
        if additional_params is None:
            additional_params = dict()
        return self.perturb(image, additional_params)

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

    @classmethod
    def is_usable(cls) -> bool:
        # Requires opencv to be installed
        return cv2_available

    def get_config(self) -> Dict[str, Any]:
        sensor = to_config_dict(self.sensor) if self.sensor else None
        scenario = to_config_dict(self.scenario) if self.scenario else None

        config = {"sensor": sensor, "scenario": scenario, "s_x": self.s_x, "s_y": self.s_y, "interp": self.interp}

        return config
