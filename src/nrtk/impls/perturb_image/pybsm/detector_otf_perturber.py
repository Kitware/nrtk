from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar

try:
    import cv2

    cv2_available = True
except ImportError:
    cv2_available = False
import numpy as np
import pybsm.radiance as radiance
from pybsm.otf.functional import detector_OTF, otf_to_psf, resample_2D
from pybsm.utils import load_database_atmosphere, load_database_atmosphere_no_interp
from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage

C = TypeVar("C", bound="DetectorOTFPerturber")


class DetectorOTFPerturber(PerturbImage):
    def __init__(
        self,
        sensor: Optional[PybsmSensor] = None,
        scenario: Optional[PybsmScenario] = None,
        w_x: Optional[float] = None,
        w_y: Optional[float] = None,
        f: Optional[float] = None,
        interp: Optional[bool] = True,
    ) -> None:
        """Initializes the DetectorOTFPerturber.

        :param sensor: pyBSM sensor object.
        :param scenario: pyBSM scenario object.
        :param w_x: Detector width in the x direction (m).
        :param w_y: Detector width in the y direction (m).
        :param f: Focal length (m).
        :param interp: a boolean determinings whether load_database_atmosphere is used with or without
                       interpoloation

        If a value is provided for w_x, w_y and/or f that value(s) will be used in
        the otf calculation.

        If both sensor and scenario parameters are provided, but not w_x, w_y and/or f, the
        value(s) of w_x, w_y and/or f will come from the sensor and scenario objects.

        If either sensor or scenario parameters are absent, default values
        will be used for both sensor and scenario parameters (except for w_x/w_y/f, as defined
        below).

        If any of w_x, w_y, or f are absent and sensor/scenario objects are also absent,
        the absent value(s) will default to 4um for w_x/w_y and 50mm for f
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
            mtf_wavelengths = wavelengths[weights > 0.0]

            D = sensor.D  # noqa: N806

            self.w_x = w_x if w_x is not None else sensor.w_x
            self.w_y = w_y if w_y is not None else sensor.w_y
            self.f = f if f is not None else sensor.f

            self.slant_range = np.sqrt(scenario.altitude**2 + scenario.ground_range**2)
            self.ifov = (sensor.p_x + sensor.p_y) / 2 / self.f
        else:
            self.w_x = w_x if w_x is not None else 4e-6
            self.w_y = w_y if w_y is not None else 4e-6
            self.f = f if f is not None else 50e-3

            # Assume visible spectrum of light
            self.ifov = -1
            self.slant_range = -1
            mtf_wavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
            # Default value for lens diameter
            D = 0.003  # noqa: N806

        self.sensor = sensor
        self.scenario = scenario
        self.interp = interp

        # Assume if nothing else cuts us off first, diffraction will set the
        # limit for spatial frequency that the imaging system is able
        # to resolve is (1/rad).
        cutoff_frequency = D / np.min(mtf_wavelengths)
        u_rng = np.linspace(-1.0, 1.0, 1501) * cutoff_frequency
        v_rng = np.linspace(1.0, -1.0, 1501) * cutoff_frequency

        # meshgrid of spatial frequencies out to the optics cutoff
        uu, vv = np.meshgrid(u_rng, v_rng)
        self.df = (abs(u_rng[1] - u_rng[0]) + abs(v_rng[0] - v_rng[1])) / 2

        self.det_OTF = detector_OTF(uu, vv, self.w_x, self.w_y, self.f)

    def perturb(self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """:raises: ValueError if 'img_gsd' not present in additional_params"""
        if additional_params is None:
            additional_params = dict()

        if self.ifov >= 0 and self.slant_range >= 0:
            if "img_gsd" not in additional_params:
                raise ValueError("'img_gsd' must be present in image metadata for this perturber")

            ref_gsd = additional_params["img_gsd"]
            psf = otf_to_psf(self.det_OTF, self.df, 2 * np.arctan(ref_gsd / 2 / self.slant_range))

            # filter the image
            blur_img = cv2.filter2D(image, -1, psf)

            # resample the image to the camera's ifov
            sim_img = resample_2D(blur_img, ref_gsd / self.slant_range, self.ifov)
        else:
            # Default is to set dxout param to same value as dxin
            psf = otf_to_psf(self.det_OTF, self.df, 1 / (self.det_OTF.shape[0] * self.df))

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
            "w_x": self.w_x,
            "w_y": self.w_y,
            "f": self.f,
            "interp": self.interp,
        }

        return config

    @classmethod
    def is_usable(cls) -> bool:
        # Requires opencv to be installed
        return cv2_available
