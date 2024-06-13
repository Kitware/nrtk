from __future__ import annotations
import cv2
import numpy as np
import pybsm.radiance as radiance
from pybsm.otf.functional import jitterOTF, otf2psf, resample2D
from pybsm.utils import loadDatabaseAtmosphere
from typing import Any, Dict, Type, Optional

from smqtk_core.configuration import from_config_dict, to_config_dict, make_default_config

from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.interfaces.perturb_image import PerturbImage


class JitterOTFPerturber(PerturbImage):
    def __init__(self,
                 name: str,
                 sensor: Optional[PybsmSensor] = None,
                 scenario: Optional[PybsmScenario] = None,
                 sx: Optional[float] = None,
                 sy: Optional[float] = None
                 ) -> None:
        '''
        :param name: string representation of object
        :param sensor: pyBSM sensor object.
        :param scenario: pyBSM scenario object
        :param sx: root-mean-squared jitter amplitudes in the x direction (rad).
        :param sy: root-mean-squared jitter amplitudes in the y direction (rad).

        If both sensor and scenario parameters are not present, then default values
        will be used for their parameters

        If neither sx, sy, sensor or scenario parameters are provided, the values
        of sx and sy will be the default of 1.0

        If sensor and scenario parameters are provided, but not sx and sy, the
        values of sx and sy will come from the sensor and scenario objects.

        If sx and sy are ever provided by the user, those values will be used
        in the otf caluclattion
        '''

        if sensor and scenario:
            atm = loadDatabaseAtmosphere(scenario.altitude,
                                         scenario.groundRange,
                                         scenario.ihaze)
            (
                _,
                _,
                spectral_weights,
            ) = radiance.reflectance2photoelectrons(atm,
                                                    sensor,
                                                    sensor.intTime)

            wavelengths = spectral_weights[0]
            weights = spectral_weights[1]

            # cut down the wavelength range to only the regions of interests
            mtfwavelengths = wavelengths[weights > 0.0]

            D = sensor.D
            self.sx = sx if sx is not None else sensor.sx
            self.sy = sy if sy is not None else sensor.sy

            self.slant_range = np.sqrt(scenario.altitude**2 +
                                       scenario.groundRange**2)
            self.ifov = (sensor.px + sensor.py) / 2 / sensor.f
        else:
            self.sx = sx if sx is not None else 1.0
            self.sy = sy if sy is not None else 1.0
            # Assume visible spectrum of light
            self.ifov = -1
            self.slant_range = -1
            mtfwavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
            # Default value for lens diameter
            D = 0.003

        # Assume if nothing else cuts us off first, diffraction will set the
        # limit for spatial frequency that the imaging system is able
        # to resolve is (1/rad).
        cutoffFrequency = D / np.min(mtfwavelengths)
        urng = np.linspace(-1.0, 1.0, 1501) * cutoffFrequency
        vrng = np.linspace(1.0, -1.0, 1501) * cutoffFrequency

        # meshgrid of spatial frequencies out to the optics cutoff
        uu, vv = np.meshgrid(urng, vrng)
        self.sensor = sensor
        self.scenario = scenario

        self.df = (abs(urng[1] - urng[0]) + abs(vrng[0] - vrng[1]))/2
        self.jitOTF = jitterOTF(uu, vv, self.sx, self.sy)
        self.name = name

    def perturb(
            self,
            image: np.ndarray,
            additional_params: Dict[str, Any] = {}
            ) -> np.ndarray:
        """
        :raises: ValueError if 'img_gsd' not present in additional_params
        """
        if self.ifov >= 0 and self.slant_range >= 0:
            if 'img_gsd' not in additional_params:
                raise ValueError("'img_gsd' must be present in image metadata\
                                  for this perturber")
            ref_gsd = additional_params['img_gsd']
            psf = otf2psf(self.jitOTF, self.df,
                          2 * np.arctan(ref_gsd / 2 / self.slant_range))

            # filter the image
            blurimg = cv2.filter2D(image, -1, psf)

            # resample the image to the camera's ifov
            sim_img = resample2D(blurimg, ref_gsd / self.slant_range,
                                 self.ifov)

        else:
            # Default is to set dxout param to same value as dxin
            psf = otf2psf(self.jitOTF, self.df,
                          1 / (self.jitOTF.shape[0] * self.df))

            sim_img = cv2.filter2D(image, -1, psf)

        return sim_img.astype(np.uint8)

    def __call__(
                self,
                image: np.ndarray,
                additional_params: Dict[str, Any] = {}
                ) -> np.ndarray:
        """
        Alias for :meth:`.NIIRS.apply`.
        """
        return self.perturb(image, additional_params)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        return cfg

    @classmethod
    def from_config(
        cls: Type[JitterOTFPerturber],
        config_dict: Dict,
        merge_default: bool = True
    ) -> JitterOTFPerturber:
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
            'name': self.name,
            'sensor': sensor,
            'scenario': scenario,
            'sx': self.sx,
            'sy': self.sy
        }

        return config
