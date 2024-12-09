"""
This module provides the `TurbulenceApertureOTFPerturber` class, which applies image perturbations
based on Optical Transfer Function (OTF) calculations considering turbulence and aperture effects.
The class supports configurations with specific sensor and scenario parameters, leveraging pyBSM
and OpenCV for realistic image simulations.

Classes:
    TurbulenceApertureOTFPerturber: Applies OTF-based perturbations with turbulence and aperture
    effects to images, utilizing pyBSM and OpenCV functionalities.

Dependencies:
    - OpenCV for image processing.
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

from collections.abc import Hashable, Iterable, Sequence
from typing import Any, TypeVar

from smqtk_image_io import AxisAlignedBoundingBox

try:
    import cv2

    cv2_available = True
except ImportError:
    cv2_available = False
import numpy as np

try:
    import pybsm.radiance as radiance
    from pybsm.otf.functional import otf_to_psf, polychromatic_turbulence_OTF, resample_2D
    from pybsm.utils import load_database_atmosphere, load_database_atmosphere_no_interp

    pybsm_available = True
except ImportError:
    pybsm_available = False

from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)
from typing_extensions import override

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage

C = TypeVar("C", bound="TurbulenceApertureOTFPerturber")


class TurbulenceApertureOTFPerturber(PerturbImage):
    """
    Implements OTF-based image perturbation with turbulence and aperture effects.

    The `TurbulenceApertureOTFPerturber` class simulates image degradation due to atmospheric
    turbulence and optical aperture effects, using pyBSM sensor and scenario configurations.
    It supports adjustable wavelengths, weights, and other environmental parameters for
    realistic perturbations.

    Attributes:
        sensor (PybsmSensor | None): Sensor configuration for the perturbation.
        scenario (PybsmScenario | None): Scenario settings applied during perturbation.
        mtf_wavelengths (Sequence[float]): Wavelengths used in MTF calculations.
        mtf_weights (Sequence[float]): Weights associated with each wavelength.
        altitude (float): Altitude of the imaging platform.
        slant_range (float): Line-of-sight distance between platform and target.
        D (float): Effective aperture diameter.
        ha_wind_speed (float): High-altitude wind speed affecting turbulence profile.
        cn2_at_1m (float): Refractive index structure parameter at ground level.
        int_time (float): Integration time for imaging.
        n_tdi (float): Number of time-delay integration stages.
        aircraft_speed (float): Apparent atmospheric velocity.
        interp (bool): Indicates whether to use interpolated atmospheric data.
    """

    def __init__(  # noqa: C901
        self,
        sensor: PybsmSensor = None,
        scenario: PybsmScenario = None,
        mtf_wavelengths: Sequence[float] = None,
        mtf_weights: Sequence[float] = None,
        altitude: float = None,
        slant_range: float = None,
        D: float = None,  # noqa: N803
        ha_wind_speed: float = None,
        cn2_at_1m: float = None,
        int_time: float = None,
        n_tdi: float = None,
        aircraft_speed: float = None,
        interp: bool = True,
        box_alignment_mode: str = "extent",
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
        :param interp: a boolean determining whether load_database_atmosphere is used with or without
                       interpolation
        :param box_alignment_mode: Mode for how to handle how bounding boxes change.
            Should be one of the following options:
                extent: a new axis-aligned bounding box that encases the transformed misaligned box
                extant: a new axis-aligned bounding box that is encased inside the transformed misaligned box
                median: median between extent and extant
            Default value is extent

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

        :raises: ImportError if pyBSM with OpenCV not found,
        installed via 'nrtk[pybsm-graphics]' or 'nrtk[pybsm-headless]'.
        :raises: ValueError if mtf_wavelengths and mtf_weights are not equal length
        :raises: ValueError if mtf_wavelengths is empty or mtf_weights is empty
        :raises: ValueError if cn2at1m <= 0.0
        """
        if not self.is_usable():
            raise ImportError(
                "pyBSM with OpenCV not found. Please install 'nrtk[pybsm-graphics]' or 'nrtk[pybsm-headless]'.",
            )

        super().__init__(box_alignment_mode=box_alignment_mode)

        if sensor and scenario:
            if interp:
                atm = load_database_atmosphere(scenario.altitude, scenario.ground_range, scenario.ihaze)
            else:
                atm = load_database_atmosphere_no_interp(scenario.altitude, scenario.ground_range, scenario.ihaze)
            _, _, spectral_weights = radiance.reflectance_to_photoelectrons(
                atm,
                sensor.create_sensor(),
                sensor.int_time,
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
            self.D = D if D is not None else sensor.D
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
            self.D = D if D is not None else 40e-3
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

        self.sensor = sensor
        self.scenario = scenario
        self.interp = interp

    @override
    def perturb(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        """
        Applies turbulence and aperture-based perturbation to the provided image.

        Args:
            image (np.ndarray): The image to be perturbed.
            boxes (Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]], optional): bounding boxes
                for detections in input image
            additional_params (dict[str, Any], optional): Additional parameters, including 'img_gsd'.

        Returns:
            np.ndarray: The perturbed image.
            Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]: unmodified bounding boxes
                for detections in input image

        Raises:
            ValueError: If 'img_gsd' is not provided in `additional_params`.
        """
        # Assume if nothing else cuts us off first, diffraction will set the
        # limit for spatial frequency that the imaging system is able
        # to resolve is (1/rad).
        cutoff_frequency = self.D / np.min(self.mtf_wavelengths)
        u_rng = np.linspace(-1.0, 1.0, 1501) * cutoff_frequency
        v_rng = np.linspace(1.0, -1.0, 1501) * cutoff_frequency

        # meshgrid of spatial frequencies out to the optics cutoff
        uu, vv = np.meshgrid(u_rng, v_rng)

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

        return sim_img.astype(np.uint8), boxes

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        Provides the default configuration for TurbulenceApertureOTFPerturber instances.

        Returns:
            dict[str, Any]: A dictionary with the default configuration values.
        """
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        return cfg

    @classmethod
    def from_config(cls: type[C], config_dict: dict, merge_default: bool = True) -> C:
        """
        Instantiates a TurbulenceApertureOTFPerturber from a configuration dictionary.

        Args:
            config_dict (dict): Configuration dictionary with initialization parameters.
            merge_default (bool, optional): Whether to merge with default configuration. Defaults to True.

        Returns:
            C: An instance of TurbulenceApertureOTFPerturber configured according to `config_dict`.
        """
        config_dict = dict(config_dict)
        sensor = config_dict.get("sensor", None)
        if sensor is not None:
            config_dict["sensor"] = from_config_dict(sensor, [PybsmSensor])
        scenario = config_dict.get("scenario", None)
        if scenario is not None:
            config_dict["scenario"] = from_config_dict(scenario, [PybsmScenario])

        return super().from_config(config_dict, merge_default=merge_default)

    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the TurbulenceApertureOTFPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
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

    @classmethod
    def is_usable(cls) -> bool:
        """
        Checks if the necessary dependencies (pyBSM and OpenCV) are available.

        Returns:
            bool: True if both pyBSM and OpenCV are available; False otherwise.
        """
        # Requires pybsm[graphics] or pybsm[headless]
        return cv2_available and pybsm_available
