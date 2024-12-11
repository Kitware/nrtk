"""
This module provides the `CircularApertureOTFPerturber` class, which applies Optical Transfer
Function (OTF) perturbations to images based on a circular aperture model. This class allows for
customizable sensor and scenario configurations, supporting realistic image perturbations with
wavelength and weight considerations.

Classes:
    CircularApertureOTFPerturber: Implements OTF-based perturbations using a circular aperture
    model, allowing for detailed wavelength and aperture-based image modifications.

Dependencies:
    - OpenCV for image filtering and processing.
    - pyBSM for radiance and OTF calculations.
    - nrtk.interfaces.perturb_image.PerturbImage as the base interface for image perturbation.

Example usage:
    sensor = PybsmSensor(...)
    scenario = PybsmScenario(...)
    perturber = CircularApertureOTFPerturber(sensor=sensor, scenario=scenario)
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
    from pybsm.otf.functional import (
        circular_aperture_OTF,
        otf_to_psf,
        resample_2D,
        weighted_by_wavelength,
    )
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

C = TypeVar("C", bound="CircularApertureOTFPerturber")


class CircularApertureOTFPerturber(PerturbImage):
    """
    Applies OTF-based image perturbation using a circular aperture model with sensor and
    scenario configurations.

    The `CircularApertureOTFPerturber` class uses a circular aperture model to simulate
    image perturbations, allowing for wavelength-specific and sensor-specific modifications
    based on the sensor and scenario configurations.

    Attributes:
        sensor (PybsmSensor | None): The sensor configuration for the perturbation.
        scenario (PybsmScenario | None): The scenario configuration used for perturbation.
        mtf_wavelengths (Sequence[float]): Sequence of wavelengths used in MTF calculations.
        mtf_weights (Sequence[float]): Sequence of weights associated with each wavelength.
        D (float): Effective aperture diameter.
        eta (float): Relative linear obscuration.
        slant_range (float): Line-of-sight distance between platform and target.
        ifov (float): Instantaneous field of view of the sensor.
        interp (bool): Specifies whether to use interpolated atmospheric data.
    """

    def __init__(  # noqa: C901
        self,
        sensor: PybsmSensor = None,
        scenario: PybsmScenario = None,
        mtf_wavelengths: Sequence[float] = None,
        mtf_weights: Sequence[float] = None,
        interp: bool = True,
        box_alignment_mode: str = "extent",
    ) -> None:
        """Initializes the CircularApertureOTFPerturber.

        :param sensor: pyBSM sensor object
        :param scenario: pyBSM scenario object
        :param mtf_wavelengths: a numpy array of wavelengths (m)
        :param mtf_weights: a numpy array of weights for each wavelength contribution (arb)
        :param interp: a boolean determining whether load_database_atmosphere is used with or without
                       interpolation
        :param box_alignment_mode: Mode for how to handle how bounding boxes change.
            Should be one of the following options:
                extent: a new axis-aligned bounding box that encases the transformed misaligned box
                extant: a new axis-aligned bounding box that is encased inside the transformed misaligned box
                median: median between extent and extant
            Default value is extent

        If both sensor and scenario parameters are absent, then default values
        will be used for their parameters

        If none of mtf_wavelengths, mtf_weights, sensor or scenario parameters are provided, the values
        of mtf_wavelengths and mtf_weights will default to [0.50e-6, 0.66e-6] and [1.0, 1.0] respectively

        If sensor and scenario parameters are provided, but not mtf_wavelengths and mtf_weights, the
        values of mtf_wavelengths and mtf_weights will come from the sensor and scenario objects.

        If mtf_wavelengths and mtf_weights are provided by the user, those values will be used
        in the otf caluclattion

        :raises: ImportError if pyBSM with OpenCV not found,
        installed via 'nrtk[pybsm-graphics]' or 'nrtk[pybsm-headless]'.
        :raises: ValueError if mtf_wavelengths and mtf_weights are not equal length
        :raises: ValueError if mtf_wavelengths is empty or mtf_weights is empty
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
            (
                _,
                _,
                spectral_weights,
            ) = radiance.reflectance_to_photoelectrons(atm, sensor.create_sensor(), sensor.int_time)

            wavelengths = spectral_weights[0]
            weights = spectral_weights[1]

            # cut down the wavelength range to only the regions of interests
            pos_weights = np.where(weights > 0.0)
            self.mtf_wavelengths = (
                np.asarray(mtf_wavelengths) if mtf_wavelengths is not None else wavelengths[pos_weights]
            )
            self.mtf_weights = np.asarray(mtf_weights) if mtf_weights is not None else weights[pos_weights]

            self.D = sensor.D
            self.eta = sensor.eta

            self.slant_range = np.sqrt(scenario.altitude**2 + scenario.ground_range**2)
            self.ifov = (sensor.p_x + sensor.p_y) / 2 / sensor.f
        else:
            self.mtf_wavelengths = (
                np.asarray(mtf_wavelengths)
                if mtf_wavelengths is not None
                else np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
            )
            self.mtf_weights = (
                np.asarray(mtf_weights) if mtf_weights is not None else np.ones(len(self.mtf_wavelengths))
            )
            # Assume visible spectrum of light
            self.ifov = -1
            self.slant_range = -1
            # Default value for lens diameter and relative linear obscuration
            self.D = 0.003
            self.eta = 0.0

        if self.mtf_wavelengths.size == 0:
            raise ValueError("mtf_wavelengths is empty")

        if self.mtf_weights.size == 0:
            raise ValueError("mtf_weights is empty")

        if self.mtf_wavelengths.size != self.mtf_weights.size:
            raise ValueError("mtf_wavelengths and mtf_weights are not the same length")

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
        Applies the circular aperture-based perturbation to the provided image.

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

        def ap_function(wavelengths: float) -> np.ndarray:
            return circular_aperture_OTF(uu, vv, wavelengths, self.D, self.eta)

        self.ap_OTF = weighted_by_wavelength(self.mtf_wavelengths, self.mtf_weights, ap_function)

        if additional_params is None:
            additional_params = dict()
        if self.ifov >= 0 and self.slant_range >= 0:
            if "img_gsd" not in additional_params:
                raise ValueError(
                    "'img_gsd' must be present in image metadata\
                                  for this perturber",
                )
            ref_gsd = additional_params["img_gsd"]
            psf = otf_to_psf(self.ap_OTF, self.df, 2 * np.arctan(ref_gsd / 2 / self.slant_range))

            # filter the image
            blur_img = cv2.filter2D(image, -1, psf)

            # resample the image to the camera's ifov
            sim_img = resample_2D(blur_img, ref_gsd / self.slant_range, self.ifov)

        else:
            # Default is to set dxout param to same value as dxin
            psf = otf_to_psf(self.ap_OTF, self.df, 1 / (self.ap_OTF.shape[0] * self.df))

            sim_img = cv2.filter2D(image, -1, psf)

        return sim_img.astype(np.uint8), boxes

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        Provides the default configuration for CircularApertureOTFPerturber instances.

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
        Instantiates a CircularApertureOTFPerturber from a configuration dictionary.

        Args:
            config_dict (dict): Configuration dictionary with initialization parameters.
            merge_default (bool, optional): Whether to merge with default configuration. Defaults to True.

        Returns:
            C: An instance of CircularApertureOTFPerturber.
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
        Returns the current configuration of the CircularApertureOTFPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """

        cfg = super().get_config()

        cfg["sensor"] = to_config_dict(self.sensor) if self.sensor else None
        cfg["scenario"] = to_config_dict(self.scenario) if self.scenario else None
        cfg["mtf_wavelengths"] = self.mtf_wavelengths
        cfg["mtf_weights"] = self.mtf_weights
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
