"""
This module defines the `DetectorOTFPerturber` class, which applies image perturbations
based on the Optical Transfer Function (OTF) of a detector, using configurations for
sensor and scenario. This class can simulate the effects of detector and environmental
parameters on images.

Classes:
    DetectorOTFPerturber: Applies OTF-based perturbations to images using specified
    sensor and scenario configurations.

Dependencies:
    - OpenCV for image processing.
    - pyBSM for radiance and OTF-related functionalities.
    - nrtk.interfaces.perturb_image.PerturbImage as the base interface for image perturbation.

Example usage:
    sensor = PybsmSensor(...)
    scenario = PybsmScenario(...)
    perturber = DetectorOTFPerturber(sensor=sensor, scenario=scenario)
    perturbed_image, boxes = perturber.perturb(image, boxes)

Notes:
    - The boxes returned from `perturb` are identical to the boxes passed in.
"""

from collections.abc import Hashable, Iterable
from typing import Any, Optional

from smqtk_image_io.bbox import AxisAlignedBoundingBox

try:
    # Multiple type ignores added for pyright's handling of guarded imports
    import cv2

    cv2_available = True
except ImportError:  # pragma: no cover
    cv2_available = False
import numpy as np

try:
    import pybsm.radiance as radiance
    from pybsm.otf.functional import detector_OTF, otf_to_psf, resample_2D
    from pybsm.utils import load_database_atmosphere, load_database_atmosphere_no_interp

    pybsm_available = True
except ImportError:  # pragma: no cover
    pybsm_available = False

from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)
from typing_extensions import Self, override

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import PyBSMAndOpenCVImportError


class DetectorOTFPerturber(PerturbImage):
    """
    Implements OTF-based image perturbation using detector specifications and atmospheric conditions.

    The `DetectorOTFPerturber` class uses sensor and scenario configurations to apply realistic
    perturbations to images. This includes adjusting for detector width, focal length, and atmospheric
    conditions using OpenCV and pyBSM functionalities.

    Attributes:
        sensor (PybsmSensor | None): The sensor configuration used to define perturbation parameters.
        scenario (PybsmScenario | None): Scenario configuration providing environmental context.
        w_x (float | None): Detector width in the x direction (meters).
        w_y (float | None): Detector width in the y direction (meters).
        f (float | None): Focal length of the detector (meters).
        interp (bool): Indicates whether atmospheric database should use interpolation.
    """

    def __init__(
        self,
        sensor: Optional[PybsmSensor] = None,
        scenario: Optional[PybsmScenario] = None,
        w_x: Optional[float] = None,
        w_y: Optional[float] = None,
        f: Optional[float] = None,
        interp: bool = True,
        box_alignment_mode: str = "extent",
    ) -> None:
        """Initializes the DetectorOTFPerturber.

        :param sensor: pyBSM sensor object.
        :param scenario: pyBSM scenario object.
        :param w_x: Detector width in the x direction (m).
        :param w_y: Detector width in the y direction (m).
        :param f: Focal length (m).
        :param interp: a boolean determining whether load_database_atmosphere is used with or without
                       interpolation
        :param box_alignment_mode: Mode for how to handle how bounding boxes change.
            Should be one of the following options:
                extent: a new axis-aligned bounding box that encases the transformed misaligned box
                extant: a new axis-aligned bounding box that is encased inside the transformed misaligned box
                median: median between extent and extant
            Default value is extent

        If a value is provided for w_x, w_y and/or f that value(s) will be used in
        the otf calculation.

        If both sensor and scenario parameters are provided, but not w_x, w_y and/or f, the
        value(s) of w_x, w_y and/or f will come from the sensor and scenario objects.

        If either sensor or scenario parameters are absent, default values
        will be used for both sensor and scenario parameters (except for w_x/w_y/f, as defined
        below).

        If any of w_x, w_y, or f are absent and sensor/scenario objects are also absent,
        the absent value(s) will default to 4um for w_x/w_y and 50mm for f

        :raises: ImportError if OpenCV or pyBSM is not found,
        install via `pip install nrtk[pybsm,graphics]` or `pip install nrtk[pybsm,headless]`.
        """
        if not self.is_usable():
            raise PyBSMAndOpenCVImportError
        super().__init__(box_alignment_mode=box_alignment_mode)
        if sensor and scenario:
            if interp:
                atm = load_database_atmosphere(scenario.altitude, scenario.ground_range, scenario.ihaze)  # type: ignore
            else:
                atm = load_database_atmosphere_no_interp(scenario.altitude, scenario.ground_range, scenario.ihaze)  # type: ignore

            _, _, spectral_weights = radiance.reflectance_to_photoelectrons(  # type: ignore
                atm,
                sensor.create_sensor(),
                sensor.int_time,
            )

            wavelengths = spectral_weights[0]
            weights = spectral_weights[1]

            # cut down the wavelength range to only the regions of interests
            self.mtf_wavelengths = wavelengths[weights > 0.0]

            self.D = sensor.D

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
            self.mtf_wavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
            # Default value for lens diameter
            self.D = 0.003

        self.sensor = sensor
        self.scenario = scenario
        self.interp = interp

    @override
    def perturb(  # noqa: C901
        self,
        image: np.ndarray,
        boxes: Optional[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]]:
        """
        Applies the OTF-based perturbation to the provided image.

        Args:
            image (np.ndarray): The image to be perturbed.
            boxes (Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]], optional): bounding boxes
                for detections in input image
            additional_params (dict[str, Any], optional): Additional parameters, including 'img_gsd'.

        Returns:
            np.ndarray: The perturbed image.
            Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]: Bounding boxes scaled to perturbed image
                shape.

        :raises:
            ValueError: If 'img_gsd' is not present in `additional_params`.
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
        self.det_OTF = detector_OTF(uu, vv, self.w_x, self.w_y, self.f)  # type: ignore

        if additional_params is None:
            additional_params = dict()

        if self.ifov >= 0 and self.slant_range >= 0:
            if "img_gsd" not in additional_params:
                raise ValueError("'img_gsd' must be present in image metadata for this perturber")

            ref_gsd = additional_params["img_gsd"]
            psf = otf_to_psf(self.det_OTF, self.df, 2 * np.arctan(ref_gsd / 2 / self.slant_range))  # type: ignore

            # filter the image
            blur_img = cv2.filter2D(image, -1, psf)  # type: ignore

            # resample the image to the camera's ifov
            if image.ndim == 3:
                resampled_img = resample_2D(blur_img[:, :, 0], ref_gsd / self.slant_range, self.ifov)  # type: ignore
                sim_img = np.empty((*resampled_img.shape, 3))
                sim_img[:, :, 0] = resampled_img
                for channel in range(1, 3):
                    sim_img[:, :, channel] = resample_2D(  # type: ignore
                        blur_img[:, :, channel],
                        ref_gsd / self.slant_range,
                        self.ifov,
                    )
            else:
                sim_img = resample_2D(blur_img, ref_gsd / self.slant_range, self.ifov)  # type: ignore
        else:
            # Default is to set dxout param to same value as dxin
            psf = otf_to_psf(self.det_OTF, self.df, 1 / (self.det_OTF.shape[0] * self.df))  # type: ignore

            sim_img = cv2.filter2D(image, -1, psf)  # type: ignore

        if boxes:
            scaled_boxes = self._rescale_boxes(boxes, image.shape, sim_img.shape)
            return sim_img.astype(np.uint8), scaled_boxes

        return sim_img.astype(np.uint8), boxes

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        Provides the default configuration for DetectorOTFPerturber instances.

        Returns:
            dict[str, Any]: A dictionary with the default configuration values.
        """
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        return cfg

    @classmethod
    def from_config(cls, config_dict: dict, merge_default: bool = True) -> Self:
        """
        Instantiates a DetectorOTFPerturber from a configuration dictionary.

        Args:
            config_dict (dict): Configuration dictionary with initialization parameters.
            merge_default (bool, optional): Whether to merge with default configuration. Defaults to True.

        Returns:
            C: An instance of DetectorOTFPerturber configured according to `config_dict`.
        """
        config_dict = dict(config_dict)
        sensor = config_dict.get("sensor", None)
        if sensor is not None:
            config_dict["sensor"] = from_config_dict(sensor, [PybsmSensor])
        scenario = config_dict.get("scenario", None)
        if scenario is not None:
            config_dict["scenario"] = from_config_dict(scenario, [PybsmScenario])

        return super().from_config(config_dict, merge_default=merge_default)

    @override
    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the DetectorOTFPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """

        cfg = super().get_config()

        cfg["sensor"] = to_config_dict(self.sensor) if self.sensor else None
        cfg["scenario"] = to_config_dict(self.scenario) if self.scenario else None
        cfg["w_x"] = self.w_x
        cfg["w_y"] = self.w_y
        cfg["f"] = self.f
        cfg["interp"] = self.interp

        return cfg

    @classmethod
    def is_usable(cls) -> bool:
        """
        Checks if the necessary dependencies (pyBSM and OpenCV) are available.

        Returns:
            bool: True if both pyBSM and OpenCV are available; False otherwise.
        """
        return cv2_available and pybsm_available
