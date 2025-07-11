"""Implements CircularApertureOTFPerturber for circular aperture OTF perturbations with sensor and scenario configs.

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

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import Any

from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import Self

try:
    # Multiple type ignores added for pyright's handling of guarded imports
    import cv2

    cv2_available: bool = True
except ImportError:  # pragma: no cover
    cv2_available: bool = False

try:
    import pybsm.radiance as radiance
    from pybsm.otf.functional import (
        circular_aperture_OTF,
        otf_to_psf,
        resample_2D,
        weighted_by_wavelength,
    )
    from pybsm.utils import load_database_atmosphere, load_database_atmosphere_no_interp

    pybsm_available: bool = True
except ImportError:  # pragma: no cover
    pybsm_available: bool = False

import numpy as np
from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)
from typing_extensions import override

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import PyBSMAndOpenCVImportError


class CircularApertureOTFPerturber(PerturbImage):
    """Applies OTF-based image perturbation using a circular aperture model with sensor and scenario configurations.

    The `CircularApertureOTFPerturber` class uses a circular aperture model to simulate
    image perturbations, allowing for wavelength-specific and sensor-specific modifications
    based on the sensor and scenario configurations.

    See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.

    Attributes:
        sensor (PybsmSensor | None):
            The sensor configuration for the perturbation.
        scenario (PybsmScenario | None):
            The scenario configuration used for perturbation.
        mtf_wavelengths (Sequence[float]):
            Sequence of wavelengths used in MTF calculations.
        mtf_weights (Sequence[float]):
            Sequence of weights associated with each wavelength.
        D (float):
            Effective aperture diameter.
        eta (float):
            Relative linear obscuration.
        slant_range (float):
            Line-of-sight distance between platform and target.
        ifov (float):
            Instantaneous field of view of the sensor.
        interp (bool):
            Specifies whether to use interpolated atmospheric data.
    """

    def __init__(  # noqa: C901
        self,
        sensor: PybsmSensor | None = None,
        scenario: PybsmScenario | None = None,
        mtf_wavelengths: Sequence[float] | None = None,
        mtf_weights: Sequence[float] | None = None,
        interp: bool = True,
        box_alignment_mode: str | None = None,
    ) -> None:
        """Initializes the CircularApertureOTFPerturber.

        Args:
            sensor:
                pyBSM sensor object
            scenario:
                pyBSM scenario object
            mtf_wavelengths:
                a numpy array of wavelengths (m)
            mtf_weights:
                a numpy array of weights for each wavelength contribution (arb)
            interp:
                a boolean determining whether load_database_atmosphere is used with or without
                interpolation.
            box_alignment_mode:
                Deprecated. Misaligned bounding boxes will always be resolved by taking the
                smallest possible box that encases the transformed misaligned box.

                .. deprecated:: 0.24.0

            If both sensor and scenario parameters are absent, then default values
            will be used for their parameters

            If none of mtf_wavelengths, mtf_weights, sensor or scenario parameters are provided, the values
            of mtf_wavelengths and mtf_weights will default to [0.50e-6, 0.66e-6] and [1.0, 1.0] respectively

            If sensor and scenario parameters are provided, but not mtf_wavelengths and mtf_weights, the
            values of mtf_wavelengths and mtf_weights will come from the sensor and scenario objects.

            If mtf_wavelengths and mtf_weights are provided by the user, those values will be used
            in the otf calculation.

        Raises:
            :raises ImportError: If OpenCV or pyBSM is not found, install via
                `pip install nrtk[pybsm,graphics]` or `pip install nrtk[pybsm,headless]`.
            :raises ValueError: If mtf_wavelengths and mtf_weights are not equal length
            :raises ValueError: If mtf_wavelengths is empty or mtf_weights is empty
        """
        if not self.is_usable():
            raise PyBSMAndOpenCVImportError
        super().__init__(box_alignment_mode=box_alignment_mode)

        # Load the pre-calculated MODTRAN atmospheric data.
        if sensor and scenario:
            if interp:
                atm = load_database_atmosphere(scenario.altitude, scenario.ground_range, scenario.ihaze)  # type: ignore
            else:
                atm = load_database_atmosphere_no_interp(  # type: ignore
                    scenario.altitude,
                    scenario.ground_range,
                    scenario.ihaze,
                )
            _, _, spectral_weights = radiance.reflectance_to_photoelectrons(  # type: ignore
                atm,
                sensor.create_sensor(),
                sensor.int_time,
            )

            # Use the spectral_weights variable for MTF wavelengths and weights
            # Note: These values are used only if mtf_wavelengths and mtf_weights
            # are missing in the input
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
            self.ifov: float = -1
            self.slant_range: float = -1
            # Default value for lens diameter and relative linear obscuration
            self.D: float = 0.003
            self.eta: float = 0.0

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
    def perturb(  # noqa: C901
        self,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Applies the circular aperture-based perturbation to the provided image.

        Args:
            image:
                The image to be perturbed.
            boxes:
                Bounding boxes for detections in input image.
            additional_params:
                Dictionary containing:
                    - "img_gsd" (float): GSD is the distance between the centers of two adjacent
                    pixels in an image, measured on the ground.

        Returns:
            :return tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
                The perturbed image and bounding boxes scaled to perturbed image shape.

        Raises:
            :raises ValueError: If 'img_gsd' is not provided in `additional_params`.
        """
        # Assume if nothing else cuts us off first, diffraction will set the
        # limit for spatial frequency that the imaging system is able
        # to resolve is (1/rad).
        cutoff_frequency = self.D / np.min(self.mtf_wavelengths)
        u_rng = np.linspace(-1.0, 1.0, 1501) * cutoff_frequency
        v_rng = np.linspace(1.0, -1.0, 1501) * cutoff_frequency

        # meshgrid of spatial frequencies out to the optics cutoff
        uu, vv = np.meshgrid(u_rng, v_rng)

        # Sample spacing for the optical transfer function
        self.df: float = (abs(u_rng[1] - u_rng[0]) + abs(v_rng[0] - v_rng[1])) / 2

        # Compute a wavelength weighted composite array based on the circular aperture OTF function.
        def ap_function(wavelengths: float) -> np.ndarray:
            return circular_aperture_OTF(uu, vv, wavelengths, self.D, self.eta)  # type: ignore

        self.ap_OTF: np.ndarray[Any, Any] = weighted_by_wavelength(self.mtf_wavelengths, self.mtf_weights, ap_function)  # type: ignore

        if additional_params is None:
            additional_params = dict()

        if self.ifov >= 0 and self.slant_range >= 0:
            if "img_gsd" not in additional_params:
                raise ValueError(
                    "'img_gsd' must be present in image metadata\
                                  for this perturber",
                )
            ref_gsd = additional_params["img_gsd"]

            # Transform an optical transfer function into a point spread function
            psf = otf_to_psf(self.ap_OTF, self.df, 2 * np.arctan(ref_gsd / 2 / self.slant_range))  # type: ignore

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
            # Transform an optical transfer function into a point spread function
            # Note: default is to set dxout param to same value as dxin to maintain the
            # image size ratio.
            psf = otf_to_psf(self.ap_OTF, self.df, 1 / (self.ap_OTF.shape[0] * self.df))  # type: ignore

            sim_img = cv2.filter2D(image, -1, psf)  # type: ignore

        # Rescale bounding boxes to the shape of the perturbed image
        if boxes:
            scaled_boxes = self._rescale_boxes(boxes, image.shape, sim_img.shape)
            return sim_img.astype(np.uint8), scaled_boxes

        return sim_img.astype(np.uint8), boxes

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Retrieves the default configuration for CircularApertureOTFPerturber instances.

        Returns:
            :return dict[str, Any]: A dictionary with the default configuration values.
        """
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        return cfg

    @classmethod
    def from_config(cls, config_dict: dict[str, Any], merge_default: bool = True) -> Self:
        """Instantiates a CircularApertureOTFPerturber from a configuration dictionary.

        Args:
            config_dict:
                Configuration dictionary with initialization parameters.
            merge_default:
                Whether to merge with default configuration. Defaults to True.

        Returns:
            :return CircularApertureOTFPerturber: An instance of CircularApertureOTFPerturber.
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
        """Returns the current configuration of the CircularApertureOTFPerturber instance.

        Returns:
            :return dict[str, Any]: Configuration dictionary with current settings.
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
        """Checks if the necessary dependencies (pyBSM and OpenCV) are available.

        Returns:
            :return bool: True if both pyBSM and OpenCV are available; False otherwise.
        """
        return cv2_available and pybsm_available
