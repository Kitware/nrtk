"""Implements DefocusOTFPerturber for optical defocus simulation via OTF using pybsm and OpenCV.

Classes:
    DefocusOTFPerturber: Simulates defocus effects in images using OTF and PSF calculations.

Dependencies:
    - pybsm: Required for radiance calculations, OTF/PSF handling, and atmosphere loading.
    - numpy: For numerical computations.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox

try:
    # Multiple type ignores added for pyright's handling of guarded imports
    import pybsm.radiance as radiance
    from pybsm.otf.functional import defocus_OTF, otf_to_psf, resample_2D
    from pybsm.utils import load_database_atmosphere, load_database_atmosphere_no_interp
    from scipy.signal import fftconvolve

    pybsm_available: bool = True
except ImportError:  # pragma: no cover
    pybsm_available: bool = False

from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)
from typing_extensions import Self, override

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import PyBSMImportError


class DefocusOTFPerturber(PerturbImage):
    """Implements image perturbation using defocus and Optical Transfer Function (OTF).

    DefocusOTFPerturber applies optical defocus perturbations to input images based on
    specified sensor and scenario configurations. The perturbation uses the Optical
    Transfer Function (OTF) and Point Spread Function (PSF) for simulation.

    See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.

    Attributes:
        sensor (PybsmSensor | None):
            The sensor configuration for the simulation.
        scenario (PybsmScenario | None):
            The scenario configuration, such as altitude and ground range.
        w_x (float | None):
            Defocus parameter in the x-direction.
        w_y (float | None):
            Defocus parameter in the y-direction.
        interp (bool):
            Whether to interpolate atmosphere data.
        mtf_wavelengths (np.ndarray):
            Array of wavelengths used for Modulation Transfer Function (MTF).
        D (float):
            Lens diameter in meters.
        slant_range (float):
            Slant range in meters, calculated from altitude and ground range.
        ifov (float):
            Instantaneous Field of View (IFOV).

    Methods:
        perturb:
            Applies the defocus effect to the input image.
        __call__:
            Alias for the perturb method.
        get_default_config:
            Provides the default configuration for the perturber.
        from_config:
            Instantiates the perturber from a configuration dictionary.
        get_config:
            Retrieves the current configuration of the perturber instance.
    """

    def __init__(
        self,
        sensor: PybsmSensor | None = None,
        scenario: PybsmScenario | None = None,
        w_x: float | None = None,
        w_y: float | None = None,
        interp: bool = True,
        box_alignment_mode: str | None = None,
    ) -> None:
        """Initializes a DefocusOTFPerturber instance with the specified parameters.

        Args:
            sensor:
                Sensor configuration for the simulation.
            scenario:
                Scenario configuration (altitude, ground range, etc.).
            w_x:
                the 1/e blur spot radii in the x direction. Defaults to the sensor's value if provided.
            w_y:
                the 1/e blur spot radii in the y direction. Defaults to the sensor's value if provided.
            interp:
                Whether to interpolate atmosphere data. Defaults to True.
            box_alignment_mode:
                Deprecated. Misaligned bounding boxes will always be resolved by taking the
                smallest possible box that encases the transformed misaligned box.

                .. deprecated:: 0.24.0

            If a value is provided for w_x and/or w_y those values will be used in the otf calculation.

            If both sensor and scenario parameters are provided, but not w_x and/or w_y, the
            value(s) of w_x and/or w_y will come from the sensor and scenario objects.

            If either sensor or scenario parameters are absent, default values will be used for both
            sensor and scenario parameters (except for w_x/w_y as defined below).

            If any of w_x or w_y are absent and sensor/scenario objects are also absent,
            the absent value(s) will default to 0.0 for both.

        Raises:
            :raises ImportError: If pyBSM is not found, install via `pip install nrtk[pybsm]`.
        """
        if not self.is_usable():
            raise PyBSMImportError
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
            self.mtf_wavelengths = wavelengths[weights > 0.0]

            self.D = sensor.D
            self.w_x = w_x if w_x is not None else sensor.w_x
            self.w_y = w_y if w_y is not None else sensor.w_y

            self.slant_range = np.sqrt(scenario.altitude**2 + scenario.ground_range**2)
            self.ifov = (sensor.p_x + sensor.p_y) / 2 / sensor.f
        else:
            self.w_x: float = w_x if w_x is not None else 0.0
            self.w_y: float = w_y if w_y is not None else 0.0
            # Assume visible spectrum of light
            self.ifov: float = -1
            self.slant_range: float = -1
            self.mtf_wavelengths: np.ndarray[np.float64, Any] = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
            # Default value for lens diameter
            self.D: float = 0.003

        self.sensor = sensor
        self.scenario = scenario
        self.interp = interp

    @override
    def perturb(  # noqa:C901
        self,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Applies the defocus aperture-based perturbation to the provided image.

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
        self.defocus_otf: np.ndarray[Any, Any] = defocus_OTF(uu, vv, self.w_x, self.w_y)  # type: ignore

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
            psf = otf_to_psf(self.defocus_otf, self.df, 2 * np.arctan(ref_gsd / 2 / self.slant_range))  # type: ignore

            # filter the image
            if image.ndim == 3:
                blur_img = np.empty((*image.shape,))
                # Perform convolution using scipy.signal.fftconvolve
                # PyRight reports that fftconvolve is possibly unbound due to
                # the guarded import at the top of this file, but an object of
                # this class is only instantiable if it has been successfully
                # imported, so we can igore this
                blur_img[:, :, 0] = fftconvolve(  # pyright: ignore [reportPossiblyUnboundVariable]
                    image[:, :, 0],
                    psf,
                    mode="same",
                )
                # resample the image to the camera's ifov
                resampled_img = resample_2D(blur_img[:, :, 0], ref_gsd / self.slant_range, self.ifov)  # type: ignore
                sim_img = np.empty((*resampled_img.shape, 3))
                sim_img[:, :, 0] = resampled_img
                for channel in range(1, 3):
                    blur_img[:, :, channel] = fftconvolve(  # pyright: ignore [reportPossiblyUnboundVariable]
                        image[:, :, channel],
                        psf,
                        mode="same",
                    )
                    sim_img[:, :, channel] = resample_2D(  # type: ignore
                        blur_img[:, :, channel],
                        ref_gsd / self.slant_range,
                        self.ifov,
                    )
            else:
                # Perform convolution using scipy.signal.fftconvolve
                blur_img = fftconvolve(image, psf, mode="same")  # pyright: ignore [reportPossiblyUnboundVariable]
                # resample the image to the camera's ifov
                sim_img = resample_2D(blur_img, ref_gsd / self.slant_range, self.ifov)  # type: ignore

        else:
            # Transform an optical transfer function into a point spread function
            # Note: default is to set dxout param to same value as dxin to maintain the
            # image size ratio.
            psf = otf_to_psf(self.defocus_otf, self.df, 1 / (self.defocus_otf.shape[0] * self.df))  # type: ignore
            if image.ndim == 2:
                sim_img = fftconvolve(image, psf, mode="same")  # pyright: ignore [reportPossiblyUnboundVariable]
            else:
                # image.ndim must be 3
                sim_img = np.zeros_like(image, dtype=float)
                for c in range(image.shape[2]):
                    sim_img[..., c] = fftconvolve(  # pyright: ignore [reportPossiblyUnboundVariable]
                        image[..., c],
                        psf,
                        mode="same",
                    )
        # Rescale bounding boxes to the shape of the perturbed image
        if boxes:
            scaled_boxes = self._rescale_boxes(boxes, image.shape, sim_img.shape)
            return sim_img.astype(np.uint8), scaled_boxes

        return sim_img.astype(np.uint8), boxes

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Retrieves the default configuration for DefocusOTFPerturber instances.

        Returns:
            :return dict[str, Any]: A dictionary with the default configuration values.
        """
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        return cfg

    @classmethod
    def from_config(cls, config_dict: dict[str, Any], merge_default: bool = True) -> Self:
        """Instantiates a DefocusOTFPerturber from a configuration dictionary.

        Args:
            config_dict:
                Configuration dictionary with initialization parameters.
            merge_default:
                Whether to merge with default configuration. Defaults to True.

        Returns:
            :return DefocusOTFPerturber: An instance of DefocusOTFPerturber.
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
        """Returns the current configuration of the DefocusOTFPerturber instance.

        Returns:
            :return dict[str, Any]: Configuration dictionary with current settings.
        """
        sensor = to_config_dict(self.sensor) if self.sensor else None
        scenario = to_config_dict(self.scenario) if self.scenario else None

        return {
            "sensor": sensor,
            "scenario": scenario,
            "w_x": self.w_x,
            "w_y": self.w_y,
            "interp": self.interp,
            "box_alignment_mode": self.box_alignment_mode,
        }

    @classmethod
    def is_usable(cls) -> bool:
        """Checks if the necessary dependencies pyBSM is available.

        Returns:
            :return bool: True if pyBSM is available; False otherwise.
        """
        return pybsm_available
