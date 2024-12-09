"""
This module implements the `DefocusOTFPerturber` class, which simulates optical defocus
using the Optical Transfer Function (OTF) in imaging systems. The class leverages the pybsm library
and OpenCV to apply perturbations to input images based on sensor and scenario configurations.

Classes:
    DefocusOTFPerturber: Simulates defocus effects in images using OTF and PSF calculations.

Dependencies:
    - pybsm: Required for radiance calculations, OTF/PSF handling, and atmosphere loading.
    - OpenCV (cv2): Used for image filtering and resampling.
    - numpy: For numerical computations.
"""

from collections.abc import Hashable, Iterable
from typing import Any, TypeVar

import numpy as np
from scipy.signal import fftconvolve
from smqtk_image_io import AxisAlignedBoundingBox

try:
    import pybsm.radiance as radiance
    from pybsm.otf.functional import defocus_OTF, otf_to_psf, resample_2D
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

C = TypeVar("C", bound="DefocusOTFPerturber")


class DefocusOTFPerturber(PerturbImage):
    """
    DefocusOTFPerturber applies optical defocus perturbations to input images based on
    specified sensor and scenario configurations. The perturbation uses the Optical
    Transfer Function (OTF) and Point Spread Function (PSF) for simulation.

    Attributes:
        sensor (PybsmSensor | None): The sensor configuration for the simulation.
        scenario (PybsmScenario | None): The scenario configuration, such as altitude and ground range.
        w_x (float | None): Defocus parameter in the x-direction.
        w_y (float | None): Defocus parameter in the y-direction.
        interp (bool): Whether to interpolate atmosphere data.
        mtf_wavelengths (np.ndarray): Array of wavelengths used for Modulation Transfer Function (MTF).
        D (float): Lens diameter in meters.
        slant_range (float): Slant range in meters, calculated from altitude and ground range.
        ifov (float): Instantaneous Field of View (IFOV).

    Methods:
        perturb: Applies the defocus effect to the input image.
        __call__: Alias for the perturb method.
        get_default_config: Provides the default configuration for the perturber.
        from_config: Instantiates the perturber from a configuration dictionary.
        is_usable: Checks if the required dependencies are available.
        get_config: Retrieves the current configuration of the perturber instance.
    """

    def __init__(
        self,
        sensor: PybsmSensor = None,
        scenario: PybsmScenario = None,
        w_x: float = None,
        w_y: float = None,
        interp: bool = True,
        box_alignment_mode: str = "extent",
    ) -> None:
        """
        Initializes a DefocusOTFPerturber instance with the specified parameters.

        Args:
            sensor (PybsmSensor | None): Sensor configuration for the simulation.
            scenario (PybsmScenario | None): Scenario configuration (altitude, ground range, etc.).
            w_x (float | None): the 1/e blur spot radii in the x direction. Defaults to the sensor's value if provided.
            w_y (float | None): the 1/e blur spot radii in the y direction. Defaults to the sensor's value if provided.
            interp (bool): Whether to interpolate atmosphere data. Defaults to True.
            box_alignment_mode (string) Mode for how to handle how bounding boxes change.
                Should be one of the following options:
                    extent: a new axis-aligned bounding box that encases the transformed misaligned box
                    extant: a new axis-aligned bounding box that is encased inside the transformed misaligned box
                    median: median between extent and extant
                Default value is extent
        Raises:
            ImportError: If pybsm or OpenCV is not available.
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
            self.mtf_wavelengths = wavelengths[weights > 0.0]

            self.D = sensor.D
            self.w_x = w_x if w_x is not None else sensor.w_x
            self.w_y = w_y if w_y is not None else sensor.w_y

            self.slant_range = np.sqrt(scenario.altitude**2 + scenario.ground_range**2)
            self.ifov = (sensor.p_x + sensor.p_y) / 2 / sensor.f
        else:
            self.w_x = w_x if w_x is not None else 0.0
            self.w_y = w_y if w_y is not None else 0.0
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
    def perturb(  # noqa:C901
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        """:raises: ValueError if 'img_gsd' not present in additional_params"""
        # Assume if nothing else cuts us off first, diffraction will set the
        # limit for spatial frequency that the imaging system is able
        # to resolve is (1/rad).
        cutoff_frequency = self.D / np.min(self.mtf_wavelengths)
        u_rng = np.linspace(-1.0, 1.0, 1501) * cutoff_frequency
        v_rng = np.linspace(1.0, -1.0, 1501) * cutoff_frequency

        # meshgrid of spatial frequencies out to the optics cutoff
        uu, vv = np.meshgrid(u_rng, v_rng)

        self.df = (abs(u_rng[1] - u_rng[0]) + abs(v_rng[0] - v_rng[1])) / 2
        self.defocus_otf = defocus_OTF(uu, vv, self.w_x, self.w_y)

        if additional_params is None:
            additional_params = dict()
        if self.ifov >= 0 and self.slant_range >= 0:
            if "img_gsd" not in additional_params:
                raise ValueError(
                    "'img_gsd' must be present in image metadata\
                                  for this perturber",
                )
            ref_gsd = additional_params["img_gsd"]
            psf = otf_to_psf(self.defocus_otf, self.df, 2 * np.arctan(ref_gsd / 2 / self.slant_range))

            # filter the image
            # Perform convolution using scipy.ndimage.convolve
            blur_img = fftconvolve(image, psf, mode="same")
            # resample the image to the camera's ifov
            sim_img = resample_2D(blur_img, ref_gsd / self.slant_range, self.ifov)

        else:
            # Default is to set dxout param to same value as dxin
            psf = otf_to_psf(self.defocus_otf, self.df, 1 / (self.defocus_otf.shape[0] * self.df))
            if image.ndim == 2:
                sim_img = fftconvolve(image, psf, mode="same")
            elif image.ndim == 3:
                sim_img = np.zeros_like(image, dtype=float)
                for c in range(image.shape[2]):
                    sim_img[..., c] = fftconvolve(image[..., c], psf, mode="same")
        return sim_img.astype(np.uint8), boxes

    @override
    def __call__(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        """Alias for :meth:`.NIIRS.apply`."""
        if additional_params is None:
            additional_params = dict()
        return self.perturb(image=image, boxes=boxes, additional_params=additional_params)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        Provides the default configuration for DefocusOTFPerturber instances.

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
        Instantiates a DefocusOTFPerturber from a configuration dictionary.

        Args:
            config_dict (dict): Configuration dictionary with initialization parameters.
            merge_default (bool, optional): Whether to merge with default configuration. Defaults to True.

        Returns:
            C: An instance of DefocusOTFPerturber configured according to `config_dict`.
        """
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
        """
        Checks if the necessary dependencies (pybsm and OpenCV) are available.

        Returns:
            bool: True if both pybsm and OpenCV are available; False otherwise.
        """
        # Requires pybsm[graphics] or pybsm[headless]
        return pybsm_available

    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the DefocusOTFPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
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
