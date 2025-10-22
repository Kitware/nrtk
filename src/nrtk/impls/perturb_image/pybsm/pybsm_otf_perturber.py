"""Implements PybsmOTFPerturber which is a base class for the pybsm and otf perturber classes.

Classes:
    PybsmOTFPerturber: Applies OTF-based perturbations to images using pyBSM.

Dependencies:
    - pyBSM for OTF and radiance calculations.
    - nrtk.interfaces.perturb_image.PerturbImage for base functionality.
"""

from __future__ import annotations

__all__ = []

import copy
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import Self, override

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import PyBSMImportError
from nrtk.utils._import_guard import import_guard

# Import checks
pybsm_available: bool = import_guard("pybsm", PyBSMImportError, ["simulation"])

from pybsm.simulation import ImageSimulator  # noqa: E402
from smqtk_core.configuration import from_config_dict, make_default_config  # noqa: E402


class PybsmOTFPerturber(PerturbImage, ABC):
    """Base handles common functionality shared across all pybsm-based OTF perturbers.

    This class handles common functionality shared across all pybsm-based OTF perturbers:
    - Sensor/scenario initialization and validation
    - Default parameter handling
    - Image perturbation workflow (GSD extraction, simulation, box rescaling)
    - Configuration management base functionality
    - Dependency checking

    Attributes:
        sensor (PybsmSensor | None):
            The sensor configuration for the perturbation.
        scenario (PybsmScenario | None):
            The scenario configuration used for perturbation.
        interp (bool):
            Specifies whether to use interpolated atmospheric data.
    """

    def __init__(  # noqa: C901
        self,
        sensor: PybsmSensor | None = None,
        scenario: PybsmScenario | None = None,
        interp: bool = True,
    ) -> None:
        """Initialize the pybsm OTF perturber.

        Args:
            sensor: pyBSM sensor configuration
            scenario: pyBSM scenario configuration
            interp: Whether to use interpolated atmospheric data
        Raises:
            :raises ImportError: pyBSM is not found, install via
                `pip install nrtk[pybsm]`.
        """
        if not self.is_usable():
            raise PyBSMImportError
        super().__init__()

        # Store original configurations
        self._interp = interp
        self._simulator: ImageSimulator
        sensor = copy.deepcopy(sensor)
        scenario = copy.deepcopy(scenario)

        if scenario is not None:
            scenario.interp = self.interp

        if sensor and scenario:
            self._use_default_psf = False
        else:
            self._use_default_psf = True

        if not sensor:
            self._sensor: PybsmSensor = self._create_default_sensor()
        else:
            self._sensor: PybsmSensor = sensor

        if not scenario:
            self._scenario: PybsmScenario = self._create_default_scenario()
        else:
            self._scenario: PybsmScenario = scenario

    @abstractmethod
    def _create_simulator(self) -> ImageSimulator:
        """Create the specific ImageSimulator for this perturber.

        Returns:
            The configured ImageSimulator instance
        """
        pass

    @override
    def perturb(
        self,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        img_gsd: float | None = None,
        **additional_params: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Apply the OTF-based perturbation to the provided image.

        Args:
            image:
                The image to be perturbed.
            boxes:
                Bounding boxes for detections in input image.
            img_gsd:
                GSD is the distance between the centers of two adjacent pixels in an image, measured on the ground.
            additional_params:
                Additional perturbation keyword arguments (currently unused).

        Returns:
            :return tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
                The perturbed image and bounding boxes scaled to perturbed image shape.

        Raises:
            :raises ValueError: If 'img_gsd' is None.
        """
        if not self._use_default_psf and img_gsd is None:
            raise ValueError("'img_gsd' must be provided for this perturber")

        # When sensor/scenario are not provided, the default psf is calculated
        # which does not use the gsd
        if self._use_default_psf:
            img_gsd = None

        _, blur_img, noisy_img = self._simulator.simulate_image(image, gsd=img_gsd)

        if self._simulator.add_noise and noisy_img is not None:  # noqa: SIM108
            out_img = noisy_img
        else:
            out_img = blur_img

        # Handle formatting and box rescaling
        return self._handle_boxes_and_format(out_img, boxes, image.shape)

    @classmethod
    def from_config(cls, config_dict: dict[str, Any], merge_default: bool = True) -> Self:
        """Instantiate from configuration dictionary."""
        config_dict = dict(config_dict)

        # Handle sensor configuration
        sensor = config_dict.get("sensor", None)
        if sensor is not None:
            config_dict["sensor"] = from_config_dict(sensor, [PybsmSensor])

        # Handle scenario configuration
        scenario = config_dict.get("scenario", None)
        if scenario is not None:
            config_dict["scenario"] = from_config_dict(scenario, [PybsmScenario])

        return super().from_config(config_dict, merge_default=merge_default)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Retrieves the default configuration.

        Returns:
            :return dict[str, Any]: A dictionary with the default configuration values.
        """
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        return cfg

    @classmethod
    def is_usable(cls) -> bool:
        """Check if dependencies are available."""
        return pybsm_available

    @property
    def mtf_wavelengths(self) -> NDArray[np.float64]:
        """Getter for mtf_wavelengths."""
        return self._simulator.mtf_wavelengths

    @property
    def mtf_weights(self) -> NDArray[np.float64]:
        """Getter for mtf_weights."""
        return self._simulator.mtf_weights

    @property
    def w_x(self) -> float:
        """Getter for w_x."""
        return self._simulator.sensor.w_x

    @property
    def w_y(self) -> float:
        """Getter for w_y."""
        return self._simulator.sensor.w_y

    @property
    def s_x(self) -> float:
        """Getter for s_x."""
        return self._simulator.sensor.s_x

    @property
    def s_y(self) -> float:
        """Getter for s_y."""
        return self._simulator.sensor.s_y

    @property
    def f(self) -> float:
        """Getter for f."""
        return self._simulator.sensor.f

    @property
    def D(self) -> float:  # noqa N802
        """Getter for D."""
        return self._simulator.sensor.D

    @property
    def int_time(self) -> float:
        """Getter for int_time."""
        return self._simulator.sensor.int_time

    @property
    def n_tdi(self) -> float:
        """Getter for n_tdi."""
        return self._simulator.sensor.n_tdi

    @property
    def slant_range(self) -> float:
        """Getter for slant_range."""
        return self._simulator.slant_range

    @property
    def ha_wind_speed(self) -> float:
        """Getter for ha_wind_speed."""
        return self._simulator.scenario.ha_wind_speed

    @property
    def cn2_at_1m(self) -> float:
        """Getter for cn2_at_1m."""
        return self._simulator.scenario.cn2_at_1m

    @property
    def aircraft_speed(self) -> float:
        """Getter for aircraft_speed."""
        return self._simulator.scenario.aircraft_speed

    @property
    def interp(self) -> bool:
        """Getter for interp."""
        return self._interp

    @property
    def sensor(self) -> PybsmSensor:
        """Getter for sensor."""
        return self._sensor

    @property
    def scenario(self) -> PybsmScenario:
        """Getter for scenario."""
        return self._scenario

    def _create_default_sensor(self) -> PybsmSensor:
        """Create a default sensor when none is provided."""
        return PybsmSensor(
            name="Sensor",
            D=275e-3,
            f=4,
            p_x=0.008e-3,
            opt_trans_wavelengths=np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6,
        )

    def _create_default_scenario(self) -> PybsmScenario:
        """Create a default scenario when none is provided."""
        return PybsmScenario(
            name="Scenario",
            ihaze=1,
            altitude=9000,
            ground_range=0,
            interp=self.interp,
        )

    def _handle_boxes_and_format(
        self,
        sim_img: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
        orig_shape: tuple,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Handle box rescaling and format conversion to uint8."""
        # Convert to uint8
        sim_img_uint8 = np.clip(sim_img, 0, 255).astype(np.uint8)

        # Rescale boxes if provided
        if boxes:
            scaled_boxes = self._rescale_boxes(boxes, orig_shape, sim_img.shape)
            return sim_img_uint8, scaled_boxes

        return sim_img_uint8, boxes
