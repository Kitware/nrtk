"""
This module provides the `PybsmPerturber` class, which applies image perturbations using
the pyBSM library. The perturbations are based on a sensor configuration and a scenario,
allowing for realistic image simulations in remote sensing or other image-processing
applications.

Classes:
    PybsmPerturber: Applies image perturbations using pyBSM based on specified sensor and
    scenario configurations.

Dependencies:
    - pybsm for simulation and reference image functionality.
    - nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario for scenario configuration.
    - nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor for sensor configuration.
    - nrtk.interfaces.perturb_image.PerturbImage as the base interface for image perturbation.

Example usage:
    sensor = PybsmSensor(...)
    scenario = PybsmScenario(...)
    perturber = PybsmPerturber(sensor=sensor, scenario=scenario)
    perturbed_image = perturber.perturb(image)
"""

import copy
from collections.abc import Hashable, Iterable
from importlib.util import find_spec
from typing import Any, TypeVar

import numpy as np
from smqtk_image_io import AxisAlignedBoundingBox

try:
    from pybsm.simulation import RefImage, simulate_image

    pybsm_available = True
except ImportError:
    pybsm_available = False

from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage

C = TypeVar("C", bound="PybsmPerturber")

DEFAULT_REFLECTANCE_RANGE = np.array([0.05, 0.5])  # It is bad standards to call np.array within argument defaults


class PybsmPerturber(PerturbImage):
    """
    Implements image perturbation using pyBSM sensor and scenario configurations.

    The `PybsmPerturber` class applies realistic perturbations to images by leveraging
    pyBSM's simulation functionalities. It takes in a sensor and scenario, along with
    other optional parameters, to simulate environmental effects on the image.

    Attributes:
        sensor (PybsmSensor): The sensor configuration for the perturbation.
        scenario (PybsmScenario): Scenario settings to apply during the perturbation.
        reflectance_range (np.ndarray): Default reflectance range for image simulation.
    """

    def __init__(  # noqa: C901
        self,
        sensor: PybsmSensor,
        scenario: PybsmScenario,
        reflectance_range: np.ndarray = DEFAULT_REFLECTANCE_RANGE,
        rng_seed: int = 1,
        box_alignment_mode: str = "extent",
        **kwargs: Any,
    ) -> None:
        """Initializes the PybsmPerturber.

        :param sensor: pyBSM sensor object.
        :param scenario: pyBSM scenario object.
        :param reflectance_range: Array of reflectances that correspond to pixel values.
        :param rng_seed: integer seed value that will be used for the random number generator
        :param box_alignment_mode: Mode for how to handle how bounding boxes change.
            Should be one of the following options:
                extent: a new axis-aligned bounding box that encases the transformed misaligned box
                extant: a new axis-aligned bounding box that is encased inside the transformed misaligned box
                median: median between extent and extant
            Default value is extent

        :raises: ImportError if pyBSM with OpenCV not found,
        installed via 'nrtk[pybsm-graphics]' or 'nrtk[pybsm-headless]'.
        :raises: ValueError if reflectance_range length != 2
        :raises: ValueError if reflectance_range not strictly ascending
        """
        if not self.is_usable():
            raise ImportError(
                "pyBSM with OpenCV not found. Please install 'nrtk[pybsm-graphics]' or 'nrtk[pybsm-headless]'.",
            )
        super().__init__(box_alignment_mode=box_alignment_mode)
        self._rng_seed = rng_seed
        self.sensor = copy.deepcopy(sensor)
        self.scenario = copy.deepcopy(scenario)

        for k in kwargs:
            if hasattr(self.sensor, k):
                setattr(self.sensor, k, kwargs[k])
            elif hasattr(self.scenario, k):
                setattr(self.scenario, k, kwargs[k])

        if reflectance_range.shape[0] != 2:
            raise ValueError(f"Reflectance range array must have length of 2, got {reflectance_range.shape[0]}")
        if reflectance_range[0] >= reflectance_range[1]:
            raise ValueError(f"Reflectance range array values must be strictly ascending, got {reflectance_range}")
        self.reflectance_range = reflectance_range

        # this is key:value record of the thetas use for perturbing
        self.thetas = copy.deepcopy(kwargs)

    @property
    def params(self) -> dict:
        """
        Retrieves the theta parameters related to the perturbation configuration.

        This method provides extra configuration details for the `PybsmPerturber` instance,
        which may include specific parameters related to the sensor, scenario, or any
        additional customizations applied during initialization.

        Returns:
            dict[str, Any]: A dictionary containing additional perturbation parameters.
        """

        return self.thetas

    def perturb(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        """:raises: ValueError if 'img_gsd' not present in additional_params"""
        if additional_params is None:  # Cannot have mutable data structure in argument default
            additional_params = dict()
        if "img_gsd" not in additional_params:
            raise ValueError("'img_gsd' must be present in image metadata for this perturber")

        ref_img = RefImage(
            image,
            additional_params["img_gsd"],
            np.array([image.min(), image.max()]),
            self.reflectance_range,
        )

        perturbed = simulate_image(ref_img, self.sensor(), self.scenario(), self._rng_seed)[-1]

        min_perturbed_val = perturbed.min()
        den = perturbed.max() - min_perturbed_val
        perturbed -= min_perturbed_val
        perturbed /= den
        perturbed *= 255

        return perturbed.astype(np.uint8), boxes

    def __str__(self) -> str:
        """
        Returns a string representation combining sensor and scenario names.

        Returns:
            str: Concatenated sensor and scenario names.
        """
        return self.sensor.name + " " + self.scenario.name

    def __repr__(self) -> str:
        """
        Returns a representation of the perturber including sensor and scenario names.

        Returns:
            str: Representation showing sensor and scenario names.
        """
        return self.sensor.name + " " + self.scenario.name

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        Provides the default configuration for PybsmPerturber instances.

        Returns:
            dict[str, Any]: A dictionary with the default configuration values.
        """
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        cfg["reflectance_range"] = cfg["reflectance_range"].tolist()
        return cfg

    @classmethod
    def from_config(cls: type[C], config_dict: dict, merge_default: bool = True) -> C:
        """
        Instantiates a PybsmPerturber from a configuration dictionary.

        Args:
            config_dict (dict): Configuration dictionary with initialization parameters.
            merge_default (bool, optional): Whether to merge with default configuration. Defaults to True.

        Returns:
            C: An instance of PybsmPerturber configured according to `config_dict`.
        """
        config_dict = dict(config_dict)

        config_dict["sensor"] = from_config_dict(config_dict["sensor"], [PybsmSensor])
        config_dict["scenario"] = from_config_dict(config_dict["scenario"], [PybsmScenario])

        # Convert input data to expected constructor types
        config_dict["reflectance_range"] = np.array(config_dict["reflectance_range"])

        return super().from_config(config_dict, merge_default=merge_default)

    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the PybsmPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = super().get_config()

        cfg["sensor"] = to_config_dict(self.sensor)
        cfg["scenario"] = to_config_dict(self.scenario)
        cfg["reflectance_range"] = self.reflectance_range.tolist()
        cfg["rng_seed"] = self._rng_seed

        return cfg

    @classmethod
    def is_usable(cls) -> bool:
        """
        Checks if the necessary dependencies (pybsm and OpenCV) are available.

        Returns:
            bool: True if both pybsm and OpenCV are available; False otherwise.
        """
        # Requires pybsm[graphics] or pybsm[headless]
        cv2_check = find_spec("cv2") is not None
        return cv2_check and pybsm_available
