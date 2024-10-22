"""
Module: PybsmPerturber
======================
This module implements the `PybsmPerturber` class, a specialized perturber for applying pyBSM-based
perturbations to images. It includes functionality for configuration handling, image perturbation,
and validation of required dependencies.
"""

from __future__ import annotations

import copy
from importlib.util import find_spec
from typing import Any, TypeVar

import numpy as np

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
from typing_extensions import override

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage

C = TypeVar("C", bound="PybsmPerturber")

DEFAULT_REFLECTANCE_RANGE = np.array([0.05, 0.5])  # It is bad standards to call np.array within argument defaults


class PybsmPerturber(PerturbImage):
    """
    Implements a perturber that uses pyBSM simulate_image for applying controlled
    perturbations to images. Supports configuration-based initialization and validation
    of pyBSM dependencies.

    Attributes:
        sensor (PybsmSensor): The pyBSM sensor object used for simulations.
        scenario (PybsmScenario): The pyBSM scenario object used for simulations.
        reflectance_range (np.ndarray): A 2-element array defining pixel reflectance range.
        thetas (dict[str, Any]): Dictionary storing parameters used for perturbations.
        _rng_seed (int): Random number generator seed.
    """

    def __init__(  # noqa: C901
        self,
        sensor: PybsmSensor,
        scenario: PybsmScenario,
        reflectance_range: np.ndarray = DEFAULT_REFLECTANCE_RANGE,
        rng_seed: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initializes the PybsmPerturber.

        :param sensor: pyBSM sensor object.
        :param scenario: pyBSM scenario object.
        :param reflectance_range: Array of reflectances that correspond to pixel values.
        :param rng_seed: integer seed value that will be used for the random number generator

        :raises: ImportError if pyBSM with OpenCV not found,
        installed via 'nrtk[pybsm-graphics]' or 'nrtk[pybsm-headless]'.
        :raises: ValueError if reflectance_range length != 2
        :raises: ValueError if reflectance_range not strictly ascending
        """
        if not self.is_usable():
            raise ImportError(
                "pyBSM with OpenCV not found. Please install 'nrtk[pybsm-graphics]' or 'nrtk[pybsm-headless]'.",
            )
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

    @override
    @property
    def params(self) -> dict:
        """
        Returns the parameters used for the perturbation.

        Returns:
            dict: A dictionary of perturbation parameters.
        """
        return self.thetas

    def perturb(self, image: np.ndarray, additional_params: dict[str, Any] = None) -> np.ndarray:
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

        return perturbed.astype(np.uint8)

    def __call__(self, image: np.ndarray, additional_params: dict[str, Any] = None) -> np.ndarray:
        """Alias for :meth:`.NIIRS.apply`."""
        if additional_params is None:
            additional_params = dict()
        return self.perturb(image, additional_params)

    def __str__(self) -> str:
        """
        Returns a string representation of the PybsmPerturber.

        Returns:
            str: The names of the sensor and scenario.
        """
        return self.sensor.name + " " + self.scenario.name

    def __repr__(self) -> str:
        """
        Returns a string representation for debugging.

        Returns:
            str: The names of the sensor and scenario.
        """
        return self.sensor.name + " " + self.scenario.name

    @override
    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        Generate the default configuration for the perturber.

        Returns:
            dict[str, Any]: Default configuration as a dictionary.
        """
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        cfg["reflectance_range"] = cfg["reflectance_range"].tolist()
        return cfg

    @override
    @classmethod
    def from_config(cls: type[C], config_dict: dict, merge_default: bool = True) -> C:
        """
        Create an instance of the perturber from a configuration dictionary.

        Args:
            config_dict (dict): Configuration dictionary.
            merge_default (bool): Whether to merge with the default configuration.

        Returns:
            PybsmPerturber: A configured perturber instance.
        """
        config_dict = dict(config_dict)

        config_dict["sensor"] = from_config_dict(config_dict["sensor"], [PybsmSensor])
        config_dict["scenario"] = from_config_dict(config_dict["scenario"], [PybsmScenario])

        # Convert input data to expected constructor types
        config_dict["reflectance_range"] = np.array(config_dict["reflectance_range"])

        return super().from_config(config_dict, merge_default=merge_default)

    @override
    def get_config(self) -> dict[str, Any]:
        """
        Get the current configuration of the perturber.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        return {
            "sensor": to_config_dict(self.sensor),
            "scenario": to_config_dict(self.scenario),
            "reflectance_range": self.reflectance_range.tolist(),
            "rng_seed": self._rng_seed,
        }

    @override
    @classmethod
    def is_usable(cls) -> bool:
        """
        Check if the pyBSM and OpenCV dependencies are available.

        Returns:
            bool: True if dependencies are available, False otherwise.
        """
        # Requires pybsm[graphics] or pybsm[headless]
        cv2_check = find_spec("cv2") is not None
        return cv2_check and pybsm_available
