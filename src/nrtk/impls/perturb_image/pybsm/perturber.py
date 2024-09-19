from __future__ import annotations

import copy
from importlib.util import find_spec
from typing import Any, Dict, Optional, Type, TypeVar

import numpy as np
from pybsm.simulation import RefImage, simulate_image
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
    def __init__(
        self,
        sensor: PybsmSensor,
        scenario: PybsmScenario,
        reflectance_range: np.ndarray = DEFAULT_REFLECTANCE_RANGE,
        **kwargs: Any,
    ) -> None:
        """Initializes the PybsmPerturber.

        :param sensor: pyBSM sensor object.
        :param scenario: pyBSM scenario object.
        :param reflectance_range: Array of reflectances that correspond to pixel values.

        :raises: ValueError if reflectance_range length != 2
        :raises: ValueError if reflectance_range not strictly ascending
        """
        if not self.is_usable():
            raise ImportError("OpenCV not found. Please install 'nrtk[graphics]' or 'nrtk[headless]'.")
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
    def params(self) -> Dict:
        return self.thetas

    def perturb(self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
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

        perturbed = simulate_image(ref_img, self.sensor(), self.scenario())[-1]

        min_perturbed_val = perturbed.min()
        den = perturbed.max() - min_perturbed_val
        perturbed -= min_perturbed_val
        perturbed /= den
        perturbed *= 255

        return perturbed.astype(np.uint8)

    def __call__(self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Alias for :meth:`.NIIRS.apply`."""
        if additional_params is None:
            additional_params = dict()
        return self.perturb(image, additional_params)

    def __str__(self) -> str:
        return self.sensor.name + " " + self.scenario.name

    def __repr__(self) -> str:
        return self.sensor.name + " " + self.scenario.name

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        cfg["reflectance_range"] = cfg["reflectance_range"].tolist()

        return cfg

    @classmethod
    def from_config(cls: Type[C], config_dict: Dict, merge_default: bool = True) -> C:
        config_dict = dict(config_dict)

        config_dict["sensor"] = from_config_dict(config_dict["sensor"], [PybsmSensor])
        config_dict["scenario"] = from_config_dict(config_dict["scenario"], [PybsmScenario])

        # Convert input data to expected constructor types
        config_dict["reflectance_range"] = np.array(config_dict["reflectance_range"])

        return super().from_config(config_dict, merge_default=merge_default)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "sensor": to_config_dict(self.sensor),
            "scenario": to_config_dict(self.scenario),
            "reflectance_range": self.reflectance_range.tolist(),
        }

        return config

    @classmethod
    def is_usable(cls) -> bool:
        # Requires pyBSM which requires opencv to be installed
        return find_spec("cv2") is not None
