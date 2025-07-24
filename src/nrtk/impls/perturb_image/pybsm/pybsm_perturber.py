"""Implements PybsmPerturber for image perturbations using pyBSM with sensor and scenario configs.

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

from __future__ import annotations

import copy
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

try:
    # Multiple type ignores added for pyright's handling of guarded imports
    from pybsm.simulation import simulate_image
    from pybsm.simulation.ref_image import RefImage

    pybsm_available: bool = True
except ImportError:  # pragma: no cover
    pybsm_available: bool = False

from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import PyBSMImportError

DEFAULT_REFLECTANCE_RANGE = np.array([0.05, 0.5])  # It is bad standards to call np.array within argument defaults


class PybsmPerturber(PerturbImage):
    """Implements image perturbation using pyBSM sensor and scenario configurations.

    The `PybsmPerturber` class applies realistic perturbations to images by leveraging
    pyBSM's simulation functionalities. It takes in a sensor and scenario, along with
    other optional parameters, to simulate environmental effects on the image.

    See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.

    Attributes:
        sensor (PybsmSensor):
            The sensor configuration for the perturbation.
        scenario (PybsmScenario):
            Scenario settings to apply during the perturbation.
        reflectance_range (np.ndarray):
            Default reflectance range for image simulation.
    """

    def __init__(  # noqa: C901
        self,
        sensor: PybsmSensor,
        scenario: PybsmScenario,
        reflectance_range: np.ndarray[Any, Any] = DEFAULT_REFLECTANCE_RANGE,
        rng_seed: int = 1,
        box_alignment_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the PybsmPerturber.

        Args:
            sensor:
                pyBSM sensor object.
            scenario:
                pyBSM scenario object.
            reflectance_range:
                Array of reflectances that correspond to pixel values.
            rng_seed:
                integer seed value that will be used for the random number generator.
            box_alignment_mode:
                Deprecated. Misaligned bounding boxes will always be resolved by taking the
                smallest possible box that encases the transformed misaligned box.

                .. deprecated:: 0.24.0
            kwargs:
                sensor and/or scenario values to modify.

        Raises:
            :raises ImportError: If pyBSM is not found, install via `pip install nrtk[pybsm]`.
            :raises ValueError: If reflectance_range length != 2
            :raises ValueError: If reflectance_range not strictly ascending
        """
        if not self.is_usable():
            raise PyBSMImportError
        super().__init__(box_alignment_mode=box_alignment_mode)
        self._rng_seed = rng_seed
        self.sensor: PybsmSensor = copy.deepcopy(sensor)
        self.scenario: PybsmScenario = copy.deepcopy(scenario)

        for k in kwargs:
            if hasattr(self.sensor, k):
                setattr(self.sensor, k, kwargs[k])
            elif hasattr(self.scenario, k):
                setattr(self.scenario, k, kwargs[k])

        if reflectance_range.shape[0] != 2:
            raise ValueError(f"Reflectance range array must have length of 2, got {reflectance_range.shape[0]}")
        if reflectance_range[0] >= reflectance_range[1]:
            raise ValueError(f"Reflectance range array values must be strictly ascending, got {reflectance_range}")
        self.reflectance_range: np.ndarray[Any, Any] = reflectance_range

        # this is key:value record of the thetas use for perturbing
        self.thetas: dict[str, Any] = copy.deepcopy(kwargs)

    @property
    def params(self) -> dict[str, Any]:
        """Retrieves the theta parameters related to the perturbation configuration.

        This method retrieves extra configuration details for the `PybsmPerturber` instance,
        which may include specific parameters related to the sensor, scenario, or any
        additional customizations applied during initialization.

        Returns:
            :return dict[str, Any]: A dictionary containing additional perturbation parameters.
        """
        return self.thetas

    @override
    def perturb(
        self,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Applies pyBSM-based perturbations to the provided image.

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
            The perturbed image and bounding boxes scaled to perturbed image shape.

        Raises:
            :raises ValueError: If 'img_gsd' is not provided in `additional_params`.
        """
        if additional_params is None:  # Cannot have mutable data structure in argument default
            additional_params = dict()
        if "img_gsd" not in additional_params:
            raise ValueError("'img_gsd' must be present in image metadata for this perturber")

        # Create a `RefImage` object using the given GSD, img_pixel and reflactance values
        ref_img = RefImage(  # type: ignore
            image,
            additional_params["img_gsd"],
            np.array([image.min(), image.max()]),
            self.reflectance_range,
        )

        # Generate a perturbed image using the given sensor and scenario parameters
        perturbed = simulate_image(ref_img, self.sensor(), self.scenario(), self._rng_seed)[-1]  # type: ignore

        # Min-Max normalization and conversion to uint8 type
        min_perturbed_val = perturbed.min()
        den = perturbed.max() - min_perturbed_val
        perturbed -= min_perturbed_val
        perturbed /= den
        perturbed *= 255

        # Rescale bounding boxes to the shape of the perturbed image
        if boxes:
            scaled_boxes = self._rescale_boxes(boxes, image.shape, perturbed.shape)
            return perturbed.astype(np.uint8), scaled_boxes

        return perturbed.astype(np.uint8), boxes

    def __str__(self) -> str:
        """Returns a string representation combining sensor and scenario names.

        Returns:
            :return str: Concatenated sensor and scenario names.
        """
        return self.sensor.name + " " + self.scenario.name

    def __repr__(self) -> str:
        """Returns a representation of the perturber including sensor and scenario names.

        Returns:
            :return str: Representation showing sensor and scenario names.
        """
        return self.sensor.name + " " + self.scenario.name

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Retrieves the default configuration for PybsmPerturber instances.

        Returns:
            :return dict[str, Any]: A dictionary with the default configuration values.
        """
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        cfg["reflectance_range"] = cfg["reflectance_range"].tolist()
        return cfg

    @classmethod
    def from_config(cls, config_dict: dict[str, Any], merge_default: bool = True) -> PybsmPerturber:
        """Instantiates a PybsmPerturber from a configuration dictionary.

        Args:
            config_dict:
                Configuration dictionary with initialization parameters.
            merge_default:
                Whether to merge with default configuration. Defaults to True.

        Returns:
            :return PybsmPerturber: An instance of PybsmPerturber.
        """
        config_dict = dict(config_dict)

        config_dict["sensor"] = from_config_dict(config_dict["sensor"], [PybsmSensor])
        config_dict["scenario"] = from_config_dict(config_dict["scenario"], [PybsmScenario])

        # Convert input data to expected constructor types
        config_dict["reflectance_range"] = np.array(config_dict["reflectance_range"])

        return super().from_config(config_dict, merge_default=merge_default)

    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the PybsmPerturber instance.

        Returns:
            :return dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = super().get_config()

        cfg["sensor"] = to_config_dict(self.sensor)
        cfg["scenario"] = to_config_dict(self.scenario)
        cfg["reflectance_range"] = self.reflectance_range.tolist()
        cfg["rng_seed"] = self._rng_seed

        return cfg

    @classmethod
    def is_usable(cls) -> bool:
        """Checks if the necessary dependency (pyBSM) is available.

        Returns:
            :return bool: True pyBSM is available; False otherwise.
        """
        return pybsm_available
