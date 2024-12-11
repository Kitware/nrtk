"""
This module defines the `PerturbImage` interface, which provides an abstract base for
implementing image perturbation algorithms. The primary purpose of this interface is to
generate perturbed versions of input images, represented as `numpy.ndarray` arrays.

Classes:
    PerturbImage: An abstract base class that specifies the structure for image perturbation
    algorithms, allowing for different perturbation techniques to be implemented.

Dependencies:
    - numpy for handling image arrays.
    - smqtk_core for configurable plugin interface capabilities.

Usage:
    To create a custom image perturbation class, inherit from `PerturbImage` and implement
    the `perturb` method, defining the specific perturbation logic.

Example:
    class CustomPerturbImage(PerturbImage):
        def perturb(self, image, additional_params=None):
            # Custom perturbation logic here
            pass

    perturber = CustomPerturbImage()
    perturbed_image = perturber(image_data)
"""

import abc
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_core import Plugfigurable
from smqtk_image_io import AxisAlignedBoundingBox


class PerturbImage(Plugfigurable):
    """Algorithm that generates a perturbed image for given input image stimulus as a ``numpy.ndarray`` type array."""

    def __init__(self, box_alignment_mode: str = "extent") -> None:
        """Initializes the PerturbImage.

        :param box_alignment_mode: Mode for how to handle how bounding boxes change.
            Should be one of the following options:
                extent: a new axis-aligned bounding box that encases the transformed misaligned box
                extant: a new axis-aligned bounding box that is encased inside the transformed misaligned box
                median: median between extent and extant
            Default value is extent
        """
        self.box_alignment_mode = box_alignment_mode

    @abc.abstractmethod
    def perturb(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        """Generate a perturbed image for the given image stimulus.

        Note perturbers that resize, rotate, or similarly affect the dimensions of an image may impact
        scoring if bounding boxes are not similarly transformed.

        :param image: Input image as a numpy array.
        :param boxes: Input bounding boxes as a Iterable of tuples containing bounding boxes. This is the single
            image output from DetectImageObjects.detect_objects
        :param additional_params: A dictionary containing perturber implementation-specific input param-values pairs.

        :return:
            Perturbed image as numpy array, including matching dtype. Implementations should impart no side
                effects upon the input image.
            Iterable of tuples containing the bounding boxes for detections in the image. If an implementation
                modifies the size of an image, it is expected to modify the bounding boxes as well.
        """
        if additional_params is None:
            additional_params = dict()
        return image, boxes

    def __call__(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        """Calls ``perturb()`` with the given input image."""
        if additional_params is None:
            additional_params = dict()
        return self.perturb(image=image, boxes=boxes, additional_params=additional_params)

    @classmethod
    def get_type_string(cls) -> str:
        """
        Returns the fully qualified type string of the `PerturbImage` class or its subclass.

        :return: A string representing the fully qualified type, in the format `<module>.<class_name>`.
                 For example, "my_module.CustomPerturbImage".
        """
        return f"{cls.__module__}.{cls.__name__}"

    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the PerturbImage instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        return {"box_alignment_mode": self.box_alignment_mode}
