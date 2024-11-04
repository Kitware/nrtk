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
from __future__ import annotations

import abc
from typing import Any

import numpy as np
from smqtk_core import Plugfigurable


class PerturbImage(Plugfigurable):
    """Algorithm that generates a perturbed image for given input image stimulus as a ``numpy.ndarray`` type array."""

    @abc.abstractmethod
    def perturb(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Generate a perturbed image for the given image stimulus.

        Note perturbers that resize, rotate, or similarly affect the dimensions of an image may impact
        scoring if bounding boxes are not similarly transformed.

        :param image: Input image as a numpy array.
        :param additional_params: A dictionary containing perturber implementation-specific input param-values pairs.

        :return: Perturbed image as numpy array, including matching dtype. Implementations should impart no side
            effects upon the input image.
        """
        if additional_params is None:
            additional_params = dict()
        return image

    def __call__(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Calls ``perturb()`` with the given input image."""
        if additional_params is None:
            additional_params = dict()
        return self.perturb(image, additional_params)

    @classmethod
    def get_type_string(cls) -> str:
        """Calls ``perturb()`` with the given input image.

        :param image: Input image as a numpy array.
        :param additional_params: A dictionary containing additional parameters for the perturbation.

        :return: Perturbed image as numpy array.
        """
        return f"{cls.__module__}.{cls.__name__}"
