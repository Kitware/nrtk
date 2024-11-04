"""
This module defines the `ImageMetric` abstract base class, an interface for computing
metrics between one or two images. Implementations of `ImageMetric` should define the
specific metric computation in the `compute` method, which can be called directly or
via the `__call__` method.

Classes:
    ImageMetric: An interface outlining the computation of a given metric for image comparison or analysis.

Dependencies:
    - numpy for numerical operations on images.
    - smqtk_core for configuration management and plugin compatibility.

Example usage:
    class SpecificMetric(ImageMetric):
        def compute(self, img_1, img_2=None, additional_params=None):
            # Define metric calculation logic here.
            pass

    metric = SpecificMetric()
    score = metric(img_1, img_2)
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
from smqtk_core import Plugfigurable


class ImageMetric(Plugfigurable):
    """This interface outlines the computation of a given metric between up to two images."""

    @abc.abstractmethod
    def compute(
        self,
        img_1: np.ndarray,
        img_2: np.ndarray | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> float:
        """Given up to two images, and additional parameters, return some given metric about the image(s).

        :param img_1: An input image in the shape (height, width, channels).
        :param img_2: An optional input image in the shape (height, width, channels)
        :param additional_params: A dictionary containing implementation-specific input param-values pairs.

        :return: Returns a single scalar value representing an implementation's computed metric. Implementations
                 should impart no side effects upon either input image or the additional parameters.
        """

    def __call__(
        self,
        img_1: np.ndarray,
        img_2: np.ndarray | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> float:
        """Calls compute() with the given input image(s) and additional parameters.

        :param img_1: An input image in the shape (height, width, channels).
        :param img_2: An optional input image in the shape (height, width, channels)
        :param additional_params: A dictionary containing implementation-specific input param-values pairs.

        :return: Returns a single scalar value representing an implementation's computed metric. Implementations
                 should impart no side effects upon either input image or the additional parameters.
        """
        return self.compute(img_1, img_2, additional_params)

    @property
    def name(self) -> str:
        """
        Returns the name of the ImageMetric instance.

        This property provides a convenient way to retrieve the name of the
        class instance, which can be useful for logging, debugging, or display purposes.

        Returns:
            str: The name of the ImageMetric instance.
        """
        return self.__class__.__name__

    def get_config(self) -> dict[str, Any]:
        """Returns the config for the interface."""
        return {}
