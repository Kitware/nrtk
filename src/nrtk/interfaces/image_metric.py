from __future__ import annotations

import abc
from typing import Any

import numpy as np
from smqtk_core import Plugfigurable
from typing_extensions import override


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
        return self.__class__.__name__

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the config for the interface."""
        return {}
