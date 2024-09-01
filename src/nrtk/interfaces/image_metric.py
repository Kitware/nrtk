import abc
from typing import Any, Dict, Optional

import numpy as np
from smqtk_core import Plugfigurable


class ImageMetric(Plugfigurable):
    """This interface outlines the computation of a given metric between up to two images."""

    @abc.abstractmethod
    def compute(
        self,
        img_1: np.ndarray,
        img_2: Optional[np.ndarray] = None,
        additional_params: Optional[Dict[str, Any]] = None,
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
        img_2: Optional[np.ndarray] = None,
        additional_params: Optional[Dict[str, Any]] = None,
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

    def get_config(self) -> Dict[str, Any]:
        """Returns the config for the interface."""
        return {}
