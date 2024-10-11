import abc
from typing import Any, Dict, Optional

import numpy as np
from smqtk_core import Plugfigurable


class PerturbImage(Plugfigurable):
    """Algorithm that generates a perturbed image for given input image stimulus as a ``numpy.ndarray`` type array."""

    @abc.abstractmethod
    def perturb(self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
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

    def __call__(self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Calls ``perturb()`` with the given input image."""
        if additional_params is None:
            additional_params = dict()
        return self.perturb(image, additional_params)

    @classmethod
    def get_type_string(cls) -> str:
        return f"{cls.__module__}.{cls.__name__}"
