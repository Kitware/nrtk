import abc

import numpy as np
from smqtk_core import Plugfigurable
from typing import Any, Dict


class PerturbImage(Plugfigurable):
    """
    Algorithm that generates a perturbed image for the given input image
    stimulus as a ``numpy.ndarray`` type array.
    """

    @abc.abstractmethod
    def perturb(
        self,
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Generate a perturbed image for the given image stimulus.

        :param image: Input image as a numpy array.
        :param additional_oarams: A dictionary containing perturber implementation-specific input param-values pairs.

        :return: Peturbed image as numpy array, including matching shape and dtype.
            Implementations should impart no side effects upon the input
            image.
        """

    def __call__(
        self,
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Calls ``perturb()`` with the given input image.
        """
        return self.perturb(image, additional_params)
