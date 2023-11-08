from typing import Any, Dict

import numpy as np

from nrtk.interfaces.perturb_image import PerturbImage


class NOPPerturber(PerturbImage):
    """
    Example implementation of the ``PerturbImage`` interface. An instance of
    this class acts as a functor to generate a perturbed image for the given
    input image stimulus.

    This class, in particular, serves as pass-through "no operation" (NOP)
    perturber.
    """

    def perturb(
        self,
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Return unperturbed image.
        """
        return np.copy(image)

    def get_config(self) -> Dict[str, Any]:
        return {}
