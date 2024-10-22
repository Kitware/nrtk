from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage


class NOPPerturber(PerturbImage):
    """Example implementation of the ``PerturbImage`` interface.

    An instance of this class acts as a functor to generate a perturbed image for the given
    input image stimulus.

    This class, in particular, serves as pass-through "no operation" (NOP)
    perturber.
    """

    @override
    def perturb(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return unperturbed image."""
        if additional_params is None:
            additional_params = dict()
        return np.copy(image)

    @override
    def get_config(self) -> dict[str, Any]:
        return {}
