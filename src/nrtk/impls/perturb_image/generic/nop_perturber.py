"""
This module provides the `NOPPerturber` class, an implementation of the `PerturbImage` interface
that performs no alteration on the input image. It serves as a pass-through or "no operation"
(NOP) perturber, returning an exact copy of the input image. This class is useful in testing or
as a baseline when no perturbation is desired.

Classes:
    NOPPerturber: An implementation of the `PerturbImage` interface that returns an unmodified
    copy of the input image.

Dependencies:
    - numpy for handling image data.
    - nrtk.interfaces for the `PerturbImage` interface.

Usage:
    Instantiate `NOPPerturber` and call `perturb` with an input image to obtain a copy of
    the original image.

Example:
    nop_perturber = NOPPerturber()
    output_image = nop_perturber.perturb(input_image)
"""

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

    def get_config(self) -> dict[str, Any]:
        """
        Get the configuration dictionary of the ComposePerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary containing perturber configurations.
        """
        return {}
