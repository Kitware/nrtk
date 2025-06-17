"""Defines NOPPerturber, a PerturbImage implementation that returns the input image unchanged for testing or baselines.

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

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
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
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Returns unperturbed image and input bounding boxes.

        Args:
            image:
                Input image as a numpy array of shape (H, W, C).
            boxes:
                List of bounding boxes in AxisAlignedBoundingBox format and their corresponding classes.
            additional_params:
                Additional parameters for perturbation.

        Returns:
            :return tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
                Returns the source image and bounding boxes.
        """
        if additional_params is None:
            additional_params = dict()
        return np.copy(image), boxes
