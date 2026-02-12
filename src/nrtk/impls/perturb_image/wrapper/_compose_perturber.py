"""Defines ComposePerturber to apply multiple PerturbImage instances sequentially for combined image perturbations.

Classes:
    ComposePerturber: A perturbation class for applying perturbations from Albumentations

Dependencies:
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for perturbation algorithms.

Example usage:
    >>> from nrtk.impls.perturb_image.photometric.enhance import BrightnessPerturber
    >>> from nrtk.impls.perturb_image.geometric.random import RandomCropPerturber
    >>> image = np.ones((256, 256, 3))
    >>> perturbers = [RandomCropPerturber(), BrightnessPerturber(factor=0.5)]
    >>> perturber = ComposePerturber(perturbers=perturbers)
    >>> perturbed_image, _ = perturber(image=image)
"""

from __future__ import annotations

__all__ = ["ComposePerturber"]

import copy
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_core.configuration import (
    from_config_dict,
    to_config_dict,
)
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import Self, override

from nrtk.interfaces.perturb_image import PerturbImage


class ComposePerturber(PerturbImage):
    """Composes multiple image perturbations by applying a list of perturbers sequentially to an input image.

    Attributes:
        perturbers (list[PerturbImage]):
            List of perturbers to apply.

    Note:
        This class has not been tested with perturber factories and is not expected
        to work with perturber factories.
    """

    def __init__(self, perturbers: list[PerturbImage] | None = None) -> None:
        """Initializes the ComposePerturber.

        This has not been tested with perturber factories and is not expected to work with perturber factories.

        Args:
            perturbers:
                List of perturbers to apply.
        """
        super().__init__()
        if perturbers is None:
            perturbers = list()
        self.perturbers = perturbers

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Apply the sequence of perturbers to the input image.

        Args:
            image:
                The input image to perturb.
            boxes:
                The bounding boxes for the input image. This is the single image
                output from DetectImageObjects.detect_objects.
            kwargs:
                Additional perturbation keyword arguments.

        Returns:
            The perturbed image and the source bounding boxes.
        """
        perturbed_image = image
        perturbed_boxes = boxes

        # Applies series of perturbations to a the given input image
        for perturber in self.perturbers:
            perturbed_image, perturbed_boxes = perturber(
                image=perturbed_image,
                boxes=perturbed_boxes,
                **kwargs,
            )

        if len(self.perturbers) == 0:
            perturbed_image = copy.deepcopy(image)
            perturbed_boxes = copy.deepcopy(boxes)

        return perturbed_image, perturbed_boxes

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary of the ComposePerturber instance."""
        cfg = super().get_config()
        cfg["perturbers"] = [to_config_dict(perturber) for perturber in self.perturbers]
        return cfg

    @classmethod
    @override
    def from_config(
        cls,
        config_dict: dict[str, Any],
        merge_default: bool = True,
    ) -> Self:
        """Create a ComposePerturber instance from a configuration dictionary.

        Args:
            config_dict:
                Configuration dictionary with perturber details.
            merge_default:
                Whether to merge with the default configuration.

        Returns:
            An instance of ComposePerturber.
        """
        config_dict = dict(config_dict)

        config_dict["perturbers"] = [
            from_config_dict(config=perturber, type_iter=PerturbImage.get_impls())
            for perturber in config_dict["perturbers"]
        ]

        return super().from_config(config_dict, merge_default=merge_default)
