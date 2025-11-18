"""Defines ComposePerturber to apply multiple PerturbImage instances sequentially for combined image perturbations."""

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
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **additional_params: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Apply the sequence of perturbers to the input image.

        Args:
            image:
                The input image to perturb.
            boxes:
                The bounding boxes for the input image. This is the single image
                output from DetectImageObjects.detect_objects.
            additional_params:
                Additional perturbation keyword arguments.

        Returns:
            :return tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
                The perturbed image and the source bounding boxes.
        """
        out_img = image
        out_boxes = boxes

        # Applies series of perturbations to a the given input image
        for perturber in self.perturbers:
            out_img, out_boxes = perturber(image=out_img, boxes=out_boxes, **additional_params)

        if len(self.perturbers) == 0:
            out_img = copy.deepcopy(image)

        return out_img, out_boxes

    def get_config(self) -> dict[str, Any]:
        """Get the configuration dictionary of the ComposePerturber instance.

        Returns:
            :return dict[str, Any]: Configuration dictionary containing perturber configurations.
        """
        cfg = super().get_config()
        cfg["perturbers"] = [to_config_dict(perturber) for perturber in self.perturbers]
        return cfg

    @classmethod
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
            :return ComposePerturber: An instance of ComposePerturber.
        """
        config_dict = dict(config_dict)

        config_dict["perturbers"] = [
            from_config_dict(perturber, PerturbImage.get_impls()) for perturber in config_dict["perturbers"]
        ]

        return super().from_config(config_dict, merge_default=merge_default)
