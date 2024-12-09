"""
This module provides the ComposePerturber class, which allows for composing multiple
image perturbations by sequentially applying a list of PerturbImage instances.
"""

from collections.abc import Hashable, Iterable
from typing import Any, TypeVar

import numpy as np
from smqtk_core.configuration import (
    from_config_dict,
    to_config_dict,
)
from smqtk_image_io import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage

C = TypeVar("C", bound="ComposePerturber")


class ComposePerturber(PerturbImage):
    """
    A class that composes multiple image perturbations by applying a list of perturbers
    sequentially to an input image.

    Note:
        This class has not been tested with perturber factories and is not expected
        to work with perturber factories.
    """

    def __init__(self, perturbers: list[PerturbImage], box_alignment_mode: str = "extent") -> None:
        """Initializes the ComposePerturber.

        This has not been tested with perturber factories and is not expected to work wit perturber factories.

        :param perturbers: list of perturbers to apply
        """
        super().__init__(box_alignment_mode=box_alignment_mode)
        self.perturbers = perturbers

    @override
    def perturb(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        """
        Apply the sequence of perturbers to the input image.

        Args:
            image (np.ndarray): The input image to perturb.
            boxes (Optional[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]): The bounding boxes for
                the input image. This is the single image output from DetectImageObjects.detect_objects
            additional_params (Optional[dict[str, Any]]): Additional parameters for perturbation.

        Returns:
            np.ndarray: The perturbed image.
        """
        out_img = image

        if additional_params is None:
            additional_params = dict()

        for perturber in self.perturbers:
            out_img, _ = perturber(image=out_img, boxes=boxes, additional_params=additional_params)

        return out_img, boxes

    def get_config(self) -> dict[str, Any]:
        """
        Get the configuration dictionary of the ComposePerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary containing perturber configurations.
        """
        cfg = super().get_config()
        cfg["perturbers"] = [to_config_dict(perturber) for perturber in self.perturbers]
        return cfg

    @classmethod
    def from_config(
        cls: type[C],
        config_dict: dict,
        merge_default: bool = True,
    ) -> C:
        """
        Create a ComposePerturber instance from a configuration dictionary.

        Args:
            config_dict (dict): Configuration dictionary with perturber details.
            merge_default (bool): Whether to merge with the default configuration.

        Returns:
            ComposePerturber: An instance of ComposePerturber.
        """
        config_dict = dict(config_dict)

        config_dict["perturbers"] = [
            from_config_dict(perturber, PerturbImage.get_impls()) for perturber in config_dict["perturbers"]
        ]

        return super().from_config(config_dict, merge_default=merge_default)
