"""This module defines the `HazePerturber` class, which implements a haze perturbation on input images.

Classes:
    HazePerturber: A perturbation class for applying haze through Shree Nayer weathering

Dependencies:
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for perturbation algorithms.
"""

from __future__ import annotations

__all__ = ["HazePerturber"]

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage


class HazePerturber(PerturbImage):
    """HazePerturber applies haze using Shree Nayar weathering.

    Attributes:
        factor (float): Strength of haze applied to an image.

    Methods:
        perturb:
            Applies haze to an input image.
        __call__:
            Calls the perturb method with the given input image.
        get_config:
            Returns the current configuration of the HazePerturber instance.
    """

    def __init__(self, factor: float = 1.0) -> None:
        """HazePerturber applies haze to an input image.

        Attributes:
            factor: Strength of haze applied to an image.
        """
        super().__init__()
        self.factor = factor

    @override
    def perturb(
        self,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        depth_map: np.ndarray[Any, Any] | None = None,
        sky_color: list[float] | None = None,
        **additional_params: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Apply haze to an image based on depth_map and sky_color.

        Args:
            image:
                Input image as a numpy array of shape (H, W, C).
            boxes:
                List of bounding boxes in AxisAlignedBoundingBox format and their corresponding classes.
            depth_map:
                Optional depth map for adding haze. If depth_map is not provided, then a depth map the size of the
                image with all values equal to 1 will be used.
            sky_color:
                Sky color to use for weathering. If sky_color is not provided, then an average pixel value will be
                calculated and used as the sky color.
            additional_params:
                Additional perturbation keyword arguments (currently unused).

        Returns:
            :return tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
                Image with haze applied and source bounding boxes.
        """
        # Use either the provided depth map or a map containing all values = 1
        if depth_map is None:
            depth_map = np.ones_like(image)
        else:
            if len(image.shape) != len(depth_map.shape):
                raise ValueError(
                    f"image dims ({len(image.shape)}) does not match depth_map dims ({len(depth_map.shape)})",
                )

        # Use either the provided sky_color map or a map containing avg. pixel values
        if sky_color is None:
            final_sky_color = np.mean(image, axis=(0, 1))
        else:
            final_sky_color = self._check_sky_color(image=image, sky_color=sky_color)

        # Beer's Law of Attenuation based on the haze factor and depth map
        attenuation = np.exp(-self.factor * depth_map)
        output = image * attenuation + final_sky_color * (1 - attenuation)

        return output.astype(np.uint8), boxes

    def _check_sky_color(self, image: np.ndarray, sky_color: list[float]) -> list[float]:
        if (len(image.shape) == 3 and len(sky_color) != 3) or (len(image.shape) == 2 and len(sky_color) != 1):
            raise ValueError(
                f"image bands ({3 if len(image.shape) == 3 else 1}) do not match sky_color bands ({len(sky_color)})",
            )
        return sky_color

    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the HazePerturber instance.

        Returns:
            :return dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = super().get_config()
        cfg["factor"] = self.factor
        return cfg
