"""
This module defines the `RandomTranslationPerturber` class, which implements a random
translation perturbation on input images. The class supports adjusting bounding
boxes to match the translated region, making it suitable for tasks involving
labeled datasets.

Classes:
    RandomTranslationPerturber: A perturbation class for applying random translation
    on images and the corresponding bounding boxes.

Dependencies:
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for perturbation algorithms.
"""

from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Optional, Union

import numpy as np
from smqtk_image_io import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image import PerturbImage


class RandomTranslationPerturber(PerturbImage):
    """
    RandomTranslationPerturber randomly translates an image and adjusts bounding boxes accordingly.
    Methods:
    perturb: Applies a random translation to an input image and adjusts bounding boxes.
    __call__: Calls the perturb method with the given input image.
    get_config: Returns the current configuration of the RandomTranslationPerturber instance.
    """

    def __init__(
        self,
        box_alignment_mode: str = "extent",
        seed: Optional[Union[int, np.random.Generator]] = 1,
        color_fill: Optional[Sequence[int]] = [0, 0, 0],
    ) -> None:
        """
        RandomTranslationPerturber applies a random translation perturbation to an input image.
        It ensures that bounding boxes are adjusted correctly to reflect the translated
        image coordinates.

        Attributes:
            rng (numpy.random.Generator): Random number generator for deterministic behavior.
            color_fill: Background color fill for RGB image.
        """
        super().__init__(box_alignment_mode=box_alignment_mode)
        self.rng = np.random.default_rng(seed)
        self.color_fill = np.array(color_fill)

    def perturb(  # noqa: C901
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        """
        Randomly translates an image and adjusts bounding boxes.

        :param image: Input image as a numpy array of shape (H, W, C).
        :param boxes: List of bounding boxes in AxisAlignedBoundingBox format and their corresponding classes.

        :param additional_params: Dictionary containing:
        - "max_translation_limit" (tuple[int, int]): Max translation magnitude
          (translate_h, translate_w) lesser than or equal to the size of the input
          image.

        :return: Translated image as numpy array with the modified bounding boxes
        """
        super().perturb(image=image)

        translate_h, translate_w = additional_params.get(
            "max_translation_limit",
            (image.shape[0], image.shape[1]),
        )
        if abs(translate_h) > image.shape[0] or abs(translate_w) > image.shape[1]:
            raise ValueError(f"Max translation limit should be less than or equal to {image.shape[:2]}")

        # Randomly select the translation magnitude for each direction
        translate_x, translate_y = (0, 0)
        if translate_w > 0:
            translate_x = self.rng.integers(-translate_w, translate_w)
        if translate_h > 0:
            translate_y = self.rng.integers(-translate_h, translate_h)

        # Apply background color fill based on the number of image dimensions
        if image.ndim == 3:
            final_image = np.full_like(image, self.color_fill.astype(image.dtype), dtype=image.dtype)
        else:
            final_image = np.zeros_like(image, dtype=image.dtype)

        # Perform the translation
        translated_image = np.roll(image.copy(), (translate_y, translate_x), axis=[0, 1])

        # Apply the valid translated image region to the final background image
        if translate_x >= 0 and translate_y >= 0:
            final_image[translate_y:, translate_x:, ...] = translated_image[translate_y:, translate_x:, ...]
        elif translate_x < 0 and translate_y >= 0:
            final_image[translate_y:, :translate_x, ...] = translated_image[translate_y:, :translate_x, ...]
        elif translate_x >= 0 and translate_y < 0:
            final_image[:translate_y, translate_x:, ...] = translated_image[:translate_y, translate_x:, ...]
        else:
            final_image[:translate_y, :translate_x, ...] = translated_image[:translate_y, :translate_x, ...]

        # Adjust bounding boxes
        adjusted_bboxes = []
        if boxes is not None:
            for bbox, metadata in boxes:
                # Shift the bounding box to align with the translated image coordinates
                shifted_min_x = bbox.min_vertex[0] + translate_x
                shifted_min_y = bbox.min_vertex[1] + translate_y
                shifted_min = (
                    shifted_min_x if shifted_min_x >= 0 else 0,
                    shifted_min_y if shifted_min_y >= 0 else 0,
                )
                shifted_max_x = bbox.max_vertex[0] + translate_x
                shifted_max_y = bbox.max_vertex[1] + translate_y
                shifted_max = (
                    shifted_max_x if shifted_max_x <= translate_w else translate_w,
                    shifted_max_y if shifted_max_y <= translate_h else translate_h,
                )
                adjusted_box = AxisAlignedBoundingBox(shifted_min, shifted_max)
                adjusted_bboxes.append((adjusted_box, metadata))
        return final_image, adjusted_bboxes

    def __call__(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Union[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]], None]]:
        """Calls `perturb` with the given input image."""
        return self.perturb(image=image, boxes=boxes, additional_params=additional_params)

    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the RandomTranslationPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = super().get_config()
        cfg["seed"] = self.rng
        cfg["color_fill"] = self.color_fill
        return cfg
