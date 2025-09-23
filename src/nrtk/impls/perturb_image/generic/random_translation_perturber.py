"""Defines RandomTranslationPerturber for random image shifts with bounding box adjustment for labeled datasets.

Classes:
    RandomTranslationPerturber: A perturbation class for applying random translation
    on images and the corresponding bounding boxes.

Dependencies:
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for perturbation algorithms.
"""

from __future__ import annotations

__all__ = ["RandomTranslationPerturber"]

from collections.abc import Hashable, Iterable, Sequence
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage


class RandomTranslationPerturber(PerturbImage):
    """RandomTranslationPerturber randomly translates an image and adjusts bounding boxes accordingly.

    Attributes:
        rng (numpy.random.Generator):
            Random number generator for deterministic behavior.
        color_fill (numpy.array):
            Background color fill for RGB image.

    Methods:
        perturb:
            Applies a random translation to an input image and adjusts bounding boxes.
        __call__:
            Calls the perturb method with the given input image.
        get_config:
            Returns the current configuration of the RandomTranslationPerturber instance.
    """

    def __init__(
        self,
        seed: int | np.random.Generator | None = 1,
        color_fill: Sequence[int] | None = [0, 0, 0],
    ) -> None:
        """RandomTranslationPerturber applies a random translation perturbation to an input image.

        It ensures that bounding boxes are adjusted correctly to reflect the translated
        image coordinates.

        Args:
            seed:
                Numpy random number generator.
            color_fill:
                Background color fill for RGB image.

        """
        super().__init__()
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.color_fill: np.ndarray[np.int64, Any] = np.array(color_fill)

    @override
    def perturb(  # noqa: C901
        self,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        max_translation_limit: tuple[int, int] | None = None,
        **additional_params: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Randomly translates an image and adjusts bounding boxes.

        Args:
            image:
                Input image as a numpy array of shape (H, W, C).
            boxes:
                List of bounding boxes in AxisAlignedBoundingBox format and their corresponding classes.
            max_translation_limit:
                Max translation magnitude (translate_h, translate_w) lesser than or equal to the size of the input
                image.
            additional_params:
                Additional perturbation keyword arguments (currently unused).

        Returns:
            :return tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
                Translated image with the modified bounding boxes.
        """
        super().perturb(image=image)

        if max_translation_limit is None:
            translate_h, translate_w = (image.shape[0], image.shape[1])
        else:
            translate_h, translate_w = max_translation_limit

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
                # Compute the shifted_min coords for the bounding box to align with
                # the translated min_vertex coordinates
                shifted_min_x = bbox.min_vertex[0] + translate_x
                shifted_min_y = bbox.min_vertex[1] + translate_y

                # Check boundary conditions for the shifted_min bounding box coordinates
                if shifted_min_x < 0:
                    shifted_min_x = 0
                elif shifted_min_x > bbox.max_vertex[0]:
                    shifted_min_x = bbox.max_vertex[0]
                if shifted_min_y < 0:
                    shifted_min_y = 0
                elif shifted_min_y > bbox.max_vertex[1]:
                    shifted_min_y = bbox.max_vertex[1]

                shifted_min = (shifted_min_x, shifted_min_y)

                # Compute the shifted_max coords for the bounding box to align with
                # the translated max_vertex coordinates
                shifted_max_x = bbox.max_vertex[0] + translate_x
                shifted_max_y = bbox.max_vertex[1] + translate_y

                # Assign boundary conditions for the shifted_max bounding box coordinates
                if shifted_max_x < 0:
                    shifted_max_x = 0
                elif shifted_max_x > bbox.max_vertex[0]:
                    shifted_max_x = bbox.max_vertex[0]
                if shifted_max_y < 0:
                    shifted_max_y = 0
                elif shifted_max_y > bbox.max_vertex[1]:
                    shifted_max_y = bbox.max_vertex[1]

                shifted_max = (shifted_max_x, shifted_max_y)

                # Apply the shifted coordinates to the output bounding box
                adjusted_box = AxisAlignedBoundingBox(shifted_min, shifted_max)
                adjusted_bboxes.append((adjusted_box, metadata))
        return final_image, adjusted_bboxes

    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the RandomTranslationPerturber instance.

        Returns:
            :return dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = super().get_config()
        cfg["seed"] = self.rng
        cfg["color_fill"] = self.color_fill
        return cfg
