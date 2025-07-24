"""Defines RandomCropPerturber for random image crops with bounding box adjustment for labeled datasets.

Classes:
    RandomCropPerturber: A perturbation class for randomly cropping images and modifying bounding boxes.

Dependencies:
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for perturbation algorithms.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image import PerturbImage


class RandomCropPerturber(PerturbImage):
    """RandomCropPerturber randomly crops an image and adjusts bounding boxes accordingly.

    Attributes:
        crop_size (tuple[int, int]): Target crop dimensions for the input image.
        seed (int | numpy.random.Generator): Random seed or number generator for deterministic behavior.

    Methods:
        perturb:
            Applies a random crop to an input image and adjusts bounding boxes.
        __call__:
            Calls the perturb method with the given input image.
        get_config:
            Returns the current configuration of the RandomCropPerturber instance.
    """

    def __init__(
        self,
        crop_size: tuple[int, int] | None = None,
        seed: int | np.random.Generator | None = 1,
        box_alignment_mode: str | None = None,
    ) -> None:
        """RandomCropPerturber applies a random cropping perturbation to an input image.

        It ensures that bounding boxes are adjusted correctly to reflect the new cropped region.

        Args:
            crop_size:
                Target crop size as (crop_height, crop_width).
            seed:
                Seed for rng.
            box_alignment_mode:
                Deprecated Mode for how to handle how bounding boxes change.
        """
        super().__init__(box_alignment_mode=box_alignment_mode)
        self.crop_size = crop_size
        self.seed = seed
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

    def perturb(
        self,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Randomly crops an image and adjusts bounding boxes.

        Args:
            image:
                Input image as a numpy array of shape (H, W, C).
            boxes:
                List of bounding boxes in AxisAlignedBoundingBox format and their corresponding classes.
            additional_params:
                Unused

        Returns:
            :return tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
                Cropped image with the modified bounding boxes.
        """
        image, boxes = super().perturb(image=image, boxes=boxes)

        if additional_params is None:
            additional_params = dict()

        # Set crop_size to half of image size if crop_size is None
        crop_size = self.crop_size if self.crop_size is not None else (image.shape[0] // 2, image.shape[1] // 2)

        crop_h, crop_w = crop_size
        orig_h, orig_w = image.shape[:2]

        # Ensure crop size is smaller than image dimensions
        crop_h = min(crop_h, orig_h)
        crop_w = min(crop_w, orig_w)

        # Randomly select the top-left corner of the crop
        crop_x = self.rng.integers(0, orig_w - crop_w)
        crop_y = self.rng.integers(0, orig_h - crop_h)

        # Perform the crop
        cropped_image = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].copy()
        # Adjust bounding boxes
        adjusted_bboxes = []
        if boxes is not None:
            for bbox, metadata in boxes:
                # Calculate intersection of the bounding box with the crop region
                crop_box = AxisAlignedBoundingBox((crop_y, crop_x), (crop_y + crop_h, crop_x + crop_w))
                intersected_box = bbox.intersection(crop_box)
                if intersected_box:
                    # Shift the intersected bounding box to align with the cropped image coordinates
                    shifted_min = (
                        intersected_box.min_vertex[0] - crop_y,
                        intersected_box.min_vertex[1] - crop_x,
                    )
                    shifted_max = (
                        intersected_box.max_vertex[0] - crop_y,
                        intersected_box.max_vertex[1] - crop_x,
                    )
                    adjusted_box = AxisAlignedBoundingBox(shifted_min, shifted_max)
                    adjusted_bboxes.append((adjusted_box, metadata))
        return cropped_image, adjusted_bboxes

    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the RandomCropPerturber instance.

        Returns:
            :return dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = super().get_config()
        cfg["seed"] = self.seed
        cfg["crop_size"] = self.crop_size
        return cfg
