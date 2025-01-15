"""
This module defines the `RandomCropPerturber` class, which implements a random cropping perturbation
on input images. The class supports adjusting bounding boxes to match the cropped region,
making it suitable for tasks involving labeled datasets.

Classes:
    RandomCropPerturber: A perturbation class for randomly cropping images and modifying bounding boxes.

Dependencies:
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for perturbation algorithms.
"""

from collections.abc import Hashable, Iterable
from typing import Any, Optional, Union

import numpy as np
from smqtk_image_io import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image import PerturbImage


class RandomCropPerturber(PerturbImage):
    """
    RandomCropPerturber randomly crops an image and adjusts bounding boxes accordingly.
    Methods:
    perturb: Applies a random crop to an input image and adjusts bounding boxes.
    __call__: Calls the perturb method with the given input image.
    get_config: Returns the current configuration of the RandomCropPerturber instance.
    """

    def __init__(self, box_alignment_mode: str = "extent", seed: Optional[Union[int, np.random.Generator]] = 1) -> None:
        """
        RandomCropPerturber applies a random cropping perturbation to an input image.
        It ensures that bounding boxes are adjusted correctly to reflect the new cropped region.

        Attributes:
            rng (numpy.random.Generator): Random number generator for deterministic behavior.
        """
        super().__init__(box_alignment_mode=box_alignment_mode)
        self.rng = np.random.default_rng(seed)

    def perturb(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        """
        Randomly crops an image and adjusts bounding boxes.

        :param image: Input image as a numpy array of shape (H, W, C).
        :param boxes: List of bounding boxes in AxisAlignedBoundingBox format and their corresponding classes.

        :param additional_params: Dictionary containing:
            - "crop_size" (Tuple[int, int]): Crop size as (crop_height, crop_width).

        :return: Cropped image as numpy array with the modified bounding boxes
        """
        super().perturb(image=image)

        # Extract additional parameters
        crop_size = additional_params.get("crop_size", (image.shape[0] // 2, image.shape[1] // 2))

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
                        intersected_box.min_vertex[0] - crop_x,
                        intersected_box.min_vertex[1] - crop_y,
                    )
                    shifted_max = (
                        intersected_box.max_vertex[0] - crop_x,
                        intersected_box.max_vertex[1] - crop_y,
                    )
                    adjusted_box = AxisAlignedBoundingBox(shifted_min, shifted_max)
                    adjusted_bboxes.append((adjusted_box, metadata))
        return cropped_image, adjusted_bboxes

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
        Returns the current configuration of the _SPNoisePerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = super().get_config()
        cfg["seed"] = self.rng
        return cfg
