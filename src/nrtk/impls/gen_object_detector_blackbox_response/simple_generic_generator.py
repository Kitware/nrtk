"""
This module defines the `SimpleGenericGenerator` class, an implementation of the
`GenerateObjectDetectorBlackboxResponse` interface for generating item-response curves
based on a set of images and ground-truth bounding box annotations.

Classes:
    SimpleGenericGenerator: An example implementation that generates object detection
    responses for a sequence of images and their associated ground-truth data.

Dependencies:
    - numpy for handling image data.
    - smqtk_image_io for bounding box operations.
    - nrtk.interfaces for blackbox response generation interface.

Example usage:
    images = [image1, image2, ...]
    ground_truth = [
        [(bbox1, {label1: score1}), (bbox2, {label2: score2})],
        ...
    ]
    generator = SimpleGenericGenerator(images, ground_truth)
    image, ground_truth, metadata = generator[0]
"""

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.gen_object_detector_blackbox_response import (
    GenerateObjectDetectorBlackboxResponse,
)


class SimpleGenericGenerator(GenerateObjectDetectorBlackboxResponse):
    """Example implementation of the ``GenerateObjectDetectorBlackboxResponse`` interface."""

    def __init__(
        self,
        images: Sequence[np.ndarray],
        ground_truth: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
    ) -> None:
        """Generate response curve for given images and ground_truth.

        :param images: Sequence of images to generate responses for.
        :param ground_truth: Sequence of sequences of detections corresponsing to each image.

        :raises ValueError: Images and ground_truth data have a size mismatch.
        """
        if len(images) != len(ground_truth):
            raise ValueError(
                "Size mismatch. ground_truth must be provided for each image.",
            )
        self.images = images
        self.ground_truth = ground_truth

    @override
    def __len__(self) -> int:
        """:return: Number of image/ground_truth pairs this generator holds."""
        return len(self.images)

    @override
    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        np.ndarray,
        Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
        dict[str, Any],
    ]:
        """Get the image and ground_truth pair for a specific index.

        :param idx: Index of desired data pair.

        :raises IndexError: The given index does not exist.

        :return: Data pair corresponding to the given index.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError
        return self.images[idx], self.ground_truth[idx], {}

    def get_config(self) -> dict[str, Any]:
        """
        Generates a serializable configuration for the instance.

        Returns:
            dict[str, Any]: Configuration dictionary containing instance parameters.
        """
        return {"images": self.images, "ground_truth": self.ground_truth}
