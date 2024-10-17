from typing import Any, Dict, Hashable, Sequence, Tuple

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.gen_object_detector_blackbox_response import (
    GenerateObjectDetectorBlackboxResponse,
)


class SimpleGenericGenerator(GenerateObjectDetectorBlackboxResponse):
    """Example implementation of the ``GenerateObjectDetectorBlackboxResponse`` interface."""

    def __init__(
        self,
        images: Sequence[np.ndarray],
        ground_truth: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
    ):
        """Generate response curve for given images and ground_truth.

        :param images: Sequence of images to generate responses for.
        :param ground_truth: Sequence of sequences of detections corresponsing to each image.

        :raises ValueError: Images and ground_truth data have a size mismatch.
        """
        if len(images) != len(ground_truth):
            raise ValueError("Size mismatch. ground_truth must be provided for each image.")
        self.images = images
        self.ground_truth = ground_truth

    def __len__(self) -> int:
        """:return: Number of image/ground_truth pairs this generator holds."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[
        np.ndarray,
        Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]],
        Dict[str, Any],
    ]:
        """Get the image and ground_truth pair for a specific index.

        :param idx: Index of desired data pair.

        :raises IndexError: The given index does not exist.

        :return: Data pair corresponding to the given index.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError
        return self.images[idx], self.ground_truth[idx], {}

    def get_config(self) -> Dict[str, Any]:
        return {"images": self.images, "ground_truth": self.ground_truth}
