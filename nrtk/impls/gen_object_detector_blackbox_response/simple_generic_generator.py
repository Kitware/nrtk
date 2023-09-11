import numpy as np
from typing import Any, Dict, Hashable, Sequence, Tuple

from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.gen_object_detector_blackbox_response import GenerateObjectDetectorBlackboxResponse


class SimpleGenericGenerator(GenerateObjectDetectorBlackboxResponse):
    """
    Example implementation of the ``GenerateObjectDetectorBlackboxResponse``
    interface.
    """

    def __init__(
        self,
        images: Sequence[np.ndarray],
        groundtruth: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]
    ):
        """
        Generate response curve for given images and groundtruth.

        :param images: Sequence of images to generate responses for.
        :param groundtruth: Sequence of sequences of detections corresponsing to each image.

        :raises ValueError: Images and groundtruth data have a size mismatch.
        """
        if len(images) != len(groundtruth):
            raise ValueError("Size mismatch. Groundtruth must be provided for each image.")
        self.images = images
        self.groundtruth = groundtruth

    def __len__(self) -> int:
        """
        :return: Number of image/groundtruth pairs this generator holds.
        """
        return len(self.images)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[np.ndarray, Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]], Dict[str, Any]]:
        """
        Get the image and groundtruth pair for a specific index.

        :param idx: Index of desired data pair.

        :raises IndexError: The given index does not exist.

        :return: Data pair corresponding to the given index.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError
        return self.images[idx], self.groundtruth[idx], {}

    def get_config(self) -> Dict[str, Any]:
        return {
            "images": self.images,
            "groundtruth": self.groundtruth
        }
