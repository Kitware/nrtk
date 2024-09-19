from importlib.util import find_spec
from typing import Any, Dict, Hashable, Sequence, Tuple

import numpy as np
from smqtk_detection import DetectImageObjects
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.gen_object_detector_blackbox_response import (
    GenerateObjectDetectorBlackboxResponse,
)
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interfaces.score_detections import ScoreDetections


class SimplePybsmGenerator(GenerateObjectDetectorBlackboxResponse):
    """Example implementation of the ``GenerateObjectDetectorBlackboxResponse`` interface."""

    def __init__(
        self,
        images: Sequence[np.ndarray],
        img_gsds: Sequence[float],
        ground_truth: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
    ):
        """Generate response curve for given images and ground_truth.

        :param images: Sequence of images to generate responses for.
        :param ground_truth: Sequence of sequences of detections corresponsing to each image.

        :raises ValueError: Images and ground_truth data have a size mismatch.
        """
        if len(images) != len(ground_truth):
            raise ValueError("Size mismatch. ground_truth must be provided for each image.")
        if len(images) != len(img_gsds):
            raise ValueError("Size mismatch. imggsd must be provided for each image.")
        self.images = images
        self.img_gsds = img_gsds
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
        return self.images[idx], self.ground_truth[idx], {"img_gsd": self.img_gsds[idx]}

    def get_config(self) -> Dict[str, Any]:
        return {
            "images": self.images,
            "img_gsds": self.img_gsds,
            "ground_truth": self.ground_truth,
        }

    def generate(
        self,
        blackbox_perturber_factories: Sequence[PerturbImageFactory],
        blackbox_detector: DetectImageObjects,
        blackbox_scorer: ScoreDetections,
        img_batch_size: int,
        verbose: bool = False,
    ) -> Tuple[Sequence[Tuple[Dict[str, Any], float]], Sequence[Sequence[float]]]:
        inter = super().generate(
            blackbox_perturber_factories=blackbox_perturber_factories,
            blackbox_detector=blackbox_detector,
            blackbox_scorer=blackbox_scorer,
            img_batch_size=img_batch_size,
            verbose=verbose,
        )

        master_key = blackbox_perturber_factories[0].theta_key
        new_curve = [(entry[0][master_key], entry[1]) for entry in inter[0]]

        return new_curve, inter[1]

    @classmethod
    def is_usable(cls) -> bool:
        # Requires pyBSM which requires opencv to be installed
        return find_spec("cv2") is not None
