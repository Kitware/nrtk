"""
This module provides an example implementation of the GenerateObjectDetectorBlackboxResponse interface
using a `SimplePybsmGenerator` class. It generates object detection responses for a series of input images
based on ground truth bounding boxes and scoring methods.

Classes:
    SimplePybsmGenerator: Implements the `GenerateObjectDetectorBlackboxResponse` interface for generating
    and scoring object detection responses with configurable image perturbation.

Dependencies:
    - smqtk_detection
    - smqtk_image_io
    - nrtk.interfaces.gen_object_detector_blackbox_response
    - nrtk.interfaces.perturb_image_factory
    - nrtk.interfaces.score_detections

Example usage:
    generator = SimplePybsmGenerator(images, img_gsds, ground_truth)
    response = generator.generate_response()

"""

from collections.abc import Hashable, Sequence
from importlib.util import find_spec
from typing import Any

import numpy as np
from smqtk_detection import DetectImageObjects
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.gen_object_detector_blackbox_response import (
    GenerateObjectDetectorBlackboxResponse,
)
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interfaces.score_detections import ScoreDetections


class SimplePybsmGenerator(GenerateObjectDetectorBlackboxResponse):
    """
    Example implementation of the `GenerateObjectDetectorBlackboxResponse` interface.

    This class generates detection response data for a sequence of images, using ground truth bounding
    boxes and configurable scoring and perturbation methods.

    Attributes:
        images (Sequence[np.ndarray]): Sequence of images to process.
        img_gsds (Sequence[float]): Ground sample distances for each image.
        ground_truth (Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]):
            Ground truth bounding boxes with associated labels and scores.

    Methods:
        generate_response(): Generates a response for each image based on ground truth and scoring.
    """

    def __init__(
        self,
        images: Sequence[np.ndarray],
        img_gsds: Sequence[float],
        ground_truth: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
    ) -> None:
        """
        Initializes the SimplePybsmGenerator with input images, ground sample distances, and ground truth.

        Args:
            images (Sequence[np.ndarray]): Sequence of images for detection.
            img_gsds (Sequence[float]): Ground sample distances (GSD) for each image.
            ground_truth (Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]):
                Ground truth data, containing bounding boxes and associated labels and scores.

        Raises:
            ImportError: If pybsm library is not available.
            ValueError: If `images` and `ground_truth` do not have the same length.
        """
        if not self.is_usable():
            raise ImportError(
                "pybsm not found. Please install 'nrtk[pybsm]', 'nrtk[pybsm-graphics]', or 'nrtk[pybsm-headless]'.",
            )
        if len(images) != len(ground_truth):
            raise ValueError("Size mismatch. ground_truth must be provided for each image.")
        if len(images) != len(img_gsds):
            raise ValueError("Size mismatch. imggsd must be provided for each image.")
        self.images = images
        self.img_gsds = img_gsds
        self.ground_truth = ground_truth

    @override
    def __len__(self) -> int:
        """:return: Number of image/ground_truth pairs this generator holds."""
        return len(self.images)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        np.ndarray,
        Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
        dict[str, Any],
    ]:
        """
        Get the image and ground truth pair for a specific index.

        Args:
            idx (int): Index of the desired data pair.

        Raises:
            IndexError: If the given index is out of range.

        Returns:
            tuple: A tuple containing the image, ground truth data, and metadata for the specified index.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError
        return self.images[idx], self.ground_truth[idx], {"img_gsd": self.img_gsds[idx]}

    def get_config(self) -> dict[str, Any]:
        """
        Generates a serializable configuration for the instance.

        Returns:
            dict[str, Any]: Configuration dictionary containing instance parameters.
        """
        return {
            "images": self.images,
            "img_gsds": self.img_gsds,
            "ground_truth": self.ground_truth,
        }

    @override
    def generate(
        self,
        blackbox_perturber_factories: Sequence[PerturbImageFactory],
        blackbox_detector: DetectImageObjects,
        blackbox_scorer: ScoreDetections,
        img_batch_size: int,
        verbose: bool = False,
    ) -> tuple[Sequence[tuple[dict[str, Any], float]], Sequence[Sequence[float]]]:
        inter = super().generate(
            blackbox_perturber_factories=blackbox_perturber_factories,
            blackbox_detector=blackbox_detector,
            blackbox_scorer=blackbox_scorer,
            img_batch_size=img_batch_size,
            verbose=verbose,
        )
        """
        Generates detection responses for the sequence of images using perturbation and scoring.

        Args:
            blackbox_perturber_factories (Sequence[PerturbImageFactory]): Factories to perturb images.
            blackbox_detector (DetectImageObjects): Object detection model.
            blackbox_scorer (ScoreDetections): Scoring function for detection evaluation.
            img_batch_size (int): Number of images to process in each batch.
            verbose (bool, optional): If True, prints verbose output.

        Returns:
            tuple: Contains two elements:
                - Sequence of tuples with metadata and score for each perturbation.
                - Sequence of scores for each perturbation level.
        """
        master_key = blackbox_perturber_factories[0].theta_key
        new_curve = [(entry[0][master_key], entry[1]) for entry in inter[0]]

        return new_curve, inter[1]

    @classmethod
    def is_usable(cls) -> bool:
        """
        Checks if the required pybsm module is available.

        Returns:
            bool: True if pybsm is installed; False otherwise.
        """
        # Requires nrtk[pybsm], nrtk[pybsm-graphics], or nrtk[pybsm-headless]
        # we don't need to check for opencv because this can run with
        # a non-opencv pybsm based perturber
        return find_spec("pybsm") is not None
