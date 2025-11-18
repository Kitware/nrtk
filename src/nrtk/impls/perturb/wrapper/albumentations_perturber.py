"""Defines AlbumentationsPerturber to apply Albumentations BasicTransforms to input images.

Classes:
    AlbumentationsPerturber: A perturbation class for applying perturbations from Albumentations

Dependencies:
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for perturbation algorithms.
    - albumentations: For the underlying perturbations
"""

from __future__ import annotations

__all__ = ["AlbumentationsPerturber"]

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import AlbumentationsImportError, OpenCVImportError
from nrtk.utils._import_guard import import_guard

albumentations_available: bool = import_guard("albumentations", AlbumentationsImportError, ["core.bbox_utils"])
cv2_available: bool = import_guard("cv2", OpenCVImportError)

import albumentations as A  # noqa N812, F401
from albumentations.core.bbox_utils import convert_bboxes_from_albumentations, convert_bboxes_to_albumentations  # noqa E402


class AlbumentationsPerturber(PerturbImage):
    """AlbumentationsPerturber applies a BasicTransform from Albumentations.

    Attributes:
        perturber (string):
            The name of the BasicTransform perturber to apply.
        parameters (dict):
            Keyword arguments that should be passed to the given perturber.
        seed (int):
            An optional seed for reproducible results.

    Methods:
        perturb:
            Applies the specified to an input image.
        __call__:
            Calls the perturb method with the given input image.
        get_config:
            Returns the current configuration of the AlbumentationsPerturber instance.
    """

    def __init__(
        self,
        perturber: str = "NoOp",
        parameters: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """AlbumentationsPerturber applies a BasicTransform from Albumentations.

        Args:
            perturber:
                The name of the BasicTransform perturber to apply.
                Will apply a NoOp if not provided.
            parameters:
                Keyword arguments that should be passed to the given perturber.
            seed (int):
                An optional seed for reproducible results.


        Raises:
            :raises ValueError: Given perturber is not available in Albumentations.
            :raises ValueError: Given perturber does not inherit BasicTransform.
        """
        if not self.is_usable():
            raise AlbumentationsImportError

        super().__init__()
        self.perturber = perturber
        self.parameters = parameters

        if not hasattr(A, self.perturber):
            raise ValueError(f'Given perturber "{self.perturber}" is not available in Albumentations')

        transformer = getattr(A, self.perturber)

        if not issubclass(transformer, A.BasicTransform):
            raise ValueError(f'Given perturber "{self.perturber}" does not inherit "BasicTransform"')

        self.transform: A.Compose = A.Compose(
            [transformer(**self.parameters) if self.parameters else transformer()],
        )

        self.seed = seed
        if seed:
            self.transform.set_random_seed(seed)

    @staticmethod
    def _aabb_to_bbox(box: AxisAlignedBoundingBox, image: np.ndarray[Any, Any]) -> list[int]:
        """Convert AxisAlignedBoundingBox to albumentations format bbox."""
        flat = np.array([[box.min_vertex[0], box.min_vertex[1], box.max_vertex[0], box.max_vertex[1]]])
        return convert_bboxes_to_albumentations(
            flat,
            "pascal_voc",
            {"height": image.shape[0], "width": image.shape[1]},
        )[0]

    @staticmethod
    def _bbox_to_aabb(box: list[int], image: np.ndarray[Any, Any]) -> AxisAlignedBoundingBox:
        """Convert albumentations format bbox to AxisAlignedBoundingBox."""
        flat = np.array([[box[0], box[1], box[2], box[3]]])
        as_aabb = convert_bboxes_from_albumentations(
            flat,
            "pascal_voc",
            {"height": image.shape[0], "width": image.shape[1]},
        )[0]
        return AxisAlignedBoundingBox((as_aabb[0], as_aabb[1]), (as_aabb[2], as_aabb[3]))

    @override
    def perturb(
        self,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **additional_params: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Apply a BasicTransform from Albumentations to an image.

        Args:
            image:
                Input image as a numpy array of shape (H, W, C).
            boxes:
                List of bounding boxes in AxisAlignedBoundingBox format and their corresponding classes.
            additional_params:
                Additional perturbation keyword arguments.

        Returns:
            :return tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
                Image with transform applied and bounding boxes with their coordinates adjusted by the transform.
        """
        # Create bboxes and labels in a format usable by Albumentations
        bboxes = list()
        labels = list()
        if boxes:
            for box in boxes:
                bboxes.append(AlbumentationsPerturber._aabb_to_bbox(box[0], image))
                labels.append(box[1])

        # Run transform
        output = self.transform(
            image=image,
            bboxes=np.array(bboxes),
        )
        output_image = output["image"].astype(np.uint8)

        # Create output_bboxes in the format expected by PerturbImage output
        output_boxes = None
        if boxes:
            output_boxes = [
                (AlbumentationsPerturber._bbox_to_aabb(bbox, output_image), label)
                for bbox, label in zip(output["bboxes"], labels, strict=False)
            ]
        return output_image, output_boxes

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the AlbumentationsPerturber instance.

        Returns:
            :return dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = super().get_config()
        cfg["perturber"] = self.perturber
        cfg["parameters"] = self.parameters
        cfg["seed"] = self.seed
        return cfg

    @classmethod
    @override
    def is_usable(cls) -> bool:
        """Checks if the required albumentations module is available.

        Returns:
            :return bool: True if albumentations is installed; False otherwise.
        """
        # Requires opencv to be installed
        return albumentations_available and cv2_available
