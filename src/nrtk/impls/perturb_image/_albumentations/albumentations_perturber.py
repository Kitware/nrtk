"""Defines AlbumentationsPerturber to apply Albumentations BasicTransforms to input images.

Classes:
    AlbumentationsPerturber: A perturbation class for applying perturbations from Albumentations

Dependencies:
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for perturbation algorithms.
    - albumentations: For the underlying perturbations

Example usage:
    >>> image = np.ones((256, 256, 3))
    >>> perturber = AlbumentationsPerturber(perturber="RandomRain", parameters={"drop_length": 40, "drop_width": 10})
    >>> perturbed_image, _ = perturber(image=image)
"""

from __future__ import annotations

__all__ = ["AlbumentationsPerturber"]

from collections.abc import Hashable, Iterable
from typing import Any

import albumentations
import numpy as np
from albumentations.core.bbox_utils import convert_bboxes_from_albumentations, convert_bboxes_to_albumentations
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces._random_perturb_image import RandomPerturbImage


class AlbumentationsPerturber(RandomPerturbImage):
    """AlbumentationsPerturber applies a BasicTransform from Albumentations.

    Attributes:
        _perturber (string):
            The name of the BasicTransform perturber to apply.
        _parameters (dict):
            Keyword arguments that should be passed to the given perturber.
        seed (int | None):
            Random seed for reproducibility. None for non-deterministic behavior.
        is_static (bool):
            If True, resets seed after each call for consistent results.
        _transform (albumentations.Compose):
            albumentations pipeline for the transformation
    """

    def __init__(
        self,
        *,
        perturber: str = "NoOp",
        parameters: dict[str, Any] | None = None,
        seed: int | None = None,
        is_static: bool = False,
    ) -> None:
        """AlbumentationsPerturber applies a BasicTransform from Albumentations.

        Args:
            perturber:
                The name of the BasicTransform perturber to apply.
                Will apply a NoOp if not provided.
            parameters:
                Keyword arguments that should be passed to the given perturber.
            seed:
                Random seed for reproducible results. Defaults to None for non-deterministic
                behavior.
            is_static:
                If True and seed is provided, resets seed after each perturb call for consistent
                results across multiple calls (useful for video frame processing).


        Raises:
            ValueError:
                Given perturber is not available in Albumentations.
            ValueError:
                Given perturber does not inherit BasicTransform.
        """
        self._perturber = perturber
        self._parameters = parameters

        if not hasattr(albumentations, self._perturber):
            raise ValueError(f'Given perturber "{self._perturber}" is not available in Albumentations')

        transformer = getattr(albumentations, self._perturber)

        if not issubclass(transformer, albumentations.BasicTransform):
            raise ValueError(f'Given perturber "{self._perturber}" does not inherit "BasicTransform"')

        self._transform: albumentations.Compose = albumentations.Compose(
            [transformer(**self._parameters) if self._parameters else transformer()],
        )

        # super().__init__ is called last so that all attributes exist when
        # _set_seed() is invoked during RandomPerturbImage.__init__.
        super().__init__(seed=seed, is_static=is_static)

    @override
    def _set_seed(self) -> None:
        """Set seed on Albumentations transform."""
        if self._seed is not None:
            self._transform.set_random_seed(self._seed)

    @staticmethod
    def _aabb_to_bbox(*, box: AxisAlignedBoundingBox, image: np.ndarray[Any, Any]) -> list[int]:
        """Convert AxisAlignedBoundingBox to albumentations format bbox."""
        flat = np.array([[box.min_vertex[0], box.min_vertex[1], box.max_vertex[0], box.max_vertex[1]]])
        return convert_bboxes_to_albumentations(
            bboxes=flat,
            source_format="pascal_voc",
            shape={"height": image.shape[0], "width": image.shape[1]},
        )[0]

    @staticmethod
    def _bbox_to_aabb(*, box: list[int], image: np.ndarray[Any, Any]) -> AxisAlignedBoundingBox:
        """Convert albumentations format bbox to AxisAlignedBoundingBox."""
        flat = np.array([[box[0], box[1], box[2], box[3]]])
        as_aabb = convert_bboxes_from_albumentations(
            bboxes=flat,
            target_format="pascal_voc",
            shape={"height": image.shape[0], "width": image.shape[1]},
        )[0]
        return AxisAlignedBoundingBox(min_vertex=(as_aabb[0], as_aabb[1]), max_vertex=(as_aabb[2], as_aabb[3]))

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Apply a BasicTransform from Albumentations to an image.

        Args:
            image:
                Input image as a numpy array of shape (H, W, C).
            boxes:
                List of bounding boxes in AxisAlignedBoundingBox format and their corresponding classes.
            kwargs:
                Additional perturbation keyword arguments.

        Returns:
            Image with transform applied and bounding boxes with their coordinates adjusted by the transform.
        """
        perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes, **kwargs)

        # Create bboxes and labels in a format usable by Albumentations
        bboxes = []
        labels = []
        if perturbed_boxes:
            for box in perturbed_boxes:
                bboxes.append(AlbumentationsPerturber._aabb_to_bbox(box=box[0], image=perturbed_image))
                labels.append(box[1])

        # Run transform
        output = self._transform(
            image=perturbed_image,
            bboxes=np.array(bboxes),
        )
        perturbed_image = output["image"].astype(np.uint8)

        # Create output_bboxes in the format expected by PerturbImage output
        if perturbed_boxes:
            perturbed_boxes = [
                (AlbumentationsPerturber._bbox_to_aabb(box=bbox, image=perturbed_image), label)
                for bbox, label in zip(output["bboxes"], labels, strict=False)
            ]
        else:
            perturbed_boxes = None

        return perturbed_image, perturbed_boxes

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the AlbumentationsPerturber instance."""
        cfg = super().get_config()
        cfg["perturber"] = self._perturber
        cfg["parameters"] = self._parameters
        return cfg
