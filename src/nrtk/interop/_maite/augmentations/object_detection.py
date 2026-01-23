"""This module contains wrappers for NRTK perturbers for object detection."""

from __future__ import annotations

__all__ = ["MAITEDetectionAugmentation"]

import copy
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interop._maite.datasets.object_detection import MAITEDetectionTarget
from nrtk.interop._maite.metadata.datum import NRTKDatumMetadata, _forward_md_keys
from nrtk.utils._exceptions import MaiteImportError
from nrtk.utils._import_guard import import_guard

maite_available: bool = import_guard(
    module_name="maite",
    exception=MaiteImportError,
    submodules=["protocols.object_detection"],
    objects=["Augmentation"],
)

from maite.protocols import AugmentationMetadata  # noqa: E402
from maite.protocols.object_detection import (  # noqa: E402
    Augmentation,
    DatumMetadataType,
    InputType,
    TargetType,
)

OBJ_DETECTION_BATCH_T = tuple[Sequence[InputType], Sequence[TargetType], Sequence[DatumMetadataType]]


class MAITEDetectionAugmentation(Augmentation):  # pyright: ignore [reportGeneralTypeIssues]
    """Implementation of MAITE Object Detection Augmentation for NRTK perturbers.

    Implementation of MAITE Augmentation for NRTK perturbers
    operating on a MAITE-protocol compliant Object Detection dataset.

    Given a set of ground truth labels alongside an image and image
    metadata, MAITEDetectionAugmentation will properly scale the
    labels in accordance with the change in the image scale due to
    applied pertubation. At this time MAITEDetectionAugmentation does
    not support any other augmentation to the labels such as cropping,
    translation, or rotation.

    Attributes:
        augmentations : Sequence[Augmentation] | None
            Optional task-specific sequence of MAITE augmentations to be applied on a given batch.
        metadata: AugmentationMetadata
            Metadata for this augmentation.
    """

    def __init__(self, *, augment: PerturbImage, augment_id: str) -> None:
        """Initialize augmentation wrapper.

        Args:
            augment:
                PerturbImage implementation to perform.
            augment_id:
                Metadata ID for this augmentation.
        """
        if not self.is_usable():
            raise MaiteImportError
        self.augment = augment
        self.metadata: AugmentationMetadata = AugmentationMetadata(id=augment_id)

    def __call__(
        self,
        batch: OBJ_DETECTION_BATCH_T,
    ) -> OBJ_DETECTION_BATCH_T:
        """Return a batch of augmented images and metadata."""
        imgs, anns, metadata = batch

        # iterate over (parallel) elements in batch
        aug_imgs = list()  # list of individual augmented inputs
        aug_dets = list()  # list of individual object detection targets
        aug_metadata = list()  # list of individual image-level metadata

        for img, img_anns, md in zip(imgs, anns, metadata, strict=False):  # pyright: ignore [reportArgumentType]
            # Perform augmentation
            aug_img = np.asarray(copy.deepcopy(img))
            aug_img = np.transpose(aug_img, (1, 2, 0))

            # format annotations for passing to perturber
            img_bboxes = [
                AxisAlignedBoundingBox(min_vertex=bbox[0:2], max_vertex=bbox[2:4]) for bbox in np.array(img_anns.boxes)
            ]  # pyright: ignore [reportAttributeAccessIssue]
            img_labels = [
                {label: score}
                for label, score in zip(np.array(img_anns.labels), np.array(img_anns.scores), strict=False)  # pyright: ignore [reportAttributeAccessIssue]
            ]

            aug_img, aug_img_anns = self.augment(
                image=np.asarray(aug_img),
                boxes=zip(img_bboxes, img_labels, strict=False),
                **dict(md),
            )
            if TYPE_CHECKING and not aug_img_anns:
                break
            aug_imgs.append(np.transpose(aug_img, (2, 0, 1)))

            # re-format annotations to MAITEDetectionTarget for returning
            aug_img_bboxes, aug_img_score_dicts = zip(*aug_img_anns, strict=False)
            aug_img_bboxes_arr = np.vstack([np.hstack((bbox.min_vertex, bbox.max_vertex)) for bbox in aug_img_bboxes])
            aug_img_labels, aug_img_scores = zip(
                *[
                    # get (label, score) pair for highest score
                    max(score_dict.items(), key=lambda x: x[1])
                    for score_dict in aug_img_score_dicts
                ],
                strict=False,
            )
            aug_dets.append(
                MAITEDetectionTarget(
                    boxes=aug_img_bboxes_arr,
                    labels=np.array(aug_img_labels),
                    scores=np.array(aug_img_scores),
                ),
            )

            perturber_configs = list()
            if "nrtk_perturber_config" in md:
                md_configs = md["nrtk_perturber_config"]
                if TYPE_CHECKING and not isinstance(md_configs, Iterable):  # pragma: no cover
                    raise RuntimeError("Expected iterable perturber config")
                perturber_configs = list(md_configs)
            perturber_configs.append(self.augment.get_config())
            aug_md = NRTKDatumMetadata(
                id=md["id"],
                nrtk_perturber_config=perturber_configs,
            )

            aug_metadata.append(_forward_md_keys(md=md, aug_md=aug_md, forwarded_keys=["id", "nrtk_perturber_config"]))

        # return batch of augmented inputs, resized bounding boxes and updated metadata
        return aug_imgs, aug_dets, aug_metadata

    @classmethod
    def is_usable(cls) -> bool:
        """Returns True if the necessary dependency (MAITE) is available."""
        return maite_available
