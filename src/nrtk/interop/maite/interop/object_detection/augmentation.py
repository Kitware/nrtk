"""This module contains wrappers for NRTK perturbers for object detection."""

from __future__ import annotations

__all__ = ["JATICDetectionAugmentation", "JATICDetectionAugmentationWithMetric"]

import copy
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.image_metric import ImageMetric
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interop.maite.interop.generic import NRTKDatumMetadata, _forward_md_keys
from nrtk.interop.maite.interop.object_detection.dataset import JATICDetectionTarget
from nrtk.utils._exceptions import MaiteImportError
from nrtk.utils._import_guard import import_guard

maite_available: bool = import_guard("maite", MaiteImportError, ["protocols.object_detection"], ["Augmentation"])

from maite.protocols import AugmentationMetadata  # noqa: E402
from maite.protocols.object_detection import (  # noqa: E402
    Augmentation,
    DatumMetadataType,
    InputType,
    TargetType,
)

OBJ_DETECTION_BATCH_T = tuple[Sequence[InputType], Sequence[TargetType], Sequence[DatumMetadataType]]


class JATICDetectionAugmentation(Augmentation):  # pyright: ignore [reportGeneralTypeIssues]
    """Implementation of JATIC Augmentation for NRTK perturbers.

    Implementation of JATIC Augmentation for NRTK perturbers
    operating on a MAITE-protocol compliant Object Detection dataset.

    Given a set of ground truth labels alongside an image and image
    metadata, JATICDetectionAugmentation will properly scale the
    labels in accordance with the change in the image scale due to
    applied pertubation. At this time JATICDetectionAugmentation does
    not support any other augmentation to the labels such as cropping,
    translation, or rotation.

    Parameters
    ----------
    augment : PerturbImage
        Augmentations to apply to an image.
    metadata: AugmentationMetadata
        Metadata for this augmentation.
    """

    def __init__(self, augment: PerturbImage, augment_id: str) -> None:
        """Initialize augmentation wrapper."""
        if not self.is_usable():
            raise MaiteImportError
        self.augment = augment
        self.metadata: AugmentationMetadata = AugmentationMetadata(id=augment_id)

    def __call__(
        self,
        batch: OBJ_DETECTION_BATCH_T,
    ) -> tuple[Sequence[InputType], Sequence[TargetType], Sequence[NRTKDatumMetadata]]:  # pyright: ignore [reportInvalidTypeForm]
        """Apply augmentations to the given data batch."""
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
            img_bboxes = [AxisAlignedBoundingBox(bbox[0:2], bbox[2:4]) for bbox in np.array(img_anns.boxes)]  # pyright: ignore [reportAttributeAccessIssue]
            img_labels = [
                {label: score}
                for label, score in zip(np.array(img_anns.labels), np.array(img_anns.scores), strict=False)  # pyright: ignore [reportAttributeAccessIssue]
            ]

            aug_img, aug_img_anns = self.augment(
                np.asarray(aug_img),
                zip(img_bboxes, img_labels, strict=False),
                **dict(md),
            )
            if TYPE_CHECKING and not aug_img_anns:
                break
            aug_imgs.append(np.transpose(aug_img, (2, 0, 1)))

            # re-format annotations to JATICDetectionTarget for returning
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
                JATICDetectionTarget(
                    aug_img_bboxes_arr,
                    np.array(aug_img_labels),
                    np.array(aug_img_scores),
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

            aug_metadata.append(_forward_md_keys(md, aug_md, forwarded_keys=["id", "nrtk_perturber_config"]))

        # return batch of augmented inputs, resized bounding boxes and updated metadata
        return aug_imgs, aug_dets, aug_metadata

    @classmethod
    def is_usable(cls) -> bool:
        """Checks if the necessary dependency (MAITE) is available.

        Returns:
            bool: True MAITE is available; False otherwise.
        """
        return maite_available


class JATICDetectionAugmentationWithMetric(Augmentation):  # pyright: ignore [reportGeneralTypeIssues]
    """Implementation of JATIC augmentation wrapper for NRTK's Image metrics.

    Implementation of JATIC augmentation for NRTK metrics operating on a MAITE-protocol
    compliant object detection dataset.

    Parameters
    ----------
    augmentations : Sequence[Augmentation] | None
        Optional task-specific sequence of JATIC augmentations to be applied on a given batch.
    metric : ImageMetric
        Image metric to be applied for a given image.
    metadata: AugmentationMetadata
        Metadata for this augmentation.
    """

    def __init__(
        self,
        augmentations: Sequence[Augmentation] | None,  # pyright: ignore [reportInvalidTypeForm]
        metric: ImageMetric,
        augment_id: str,
    ) -> None:
        """Initialize augmentation with metric wrapper."""
        if not self.is_usable():
            raise MaiteImportError
        self.augmentations = augmentations
        self.metric = metric
        self.metadata: AugmentationMetadata = AugmentationMetadata(id=augment_id)

    def _apply_augmentations(
        self,
        batch: OBJ_DETECTION_BATCH_T,
    ) -> tuple[Sequence[InputType] | Sequence[None], Sequence[TargetType], Sequence[DatumMetadataType]]:  # pyright: ignore [reportInvalidTypeForm]
        """Apply augmentations to given batch."""
        if self.augmentations:
            aug_batch = batch
            for aug in self.augmentations:
                aug_batch = aug(aug_batch)
        else:
            imgs, anns, metadata = batch
            aug_batch = [None] * len(imgs), anns, metadata  # pyright: ignore [reportArgumentType]

        return aug_batch

    def __call__(
        self,
        batch: OBJ_DETECTION_BATCH_T,
    ) -> tuple[Sequence[InputType], Sequence[TargetType], Sequence[NRTKDatumMetadata]]:  # pyright: ignore [reportInvalidTypeForm]
        """Compute a specified image metric on the given batch."""
        imgs, _, _ = batch
        metric_aug_metadata = list()  # list of individual image-level metric metadata

        aug_imgs, aug_dets, aug_metadata = self._apply_augmentations(batch)

        for img, aug_img, aug_md in zip(imgs, aug_imgs, aug_metadata, strict=False):  # pyright: ignore [reportArgumentType]
            # Convert from channels-first to channels-last
            img_1 = img
            img_2 = None if aug_img is None else aug_img

            # Compute Image metric values
            metric_value = self.metric(img_1=img_1, img_2=img_2, additional_params=aug_md)  # type: ignore
            metric_name = self.metric.__class__.__name__

            existing_metrics = list()
            if "nrtk_metric" in aug_md:
                existing_metrics = list(aug_md["nrtk_metric"])
            existing_metrics.append((metric_name, metric_value))
            metric_aug_md = NRTKDatumMetadata(
                id=aug_md["id"],
                nrtk_metric=existing_metrics,
            )

            metric_aug_metadata.append(
                _forward_md_keys(aug_md, metric_aug_md, forwarded_keys=["id", "nrtk_metric"]),
            )

        # return batch of augmented/original images, detections and metric-updated metadata
        if self.augmentations:
            # type ignore was included to handle the dual Sequence[ArrrayLike] | List[None]
            # case for the augmented images.
            return aug_imgs, aug_dets, metric_aug_metadata  # type: ignore
        return imgs, aug_dets, metric_aug_metadata

    @classmethod
    def is_usable(cls) -> bool:
        """Checks if the necessary dependency (MAITE) is available.

        Returns:
            bool: True MAITE is available; False otherwise.
        """
        return maite_available
