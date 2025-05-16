"""This module contains wrappers for NRTK perturbers for object detection"""

import copy
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from maite.protocols import AugmentationMetadata
from maite.protocols.object_detection import (
    Augmentation,
    DatumMetadataBatchType,
    InputBatchType,
    TargetBatchType,
)
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.image_metric import ImageMetric
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interop.maite.interop.generic import NRTKDatumMetadata, _forward_md_keys
from nrtk.interop.maite.interop.object_detection.dataset import JATICDetectionTarget

OBJ_DETECTION_BATCH_T = tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]


class JATICDetectionAugmentation(Augmentation):
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
        """Initialize augmentation wrapper"""
        self.augment = augment
        self.metadata = AugmentationMetadata(id=augment_id)

    def __call__(
        self,
        batch: OBJ_DETECTION_BATCH_T,
    ) -> tuple[InputBatchType, TargetBatchType, Sequence[NRTKDatumMetadata]]:
        """Apply augmentations to the given data batch."""
        imgs, anns, metadata = batch

        # iterate over (parallel) elements in batch
        aug_imgs = list()  # list of individual augmented inputs
        aug_dets = list()  # list of individual object detection targets
        aug_metadata = list()  # list of individual image-level metadata

        for img, img_anns, md in zip(imgs, anns, metadata):
            # Perform augmentation
            aug_img = copy.deepcopy(img)
            aug_img = np.transpose(aug_img, (1, 2, 0))

            # format annotations for passing to perturber
            img_bboxes = [AxisAlignedBoundingBox(bbox[0:2], bbox[2:4]) for bbox in np.array(img_anns.boxes)]
            img_labels = [{label: score} for label, score in zip(np.array(img_anns.labels), np.array(img_anns.scores))]

            aug_img, aug_img_anns = self.augment(
                np.asarray(aug_img),
                zip(img_bboxes, img_labels),
                additional_params=dict(md),
            )
            aug_imgs.append(np.transpose(aug_img, (2, 0, 1)))

            # re-format annotations to JATICDetectionTarget for returning
            aug_img_bboxes, aug_img_score_dicts = zip(*aug_img_anns)
            aug_img_bboxes_arr = np.vstack([np.hstack((bbox.min_vertex, bbox.max_vertex)) for bbox in aug_img_bboxes])
            aug_img_labels, aug_img_scores = zip(
                *[
                    # get (label, score) pair for highest score
                    max(score_dict.items(), key=lambda x: x[1])
                    for score_dict in aug_img_score_dicts
                ],
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
                # TODO: Remove ignore after switch to pyright, mypy doesn't have good typed dict support  # noqa: FIX002
                perturber_configs = list(md["nrtk_perturber_config"])  # type: ignore
            perturber_configs.append(self.augment.get_config())
            aug_md = NRTKDatumMetadata(
                id=md["id"],
                nrtk_perturber_config=perturber_configs,
            )

            aug_metadata.append(_forward_md_keys(md, aug_md, forwarded_keys=["id", "nrtk_perturber_config"]))

        # return batch of augmented inputs, resized bounding boxes and updated metadata
        return aug_imgs, aug_dets, aug_metadata


class JATICDetectionAugmentationWithMetric(Augmentation):
    """Implementation of JATIC augmentation wrapper for NRTK's Image metrics.

    Implementation of JATIC augmentation for NRTK metrics operating on a MAITE-protocol
    compliant object detection dataset.

    Parameters
    ----------
    augmentations : Optional[Sequence[Augmentation]]
        Optional task-specific sequence of JATIC augmentations to be applied on a given batch.
    metric : ImageMetric
        Image metric to be applied for a given image.
    metadata: AugmentationMetadata
        Metadata for this augmentation.
    """

    def __init__(
        self,
        augmentations: Optional[Sequence[Augmentation]],
        metric: ImageMetric,
        augment_id: str,
    ) -> None:
        """Initialize augmentation with metric wrapper"""
        self.augmentations = augmentations
        self.metric = metric
        self.metadata = AugmentationMetadata(id=augment_id)

    def _apply_augmentations(
        self,
        batch: OBJ_DETECTION_BATCH_T,
    ) -> tuple[Union[InputBatchType, Sequence[None]], TargetBatchType, DatumMetadataBatchType]:
        """Apply augmentations to given batch"""

        if self.augmentations:
            aug_batch = batch
            for aug in self.augmentations:
                aug_batch = aug(aug_batch)
        else:
            imgs, anns, metadata = batch
            aug_batch = [None] * len(imgs), anns, metadata

        return aug_batch

    def __call__(
        self,
        batch: OBJ_DETECTION_BATCH_T,
    ) -> tuple[InputBatchType, TargetBatchType, Sequence[NRTKDatumMetadata]]:
        """Compute a specified image metric on the given batch."""
        imgs, _, _ = batch
        metric_aug_metadata = list()  # list of individual image-level metric metadata

        aug_imgs, aug_dets, aug_metadata = self._apply_augmentations(batch)

        for img, aug_img, aug_md in zip(imgs, aug_imgs, aug_metadata):
            # Convert from channels-first to channels-last
            img_1 = img
            img_2 = None if aug_img is None else aug_img

            # Compute Image metric values
            metric_value = self.metric(img_1=img_1, img_2=img_2, additional_params=aug_md)  # type: ignore
            metric_name = self.metric.__class__.__name__

            existing_metrics = list()
            if "nrtk_metric" in aug_md:
                # TODO: Remove ignore after switch to pyright, mypy doesn't have good typed dict support  # noqa: FIX002
                existing_metrics = list(aug_md["nrtk_metric"])  # type: ignore
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
