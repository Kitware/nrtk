"""This module contains wrappers for NRTK perturbers for object detection"""

import copy
from collections.abc import Sequence
from typing import Optional

import numpy as np
from maite.protocols import ArrayLike
from maite.protocols.object_detection import (
    Augmentation,
    DatumMetadataBatchType,
    InputBatchType,
    TargetBatchType,
)

from nrtk.interfaces.image_metric import ImageMetric
from nrtk.interfaces.perturb_image import PerturbImage
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
    name: Optional[str]
        Name of the augmentation. Will appear in metadata key.
    """

    def __init__(self, augment: PerturbImage, name: Optional[str] = None) -> None:
        """Initialize augmentation wrapper"""
        self.augment = augment
        self.name = name

    def __call__(self, batch: OBJ_DETECTION_BATCH_T) -> OBJ_DETECTION_BATCH_T:
        """Apply augmentations to the given data batch."""
        imgs, anns, metadata = batch

        # iterate over (parallel) elements in batch
        aug_imgs = list()  # list of individual augmented inputs
        aug_dets = list()  # list of individual object detection targets
        aug_metadata = list()  # list of individual image-level metadata

        for img, ann, md in zip(imgs, anns, metadata):
            # Perform augmentation
            aug_img = copy.deepcopy(img)
            aug_img = np.transpose(aug_img, (1, 2, 0))
            height, width = aug_img.shape[0:2]  # type: ignore
            aug_img, _ = self.augment(np.asarray(aug_img), additional_params=md)
            aug_height, aug_width = aug_img.shape[0:2]
            aug_imgs.append(np.transpose(aug_img, (2, 0, 1)))

            # Resize bounding boxes
            y_aug_boxes = copy.deepcopy(np.asarray(ann.boxes))
            y_aug_labels = copy.deepcopy(np.asarray(ann.labels))
            y_aug_scores = copy.deepcopy(np.asarray(ann.scores))
            y_aug_boxes[:, 0] *= aug_width / width
            y_aug_boxes[:, 1] *= aug_height / height
            y_aug_boxes[:, 2] *= aug_width / width
            y_aug_boxes[:, 3] *= aug_height / height
            aug_dets.append(JATICDetectionTarget(y_aug_boxes, y_aug_labels, y_aug_scores))

            m_aug = copy.deepcopy(md)
            key_name = f"::{self.name}" if self.name else ""
            m_aug.update({f"nrtk::perturber{key_name}": self.augment.get_config()})
            aug_metadata.append(m_aug)

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
    """

    def __init__(self, augmentations: Optional[Sequence[Augmentation]], metric: ImageMetric) -> None:
        """Initialize augmentation with metric wrapper"""
        self.augmentations = augmentations
        self.metric = metric

    def __call__(self, batch: OBJ_DETECTION_BATCH_T) -> OBJ_DETECTION_BATCH_T:
        """Compute a specified image metric on the given batch."""
        imgs, dets, metadata = batch
        metric_aug_metadata = list()  # list of individual image-level metric metadata

        aug_imgs: Sequence[Optional[ArrayLike]] = list()
        if self.augmentations:
            aug_batch = batch
            for aug in self.augmentations:
                aug_batch = aug(aug_batch)
            aug_imgs, aug_dets, aug_metadata = aug_batch
        else:
            aug_imgs, aug_dets, aug_metadata = [None] * len(imgs), dets, metadata

        for img, aug_img, aug_md in zip(imgs, aug_imgs, aug_metadata):
            # Convert from channels-first to channels-last
            img_1 = img
            img_2 = None if aug_img is None else aug_img

            # Compute Image metric values
            metric_value = self.metric(img_1=img_1, img_2=img_2, additional_params=aug_md)  # type: ignore
            metric_aug_md = copy.deepcopy(aug_md)
            metric_name = self.metric.__class__.__name__
            metric_aug_md.update({"nrtk::" + metric_name: metric_value})
            metric_aug_metadata.append(metric_aug_md)

        # return batch of augmented/original images, detections and metric-updated metadata
        if self.augmentations:
            # type ignore was included to handle the dual Sequence[ArrrayLike] | List[None]
            # case for the augmented images.
            return aug_imgs, aug_dets, metric_aug_metadata  # type: ignore
        return imgs, aug_dets, metric_aug_metadata
