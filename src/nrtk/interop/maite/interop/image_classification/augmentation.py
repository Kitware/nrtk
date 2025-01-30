"""This module contains wrappers for NRTK perturbers for image classification"""

import copy
from collections.abc import Sequence
from typing import Optional

import numpy as np
from maite.protocols import ArrayLike
from maite.protocols.image_classification import (
    Augmentation,
    DatumMetadataBatchType,
    InputBatchType,
    TargetBatchType,
)

from nrtk.interfaces.image_metric import ImageMetric
from nrtk.interfaces.perturb_image import PerturbImage

IMG_CLASSIFICATION_BATCH_T = tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]


class JATICClassificationAugmentation(Augmentation):
    """Implementation of JATIC Augmentation for NRTK perturbers.

    Implementation of JATIC Augmentation for NRTK perturbers operating on a MAITE-protocol
    compliant Image Classification dataset.

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

    def __call__(self, batch: IMG_CLASSIFICATION_BATCH_T) -> IMG_CLASSIFICATION_BATCH_T:
        """Apply augmentations to the given data batch."""
        imgs, anns, metadata = batch

        # iterate over (parallel) elements in batch
        aug_imgs = list()  # list of individual augmented inputs
        aug_anns = list()  # list of individual augmented annotations
        aug_metadata = list()  # list of individual augmented image-level metadata

        for img, ann, md in zip(imgs, anns, metadata):
            # Perform augmentation
            aug_img = np.transpose(np.asarray(copy.deepcopy(img)), (1, 2, 0))  # Convert to channels-last
            aug_img, _ = self.augment(aug_img, additional_params=md)
            if aug_img.ndim > 2:
                # Convert back to channels first
                aug_img = np.transpose(aug_img, (2, 0, 1))
            aug_imgs.append(aug_img)

            aug_ann = copy.deepcopy(ann)
            aug_anns.append(aug_ann)

            aug_md = copy.deepcopy(md)
            key_name = f"::{self.name}" if self.name else ""
            aug_md.update({f"nrtk::perturber{key_name}": self.augment.get_config()})
            aug_metadata.append(aug_md)

        # return batch of augmented inputs, class labels and updated metadata
        return aug_imgs, aug_anns, aug_metadata


class JATICClassificationAugmentationWithMetric(Augmentation):
    """Implementation of JATIC augmentation wrapper for NRTK's Image metrics.

    Implementation of JATIC augmentation for NRTK metrics operating on a MAITE-protocol
    compliant image classification dataset.

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

    def __call__(self, batch: IMG_CLASSIFICATION_BATCH_T) -> IMG_CLASSIFICATION_BATCH_T:
        """Compute a specified image metric on the given batch."""
        imgs, anns, metadata = batch
        metric_aug_metadata = list()  # list of individual image-level metric metadata

        aug_imgs: Sequence[Optional[ArrayLike]] = list()
        if self.augmentations:
            aug_batch = batch
            for aug in self.augmentations:
                aug_batch = aug(aug_batch)
            aug_imgs, aug_anns, aug_metadata = aug_batch
        else:
            aug_imgs, aug_anns, aug_metadata = [None] * len(imgs), anns, metadata

        for img, aug_img, aug_md in zip(imgs, aug_imgs, aug_metadata):
            # Convert from channels-first to channels-last
            img_1 = np.transpose(img, (1, 2, 0))
            img_2 = None if aug_img is None else np.transpose(aug_img, (1, 2, 0))

            # Compute Image metric values
            metric_value = self.metric(img_1=img_1, img_2=img_2, additional_params=aug_md)
            metric_aug_md = copy.deepcopy(aug_md)
            metric_name = self.metric.__class__.__name__
            metric_aug_md.update({"nrtk::" + metric_name: metric_value})
            metric_aug_metadata.append(metric_aug_md)

        # return batch of augmented/original images, annotations and metric-updated metadata
        if self.augmentations:
            # type ignore was included to handle the dual Sequence[ArrrayLike] | List[None]
            # case for the augmented images.
            return aug_imgs, aug_anns, metric_aug_metadata  # type: ignore
        return imgs, aug_anns, metric_aug_metadata
