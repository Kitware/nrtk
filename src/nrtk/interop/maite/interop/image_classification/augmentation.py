"""This module contains wrappers for NRTK perturbers for image classification."""

from __future__ import annotations

__all__ = ["JATICClassificationAugmentation", "JATICClassificationAugmentationWithMetric"]

import copy
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from nrtk.interfaces.image_metric import ImageMetric
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interop.maite.interop.generic import NRTKDatumMetadata, _forward_md_keys
from nrtk.utils._exceptions import MaiteImportError
from nrtk.utils._import_guard import import_guard

maite_available: bool = import_guard(
    "maite",
    MaiteImportError,
    ["protocols", "protocols.image_classification"],
    ["Augmentation"],
)
from maite.protocols import AugmentationMetadata  # noqa: E402
from maite.protocols.image_classification import (  # noqa: E402
    Augmentation,
    DatumMetadataType,
    InputType,
    TargetType,
)

IMG_CLASSIFICATION_BATCH_T = tuple[Sequence[InputType], Sequence[TargetType], Sequence[DatumMetadataType]]


class JATICClassificationAugmentation(Augmentation):  # pyright:  ignore [reportGeneralTypeIssues]
    """Implementation of JATIC Augmentation for NRTK perturbers.

    Implementation of JATIC Augmentation for NRTK perturbers operating on a MAITE-protocol
    compliant Image Classification dataset.

    Parameters
    ----------
    augment : PerturbImage
        Augmentations to apply to an image.
    name: str
        Name of the augmentation. Will appear in metadata key.
    """

    def __init__(self, augment: PerturbImage, augment_id: str) -> None:
        """Initialize augmentation wrapper."""
        if not self.is_usable():
            raise MaiteImportError
        self.augment = augment
        self.metadata: AugmentationMetadata = AugmentationMetadata(id=augment_id)

    def __call__(
        self,
        batch: IMG_CLASSIFICATION_BATCH_T,
    ) -> tuple[Sequence[InputType], Sequence[TargetType], Sequence[NRTKDatumMetadata]]:  # pyright: ignore [reportInvalidTypeForm]
        """Apply augmentations to the given data batch."""
        imgs, anns, metadata = batch

        # iterate over (parallel) elements in batch
        aug_imgs = list()  # list of individual augmented inputs
        aug_anns = list()  # list of individual augmented annotations
        aug_metadata = list()  # list of individual augmented image-level metadata

        for img, ann, md in zip(imgs, anns, metadata, strict=False):  # pyright: ignore [reportArgumentType]
            # Perform augmentation
            aug_img = np.transpose(np.asarray(copy.deepcopy(img)), (1, 2, 0))  # Convert to channels-last
            aug_img, _ = self.augment(aug_img, boxes=None, **dict(md))
            if aug_img.ndim > 2:
                # Convert back to channels first
                aug_img = np.transpose(aug_img, (2, 0, 1))
            aug_imgs.append(aug_img)

            aug_ann = copy.deepcopy(ann)
            aug_anns.append(aug_ann)

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

        # return batch of augmented inputs, class labels and updated metadata
        return aug_imgs, aug_anns, aug_metadata

    @classmethod
    def is_usable(cls) -> bool:
        """Checks if the necessary dependency (MAITE) is available.

        Returns:
            bool: True MAITE is available; False otherwise.
        """
        return maite_available


class JATICClassificationAugmentationWithMetric(Augmentation):  # pyright:  ignore [reportGeneralTypeIssues]
    """Implementation of JATIC augmentation wrapper for NRTK's Image metrics.

    Implementation of JATIC augmentation for NRTK metrics operating on a MAITE-protocol
    compliant image classification dataset.

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
        self.augmentations = augmentations
        self.metric = metric
        self.metadata: AugmentationMetadata = AugmentationMetadata(id=augment_id)

    def _apply_augmentations(
        self,
        batch: IMG_CLASSIFICATION_BATCH_T,
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
        batch: IMG_CLASSIFICATION_BATCH_T,
    ) -> tuple[Sequence[InputType], Sequence[TargetType], Sequence[NRTKDatumMetadata]]:  # pyright: ignore [reportInvalidTypeForm]
        """Compute a specified image metric on the given batch."""
        imgs, _, _ = batch
        metric_aug_metadata = list()  # list of individual image-level metric metadata

        aug_imgs, aug_anns, aug_metadata = self._apply_augmentations(batch)

        for img, aug_img, aug_md in zip(imgs, aug_imgs, aug_metadata, strict=False):  # pyright: ignore [reportArgumentType]
            # Convert from channels-first to channels-last
            img_1 = np.transpose(np.asarray(img), (1, 2, 0))
            img_2 = None if aug_img is None else np.transpose(aug_img, (1, 2, 0))

            # Compute Image metric values
            metric_value = self.metric(img_1=img_1, img_2=img_2, additional_params=dict(aug_md))
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

        # return batch of augmented/original images, annotations and metric-updated metadata
        if self.augmentations:
            # type ignore was included to handle the dual Sequence[ArrrayLike] | List[None]
            # case for the augmented images.
            return aug_imgs, aug_anns, metric_aug_metadata  # type: ignore
        return imgs, aug_anns, metric_aug_metadata
