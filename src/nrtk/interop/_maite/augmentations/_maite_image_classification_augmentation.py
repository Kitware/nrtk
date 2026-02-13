"""This module contains wrappers for NRTK perturbers for image classification."""

from __future__ import annotations

__all__ = ["MAITEImageClassificationAugmentation"]

import copy
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from maite.protocols import AugmentationMetadata
from maite.protocols.image_classification import (
    Augmentation,
    DatumMetadataType,
    InputType,
    TargetType,
)

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interop._maite.metadata import NRTKDatumMetadata
from nrtk.interop._maite.metadata._nrtk_datum_metadata import _forward_md_keys

IMG_CLASSIFICATION_BATCH_T = tuple[Sequence[InputType], Sequence[TargetType], Sequence[DatumMetadataType]]


class MAITEImageClassificationAugmentation(Augmentation):  # pyright:  ignore [reportGeneralTypeIssues]
    """Implementation of MAITE Image Classification Augmentation for NRTK perturbers.

    Implementation of MAITE Augmentation for NRTK perturbers operating on a MAITE-protocol
    compliant Image Classification dataset.

    Attributes:
        augment: PerturbImage
            Augmentations to apply to an image.
        name: str
            Name of the augmentation. Will appear in metadata key.
    """

    def __init__(self, *, augment: PerturbImage, augment_id: str) -> None:
        """Initialize augmentation wrapper.

        Args:
            augment:
                PerturbImage implementation to apply to an image.
            augment_id:
                Metadata ID for this augmentation.
        """
        self.augment = augment
        self.metadata: AugmentationMetadata = AugmentationMetadata(id=augment_id)

    def __call__(
        self,
        batch: IMG_CLASSIFICATION_BATCH_T,
    ) -> IMG_CLASSIFICATION_BATCH_T:
        """Return a batch of augmented images and metadata."""
        imgs, anns, metadata = batch

        # iterate over (parallel) elements in batch
        aug_imgs = []  # list of individual augmented inputs
        aug_anns = []  # list of individual augmented annotations
        aug_metadata = []  # list of individual augmented image-level metadata

        for img, ann, md in zip(imgs, anns, metadata, strict=False):  # pyright: ignore [reportArgumentType]
            # Perform augmentation
            aug_img = np.transpose(np.asarray(copy.deepcopy(img)), (1, 2, 0))  # Convert to channels-last
            aug_img, _ = self.augment(image=aug_img, boxes=None, **dict(md))
            if aug_img.ndim > 2:
                # Convert back to channels first
                aug_img = np.transpose(aug_img, (2, 0, 1))
            aug_imgs.append(aug_img)

            aug_ann = copy.deepcopy(ann)
            aug_anns.append(aug_ann)

            perturber_configs = []
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

        # return batch of augmented inputs, class labels and updated metadata
        return aug_imgs, aug_anns, aug_metadata
