"""This module contains wrappers for converting a generic dataset to a MAITE dataset for image classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np

InputType: type = object
TargetType: type = object
DatumMetadataType: type = TypedDict
Dataset: type = object
try:
    # Multiple type ignores added for pyright's handling of guarded imports
    from maite.protocols import DatasetMetadata
    from maite.protocols.image_classification import (
        Dataset,
        DatumMetadataType,
        InputType,
        TargetType,
    )

    maite_available: bool = True
except ImportError:  # pragma: no cover
    maite_available: bool = False

IMG_CLASSIFICATION_DATUM_T = tuple[InputType, TargetType, DatumMetadataType]  # pyright:  ignore [reportPossiblyUnboundVariable]


class JATICImageClassificationDataset(Dataset):  # pyright: ignore [reportGeneralTypeIssues]
    """Implementation of the JATIC Image Classification dataset wrapper for dataset images of varying sizes.

    Parameters
    ----------
    imgs : Sequence[np.ndarray]
        Sequence of images.
    labels : Sequence[ArrayLike]
        Sequence of labels for each image.
    datum_metadata : Sequence[DatumMetadataType]
        Sequence of custom metadata for each image.
    metadata : DatasetMetadata
        Metadata for this dataset.
    """

    def __init__(
        self,
        imgs: Sequence[np.ndarray[Any, Any]],
        labels: Sequence[TargetType],  # pyright: ignore [reportInvalidTypeForm]
        datum_metadata: Sequence[DatumMetadataType],  # pyright: ignore [reportInvalidTypeForm]
        dataset_id: str,
        index2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize MAITE-compliant dataset.

        Args:
            imgs (Sequence[np.ndarray]): Sequence of images in the dataset.
            labels (Sequence[TargetType]): Sequence of labels for the images.
            datum_metadata (Sequence[DatumMetadataType]): Sequence of metadata dictionaries.
            dataset_id (str): Dataset ID.
            index2label (dict[int, str] | None): Mapping from class index to label.
        """
        self.imgs = imgs
        self.labels = labels
        self.datum_metadata = datum_metadata
        if index2label is not None:
            self.metadata: DatasetMetadata = {
                "id": dataset_id,
                "index2label": index2label,
            }
        else:
            self.metadata = {
                "id": dataset_id,
            }

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.imgs)

    def __getitem__(self, index: int) -> IMG_CLASSIFICATION_DATUM_T:
        """Returns the dataset object at the given index."""
        return self.imgs[index], self.labels[index], self.datum_metadata[index]
