"""MAITE Object Detection dataset wrapper classes.

These classes require the ``maite`` extra.
"""

from __future__ import annotations

__all__ = [
    "MAITEObjectDetectionDataset",
    "MAITEObjectDetectionTarget",
    "OBJ_DETECTION_DATUM_T",
    "_xywh_bbox_xform",
]

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from maite.protocols import DatasetMetadata
from maite.protocols.object_detection import (
    Dataset,
    DatumMetadataType,
    InputType,
    TargetType,
)

from nrtk.utils._logging import setup_logging

logger: logging.Logger = setup_logging(name=__name__)


OBJ_DETECTION_DATUM_T = tuple[InputType, TargetType, DatumMetadataType]


@dataclass
class MAITEObjectDetectionTarget:
    """Dataclass for the datum-level MAITE output detection format."""

    boxes: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]
    scores: np.ndarray[Any, Any]


class MAITEObjectDetectionDataset(Dataset):  # pyright: ignore [reportGeneralTypeIssues]
    """Implementation of the MAITE Object Detection dataset wrapper for dataset images of varying sizes.

    Attributes:
        imgs: Sequence[np.ndarray]
            Sequence of images.
        dets: Sequence[ObjectDetectionTarget]
            Sequence of detections for each image.
        datum_metadata: Sequence[DatumMetadataType]
            Sequence of metadata for each image.
        dataset_id: str
            Dataset ID.
        index2label: dict[int, str] | None
            Mapping from class index to label.
    """

    def __init__(
        self,
        *,
        imgs: Sequence[np.ndarray[Any, Any]],
        dets: Sequence[TargetType],  # pyright: ignore [reportInvalidTypeForm]
        datum_metadata: Sequence[DatumMetadataType],  # pyright: ignore [reportInvalidTypeForm]
        dataset_id: str,
        index2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize MAITE-compliant dataset.

        Args:
            imgs:
                Sequence of images in the dataset.
            dets:
                Sequence of detection targets for the images.
            datum_metadata:
                Sequence of metadata dictionaries.
            dataset_id:
                Dataset ID.
            index2label:
                Mapping from class index to label.
        """
        self.imgs = imgs
        self.dets = dets
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

    def __getitem__(self, index: int) -> OBJ_DETECTION_DATUM_T:
        """Returns the dataset object at the given index."""
        return self.imgs[index], self.dets[index], self.datum_metadata[index]


def _xywh_bbox_xform(*, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    """Transform bounding box from xyxy format to xywh format."""
    return x1, y1, x2 - x1, y2 - y1
