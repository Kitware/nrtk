"""This module contains wrappers for converting a COCO dataset or a generic dataset to a MAITE dataset."""

from __future__ import annotations

__all__ = ["MAITEObjectDetectionDataset", "COCOMAITEObjectDetectionDataset", "dataset_to_coco"]

import copy
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image  # type: ignore
from typing_extensions import ReadOnly

from nrtk.utils._exceptions import KWCocoImportError, MaiteImportError
from nrtk.utils._import_guard import import_guard
from nrtk.utils._logging import setup_logging

maite_available: bool = import_guard(
    module_name="maite",
    exception=MaiteImportError,
    submodules=["protocols.object_detection"],
    objects=["Dataset", "DatumMetadata"],
)
kwcoco_available: bool = import_guard(module_name="kwcoco", exception=KWCocoImportError)
from kwcoco import CocoDataset  # type: ignore  # noqa: E402
from maite.protocols import DatasetMetadata, DatumMetadata  # noqa: E402
from maite.protocols.object_detection import (  # noqa: E402
    Dataset,
    DatumMetadataType,
    InputType,
    TargetType,
)

logger: logging.Logger = setup_logging(name=__name__)


OBJ_DETECTION_DATUM_T = tuple[InputType, TargetType, DatumMetadataType]


@dataclass
class MAITEObjectDetectionTarget:
    """Dataclass for the datum-level MAITE output detection format."""

    boxes: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]
    scores: np.ndarray[Any, Any]


class COCOMetadata(DatumMetadata):  # pyright: ignore [reportGeneralTypeIssues]
    """TypedDict for COCO-detection datum-level metdata."""

    # pyright fails when failing to import maite.protocols
    ann_ids: ReadOnly[Sequence[int]]  # pyright: ignore [reportInvalidTypeForm]
    image_info: ReadOnly[dict[str, Any]]  # pyright: ignore [reportInvalidTypeForm]


class COCOMAITEObjectDetectionDataset(Dataset):  # pyright: ignore [reportGeneralTypeIssues]
    """Dataset class to convert a COCO dataset to a dataset compliant with MAITE's Object Detection protocol.

    Attributes:
        metadata: DatasetMetadata
            Metadata of this dataset.
    """

    def __init__(  # noqa: C901
        self,
        *,
        kwcoco_dataset: CocoDataset,  # pyright: ignore [reportGeneralTypeIssues]
        image_metadata: Sequence[DatumMetadataType],  # pyright: ignore [reportInvalidTypeForm]
        skip_no_anns: bool = False,
        dataset_id: str | None = None,
    ) -> None:
        """Initialize MAITE-compliant dataset from a COCO dataset.

        Args:
            kwcoco_dataset:
                The COCO dataset object.
            image_metadata:
                Metadata for each image.
            skip_no_anns:
                Whether to skip images without annotations. Defaults to False.
            dataset_id:
                Dataset ID, defaults to filepath.

        Raises:
            ImportError:
                required dependencies are not installed.
            ValueError:
                metadata is missing for any image in the dataset.
        """
        if not kwcoco_available:
            raise KWCocoImportError

        if not maite_available:
            raise MaiteImportError

        self._kwcoco_dataset = kwcoco_dataset

        self._image_ids = list()
        self._annotations = dict()

        for _, img_id in enumerate(kwcoco_dataset.imgs.keys()):
            bboxes = np.empty((0, 4))
            labels = []
            scores = []

            if img_id in kwcoco_dataset.gid_to_aids and len(kwcoco_dataset.gid_to_aids[img_id]) > 0:
                det_ids = kwcoco_dataset.gid_to_aids[img_id]
                for det_id in det_ids:
                    ann = kwcoco_dataset.anns[det_id]

                    labels.append(ann["category_id"])

                    if "score" in ann:
                        scores.append(ann["score"])
                    elif "prob" in ann:
                        scores.append(max(ann["prob"]))
                    else:
                        scores.append(1.0)

                    x, y, w, h = ann["bbox"]
                    bbox = [x, y, x + w, y + h]
                    bboxes = np.vstack((bboxes, bbox))
            elif skip_no_anns:
                continue

            img_file_path = Path(kwcoco_dataset.get_image_fpath(img_id))
            if not img_file_path.exists():
                continue
            self._image_ids.append(img_id)
            self._annotations[img_id] = MAITEObjectDetectionTarget(
                boxes=bboxes,
                labels=np.asarray(labels),
                scores=np.asarray(scores),
            )

        self._image_metadata = {
            image_md["id"]: copy.deepcopy(image_md)
            for image_md in image_metadata
            if "id" in image_md and image_md["id"] in self._image_ids
        }
        if len(self._image_metadata) != len(self._image_ids):
            raise ValueError("Image metadata length mismatch, metadata needed for every image.")

        self.metadata: DatasetMetadata = DatasetMetadata(
            id=dataset_id if dataset_id else kwcoco_dataset.fpath,
            index2label={c["id"]: c["name"] for c in kwcoco_dataset.cats.values()},
        )

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self._image_ids)

    def __getitem__(self, index: int) -> tuple[InputType, TargetType, COCOMetadata]:  # pyright: ignore [reportInvalidTypeForm]
        """Returns the dataset object at the given index."""
        image_id = self._image_ids[index]
        img_file_path = Path(self._kwcoco_dataset.get_image_fpath(image_id))
        image = Image.open(img_file_path)
        width, height = image.size

        gid_to_aids = self._kwcoco_dataset.gid_to_aids

        image_md: COCOMetadata = {  # pyright: ignore [reportAssignmentType]
            "id": image_id,
            "ann_ids": (list(gid_to_aids[image_id]) if image_id in gid_to_aids else list()),
            "image_info": dict(width=width, height=height, file_name=str(img_file_path)),
        }

        # Forward input metadata, checking for clobbering
        forwarded_keys = ["id"]
        for key, value in self._image_metadata[image_id].items():
            if key in forwarded_keys:
                continue
            if key in image_md and key not in forwarded_keys:
                raise KeyError(f"'{key}' already present in metadata, erroring out to prevent overwrite")
            image_md[key] = value

        image_array = np.asarray(image)
        if image.mode == "L":
            image_array = np.expand_dims(image_array, axis=2)

        return (
            np.asarray(np.transpose(image_array, axes=(2, 0, 1))),
            self._annotations[image_id],
            image_md,
        )

    def get_img_path_list(self) -> list[Path]:
        """Returns the sorted list of absolute paths for the input images."""
        return sorted([Path(self._kwcoco_dataset.get_image_fpath(img_id)) for img_id in self._image_ids])

    def get_categories(self) -> list[dict[str, Any]]:
        """Returns the list of categories for this dataset."""
        return list(self._kwcoco_dataset.cats.values())

    @classmethod
    def is_usable(cls) -> bool:
        """Returns True if the required kwcoco and MAITE modules are available."""
        return kwcoco_available and maite_available


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
        if not self.is_usable():
            raise MaiteImportError
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

    @classmethod
    def is_usable(cls) -> bool:
        """Returns True if the required MAITE module is available."""
        return maite_available


def _xywh_bbox_xform(*, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    """Transform bounding box from xyxy format to xywh format."""
    return x1, y1, x2 - x1, y2 - y1


def _create_annotations(dataset_categories: list[dict[str, Any]]) -> CocoDataset:  # pyright: ignore [reportReturnType]
    """Create a new CocoDataset with the given categories."""
    annotations = CocoDataset()  # pyright: ignore [reportCallIssue]
    for cat in dataset_categories:
        annotations.add_category(name=cat["name"], supercategory=cat["supercategory"], id=cat["id"])
    return annotations


def dataset_to_coco(  # noqa: C901
    *,
    dataset: Dataset,  # pyright: ignore [reportInvalidTypeForm]
    output_dir: Path,
    img_filenames: list[Path],
    dataset_categories: list[dict[str, Any]],
) -> None:
    """Save dataset object to file as a COCO formatted dataset.

    Args:
        dataset:
            MAITE-compliant object detection dataset
        output_dir:
            The location where data will be saved.
        img_filenames:
            Filenames of images to be saved.
        dataset_categories:
            A list of the categories related to this dataset.
            Each dictionary should contain the following keys: id, name, supercategory.

    Raises:
        KWCocoImportError:
            KWCOCO is not available.
        MaiteImportError:
            MAITE is not available.
        ValueError:
            Image filename and dataset length mismatch.
    """
    if not kwcoco_available:
        raise KWCocoImportError
    if not maite_available:
        raise MaiteImportError
    if len(img_filenames) != len(dataset):
        raise ValueError(f"Image filename and dataset length mismatch ({len(img_filenames)} != {len(dataset)})")

    mod_metadata = list()

    annotations = _create_annotations(dataset_categories)

    for i in range(len(dataset)):
        image, dets, metadata = dataset[i]
        filename = output_dir / img_filenames[i]
        filename.parent.mkdir(parents=True, exist_ok=True)

        im = Image.fromarray(np.transpose(np.asarray(image), (1, 2, 0)))
        im.save(filename)

        labels = np.asarray(dets.labels)
        bboxes = np.asarray(dets.boxes)
        annotations.add_images([{"id": i, "file_name": str(filename)}])
        for lbl, bbox in zip(labels, bboxes, strict=False):
            annotations.add_annotation(
                image_id=i,
                category_id=int(lbl),
                bbox=list(
                    _xywh_bbox_xform(
                        x1=int(bbox[0]),
                        y1=int(bbox[1]),
                        x2=int(bbox[2]),
                        y2=int(bbox[3]),
                    ),
                ),
            )

        mod_metadata.append(metadata)
    logger.info(f"Saved perturbed images to {output_dir}")

    metadata_file = output_dir / "image_metadata.json"

    with open(metadata_file, "w") as f:
        json.dump(mod_metadata, f)
    logger.info(f"Saved image metadata to {metadata_file}")

    annotations_file = output_dir / "annotations.json"
    annotations.dump(annotations_file)
    logger.info(f"Saved annotations to {annotations_file}")
