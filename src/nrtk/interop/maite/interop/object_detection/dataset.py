"""
This module contains wrappers for converting a COCO dataset or
a generic dataset to a MAITE dataset for object detection
"""

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from maite.protocols.object_detection import (
    Dataset,
    DatumMetadataType,
    InputType,
    TargetType,
)
from PIL import Image  # type: ignore

try:
    import kwcoco  # type: ignore

    is_usable = True
except ImportError:
    is_usable = False

OBJ_DETECTION_DATUM_T = tuple[InputType, TargetType, DatumMetadataType]

LOG = logging.getLogger(__name__)


@dataclass
class JATICDetectionTarget:
    """Dataclass for the datum-level JATIC output detection format."""

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


def _xyxy_bbox_xform(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    return x, y, x + w, y + h


def _coco_to_maite_detections(coco_annotation: list) -> TargetType:
    num_anns = len(coco_annotation)
    boxes = np.zeros((num_anns, 4))
    for i, anns in enumerate(coco_annotation):
        box = list(map(int, anns["bbox"]))
        # convert box from xywh in xyxy format
        boxes[i, :] = np.asarray(_xyxy_bbox_xform(x=box[0], y=box[1], w=box[2], h=box[3]))

    labels = np.stack([int(anns["category_id"]) for anns in coco_annotation])
    scores = np.ones(num_anns)

    return JATICDetectionTarget(boxes, labels, scores)


if not is_usable:
    LOG.warning("COCOJATICObjectDetectionDataset requires additional dependencies, please install 'nrtk-jatic[tools]'")
else:

    class COCOJATICObjectDetectionDataset(Dataset):
        """Dataset class to convert a COCO dataset to a dataset compliant with JATIC's Object Detection protocol.

        Parameters
        ----------
        root : str
            The root directory of the dataset.
        kwcoco_dataset : kwcoco.CocoDataset
            The kwcoco COCODataset object.
        image_metadata : list[dict[str, Any]]
            A list of per-image metadata. Any metadata required by a perturber should be provided.
        """

        def __init__(
            self,
            root: str,
            kwcoco_dataset: kwcoco.CocoDataset,
            image_metadata: list[dict[str, Any]],
        ) -> None:
            """Initialize MAITE-compliant dataset from a COCO dataset"""
            self._root: Path = Path(root)
            image_dir = self._root / "images"
            self.all_img_paths = [image_dir / val["file_name"] for key, val in kwcoco_dataset.imgs.items()]
            self.all_image_ids = sorted({p.stem for p in self.all_img_paths})

            # Get all image filenames from the kwcoco object
            anns_image_ids = [
                {"coco_image_id": val["id"], "filename": val["file_name"]} for key, val in kwcoco_dataset.imgs.items()
            ]
            anns_image_ids = sorted(anns_image_ids, key=lambda d: d["filename"])

            # store sorted image paths
            self._images = sorted([p for p in self.all_img_paths if p.stem in self.all_image_ids])

            self._annotations = {}
            for image_id, anns_img_id in zip(self.all_image_ids, anns_image_ids):
                image_annotations = [
                    sub for sub in list(kwcoco_dataset.anns.values()) if sub["image_id"] == anns_img_id["coco_image_id"]
                ]
                # Convert annotations to maite detections format
                self._annotations[image_id] = _coco_to_maite_detections(image_annotations)

            self.classes = list(kwcoco_dataset.cats.values())

            self._image_metadata = copy.deepcopy(image_metadata)
            if len(self._image_metadata) != len(self.all_img_paths):
                raise ValueError("Image metadata length mismatch, metadata needed for every image")

        def __len__(self) -> int:
            """Returns the number of images in the dataset."""
            return len(self._images)

        def __getitem__(self, index: int) -> OBJ_DETECTION_DATUM_T:
            """Returns the dataset object at the given index."""
            image_path = self._images[index]
            image = Image.open(image_path)
            image_id = image_path.stem
            width, height = image.size
            annotation = self._annotations[image_id].labels
            num_objects = np.asarray(annotation).shape[0]
            uniq_objects = np.unique(annotation)
            num_unique_classes = uniq_objects.shape[0]
            unique_classes = [self.classes[int(idx)]["name"] for idx in uniq_objects.tolist()]

            self._image_metadata[index].update(
                dict(
                    id=image_id,
                    image_info=dict(
                        width=width,
                        height=height,
                        num_objects=num_objects,
                        num_unique_classes=num_unique_classes,
                        unique_classes=unique_classes,
                    ),
                ),
            )

            input_img, dets, metadata = (
                np.transpose(np.asarray(image), (2, 0, 1)),
                self._annotations[image_id],
                self._image_metadata[index],
            )

            return input_img, dets, metadata

        def get_img_path_list(self) -> list[Path]:
            """Returns the sorted list of absolute paths for the input images."""
            return sorted(self.all_img_paths)

        def get_categories(self) -> list[dict[str, Any]]:
            """Returns the list of categories for this dataset."""
            return self.classes


class JATICObjectDetectionDataset(Dataset):
    """Implementation of the JATIC Object Detection dataset wrapper for dataset images of varying sizes.

    Parameters
    ----------
    imgs : Sequence[np.ndarray]
        Sequence of images.
    dets : Sequence[ObjectDetectionTarget]
        Sequence of detections for each image.
    metadata : Sequence[dict[str, Any]]
        Sequence of custom metadata for each image.
    """

    def __init__(
        self,
        imgs: Sequence[np.ndarray],
        dets: Sequence[TargetType],
        metadata: Sequence[DatumMetadataType],
    ) -> None:
        """Initialize MAITE-compliant dataset"""
        self.imgs = imgs
        self.dets = dets
        self.metadata = metadata

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.imgs)

    def __getitem__(self, index: int) -> OBJ_DETECTION_DATUM_T:
        """Returns the dataset object at the given index."""
        return self.imgs[index], self.dets[index], self.metadata[index]
