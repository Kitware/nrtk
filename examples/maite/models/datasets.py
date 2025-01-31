"""This module contains VisDroneDataset, which is a MAITE wrapper for VisDrone dataset"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import maite.protocols as pr
import numpy as np
import torch
from PIL import Image
from torchvision.ops.boxes import box_convert
from torchvision.transforms.functional import pil_to_tensor


@dataclass
class ObjectDetectionData:
    """Dataclass for object detection"""

    boxes: pr.ArrayLike
    labels: pr.ArrayLike
    scores: pr.ArrayLike

    def __iter__(self) -> Iterator[ObjectDetectionData]:
        """Return an iterator for object detection data"""
        self.n = 0
        return self

    def __next__(self) -> tuple[pr.ArrayLike, pr.ArrayLike, pr.ArrayLike]:
        """Get next object detection data"""
        if self.n < len(self.boxes):
            self.n += 1
            return self.boxes[self.n - 1], self.labels[self.n - 1], self.scores[self.n - 1]
        raise StopIteration


@dataclass
class VisdroneDatumMetadata:
    """Dataclass for Visdrone metadata"""

    id: str
    image_info: dict  # user defined datum metadata


def _load_annotations(annotation_path: Path) -> pr.HasDataBoxesLabels:
    boxes = []
    labels = []
    with open(annotation_path) as file:
        for row in [x.split(",") for x in file.read().strip().splitlines()]:
            # skip VisDrone "ignored regions"
            if row[4] == "0":
                continue

            # convert box from xywh in xyxy format
            box = torch.tensor(list(map(int, row[:4])))
            box = box_convert(box, "xywh", "xyxy")
            boxes.append(box)

            # change class label from 1...10 to 0...9
            label = int(row[5]) - 1
            labels.append(label)
    return ObjectDetectionData(
        boxes=torch.stack(boxes).float(),
        labels=torch.tensor(labels).int(),
        scores=torch.ones(len(labels)).int(),
    )


class VisDroneDataset:
    """VisDrone Dataset.

    Parameters
    ----------
    root : Path | str
        The root directory of the dataset.
    subset_ids : Sequence[str], optional
        Sequence of image IDs (i.e. image filenames modulo extension) to
        restrict dataset to.

    Methods:
    -------
    __len__() -> int
        Returns the number of images in the dataset.
    __getitem__(index: int) -> SupportsObjectDetection
        Returns the image and annotations at the given index.
    __getitem__(index: slice) -> VisDroneDataset
        Returns subset of original dataset as another VisDroneDataset object
    """

    classes = (
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
    )

    def __init__(self, root: Path | str, subset_ids: Sequence[str] | None = None) -> None:
        """Load and initialize VisDroneDataset"""
        self._root: Path = Path(root)

        # populate self._images and self._annotations
        # self._images is a sorted list of paths to image files
        # self._annotations is a dictionary mapping image_id (really the image filename
        # less extension) to its annotation (the HasDataBoxesLabels object for that image)

        # verify root has `images` and `annotations` subdirectories
        image_dir = self._root / "images"
        annotation_dir = self._root / "annotations"

        self._check_dirs(image_dir, annotation_dir)

        all_image_paths = list(image_dir.glob("**/*.jpg"))
        all_image_ids = {p.stem for p in all_image_paths}

        all_annotation_paths = list(annotation_dir.glob("**/*.txt"))
        all_annotation_ids = {p.stem for p in all_annotation_paths}

        # restrict to subset if requested (or default to whole directory); ids look like "0000006_00159_d_0000001"
        keep_ids = {p.stem for p in all_image_paths}
        if subset_ids is not None:
            keep_ids = set(subset_ids)

        self._check_ids(keep_ids, all_image_ids, all_annotation_ids)

        # store sorted image paths
        self._images = sorted([p for p in all_image_paths if p.stem in keep_ids])

        # load all annotations into map keyed by image id
        self._annotations = {}
        for annotation_path in annotation_dir.glob("**/*.txt"):
            image_id = annotation_path.stem
            if image_id in keep_ids:
                self._annotations[image_id] = _load_annotations(annotation_path)

        self._augmentation_func = None
        self._reshape = False
        # load pre-computered brisque score
        self.brisque_scores = dict()
        with open(self._root / "brisque.csv") as bscore_file:
            reader = csv.reader(bscore_file)
            next(reader)  # skip header
            self.brisque_scores = {rows[0]: rows[1] for rows in reader}

    @staticmethod
    def _check_dirs(image_dir: Path, annotation_dir: Path) -> None:
        """Validate existence of image_dir and annotation_dir"""
        if not image_dir.is_dir():
            raise IndexError("Subdirectory `image` doesn't exist")
        if not annotation_dir.is_dir():
            raise IndexError("Subdirectory `annotation` doesn't exist")

    @staticmethod
    def _check_ids(keep_ids: set, all_image_ids: set, all_annotation_ids: set) -> None:
        """Verify that all ids have image and annotation files"""
        for image_id in keep_ids:
            if image_id not in all_image_ids:
                raise IndexError(f"No image file for id: {image_id}")
            if image_id not in all_annotation_ids:
                raise IndexError(f"No annotation file for id: {image_id}")

    def __len__(self) -> int:
        """Returns the number of images in the dataset"""
        return len(self._images)

    def set_augmentation(
        self,
        agm: pr.Augmentation[..., pr.SupportsObjectDetection],
    ) -> None:
        """Sets _augmentation_func for the given value."""
        self._augmentation_func = agm

    def set_reshape(self, value: bool) -> None:
        """Sets _reshape for the given value."""
        self._reshape = value

    @overload
    def __getitem__(self, index: slice) -> VisDroneDataset: ...

    @overload
    def __getitem__(self, index: int) -> pr.SupportsObjectDetection: ...

    def __getitem__(self, index: int | slice) -> VisDroneDataset | pr.SupportsObjectDetection:
        """Returns the dataset object at the given index or subset of dataset at the given slice"""
        if isinstance(index, int):
            image_path = self._images[index]
            image = Image.open(image_path)
            if self._reshape:
                img_array = np.array(image)
                image = torch.from_numpy(img_array)
            else:
                image = pil_to_tensor(image)
            image_id = image_path.stem
            # num_objects = len(self._annotations[image_id].labels)
            # uniq_objects = torch.unique(self._annotations[image_id].labels)
            # num_unique_classes = list(uniq_objects.size())[0]
            # unique_classes = [VisDroneDataset.classes[idx] for idx in uniq_objects.tolist()]

            labeled_datum: pr.SupportsObjectDetection = (
                image,
                self._annotations[image_id],
                {"id": image_id},
            )
            if self._augmentation_func is not None:
                labeled_datum = self._augmentation_func(labeled_datum)

            return labeled_datum
        if isinstance(index, slice):
            images_subset = self._images[index]
            return VisDroneDataset(root=self._root, subset_ids=[p.stem for p in images_subset])
        raise IndexError("Bad index")
