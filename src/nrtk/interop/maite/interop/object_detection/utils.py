"""This module contains dataset_to_coco, which converts a MAITE dataset to a COCO dataset"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from maite.protocols.object_detection import Dataset
from PIL import Image  # type: ignore

try:
    import kwcoco  # type: ignore

    is_usable = True
except ImportError:
    is_usable = False


def _xywh_bbox_xform(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    return x1, y1, x2 - x1, y2 - y1


def _create_annotations(dataset_categories: list[dict[str, Any]]) -> kwcoco.CocoDataset:
    annotations = kwcoco.CocoDataset()
    for cat in dataset_categories:
        annotations.add_category(name=cat["name"], supercategory=cat["supercategory"], id=cat["id"])
    return annotations


def dataset_to_coco(
    dataset: Dataset,
    output_dir: Path,
    img_filenames: list[Path],
    dataset_categories: list[dict[str, Any]],
) -> None:
    """Save dataset object to file as a COCO formatted dataset.

    :param dataset: MAITE-compliant object detection dataset
    :param output_dir: The location where data will be saved.
    :param img_filenames: Filenames of images to be saved.
    :param dataset_categories: A list of the categories related to this dataset.
        Each dictionary should contain the following keys: id, name, supercategory.
    """
    if len(img_filenames) != len(dataset):
        raise ValueError(f"Image filename and dataset length mismatch ({len(img_filenames)} != {len(dataset)})")
    if not is_usable:
        raise ImportError("This tool requires additional dependencies, please install `nrtk-jatic[tools]`")
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
        for lbl, bbox in zip(labels, bboxes):
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
    logging.info(f"Saved perturbed images to {output_dir}")

    metadata_file = output_dir / "image_metadata.json"

    with open(metadata_file, "w") as f:
        json.dump(mod_metadata, f)
    logging.info(f"Saved image metadata to {metadata_file}")

    annotations_file = output_dir / "annotations.json"
    annotations.dump(annotations_file)
    logging.info(f"Saved annotations to {annotations_file}")
