"""Utilities for interoperability with MAITE OD dataset protocols."""

__all__ = ["VisDroneObjectDetectionDataset", "stratified_sample_dataset"]

# Python imports
import csv
import os
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from typing_extensions import ReadOnly

# Local Import
from nrtk.utils._exceptions import NRTKXAITKHelperImportError
from nrtk.utils._import_guard import import_guard

PIL_available: bool = import_guard("PIL", NRTKXAITKHelperImportError)
maite_available: bool = import_guard(
    "maite",
    NRTKXAITKHelperImportError,
    ["protocols.object_detection"],
    ["Dataset", "DatumMetadataType", "InputType", "TargetType"],
) and import_guard(
    "maite",
    NRTKXAITKHelperImportError,
    ["protocols"],
    ["DatasetMetadata", "DatumMetadata"],
)
torch_available: bool = import_guard("torch", NRTKXAITKHelperImportError) and import_guard(
    "torch",
    NRTKXAITKHelperImportError,
    ["utils.data"],
    ["Subset"],
)
nrtk_xaitk_helpers_available: bool = maite_available and PIL_available and torch_available
import torch  # noqa: E402
from maite.protocols import DatasetMetadata, DatumMetadata  # noqa: E402
from maite.protocols.object_detection import Dataset, DatumMetadataType, InputType, TargetType  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import Subset  # noqa: E402


@dataclass
class YOLODetectionTarget:
    """A helper class to represent object detection results in the format expected by YOLO-based models.

    Attributes:
        boxes (torch.Tensor): A tensor containing the bounding boxes for detected objects in
            [x_min, y_min, x_max, y_max] format.
        labels (torch.Tensor): A tensor containing the class labels for the detected objects.
            These may be floats for compatibility with specific datasets or tools.
        scores (torch.Tensor): A tensor containing the confidence scores for the detected objects.
    """

    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


class VisDroneMetadata(DatumMetadata):  # pyright: ignore [reportGeneralTypeIssues]
    """TypedDict for VisDrone-detection datum-level metdata."""

    # pyright fails when failing to import maite.protocols
    filename: ReadOnly[Sequence[str]]  # pyright: ignore [reportInvalidTypeForm]


class VisDroneObjectDetectionDataset(Dataset):
    """A MAITE-compliant dataset wrapper around the VisDrone detection data following the Dataset protocol.

    Each item yields (image, target, metadata):
      - image: (C,H,W) float array
      - target: dict with bounding boxes, labels, etc.
      - metadata: dict with 'id' and other fields like 'filename'

    Example usage:
        dataset = VisDroneObjectDetectionDataset(
            images_dir="VisDrone2019-DET/train/images",
            annotations_dir="VisDrone2019-DET/train/annotations",
            dataset_id="VisDrone-DET-train"
        )
        img, tgt, meta = dataset[0]
    """

    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        dataset_id: str = "VisDrone-DET-test",
    ) -> None:
        """Wraps a VisDrone dataset to comply with MAITE Dataset protocol.

        :param images_dir: Path to the folder containing .jpg images
        :param annotations_dir: Path to the folder containing corresponding .txt annotations
        :param dataset_id: String for 'id' in dataset-level metadata
        """
        if not nrtk_xaitk_helpers_available:
            raise NRTKXAITKHelperImportError
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.max_size = 1360

        # Gather all .jpg filenames
        self.image_files: list[str] = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".jpg"))

        # Per the protocol, define dataset-level metadata with at least 'id'
        self.metadata: DatasetMetadata = DatasetMetadata(
            id=dataset_id,
            index2label={-1: "N/A"}
            | {
                idx: c
                for idx, c in enumerate(
                    [
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
                        "N/A",
                    ],
                )
            },
        )

    def __len__(self) -> int:
        """Return length of the dataset."""
        # The dataset length is the number of images we have
        return len(self.image_files)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[InputType, TargetType, DatumMetadataType]:
        """Retrieve an indexed sample from the dataset.

        Returns:
            tuple[InputType, TargetType, DatumMetadataType]:
                - image: (C,H,W) float32 array in [0..1]
                - yolo_target: YOLODetectionTarget, with:
                    boxes:  shape (N,4)
                    labels: shape (N,)
                    scores: shape (N,)
                - metadata: dict with 'id', 'filename'
        """
        # 1. Load the image
        image_filename = self.image_files[index]
        img_path = os.path.join(self.images_dir, image_filename)

        with Image.open(img_path) as img_pil:
            img_np = np.array(img_pil, dtype=np.uint8)  # (H,W,C)
            img_np = img_np[:, :, ::-1]
            img_np = np.transpose(img_np, (2, 0, 1))
        # 2. Read annotation file
        annotation_filename = image_filename.replace(".jpg", ".txt")
        ann_path = os.path.join(self.annotations_dir, annotation_filename)

        boxes_list = []
        labels_list = []
        scores_list = []  # If you have real scores for GT, otherwise 1.0

        # Parse annotations if file exists
        if os.path.isfile(ann_path):
            with open(ann_path) as f:
                reader = csv.reader(f)
                for vals in reader:
                    if len(vals) < 8:
                        continue
                    bbox_top = float(vals[0])
                    bbox_left = float(vals[1])
                    bbox_width = float(vals[2])
                    bbox_height = float(vals[3])
                    score = float(vals[4])  # GT is 1, for detection results might vary
                    category = int(vals[5]) - 1

                    x1 = bbox_top
                    y1 = bbox_left
                    x2 = bbox_top + bbox_width
                    y2 = bbox_left + bbox_height

                    boxes_list.append([x1, y1, x2, y2])
                    labels_list.append(int(category))  # type: ignore
                    scores_list.append(score)

        # Convert to Tensors
        if len(boxes_list) > 0:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.int32)
            labels_tensor = torch.tensor(labels_list, dtype=torch.int8)
            scores_tensor = torch.tensor(scores_list, dtype=torch.float32)
        else:
            # no bounding boxes
            boxes_tensor = torch.zeros((0, 4), dtype=torch.int32)
            labels_tensor = torch.zeros((0,), dtype=torch.int8)
            scores_tensor = torch.zeros((0,), dtype=torch.float32)

        # Build the YOLODetectionTarget (just one per image)
        yolo_target = YOLODetectionTarget(
            boxes=boxes_tensor,
            labels=labels_tensor,
            scores=scores_tensor,
        )

        # 3. Build per-datum metadata
        datum_metadata = VisDroneMetadata(
            id=str(index),
            filename=image_filename,
        )

        return (img_np, yolo_target, datum_metadata)


def stratified_sample_dataset(
    dataset: VisDroneObjectDetectionDataset,
    subset_size: int,
    seed: int = 42,
) -> VisDroneObjectDetectionDataset:
    """Creates a stratified random subset of the dataset while preserving label distribution as closely as possible.

    :param dataset: The original VisDroneObjectDetectionDataset instance.
    :param subset_size: The number of samples to select.
    :param seed: The random seed for reproducibility.
    :return: A new VisDroneObjectDetectionDataset instance containing the stratified subset.
    """
    random.seed(seed)
    np.random.default_rng(seed)

    # Create a mapping from label to indices
    label_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, target, _ = dataset[idx]
        unique_labels = target.labels.tolist()  # pyright: ignore [reportAttributeAccessIssue]

        for label in unique_labels:
            label_to_indices[label].append(idx)

    # Determine the number of samples per label based on their proportion in the dataset
    total_samples = sum(len(indices) for indices in label_to_indices.values())
    selected_indices = []

    for indices in label_to_indices.values():
        num_samples = max(1, int((len(indices) / total_samples) * subset_size))
        selected_indices.extend(random.sample(indices, min(num_samples, len(indices))))

    # If we have fewer samples than required (due to rounding), randomly add more from the available pool
    if len(selected_indices) < subset_size:
        remaining_indices = list(set(range(len(dataset))) - set(selected_indices))
        additional_samples = subset_size - len(selected_indices)
        selected_indices.extend(random.sample(remaining_indices, additional_samples))

    # Create a new dataset object as a subset
    subset_dataset: VisDroneObjectDetectionDataset = Subset(dataset, selected_indices)  # pyright: ignore [reportAssignmentType, reportArgumentType]

    return subset_dataset
