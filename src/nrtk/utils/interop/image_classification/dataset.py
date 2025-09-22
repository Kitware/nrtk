"""Utilities for interoperability with MAITE IC dataset protocols."""

# Python imports
from __future__ import annotations

__all__ = ["HuggingFaceMaiteDataset", "create_data_subset"]

import math
import random
from collections.abc import Callable

# 3rd party imports
import numpy as np

# Local imports
from nrtk.utils._exceptions import NRTKXAITKHelperImportError
from nrtk.utils._import_guard import import_guard

PIL_available: bool = import_guard("PIL", NRTKXAITKHelperImportError)
maite_available: bool = import_guard(
    "maite",
    NRTKXAITKHelperImportError,
    ["protocols.image_classification"],
    ["Dataset", "DatumMetadataType", "InputType", "TargetType"],
)
datasets_available: bool = import_guard("datasets", NRTKXAITKHelperImportError)
nrtk_xaitk_helpers_available: bool = maite_available and PIL_available and datasets_available
from datasets import Dataset as HFDataset  # noqa: E402
from datasets import load_dataset  # noqa: E402
from maite.protocols import ModelMetadata  # noqa: E402
from maite.protocols.image_classification import Dataset, DatumMetadataType, InputType, TargetType  # noqa: E402
from PIL import Image  # noqa: E402


def create_data_subset(
    dataset_name: str,
    split: str = "test",
    fraction: float = 1.0,
) -> HFDataset:
    """Create a subset of a Hugging Face dataset with approximately the same label distribution.

    Args:
        dataset_name (str):
            Name of the Hugging Face dataset to load.
        split (str):
            The dataset split to use (e.g., "train", "test").
        fraction (float):
            Fraction of the dataset to sample, between 0.0 and 1.0.

    Returns:
        HFDataset: A subset of the dataset with approximately the same label distribution.
    """
    if not nrtk_xaitk_helpers_available:
        raise NRTKXAITKHelperImportError

    # Load the dataset
    ds = load_dataset(dataset_name, split=split)  # pyright: ignore [reportPossiblyUnboundVariable]

    # Count total examples and label frequencies
    total_examples: int = ds.num_rows  # type: ignore

    label_counts = {}
    for sample in ds:
        lbl = sample["label"]  # type: ignore
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # Calculate expected number of samples based on the fraction parameter
    expected_total_samples = math.floor(fraction * total_examples)

    sample_indices = []
    # For each label, sample based on the minimum number of samples
    min_samples_per_label = math.floor(fraction * expected_total_samples)
    for lbl in label_counts:
        # All indices in ds that have label == lbl
        lbl_indices = [i for i, ex in enumerate(ds) if ex["label"] == lbl]
        # How many to sample from this label
        desired_count_for_label = math.floor(fraction * min_samples_per_label)
        # Randomly sample these indices
        sampled_lbl_indices = random.sample(lbl_indices, min(desired_count_for_label, len(lbl_indices)))
        sample_indices.extend(sampled_lbl_indices)

    # If rounding down causes the total to be less than samples_per_label,
    # you can optionally add more examples from random labels until you reach the target
    current_len = len(sample_indices)
    if current_len < expected_total_samples:
        # Collect the remaining needed
        needed = expected_total_samples - current_len
        # Get all unused indices
        unused_indices = list(set(range(total_examples)) - set(sample_indices))
        # Randomly sample from the unused indices
        sample_indices.extend(random.sample(unused_indices, min(needed, len(unused_indices))))

    # Finally, select the subset
    return ds.select(sample_indices)  # type: ignore


class HuggingFaceMaiteDataset(Dataset):  # pyright:  ignore [reportGeneralTypeIssues]
    """Hugging Face dataset wrapper for MAITE image classification protocol."""

    def __init__(self, hf_dataset: HFDataset, dataset_name: str) -> None:
        """Wraps a Hugging Face dataset to comply with MAITE Dataset protocol.

        Args:
            hf_dataset (HFDataset):
                The HuggingFace (HF) dataset object.
            dataset_name (str):
                Name for the HF dataset (used in metadata).
        """
        if not self.is_usable():
            raise NRTKXAITKHelperImportError

        self.hf_dataset = hf_dataset

        # Extract class labels mapping (assumes dataset has 'label' column)
        self.index2label: Callable = hf_dataset.features["label"].int2str
        self.num_classes: int = len(set(hf_dataset["label"]))
        # Set up required metadata
        self.metadata: ModelMetadata = {
            "id": dataset_name,
            "index2label": {i: self.index2label(i) for i in range(self.num_classes)},
        }

    def __len__(self) -> int:
        """The input dataset size.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.hf_dataset)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[InputType, TargetType, DatumMetadataType]:  # pyright: ignore [reportInvalidTypeForm]
        """Retrieve an indexed sample from the dataset.

        Args:
            idx (int):
                Index of the sample.

        Returns:
            tuple[InputType, TargetType, DatumMetadataType]:
                Tuple comprising the image tensor, one-hot encoded target and metadata.
        """
        # Load data from dataset
        sample = self.hf_dataset[idx]

        # Convert image (assumes dataset stores images as PIL or arrays)
        image = sample["image"]
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to (C, H, W) format (assuming RGB input)
        if image.shape[-1] == 3:  # If shape is (H, W, C), transpose
            image = image.transpose(2, 0, 1)

        # Convert label to one-hot encoding
        label = sample["label"]
        target = np.zeros(self.num_classes, dtype=np.float32)
        target[label] = 1.0

        # Define datum metadata (assuming minimal metadata requirement)
        datum_metadata: DatumMetadataType = {"id": str(idx)}  # pyright: ignore [reportInvalidTypeForm]

        return image, target, datum_metadata

    @classmethod
    def is_usable(cls) -> bool:
        """Checks if the necessary dependencies for NRTK-XAITK workflow are available.

        Returns:
            bool: True if NRTK-XAITK helper utils are available; False otherwise.
        """
        return nrtk_xaitk_helpers_available
