"""Utilities for interoperability with MAITE IC model protocols."""

__all__ = ["HuggingFaceMaiteModel"]

# Python imports
from collections.abc import Hashable, Sequence
from typing import Any

# 3rd party imports
import numpy as np

# Local imports
from nrtk.utils._exceptions import NRTKXAITKHelperImportError
from nrtk.utils._import_guard import import_guard

maite_available: bool = import_guard(
    "maite",
    NRTKXAITKHelperImportError,
    ["protocols.image_classification"],
    ["Model", "InputType", "TargetType"],
)
torch_available: bool = import_guard("torch", NRTKXAITKHelperImportError, fake_spec=True)
PIL_available: bool = import_guard("PIL", NRTKXAITKHelperImportError)
transformers_available: bool = import_guard("transformers", NRTKXAITKHelperImportError)
import torch  # noqa: E402
from maite.protocols import ModelMetadata  # noqa: E402
from maite.protocols.image_classification import (  # noqa: E402
    InputType,
    Model,
    TargetType,
)
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageClassification, AutoProcessor  # noqa: E402

nrtk_xaitk_helpers_available: bool = maite_available and torch_available and PIL_available and transformers_available


class HuggingFaceMaiteModel(Model):  # pyright: ignore [reportGeneralTypeIssues]
    """Hugging Face model wrapper that conforms to the MAITE ic.Model protocol."""

    def __init__(self, model_name: str, device: str) -> None:
        """Initialize a Hugging Face model wrapper that conforms to the MAITE Model protocol.

        Args:
            model_name (str):
                Name of the Hugging Face model (e.g., "facebook/deit-base-distilled-patch16-224")
            device (str):
                Device to run the model on, e.g., "cuda" or "cpu"
        Raises:
            ImportError:
                Helper functions for the NRTK-XAITK workflow must be installed.
                Please install via `nrtk[maite,Pillow,notebook-testing]`.
        """
        if not self.is_usable():
            raise NRTKXAITKHelperImportError
        self.model_name = model_name
        self.model: Any = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.feature_layer: torch.nn.Sequential = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.processor: Any = AutoProcessor.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode

        # Set up metadata required by MAITE
        self.metadata: ModelMetadata = {"id": model_name}

        # Move model to GPU if available
        self.device = device
        self.model.to(self.device)

    @property
    def id2label(self) -> dict[int, Hashable]:
        """Get the mapping from model label indices to label names.

        Returns:
            dict[int, Hashable]: Mapping from label index to label name
        """
        return self.model.config.id2label

    @property
    def model2dataset_mapping(self) -> dict[int, int]:
        """Get the mapping from model label indices to dataset label indices.

        Returns:
            dict[int, int]: Mapping from model label index to dataset label index
        """
        return {
            0: 3,  # Model's "AnnualCrop" -> Dataset's index 3
            1: 0,  # Model's "Forest" -> Dataset's index 0
            2: 5,  # Model's "HerbaceousVegetation" -> Dataset's index 5
            3: 2,  # Model's "Highway" -> Dataset's index 2
            4: 6,  # Model's "Industrial" -> Dataset's index 6
            5: 9,  # Model's "Pasture" -> Dataset's index 9
            6: 8,  # Model's "PermanentCrop" -> Dataset's index 8
            7: 7,  # Model's "Residential" -> Dataset's index 7
            8: 1,  # Model's "River" -> Dataset's index 1
            9: 4,  # Model's "SeaLake" -> Dataset's index 4
        }

    @staticmethod
    def _remap_model_output(
        model_probs: Sequence[np.ndarray],
        model_to_dataset_mapping: dict[int, int],
    ) -> Sequence[TargetType]:  # pyright: ignore [reportInvalidTypeForm]
        """Reorders model output probabilities to match dataset label indices.

        Args:
            model_probs (Sequence[np.ndarray]):
                The model's per-class predicted probabilities.
            model_to_dataset_mapping (dict[int, int]):
                Mapping from model label indices to dataset label indices.

        Returns:
            Sequence[np.ndarray]:
                The reordered probabilities corresponding to the dataset label indices.
        """
        # Create an empty array to store the reordered probabilities
        output = []
        for probs in model_probs:
            dataset_probs = np.zeros_like(probs)

            # Reorder the probabilities according to the mapping
            for model_idx, dataset_idx in model_to_dataset_mapping.items():
                dataset_probs[dataset_idx] = probs[model_idx]
            output.append(dataset_probs)
        return output

    def __call__(
        self,
        batch: Sequence[InputType],  # pyright: ignore [reportInvalidTypeForm]
    ) -> Sequence[TargetType]:  # pyright: ignore [reportInvalidTypeForm]
        """Perform model inference on a batch of images.

        Args:
            batch (Sequence[np.ndarray]): A sequence of images with shape (C, H, W).

        Returns:
            Sequence[np.ndarray]: A sequence of arrays containing class probabilities.
        """
        # Convert batch to PIL images (assuming input format is ArrayLike)
        pil_images = [Image.fromarray(np.asarray(img).astype(np.uint8).transpose(1, 2, 0)) for img in batch]

        # Preprocess images
        inputs = self.processor(images=pil_images, return_tensors="pt")

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

        return self._remap_model_output(
            [probs for probs in probabilities],
            self.model2dataset_mapping,
        )  # Returning probabilities for each image

    def extract_features(
        self,
        batch: Sequence[InputType],  # pyright: ignore [reportInvalidTypeForm]
    ) -> TargetType:  # pyright: ignore [reportInvalidTypeForm]
        """Extract features from an input image.

        Args:
            batch (Sequence[np.ndarray]): A sequence of images with shape (C, H, W).

        Returns:
            torch.Tensor: Feature vector as a torch tensor
        """
        # Convert batch to PIL images (assuming input format is ArrayLike)
        pil_images = [Image.fromarray(np.asarray(img).astype(np.uint8).transpose(1, 2, 0)) for img in batch]
        # Preprocess image
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        return torch.tensor(
            np.array(
                self._remap_model_output(
                    [logit for logit in logits.cpu().numpy()],
                    self.model2dataset_mapping,
                ),
            ),
        )

    @classmethod
    def is_usable(cls) -> bool:
        """Checks if the necessary dependencies for NRTK-XAITK workflow are available.

        Returns:
            bool: True if NRTK-XAITK helper utils are available; False otherwise.
        """
        return nrtk_xaitk_helpers_available
