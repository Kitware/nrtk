"""This module contains the input schema for NRTK perturber API."""

__all__ = ["NRTKPerturbInputSchema"]

from typing import Any

from pydantic import BaseModel


# pyright warns about inheritance from BaseModel which is ambiguous
class NRTKPerturbInputSchema(BaseModel):  # pyright: ignore [reportGeneralTypeIssues]
    """Input schema for NRTK perturber API."""

    # Header
    id: str
    name: str

    # Dataset Params
    dataset_dir: str
    label_file: str
    output_dir: str
    image_metadata: list[dict[str, Any]]

    # NRTK Perturber
    config_file: str

    class Config:
        arbitrary_types_allowed: bool = True
        schema_extra: dict[str, Any] = {
            "examples": [
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "path/to/dataset/dir",
                    "output_dir": "path/to/output/dir",
                    "label_file": "path/to/label_file",
                    "image_metadata": [{"id": idx, "gsd": idx} for idx in range(11)],
                    "is_factory": True,
                    "config": "path/to/config_file",
                },
            ],
        }

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
