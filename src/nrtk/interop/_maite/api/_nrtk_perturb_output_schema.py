"""This module contains the output schema for NRTK perturber API."""

__all__ = ["NRTKPerturbOutputSchema"]

from typing import Any

from pydantic import BaseModel

from nrtk.interop._maite.api._dataset_schema import DatasetSchema


class NRTKPerturbOutputSchema(BaseModel):  # pyright: ignore [reportGeneralTypeIssues]
    """Output schema for NRTK perturber API."""

    message: str
    datasets: list[DatasetSchema]

    class Config:
        arbitrary_types_allowed: bool = True
        schema_extra: dict[str, Any] = {
            "examples": [
                {
                    "message": "response message",
                    "datasets": [
                        {
                            "root_dir": "path/to/root/dir0",
                            "label_file": "path/from/root_dir/to/label/file",
                            "metadata_file": "path/from/root_dir/to/metadata/file",
                        },
                    ],
                },
            ],
        }

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
