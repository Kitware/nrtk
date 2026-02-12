"""This module contains the dataset schema for NRTK perturber API."""

__all__ = ["DatasetSchema"]

from typing import Any

from pydantic import BaseModel


class DatasetSchema(BaseModel):  # pyright: ignore [reportGeneralTypeIssues]
    """Dataset schema for NRTK perturber API."""

    root_dir: str
    label_file: str
    metadata_file: str

    class Config:
        arbitrary_types_allowed: bool = True
        schema_extra: dict[str, Any] = {
            "examples": [
                {
                    "root_dir": "path/to/root/dir",
                    "label_file": "path/from/root_dir/to/label/file",
                    "metadata_file": "path/from/root_dir/to/metadata/file",
                },
            ],
        }

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
