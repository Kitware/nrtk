"""This module contains schemas for NRTK perturber API"""

from typing import Any

from nrtk.utils._exceptions import PydanticImportError

BaseModel: type = object
try:
    # TODO: Remove once mypy is dropped (no redef) # noqa: FIX002
    from pydantic import BaseModel  # type: ignore

    pydantic_available = True
except ImportError:  # pragma: no cover
    pydantic_available = False


class NrtkPerturbInputSchema(BaseModel):
    """Input schema for NRTK perturber API"""

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

    class Config:  # noqa: D106
        arbitrary_types_allowed = True
        schema_extra = {
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
        """Raise import error if pydantic isn't available"""
        if not pydantic_available:
            raise PydanticImportError

        super().__init__(**data)


class DatasetSchema(BaseModel):
    """Dataset schema for NRTK perturber API"""

    root_dir: str
    label_file: str
    metadata_file: str

    class Config:  # noqa: D106
        arbitrary_types_allowed = True
        schema_extra = {
            "examples": [
                {
                    "root_dir": "path/to/root/dir",
                    "label_file": "path/from/root_dir/to/label/file",
                    "metadata_file": "path/from/root_dir/to/metadata/file",
                },
            ],
        }

    def __init__(self, /, **data: Any) -> None:
        """Raise import error if pydantic isn't available"""
        if not pydantic_available:
            raise PydanticImportError

        super().__init__(**data)


class NrtkPerturbOutputSchema(BaseModel):
    """Output schema for NRTK perturber API"""

    message: str
    datasets: list[DatasetSchema]

    class Config:  # noqa: D106
        arbitrary_types_allowed = True
        schema_extra = {
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
        """Raise import error if pydantic isn't available"""
        if not pydantic_available:
            raise PydanticImportError

        super().__init__(**data)
