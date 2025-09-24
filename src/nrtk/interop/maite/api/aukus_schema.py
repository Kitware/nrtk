"""This module contains schemas for AUKUS API."""

from __future__ import annotations

__all__ = ["AukusdataCollectionSchema", "AukusDatasetSchema"]

from typing import Any

from nrtk.utils._exceptions import PydanticImportError
from nrtk.utils._import_guard import import_guard

pydantic_available: bool = import_guard("pydantic", PydanticImportError)
from pydantic import BaseModel  # noqa: E402


# pyright warns about inheritance from BaseSettings which is ambiguous
class AukusdataCollectionSchema(BaseModel):  # pyright: ignore [reportGeneralTypeIssues]
    """Collection schema for AUKUS API."""

    # header params
    doc_type: str
    doc_version: str
    ism: dict[str, Any]
    last_update_time: str
    id: str
    name: str
    uri: str

    # Required Data Collection Params
    size: int
    description: str

    # Optional Data Collection Params
    local_region: str | None = None
    collection_date_time: str | None = None
    data_entries: int | None = None
    source: dict[str, str] | None = None
    data_formats: list[dict[str, Any]] | None = None

    def __init__(self, /, **data: Any) -> None:
        """Raise import error if pydantic isn't available."""
        if not pydantic_available:
            raise PydanticImportError

        super().__init__(**data)


# pyright warns about inheritance from BaseModel which is ambiguous
class AukusDatasetSchema(BaseModel):  # pyright: ignore [reportGeneralTypeIssues]
    """Dataset schema for AUKUS API."""

    # header params
    doc_type: str
    doc_version: str
    ism: dict[str, Any]
    last_update_time: str
    id: str
    name: str
    uri: str

    # Required Dataset Params
    size: str
    description: str
    data_collections: list[AukusdataCollectionSchema]
    data_format: str
    labels: list[dict[str, Any]]

    # NRTk specific param
    nrtk_config: str
    image_metadata: list[dict[str, Any]]
    output_dir: str

    # Optional Dataset Params
    tags: list[str] | None = None

    def __init__(self, /, **data: Any) -> None:
        """Raise import error if pydantic isn't available."""
        if not pydantic_available:
            raise PydanticImportError

        super().__init__(**data)
