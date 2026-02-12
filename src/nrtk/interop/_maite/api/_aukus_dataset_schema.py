"""This module contains the dataset schema for AUKUS API."""

from __future__ import annotations

__all__ = ["AukusDatasetSchema"]

from typing import Any

from pydantic import BaseModel

from nrtk.interop._maite.api._aukus_data_collection_schema import AukusDataCollectionSchema


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
    data_collections: list[AukusDataCollectionSchema]
    data_format: str
    labels: list[dict[str, Any]]

    # NRTk specific param
    nrtk_config: str
    image_metadata: list[dict[str, Any]]
    output_dir: str

    # Optional Dataset Params
    tags: list[str] | None = None

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
