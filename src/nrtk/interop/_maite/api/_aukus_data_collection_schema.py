"""This module contains the collection schema for AUKUS API."""

from __future__ import annotations

__all__ = ["AukusDataCollectionSchema"]

from typing import Any

from pydantic import BaseModel


# pyright warns about inheritance from BaseSettings which is ambiguous
class AukusDataCollectionSchema(BaseModel):  # pyright: ignore [reportGeneralTypeIssues]
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
        super().__init__(**data)
