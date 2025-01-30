"""This module contains schemas for AUKUS API"""

from typing import Any, Optional

from pydantic import BaseModel


class AukusdataCollectionSchema(BaseModel):
    """Collection schema for AUKUS API"""

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
    local_region: Optional[str] = None
    collection_date_time: Optional[str] = None
    data_entries: Optional[int] = None
    source: Optional[dict[str, str]] = None
    data_formats: Optional[list[dict[str, Any]]] = None


class AukusDatasetSchema(BaseModel):
    """Dataset schema for AUKUS API"""

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
    tags: Optional[list[str]] = None
