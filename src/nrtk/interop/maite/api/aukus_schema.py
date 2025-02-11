"""This module contains schemas for AUKUS API"""

from typing import Any, Optional

from nrtk.utils._exceptions import PydanticImportError

BaseModel: type = object
try:
    # TODO: Remove once mypy is dropped (no redef)  # noqa: FIX002
    from pydantic import BaseModel  # type: ignore

    pydantic_available = True
except ImportError:  # pragma: no cover
    pydantic_available = False


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

    def __init__(self, /, **data: Any) -> None:
        """Raise import error if pydantic isn't available"""
        if not pydantic_available:
            raise PydanticImportError

        super().__init__(**data)


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

    def __init__(self, /, **data: Any) -> None:
        """Raise import error if pydantic isn't available"""
        if not pydantic_available:
            raise PydanticImportError

        super().__init__(**data)
