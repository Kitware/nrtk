"""This module contains handle_aukus_post, which is the endpoint for AUKUS API requests."""

from __future__ import annotations

__all__ = ["handle_aukus_post"]

import copy
import os
from collections.abc import Callable
from pathlib import Path

import requests

from nrtk.interop.maite.api.aukus_schema import AukusDatasetSchema
from nrtk.interop.maite.api.schema import NrtkPerturbInputSchema
from nrtk.utils._exceptions import FastApiImportError, PydanticSettingsImportError
from nrtk.utils._import_guard import import_guard

fastapi_available: bool = import_guard("fastapi", FastApiImportError, ["encoders"])
pydantic_settings_available: bool = import_guard(
    "pydantic_settings",
    PydanticSettingsImportError,
    objects=["BaseSettings"],
)
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.encoders import jsonable_encoder  # noqa: E402
from pydantic_settings import BaseSettings  # noqa: E402


# pyright warns about inheritance from BaseSettings which is ambiguous
class Settings(BaseSettings):  # pyright: ignore [reportGeneralTypeIssues]
    """Dataclass for NRTK API settings."""

    NRTK_IP: str = "http://localhost:8888/"

    print("\nTo access the server, open this URL in a browser:")
    print("\thttp://localhost:8888/")
    print("To use a different URL in the browser, define the following environment variable:")
    print('\t`NRTK_IP="http://<hostname>:<port>/"`\n')


settings: Settings = Settings()


class _UnusableFastApi:
    def post(self, _: str) -> Callable:
        def wrapper(_: str) -> None:
            return None

        return wrapper


if fastapi_available:
    AUKUS_app: FastAPI | _UnusableFastApi = FastAPI()
else:
    AUKUS_app = _UnusableFastApi()


def _check_input(data: AukusDatasetSchema) -> None:
    """Check input data and raise HTTPException if needed."""
    if not fastapi_available:
        raise FastApiImportError
    if data.data_format != "COCO":
        raise HTTPException(status_code=400, detail="Labels provided in incorrect format.")

    if not settings.NRTK_IP:
        raise HTTPException(status_code=400, detail="Provide NRTK_IP in AUKUS_app.env.")
    # Read NRTK configuration file and add relevant data to internalJSON
    if not os.path.isfile(data.nrtk_config):
        raise HTTPException(status_code=400, detail="Provided NRTK config is not a valid file.")


@AUKUS_app.post("/")
def handle_aukus_post(data: AukusDatasetSchema) -> list[AukusDatasetSchema]:
    """Format AUKUS request data to NRTK API format and return NRTK API data in AUKUS format."""
    if not fastapi_available:
        raise FastApiImportError

    if not pydantic_settings_available:
        raise PydanticSettingsImportError

    _check_input(data)
    annotation_file = Path(data.uri) / data.labels[0]["iri"]

    nrtk_input = NrtkPerturbInputSchema(
        id=data.id,
        name=data.name,
        dataset_dir=data.uri,
        label_file=str(annotation_file),
        output_dir=data.output_dir,
        image_metadata=data.image_metadata,
        config_file=data.nrtk_config,
    )

    # Call 'handle_post' function with processed data and get the result
    out = requests.post(settings.NRTK_IP, json=jsonable_encoder(nrtk_input), timeout=3600).json()

    # Process the result and construct return JSONs
    return_jsons = []
    for i in range(len(out["datasets"])):
        dataset = out["datasets"][i]
        dataset_json = copy.deepcopy(data)
        dataset_json.uri = dataset["root_dir"]
        if dataset_json.labels:
            dataset_json.labels = [
                {
                    "name": f"{dataset_json.labels[0]['name']}_pertubation_{i}",
                    "iri": dataset["label_file"],
                    "objectCount": dataset_json.labels[0]["objectCount"],
                },
            ]
        return_jsons.append(dataset_json)

    return return_jsons
