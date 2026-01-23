from collections.abc import Generator
from pathlib import Path

import py  # type: ignore
import pytest
import responses
from starlette.testclient import TestClient

from nrtk.interop._maite.api.aukus_app import AUKUS_app, Settings
from nrtk.interop._maite.api.aukus_schema import AukusDatasetSchema
from nrtk.interop._maite.api.schema import DatasetSchema, NrtkPerturbOutputSchema
from nrtk.utils._exceptions import FastApiImportError
from nrtk.utils._import_guard import import_guard
from tests.interop.maite import DATASET_FOLDER, NRTK_PYBSM_CONFIG

is_usable: bool = import_guard(module_name="fastapi", exception=FastApiImportError, submodules=["encoders"])
from fastapi.encoders import jsonable_encoder  # noqa: E402


@pytest.fixture
def test_aukus_client() -> Generator:
    # Create a test client for the FastAPI application
    with TestClient(AUKUS_app) as client:  # pyright: ignore [reportArgumentType]
        yield client


@pytest.mark.skipif(not is_usable, reason="fastapi and/or pydantic not found. Please install via `nrtk[maite]`")
class TestAukusApp:
    @responses.activate
    def test_handle_aukus_post(self, test_aukus_client: TestClient, tmpdir: py.path.local) -> None:
        aukus_dataset = AukusDatasetSchema(
            doc_type="Dataset Metadata",
            doc_version="0.1",
            ism={
                "ownerProducer": ["USA"],
                "disseminationControls": ["U"],
                "classification": "U",
                "releasableTo": ["USA"],
            },
            last_update_time="2024-04-08T12:00:00.0Z",
            id="test_id",
            name="UnityExample",
            uri=str(DATASET_FOLDER),
            size="11",
            description="AUKUS Test",
            data_collections=[],
            data_format="COCO",
            nrtk_config=str(NRTK_PYBSM_CONFIG),
            image_metadata=[{"gsd": gsd} for gsd in range(11)],
            output_dir=str(tmpdir),
            labels=[
                {
                    "name": "AUKUS",
                    "iri": "annotations/COCO_annotations_VisDrone_TINY.json",
                    "objectCount": 100,
                },
            ],
            tags=["training", "synthetic"],
        )

        responses.add(
            method="POST",
            url=Settings().NRTK_IP,
            json=jsonable_encoder(
                NrtkPerturbOutputSchema(
                    message="Data received successfully",
                    datasets=[
                        DatasetSchema(
                            root_dir="test_path/perturb1",
                            label_file="annotations.json",
                            metadata_file="image_metadata.json",
                        ),
                    ]
                    * 4,
                ),
            ),
        )
        response = test_aukus_client.post("/", json=jsonable_encoder(aukus_dataset))

        # Check if the response status code is 200 OK
        assert response.status_code == 200
        base_path = Path("test_path/perturb1")
        label_file = base_path / "annotations.json"

        # Check if the response data contains the expected message
        for dataset in response.json():
            assert dataset["labels"][0]["iri"] == label_file.name
            assert dataset["uri"] == str(base_path)

    def test_bad_data_format_post(self, test_aukus_client: TestClient, tmpdir: py.path.local) -> None:
        aukus_dataset = AukusDatasetSchema(
            doc_type="Dataset Metadata",
            doc_version="0.1",
            ism={
                "ownerProducer": ["USA"],
                "disseminationControls": ["U"],
                "classification": "U",
                "releasableTo": ["USA"],
            },
            last_update_time="2024-04-08T12:00:00.0Z",
            id="test_id",
            name="UnityExample",
            uri=str(DATASET_FOLDER),
            size="11",
            description="AUKUS Test",
            data_collections=[],
            data_format="YOLO",
            nrtk_config=str(NRTK_PYBSM_CONFIG),
            image_metadata=[{"gsd": gsd} for gsd in range(11)],
            output_dir=str(tmpdir),
            labels=[
                {
                    "name": "AUKUS",
                    "iri": "annotations/COCO_annotations_VisDrone_TINY.json",
                    "objectCount": 100,
                },
            ],
            tags=["training", "synthetic"],
        )

        response = test_aukus_client.post("/", json=jsonable_encoder(aukus_dataset))

        assert response.status_code == 400
        assert response.json()["detail"] == "Labels provided in incorrect format."

    def test_bad_nrtk_config_post(self, test_aukus_client: TestClient, tmpdir: py.path.local) -> None:
        aukus_dataset = AukusDatasetSchema(
            doc_type="Dataset Metadata",
            doc_version="0.1",
            ism={
                "ownerProducer": ["USA"],
                "disseminationControls": ["U"],
                "classification": "U",
                "releasableTo": ["USA"],
            },
            last_update_time="2024-04-08T12:00:00.0Z",
            id="test_id",
            name="UnityExample",
            uri=str(DATASET_FOLDER),
            size="11",
            description="AUKUS Test",
            data_collections=[],
            data_format="COCO",
            nrtk_config="",
            image_metadata=[{"gsd": gsd} for gsd in range(11)],
            output_dir=str(tmpdir),
            labels=[
                {
                    "name": "AUKUS",
                    "iri": "annotations/COCO_annotations_VisDrone_TINY.json",
                    "objectCount": 100,
                },
            ],
            tags=["training", "synthetic"],
        )

        response = test_aukus_client.post("/", json=jsonable_encoder(aukus_dataset))

        assert response.status_code == 400
        assert response.json()["detail"] == "Provided NRTK config is not a valid file."
