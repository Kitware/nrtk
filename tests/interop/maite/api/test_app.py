import os
import unittest.mock as mock
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import py  # type: ignore
import pytest
from fastapi.encoders import jsonable_encoder
from starlette.testclient import TestClient

from nrtk.interop._maite.api._app import app
from nrtk.interop._maite.api._nrtk_perturb_input_schema import NRTKPerturbInputSchema
from nrtk.interop._maite.datasets import (
    MAITEObjectDetectionDataset,
    MAITEObjectDetectionTarget,
)
from tests.interop.maite import BAD_NRTK_CONFIG, DATASET_FOLDER, LABEL_FILE, NRTK_PYBSM_CONFIG

random = np.random.default_rng()

TEST_RETURN_VALUE = [  # repeated test return value for 3 tests, saved to var to save space
    (
        "perturb1",
        MAITEObjectDetectionDataset(
            imgs=[random.integers(0, 255, size=(3, 3, 3), dtype=np.uint8)] * 11,
            dets=[
                MAITEObjectDetectionTarget(
                    boxes=random.random((2, 4)),
                    labels=random.random(2),
                    scores=random.random(2),
                ),
            ]
            * 11,
            datum_metadata=[{"id": idx} for idx in range(11)],
            dataset_id="dummy dataset",
        ),
    ),
]


@pytest.fixture
def test_client() -> Generator:
    # Create a test client for the FastAPI application
    with TestClient(app) as client:  # pyright: ignore [reportArgumentType]
        yield client


@pytest.mark.maite
@pytest.mark.tools
class TestApp:
    @mock.patch("nrtk.interop._maite.api._app.build_factory")
    @mock.patch("nrtk.interop._maite.api._app.nrtk_perturber", return_value=TEST_RETURN_VALUE)
    def test_handle_post_pybsm(
        self,
        nrtk_perturber_patch: MagicMock,
        build_factory_patch: MagicMock,  # noqa: ARG002
        test_client: TestClient,
        tmpdir: py.path.local,
    ) -> None:
        """Check for an appropriate response to a "good" request."""
        # Test data to be sent in the POST request
        test_data = NRTKPerturbInputSchema(
            id="0",
            name="Example",
            dataset_dir=str(DATASET_FOLDER),
            label_file=str(LABEL_FILE),
            output_dir=str(tmpdir),
            image_metadata=[{"id": idx, "gsd": idx} for idx in range(11)],
            config_file=str(NRTK_PYBSM_CONFIG),
        )

        # Send a POST request to the API endpoint
        response = test_client.post("/", json=jsonable_encoder(test_data))

        # Confirm mocked nrtk_perturber was called with the correct arguments
        kwargs = nrtk_perturber_patch.call_args.kwargs
        assert len(kwargs["maite_dataset"]) == 11

        # Check if the response status code is 200 OK
        assert response.status_code == 200

        # Check if the response data contains the expected message
        assert response.json()["message"] == "Data received successfully"

        # Check if the response data contains the processed data
        base_path = Path(tmpdir) / "perturb1"
        image_dir = base_path / "images"
        label_file = base_path / "annotations.json"
        metadata_file = base_path / "image_metadata.json"
        assert response.json()["datasets"] == [
            {
                "root_dir": str(base_path),
                "label_file": label_file.name,
                "metadata_file": metadata_file.name,
            },
        ]
        assert image_dir.is_dir()
        assert label_file.is_file()
        assert metadata_file.is_file()
        # Check that the correct number of images are in the dir
        assert len(os.listdir(os.path.join(str(image_dir)))) == 11

    @mock.patch("nrtk.interop._maite.api._app.build_factory")
    def test_bad_gsd_post(self, build_factory_patch: MagicMock, test_client: TestClient, tmpdir: py.path.local) -> None:  # noqa: ARG002
        """Test that an error response is appropriately propagated to the user."""
        test_data = NRTKPerturbInputSchema(
            id="0",
            name="Example",
            dataset_dir=str(DATASET_FOLDER),
            label_file=str(LABEL_FILE),
            output_dir=str(tmpdir),
            image_metadata=[{"id": idx, "gsd": idx} for idx in range(3)],  # incorrect number of gsds
            config_file=str(NRTK_PYBSM_CONFIG),
        )

        # Send a POST request to the API endpoint
        response = test_client.post("/", json=jsonable_encoder(test_data))

        # Check response status code
        assert response.status_code == 400

        # Check that we got the correct error message
        assert response.json()["detail"] == "Image metadata length mismatch, metadata needed for every image."

    def test_no_config_post(self, test_client: TestClient, tmpdir: py.path.local) -> None:
        """Test that an error response is appropriately propagated to the user."""
        test_data = NRTKPerturbInputSchema(
            id="0",
            name="Example",
            dataset_dir=str(DATASET_FOLDER),
            label_file=str(LABEL_FILE),
            output_dir=str(tmpdir),
            image_metadata=[{"id": idx, "gsd": idx} for idx in range(11)],
            config_file="/bad/path/",
        )

        # Send a POST request to the API endpoint
        response = test_client.post("/", json=jsonable_encoder(test_data))

        # Check response status code
        assert response.status_code == 400

        # Check that we got the correct error message
        assert response.json()["detail"] == "Config file at /bad/path/ was not found"

    def test_bad_config_post(self, test_client: TestClient, tmpdir: py.path.local) -> None:
        """Test that an error response is appropriately propagated to the user."""
        test_data = NRTKPerturbInputSchema(
            id="0",
            name="Example",
            dataset_dir=str(DATASET_FOLDER),
            label_file=str(LABEL_FILE),
            output_dir=str(tmpdir),
            image_metadata=[{"id": idx, "gsd": idx} for idx in range(11)],
            config_file=str(BAD_NRTK_CONFIG),
        )

        # Send a POST request to the API endpoint
        response = test_client.post("/", json=jsonable_encoder(test_data))

        # Check response status code
        assert response.status_code == 400

        # Check that we got the correct error message
        assert (
            response.json()["detail"]
            == "Configuration dictionary given does not have an implementation type specification."
        )
