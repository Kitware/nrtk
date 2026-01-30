import os
import unittest.mock as mock
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import py  # type: ignore
import pytest

from nrtk.interop._maite.api.app import app
from nrtk.interop._maite.api.schema import NrtkPerturbInputSchema
from nrtk.interop._maite.datasets.object_detection import (
    MAITEObjectDetectionDataset,
    MAITEObjectDetectionTarget,
)
from nrtk.utils._exceptions import FastApiImportError
from nrtk.utils._import_guard import import_guard, is_available
from tests.interop.maite import BAD_NRTK_CONFIG, DATASET_FOLDER, LABEL_FILE, NRTK_PYBSM_CONFIG

# Guard import - starlette.testclient requires httpx which may not be installed
httpx = pytest.importorskip("httpx")
from starlette.testclient import TestClient  # noqa: E402

app_deps_available: bool = import_guard(module_name="fastapi", exception=FastApiImportError, submodules=["encoders"])
from fastapi.encoders import jsonable_encoder  # noqa: E402

deps = ["kwcoco", "maite"]
is_usable = all(is_available(dep) for dep in deps)
random = np.random.default_rng()

if is_usable:
    # MAITE is required for MAITEObjectDetectionDataset
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
else:
    TEST_RETURN_VALUE = list()


@pytest.fixture
def test_client() -> Generator:
    # Create a test client for the FastAPI application
    with TestClient(app) as client:  # pyright: ignore [reportArgumentType]
        yield client


@pytest.mark.skipif(
    not app_deps_available,
    reason="fastapi and/or pydantic not found. Please install via `nrtk[maite]`",
)
class TestApp:
    @pytest.mark.skipif(not is_usable, reason="Extra 'nrtk[tools]' not installed.")
    @mock.patch("nrtk.interop._maite.api.app.nrtk_perturber", return_value=TEST_RETURN_VALUE)
    def test_handle_post_pybsm(self, patch: MagicMock, test_client: TestClient, tmpdir: py.path.local) -> None:
        """Check for an appropriate response to a "good" request."""
        # Test data to be sent in the POST request
        test_data = NrtkPerturbInputSchema(
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
        kwargs = patch.call_args.kwargs
        assert len(kwargs["maite_dataset"]) == 11

        factory_config = kwargs["perturber_factory"].get_config()

        np.testing.assert_equal(
            factory_config,
            {
                "perturber": "nrtk.impls.perturb_image.optical.pybsm_perturber.PybsmPerturber",
                "theta_keys": ["f", "D", "p_x"],
                "perturber_kwargs": {
                    "sensor_name": "L32511x",
                    "D": 0.004,
                    "f": 0.014285714285714287,
                    "p_x": 2e-05,
                    "opt_trans_wavelengths": [3.8e-07, 7e-07],
                    "optics_transmission": None,
                    "eta": 0.4,
                    "w_x": None,
                    "w_y": None,
                    "int_time": 0.03,
                    "dark_current": 0.0,
                    "read_noise": 25.0,
                    "max_n": 96000.0,
                    "bit_depth": 11.9,
                    "max_well_fill": 0.005,
                    "s_x": 0.0,
                    "s_y": 0.0,
                    "qe_wavelengths": [
                        3e-07,
                        4e-07,
                        5e-07,
                        6e-07,
                        7e-07,
                        8e-07,
                        9e-07,
                        1e-06,
                        1.1e-06,
                    ],
                    "qe": [0.05, 0.6, 0.75, 0.85, 0.85, 0.75, 0.5, 0.2, 0.0],
                    "scenario_name": "niceday",
                    "ihaze": 2,
                    "altitude": 75,
                    "ground_range": 0,
                    "aircraft_speed": 0.0,
                    "target_reflectance": 0.15,
                    "target_temperature": 295.0,
                    "background_reflectance": 0.07,
                    "background_temperature": 293.0,
                    "ha_wind_speed": 21.0,
                    "cn2_at_1m": 0,
                },
                "thetas": [[0.014, 0.012], [0.001, 0.003], [2e-05]],
            },
        )

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

    @pytest.mark.skipif(not is_usable, reason="Extra 'nrtk[tools]' not installed.")
    def test_bad_gsd_post(self, test_client: TestClient, tmpdir: py.path.local) -> None:
        """Test that an error response is appropriately propagated to the user."""
        test_data = NrtkPerturbInputSchema(
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
        test_data = NrtkPerturbInputSchema(
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
        test_data = NrtkPerturbInputSchema(
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

    @mock.patch("nrtk.interop._maite.api.app.fastapi_available", False)
    def test_missing_deps(self, test_client: TestClient, tmpdir: py.path.local) -> None:
        """Test that an exception is raised when required dependencies are not installed."""
        test_data = NrtkPerturbInputSchema(
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

        # Check response status code
        assert response.status_code == 400

        # Check that we got the correct error message
        assert response.json()["detail"] == str(FastApiImportError())
