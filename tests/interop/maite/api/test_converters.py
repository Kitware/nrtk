import os
import unittest.mock as mock
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from nrtk.interop._maite.api._converters import build_factory, load_COCOMAITE_dataset
from nrtk.interop._maite.api._nrtk_perturb_input_schema import NRTKPerturbInputSchema
from tests.interop.maite import (
    BAD_NRTK_CONFIG,
    DATASET_FOLDER,
    EMPTY_NRTK_CONFIG,
    LABEL_FILE,
    NRTK_PYBSM_CONFIG,
)

_EXPECTED_PYBSM_CONFIG: dict[str, Any] = {
    "perturber": "nrtk.impls.perturb_image.optical.PybsmPerturber",
    "theta_keys": ["f", "D", "p_x"],
    "thetas": [[0.014, 0.012], [0.001, 0.003], [0.00002]],
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
        "qe_wavelengths": [3e-07, 4e-07, 5e-07, 6e-07, 7e-07, 8e-07, 9e-07, 1e-06, 1.1e-06],
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
}


@pytest.mark.maite
@pytest.mark.tools
class TestAPIConversionFunctions:
    @mock.patch("nrtk.interop._maite.api._converters.from_config_dict")
    def test_build_factory(self, mock_from_config_dict: MagicMock) -> None:
        """Test if build_factory calls from_config_dict with the loaded config."""
        mock_factory = MagicMock()
        mock_factory.get_config.return_value = _EXPECTED_PYBSM_CONFIG
        mock_from_config_dict.return_value = mock_factory

        schema = NRTKPerturbInputSchema.model_validate(
            {
                "id": "0",
                "name": "Example",
                "dataset_dir": "",
                "label_file": "",
                "output_dir": "",
                "image_metadata": [],
                "config_file": str(NRTK_PYBSM_CONFIG),
            },
        )
        factory = build_factory(schema)
        np.testing.assert_equal(factory.get_config(), _EXPECTED_PYBSM_CONFIG)
        mock_from_config_dict.assert_called_once()

    @pytest.mark.parametrize(
        "data",
        [
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "",  # Not used in this test
                    "label_file": "",  # Not used in this test
                    "output_dir": "",  # Not used in this test
                    "image_metadata": [],  # Not used in this test
                    "config_file": "",
                }
            ),
        ],
    )
    def test_build_factory_no_config(self, data: dict[str, Any]) -> None:
        """Test if build_factory throws ."""
        schema = NRTKPerturbInputSchema.model_validate(data)
        with pytest.raises(FileNotFoundError):
            build_factory(schema)

    @pytest.mark.parametrize(
        "data",
        [
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "",  # Not used in this test
                    "label_file": "",  # Not used in this test
                    "output_dir": "",  # Not used in this test
                    "image_metadata": [],  # Not used in this test
                    "config_file": str(BAD_NRTK_CONFIG),
                }
            ),
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "",  # Not used in this test
                    "label_file": "",  # Not used in this test
                    "output_dir": "",  # Not used in this test
                    "image_metadata": [],  # Not used in this test
                    "config_file": str(EMPTY_NRTK_CONFIG),
                }
            ),
        ],
    )
    def test_build_factory_bad_config(self, data: dict[str, Any]) -> None:
        """Test if build_factory throws ."""
        schema = NRTKPerturbInputSchema.model_validate(data)
        with pytest.raises(ValueError, match=r"Config"):
            build_factory(schema)

    @pytest.mark.parametrize(
        "data",
        [
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": str(DATASET_FOLDER),
                    "label_file": str(LABEL_FILE),
                    "output_dir": "",  # Not used in this test
                    "image_metadata": [{"id": idx, "gsd": idx} for idx in range(11)],
                    "config_file": "",  # Not used in this test
                }
            ),
        ],
    )
    def test_load_COCOMAITE_dataset(self, data: dict[str, Any]) -> None:  # noqa: N802
        """Test if load_COCOMAITE_dataset returns the expected dataset."""
        schema = NRTKPerturbInputSchema.model_validate(data)
        dataset = load_COCOMAITE_dataset(schema)
        # Check all images metadata for gsd
        for i in range(len(dataset)):
            assert dict(dataset[i][2])["gsd"] == dict(data["image_metadata"][i])["gsd"]
        # Check number of image matches
        assert len(dataset) == len(os.listdir(os.path.join(data["dataset_dir"], "images")))

    @pytest.mark.parametrize(
        "data",
        [
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": str(DATASET_FOLDER),
                    "label_file": str(LABEL_FILE),
                    "output_dir": "",  # Not used in this test
                    "image_metadata": [{"gsd": idx} for idx in range(11)],
                    "config_file": "",  # Not used in this test
                }
            ),
        ],
    )
    def test_load_COCOMAITE_dataset_bad_metadata(self, data: dict[str, Any]) -> None:  # noqa: N802
        """Test that ValueError appropriately raised when bad metadata is provided."""
        schema = NRTKPerturbInputSchema.model_validate(data)

        with pytest.raises(ValueError, match=r"ID not present in image metadata. Is it a DatumMetadataType?"):
            _ = load_COCOMAITE_dataset(schema)
