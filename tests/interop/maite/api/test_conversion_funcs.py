import os
import unittest.mock as mock
from typing import Any

import numpy as np
import pytest

from nrtk.impls.perturb.optical.pybsm_perturber import PybsmPerturber
from nrtk.interop.maite.api.converters import build_factory
from nrtk.interop.maite.api.schema import NrtkPerturbInputSchema
from nrtk.utils._exceptions import KWCocoImportError, MaiteImportError
from nrtk.utils._import_guard import import_guard
from tests.interop.maite import (
    BAD_NRTK_CONFIG,
    DATASET_FOLDER,
    EMPTY_NRTK_CONFIG,
    LABEL_FILE,
    NRTK_PYBSM_CONFIG,
)

maite_available: bool = import_guard(module_name="maite", exception=MaiteImportError)
kwcoco_available: bool = import_guard(module_name="kwcoco", exception=KWCocoImportError)
is_usable = maite_available and kwcoco_available
from nrtk.interop.maite.api.converters import load_COCOJATIC_dataset  # noqa: E402


@pytest.mark.skipif(
    not PybsmPerturber.is_usable(),
    reason="pybsm not found. Please install `nrtk[pybsm]`.",
)
class TestAPIConversionFunctions:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "",  # Not used in this test
                    "label_file": "",  # Not used in this test
                    "output_dir": "",  # Not used in this test
                    "image_metadata": [],  # Not used in this test
                    "config_file": str(NRTK_PYBSM_CONFIG),
                },
                {
                    "perturber": PybsmPerturber.get_type_string(),
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
                },
            ),
        ],
    )
    def test_build_factory(self, data: dict[str, Any], expected: dict[str, Any]) -> None:
        """Test if _build_pybsm_factory returns the expected factory."""
        schema = NrtkPerturbInputSchema.model_validate(data)
        factory = build_factory(schema)
        np.testing.assert_equal(factory.get_config(), expected)

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
        schema = NrtkPerturbInputSchema.model_validate(data)
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
        schema = NrtkPerturbInputSchema.model_validate(data)
        with pytest.raises(ValueError, match=r"Config"):
            build_factory(schema)

    @pytest.mark.skipif(not is_usable, reason="Extra 'nrtk-jatic[tools]' not installed.")
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
    def test_load_COCOJATIC_dataset(self, data: dict[str, Any]) -> None:  # noqa: N802
        """Test if load_COCOJATIC_dataset returns the expected dataset."""
        schema = NrtkPerturbInputSchema.model_validate(data)
        dataset = load_COCOJATIC_dataset(schema)
        # Check all images metadata for gsd
        for i in range(len(dataset)):
            assert dict(dataset[i][2])["gsd"] == dict(data["image_metadata"][i])["gsd"]
        # Check number of image matches
        assert len(dataset) == len(os.listdir(os.path.join(data["dataset_dir"], "images")))

    @mock.patch("nrtk.interop.maite.api.converters.is_usable", False)
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
    def test_load_COCOJATIC_dataset_not_usable(self, data: dict[str, Any]) -> None:  # noqa: N802
        """Test that ImportError appropriately raised when imports missing."""
        schema = NrtkPerturbInputSchema.model_validate(data)

        with pytest.raises(KWCocoImportError):
            _ = load_COCOJATIC_dataset(schema)

    @pytest.mark.skipif(not is_usable, reason="Extra 'nrtk-jatic[tools]' not installed.")
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
    def test_load_COCOJATIC_dataset_bad_metadata(self, data: dict[str, Any]) -> None:  # noqa: N802
        """Test that ValueError appropriately raised when bad metadata is provided."""
        schema = NrtkPerturbInputSchema.model_validate(data)

        with pytest.raises(ValueError, match=r"ID not present in image metadata. Is it a DatumMetadataType?"):
            _ = load_COCOJATIC_dataset(schema)
