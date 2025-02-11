import os
import unittest.mock as mock
from typing import Any

import numpy as np
import pytest
from smqtk_core.configuration import to_config_dict

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interop.maite.api.converters import build_factory
from nrtk.interop.maite.api.schema import NrtkPerturbInputSchema
from nrtk.utils._exceptions import KWCocoImportError
from tests.interop.maite import (
    BAD_NRTK_CONFIG,
    DATASET_FOLDER,
    EMPTY_NRTK_CONFIG,
    LABEL_FILE,
    NRTK_PYBSM_CONFIG,
)

try:
    from nrtk.interop.maite.api.converters import load_COCOJATIC_dataset

    is_usable = True
except ImportError:
    is_usable = False


@pytest.mark.skipif(
    not PybsmScenario.is_usable() or not PybsmSensor.is_usable(),
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
                    "theta_keys": ["f", "D", "p_x"],
                    "thetas": [[0.014, 0.012], [0.001, 0.003], [0.00002]],
                    "sensor": to_config_dict(
                        PybsmSensor(
                            name="L32511x",
                            D=0.004,
                            f=0.014285714285714287,
                            p_x=0.00002,
                            opt_trans_wavelengths=np.asarray([3.8e-7, 7.0e-7]),
                            eta=0.4,
                            int_time=0.03,
                            read_noise=25.0,
                            max_n=96000.0,
                            bit_depth=11.9,
                            max_well_fill=0.005,
                            da_x=0.0001,
                            da_y=0.0001,
                            qe_wavelengths=np.asarray(
                                [
                                    3.0e-7,
                                    4.0e-7,
                                    5.0e-7,
                                    6.0e-7,
                                    7.0e-7,
                                    8.0e-7,
                                    9.0e-7,
                                    1.0e-6,
                                    1.1e-6,
                                ],
                            ),
                            qe=np.asarray([0.05, 0.6, 0.75, 0.85, 0.85, 0.75, 0.5, 0.2, 0]),
                        ),
                    )
                    if PybsmSensor.is_usable()
                    else dict(),
                    "scenario": to_config_dict(
                        PybsmScenario(
                            name="niceday",
                            ihaze=2,
                            altitude=75,
                            ground_range=0,
                            cn2_at_1m=0,
                        ),
                    )
                    if PybsmScenario.is_usable()
                    else dict(),
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
            assert dataset[i][2]["gsd"] == data["image_metadata"][i]["gsd"]
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
        """Test that ImportError appropriately raised when imports missing"""
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
        """Test that ValueError appropriately raised when bad metadata is provided"""
        schema = NrtkPerturbInputSchema.model_validate(data)

        with pytest.raises(ValueError, match=r"ID not present in image metadata. Is it a DatumMetadataType?"):
            _ = load_COCOJATIC_dataset(schema)
