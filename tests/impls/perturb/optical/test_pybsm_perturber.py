import unittest.mock as mock
from collections.abc import Hashable, Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb.optical.pybsm_perturber import PybsmPerturber
from nrtk.utils._exceptions import PyBSMImportError
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE
from tests.impls.perturb.test_perturber_utils import pybsm_perturber_assertions
from tests.impls.test_pybsm_utils import create_sample_sensor_and_scenario

np.random.seed(42)  # noqa: NPY002


@pytest.mark.skipif(not PybsmPerturber.is_usable(), reason=str(PyBSMImportError()))
class TestPyBSMPerturber:
    def test_regression(self, psnr_tiff_snapshot: SnapshotAssertion) -> None:
        """Regression testing results to detect API changes."""
        image = np.array(Image.open(INPUT_IMG_FILE))
        img_gsd = 3.19 / 160.0
        sensor_and_scenario = create_sample_sensor_and_scenario()

        sensor_and_scenario["ground_range"] = 10000

        # Test perturb interface directly
        inst = PybsmPerturber(**sensor_and_scenario)
        out_img = pybsm_perturber_assertions(
            perturb=inst,
            image=image,
            expected=None,
            img_gsd=img_gsd,
        )
        Image.fromarray(out_img).save("test.tiff")

        psnr_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        ("param_name", "param_value"),
        [
            ("ground_range", 10000),
            ("ground_range", 20000),
            ("altitude", 10000),
        ],
    )
    def test_default_seed_reproducibility(self, param_name: str, param_value: int) -> None:
        """Ensure results are reproducible with default seed (no seed parameter provided)."""
        # Test perturb interface directly without providing rng_seed (uses default=1)
        image = np.array(Image.open(INPUT_IMG_FILE))
        sensor_and_scenario = create_sample_sensor_and_scenario()
        sensor_and_scenario[param_name] = param_value
        # For type: ignore below, see https://github.com/microsoft/pyright/issues/5545#issuecomment-1644027877
        inst = PybsmPerturber(**sensor_and_scenario)
        img_gsd = 3.19 / 160.0
        out_image = pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
            img_gsd=img_gsd,
        )
        # Create another instance without seed and ensure perturbed image is the same
        inst2 = PybsmPerturber(**sensor_and_scenario)
        pybsm_perturber_assertions(
            perturb=inst2.perturb,
            image=image,
            expected=out_image,
            img_gsd=img_gsd,
        )

    @pytest.mark.parametrize(
        ("param_name", "param_value", "rng_seed"),
        [
            ("ground_range", 30000, 2),
            ("altitude", 10000, 2),
            ("ihaze", 2, 2),
        ],
    )
    def test_reproducibility(self, param_name: str, param_value: int, rng_seed: int) -> None:
        """Ensure results are reproducible when explicit seed is provided."""
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE))
        sensor_and_scenario = create_sample_sensor_and_scenario()
        sensor_and_scenario[param_name] = param_value
        # For type: ignore below, see https://github.com/microsoft/pyright/issues/5545#issuecomment-1644027877
        inst = PybsmPerturber(rng_seed=rng_seed, **sensor_and_scenario)
        img_gsd = 3.19 / 160.0
        out_image = pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
            img_gsd=img_gsd,
        )
        # Create another instance with same seed and ensure perturbed image is the same
        inst2 = PybsmPerturber(rng_seed=rng_seed, **sensor_and_scenario)
        pybsm_perturber_assertions(
            perturb=inst2.perturb,
            image=image,
            expected=out_image,
            img_gsd=img_gsd,
        )

    def test_configuration(self) -> None:  # noqa: C901
        """Test configuration stability."""
        sensor_and_scenario = create_sample_sensor_and_scenario()
        inst = PybsmPerturber(**sensor_and_scenario)
        for i in configuration_test_helper(inst):
            assert i.sensor is not None
            assert i.scenario is not None

            for param_name, param_value in sensor_and_scenario.items():
                if hasattr(i.sensor, param_name):
                    if type(param_value) is np.ndarray:
                        assert np.allclose(i.sensor.__getattribute__(param_name), param_value)
                    else:
                        assert i.sensor.__getattribute__(param_name) == param_value
                if hasattr(i.scenario, param_name):
                    if type(param_value) is np.ndarray:
                        assert np.allclose(i.scenario.__getattribute__(param_name), param_value)
                    else:
                        assert i.scenario.__getattribute__(param_name) == param_value

            assert np.array_equal(i._reflectance_range, inst._reflectance_range)

    @pytest.mark.parametrize(
        ("reflectance_range", "expectation"),
        [
            (np.array([0.05, 0.5]), does_not_raise()),
            (np.array([0.01, 0.5]), does_not_raise()),
            (
                np.array([0.05]),
                pytest.raises(ValueError, match=r"Reflectance range array must have length of 2"),
            ),
            (
                np.array([0.5, 0.05]),
                pytest.raises(
                    ValueError,
                    match=r"Reflectance range array values must be strictly ascending",
                ),
            ),
        ],
    )
    def test_configuration_bounds(self, reflectance_range: np.ndarray, expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        with expectation:
            PybsmPerturber(reflectance_range=reflectance_range)

    @pytest.mark.parametrize(
        ("altitude", "expectation"),
        [
            (24500, does_not_raise()),
            (25000, does_not_raise()),
            (
                24499,
                pytest.raises(ValueError, match=r"Invalid altitude value"),
            ),
        ],
    )
    def test_altitude_bounds(self, altitude: float, expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        with expectation:
            PybsmPerturber(altitude=altitude)

    @pytest.mark.parametrize(
        ("additional_params", "expectation"),
        [
            ({"img_gsd": 3.19 / 160.0}, does_not_raise()),
            (
                {},
                pytest.raises(
                    ValueError,
                    match=r"'img_gsd' must be provided for this perturber",
                ),
            ),
        ],
    )
    def test_additional_params(self, additional_params: dict[str, Any], expectation: AbstractContextManager) -> None:
        """Test variations of additional params."""
        sensor_and_scenario = create_sample_sensor_and_scenario()
        perturber = PybsmPerturber(reflectance_range=np.array([0.05, 0.5]), **sensor_and_scenario)
        image = np.array(Image.open(INPUT_IMG_FILE))
        with expectation:
            _ = perturber(image, **additional_params)

    @pytest.mark.parametrize(
        "boxes",
        [
            None,
            [(AxisAlignedBoundingBox((0, 0), (123, 236)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox((132, 222), (444, 352)), {"test": 0.7}),
                (AxisAlignedBoundingBox((231, 111), (333, 212)), {"test2": 1.0}),
            ],
        ],
    )
    def test_perturb_with_boxes(
        self,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test that bounding boxes scale as expected during perturb."""
        image = np.array(Image.open(INPUT_IMG_FILE))
        sensor_and_scenario = create_sample_sensor_and_scenario()
        inst = PybsmPerturber(**sensor_and_scenario)
        _, out_boxes = inst.perturb(image, boxes=boxes, img_gsd=(3.19 / 160))
        assert out_boxes == snapshot

    @mock.patch.object(PybsmPerturber, "is_usable")
    def test_missing_deps(self, mock_is_usable: MagicMock) -> None:
        """Test that an exception is raised when required dependencies are not installed."""
        mock_is_usable.return_value = False
        assert not PybsmPerturber.is_usable()
        with pytest.raises(PyBSMImportError):
            PybsmPerturber()
