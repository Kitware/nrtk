from __future__ import annotations

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

from nrtk.impls.perturb_image.optical.jitter_otf_perturber import JitterOTFPerturber
from nrtk.utils._exceptions import PyBSMImportError
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.test_perturber_utils import pybsm_perturber_assertions
from tests.utils.test_pybsm import create_sample_sensor_and_scenario


@pytest.mark.skipif(not JitterOTFPerturber.is_usable(), reason=str(PyBSMImportError()))
class TestJitterOTFPerturber:
    def test_interp_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        img_gsd = 3.19 / 160.0
        sensor_and_scenario = create_sample_sensor_and_scenario()
        # Test perturb interface directly
        inst = JitterOTFPerturber(interp=True, **sensor_and_scenario)
        inst2 = JitterOTFPerturber(interp=True, **sensor_and_scenario)
        out_image = pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
            img_gsd=img_gsd,
        )

        pybsm_perturber_assertions(
            perturb=inst2.perturb,
            image=image,
            expected=out_image,
            img_gsd=img_gsd,
        )

    @pytest.mark.parametrize("s_x", [0.5, 1.5])
    @pytest.mark.parametrize("s_y", [0.5, 1.5])
    @pytest.mark.parametrize("interp", [True, False])
    def test_provided_reproducibility(self, s_x: float, s_y: float, interp: bool) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        sensor_and_scenario = create_sample_sensor_and_scenario()
        sensor_and_scenario["s_x"] = s_x
        sensor_and_scenario["s_y"] = s_y
        inst = JitterOTFPerturber(interp=interp, **sensor_and_scenario)
        img_gsd = 3.19 / 160.0
        out_image = pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
            img_gsd=img_gsd,
        )
        pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
            img_gsd=img_gsd,
        )

    def test_default_reproducibility(self) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        inst = JitterOTFPerturber()
        img_gsd = 3.19 / 160.0
        out_image = pybsm_perturber_assertions(perturb=inst.perturb, image=image, expected=None, img_gsd=img_gsd)
        pybsm_perturber_assertions(perturb=inst.perturb, image=image, expected=out_image, img_gsd=img_gsd)

    @pytest.mark.parametrize(
        ("additional_params", "expectation"),
        [
            ({"img_gsd": 3.19 / 160.0}, does_not_raise()),
            (
                {},
                pytest.raises(ValueError, match=r"'img_gsd' must be provided"),
            ),
        ],
    )
    def test_provided_additional_params(
        self,
        additional_params: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Test variations of additional params."""
        sensor_and_scenario = create_sample_sensor_and_scenario()
        perturber = JitterOTFPerturber(interp=True, **sensor_and_scenario)
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        with expectation:
            _ = perturber(image=image, **additional_params)

    @pytest.mark.parametrize(
        ("additional_params", "expectation"),
        [
            ({"img_gsd": 3.19 / 160.0}, does_not_raise()),
            (
                {},
                pytest.raises(ValueError, match=r"'img_gsd' must be provided for this perturber"),
            ),
        ],
    )
    def test_default_additional_params(
        self,
        additional_params: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Test variations of additional params."""
        perturber = JitterOTFPerturber()
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        with expectation:
            _ = perturber(image=image, **additional_params)

    @pytest.mark.parametrize("s_x", [0.5, 1.5])
    @pytest.mark.parametrize("s_y", [0.5, 1.5])
    def test_provided_sx_sy_reproducibility(
        self,
        s_x: float,
        s_y: float,
    ) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        sensor_and_scenario = create_sample_sensor_and_scenario()
        sensor_and_scenario["s_x"] = s_x
        sensor_and_scenario["s_y"] = s_y
        inst = JitterOTFPerturber(**sensor_and_scenario)
        img_gsd = 3.19 / 160.0
        out_image = pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
            img_gsd=img_gsd,
        )
        pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
            img_gsd=img_gsd,
        )

    @pytest.mark.parametrize("s_x", [0.5])
    @pytest.mark.parametrize("s_y", [0.5])
    def test_sx_sy_configuration(
        self,
        s_x: float,
        s_y: float,
    ) -> None:
        """Test configuration stability."""
        inst = JitterOTFPerturber(s_x=s_x, s_y=s_y)
        for i in configuration_test_helper(inst):
            assert i.s_x == s_x
            assert i.s_y == s_y

    def test_sensor_scenario_configuration(self) -> None:
        """Test configuration stability."""
        sensor_and_scenario = create_sample_sensor_and_scenario()
        inst = JitterOTFPerturber(**sensor_and_scenario)
        for i in configuration_test_helper(inst):
            assert i.s_x == sensor_and_scenario["s_x"]
            assert i.s_y == sensor_and_scenario["s_y"]

    @pytest.mark.parametrize("s_x", [0.5])
    @pytest.mark.parametrize("s_y", [0.5])
    @pytest.mark.parametrize("interp", [True, False])
    def test_overall_configuration(
        self,
        s_x: float,
        s_y: float,
        interp: bool,
    ) -> None:
        """Test configuration stability."""
        sensor_and_scenario = create_sample_sensor_and_scenario()
        sensor_and_scenario["s_x"] = s_x
        sensor_and_scenario["s_y"] = s_y
        inst = JitterOTFPerturber(interp=interp, **sensor_and_scenario)
        for i in configuration_test_helper(inst):
            assert i.interp == interp
            assert i.s_x == s_x
            assert i.s_y == s_y

    @pytest.mark.parametrize(
        ("use_sensor_scenario", "s_x", "s_y", "interp", "is_rgb"),
        [
            (False, None, None, None, True),
            (True, None, None, None, False),
            (False, None, None, None, False),
            (True, None, None, None, True),
            (True, 0.5, 0.5, False, True),
            (False, 0.5, 0.5, True, False),
            (True, 0.5, 0.5, False, False),
            (False, 0.5, 0.5, True, True),
        ],
    )
    def test_regression(
        self,
        psnr_tiff_snapshot: SnapshotAssertion,
        use_sensor_scenario: bool,
        s_x: float | None,
        s_y: float | None,
        interp: bool,
        is_rgb: bool,
    ) -> None:
        """Regression testing results to detect API changes."""
        img = np.array(Image.open(INPUT_IMG_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)
        img_md = {"img_gsd": 3.19 / 160.0}

        sensor_and_scenario = {}
        if use_sensor_scenario:
            sensor_and_scenario = create_sample_sensor_and_scenario()

        if s_x is not None:
            sensor_and_scenario["s_x"] = s_x
        if s_y is not None:
            sensor_and_scenario["s_y"] = s_y

        inst = JitterOTFPerturber(interp=interp, **sensor_and_scenario)

        out_img = pybsm_perturber_assertions(perturb=inst, image=img, expected=None, **img_md)
        psnr_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        "boxes",
        [
            None,
            [(AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"test": 0.0}),
                (AxisAlignedBoundingBox(min_vertex=(2, 2), max_vertex=(3, 3)), {"test2": 1.0}),
            ],
        ],
    )
    def test_perturb_with_boxes(self, boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]) -> None:
        """Test that bounding boxes do not change during perturb."""
        inst = JitterOTFPerturber()
        _, out_boxes = inst.perturb(image=np.ones((256, 256, 3)), boxes=boxes, img_gsd=(3.19 / 160))
        assert boxes == out_boxes


@mock.patch.object(JitterOTFPerturber, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not JitterOTFPerturber.is_usable()
    with pytest.raises(PyBSMImportError):
        JitterOTFPerturber()
