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

from nrtk.impls.perturb_image.optical.defocus_otf_perturber import DefocusOTFPerturber
from nrtk.utils._exceptions import PyBSMImportError
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.test_perturber_utils import bbox_perturber_assertions, pybsm_perturber_assertions
from tests.utils.test_pybsm import create_sample_sensor_and_scenario


@pytest.mark.skipif(not DefocusOTFPerturber.is_usable(), reason=str(PyBSMImportError()))
class TestDefocusOTFPerturber:
    def test_interp_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        img_gsd = 3.19 / 160.0
        sensor_and_scenario = create_sample_sensor_and_scenario()
        # Test perturb interface directly
        inst = DefocusOTFPerturber(interp=True, **sensor_and_scenario)
        inst2 = DefocusOTFPerturber(interp=False, **sensor_and_scenario)
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

    @pytest.mark.parametrize("w_x", [0.5, 1.5])
    @pytest.mark.parametrize("w_y", [0.5, 1.5])
    @pytest.mark.parametrize("interp", [True, False])
    def test_provided_reproducibility(self, w_x: float, w_y: float, interp: bool) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        sensor_and_scenario = create_sample_sensor_and_scenario()
        sensor_and_scenario["w_x"] = w_x
        sensor_and_scenario["w_y"] = w_y
        inst = DefocusOTFPerturber(interp=interp, **sensor_and_scenario)
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
        inst = DefocusOTFPerturber()
        img_gsd = 3.19 / 160.0
        out_image = pybsm_perturber_assertions(perturb=inst.perturb, image=image, expected=None, img_gsd=img_gsd)
        pybsm_perturber_assertions(perturb=inst.perturb, image=image, expected=out_image, img_gsd=img_gsd)

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
    def test_provided_additional_params(
        self,
        additional_params: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Test variations of additional params."""
        sensor_and_scenario = create_sample_sensor_and_scenario()
        perturber = DefocusOTFPerturber(**sensor_and_scenario)
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        with expectation:
            _ = perturber(image=image, boxes=None, **additional_params)

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
        perturber = DefocusOTFPerturber()
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        with expectation:
            _ = perturber(image=image, **additional_params)

    @pytest.mark.parametrize("w_x", [0.5, 1.5])
    @pytest.mark.parametrize("w_y", [0.5, 1.5])
    def test_provided_wx_wy_reproducibility(
        self,
        w_x: float,
        w_y: float,
    ) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        sensor_and_scenario = create_sample_sensor_and_scenario()
        sensor_and_scenario["w_x"] = w_x
        sensor_and_scenario["w_y"] = w_y
        inst = DefocusOTFPerturber(**sensor_and_scenario)
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

    @pytest.mark.parametrize("w_x", [0.5])
    @pytest.mark.parametrize("w_y", [0.5])
    def test_wx_wy_configuration(
        self,
        w_x: float,
        w_y: float,
    ) -> None:
        """Test configuration stability."""
        inst = DefocusOTFPerturber(w_x=w_x, w_y=w_y)
        for i in configuration_test_helper(inst):
            assert i.w_x == w_x
            assert i.w_y == w_y

    def test_sensor_scenario_configuration(self) -> None:
        """Test configuration stability."""
        sensor_and_scenario = create_sample_sensor_and_scenario()
        inst = DefocusOTFPerturber(**sensor_and_scenario)
        for i in configuration_test_helper(inst):
            assert i.w_x == sensor_and_scenario["w_x"]
            assert i.w_y == sensor_and_scenario["w_y"]

    @pytest.mark.parametrize("w_x", [0.5])
    @pytest.mark.parametrize("w_y", [0.5])
    @pytest.mark.parametrize("interp", [True, False])
    def test_overall_configuration(
        self,
        w_x: float,
        w_y: float,
        interp: bool,
    ) -> None:
        """Test configuration stability."""
        sensor_and_scenario = create_sample_sensor_and_scenario()
        sensor_and_scenario["w_x"] = w_x
        sensor_and_scenario["w_y"] = w_y
        inst = DefocusOTFPerturber(interp=interp, **sensor_and_scenario)
        for i in configuration_test_helper(inst):
            assert i.w_x == w_x
            assert i.w_y == w_y
            assert i.interp == interp

    @pytest.mark.parametrize(
        ("use_sensor_scenario", "w_x", "w_y", "interp", "is_rgb"),
        [
            (False, None, None, None, False),
            (True, None, None, None, True),
            (False, None, None, None, True),
            (True, None, None, None, False),
            (True, 0.5, 1.5, False, True),
            (False, 0.5, 1.5, True, False),
            (True, 0.5, 1.5, False, False),
            (False, 0.5, 1.5, True, True),
        ],
    )
    def test_regression(
        self,
        psnr_tiff_snapshot: SnapshotAssertion,
        ssim_tiff_snapshot: SnapshotAssertion,
        use_sensor_scenario: bool,
        w_x: float | None,
        w_y: float | None,
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

        if w_x is not None:
            sensor_and_scenario["w_x"] = w_x
        if w_y is not None:
            sensor_and_scenario["w_y"] = w_y

        inst = DefocusOTFPerturber(interp=interp, **sensor_and_scenario)

        out_img = pybsm_perturber_assertions(perturb=inst, image=img, expected=None, **img_md)
        psnr_tiff_snapshot.assert_match(out_img)
        ssim_tiff_snapshot.assert_match(out_img)

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
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        img_gsd = 3.19 / 160.0
        sensor_and_scenario = create_sample_sensor_and_scenario()
        # Test perturb interface directly
        inst = DefocusOTFPerturber(interp=True, **sensor_and_scenario)
        bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
            boxes=boxes,
            img_gsd=img_gsd,
        )


@mock.patch.object(DefocusOTFPerturber, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False

    assert not DefocusOTFPerturber.is_usable()

    with pytest.raises(PyBSMImportError):
        DefocusOTFPerturber()
