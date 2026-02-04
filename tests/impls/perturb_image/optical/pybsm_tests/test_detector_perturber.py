from __future__ import annotations

from collections.abc import Hashable, Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.optical.otf import DetectorPerturber
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin
from tests.impls.perturb_image.test_perturber_utils import pybsm_perturber_assertions
from tests.utils.test_pybsm import create_sample_sensor_and_scenario


@pytest.mark.pybsm
class TestDetectorPerturber(PerturberTestsMixin):
    impl_class = DetectorPerturber

    def test_interp_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        img_gsd = 3.19 / 160.0
        sensor_and_scenario = create_sample_sensor_and_scenario()
        # Test perturb interface directly
        inst = DetectorPerturber(interp=True, **sensor_and_scenario)
        inst2 = DetectorPerturber(interp=True, **sensor_and_scenario)
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

    @pytest.mark.parametrize(
        ("use_sensor_scenario", "w_x", "w_y", "f", "interp"),
        [(False, None, None, None, None), (True, 3e-6, 20e-6, 30e-3, True), (True, 3e-6, 20e-6, 30e-3, False)],
    )
    def test_reproducibility(
        self,
        use_sensor_scenario: bool,
        w_x: float | None,
        w_y: float | None,
        f: float | None,
        interp: bool,
    ) -> None:
        """Ensure results are reproducible."""
        img = np.array(Image.open(INPUT_IMG_FILE_PATH))
        img_md = {"img_gsd": 3.19 / 160.0}

        sensor_and_scenario = {}
        if use_sensor_scenario:
            sensor_and_scenario = create_sample_sensor_and_scenario()

        if w_x is not None:
            sensor_and_scenario["w_x"] = w_x
        if w_y is not None:
            sensor_and_scenario["w_y"] = w_y
        if f is not None:
            sensor_and_scenario["f"] = f

        inst = DetectorPerturber(interp=interp, **sensor_and_scenario)

        out_img = pybsm_perturber_assertions(perturb=inst, image=img, expected=None, **img_md)

        pybsm_perturber_assertions(perturb=inst, image=img, expected=out_img, **img_md)

    @pytest.mark.parametrize(
        ("use_sensor_scenario", "kwargs", "expectation"),
        [
            (True, {"img_gsd": 3.19 / 160.0}, does_not_raise()),
            (
                True,
                dict(),
                pytest.raises(ValueError, match=r"'img_gsd' must be provided for this perturber"),
            ),
            (False, {"img_gsd": 3.19 / 160.0}, does_not_raise()),
        ],
    )
    def test_kwargs(
        self,
        use_sensor_scenario: bool,
        kwargs: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Test that exceptions are appropriately raised based on available metadata."""
        sensor_and_scenario = {}
        if use_sensor_scenario:
            sensor_and_scenario = create_sample_sensor_and_scenario()
        perturber = DetectorPerturber(**sensor_and_scenario)
        img = np.array(Image.open(INPUT_IMG_FILE_PATH))
        with expectation:
            _ = perturber.perturb(image=img, **kwargs)

    @pytest.mark.parametrize(
        ("use_sensor_scenario", "w_x", "w_y", "f", "interp"),
        [
            (False, None, None, None, None),
            (True, None, None, None, None),
            (True, 3e-6, 20e-6, 30e-3, False),
            (False, 3e-6, 20e-6, 30e-3, True),
        ],
    )
    def test_configuration(  # noqa: C901
        self,
        use_sensor_scenario: bool,
        w_x: float | None,
        w_y: float | None,
        f: float | None,
        interp: bool,
    ) -> None:
        """Test configuration stability."""
        sensor_and_scenario = {}
        if use_sensor_scenario:
            sensor_and_scenario = create_sample_sensor_and_scenario()

        if w_x is not None:
            sensor_and_scenario["w_x"] = w_x
        if w_y is not None:
            sensor_and_scenario["w_y"] = w_y
        if f is not None:
            sensor_and_scenario["f"] = f

        inst = DetectorPerturber(interp=interp, **sensor_and_scenario)
        for i in configuration_test_helper(inst):
            if w_x is not None:
                assert i.w_x == w_x
            elif use_sensor_scenario:
                assert i.w_x == sensor_and_scenario["w_x"]
            else:  # Default value
                assert i.w_x == 4e-6

            if w_y is not None:
                assert i.w_y == w_y
            elif use_sensor_scenario:
                assert i.w_y == sensor_and_scenario["w_y"]
            else:  # Default value
                assert i.w_y == 4e-6

            if f is not None:
                assert i.f == f
            elif use_sensor_scenario:
                assert i.f == sensor_and_scenario["f"]
            else:  # Default value
                assert i.f == 50e-3

    @pytest.mark.parametrize(
        ("use_sensor_scenario", "w_x", "w_y", "f", "interp", "is_rgb"),
        [
            (False, None, None, None, None, True),
            (True, None, None, None, None, False),
            (False, None, None, None, None, False),
            (True, None, None, None, None, True),
            (True, 3e-6, 20e-6, 30e-3, False, True),
            (False, 3e-6, 20e-6, 30e-3, True, False),
            (True, 3e-6, 20e-6, 30e-3, False, False),
            (False, 3e-6, 20e-6, 30e-3, True, True),
        ],
    )
    def test_regression(  # noqa: C901
        self,
        ssim_tiff_snapshot: SnapshotAssertion,
        use_sensor_scenario: bool,
        w_x: float | None,
        w_y: float | None,
        f: float | None,
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
            # For the small f override, set scenario to a shorter altiude/range
            # to avoid downsampling image to a single pixel
            if f is not None:
                sensor_and_scenario["altitude"] = 500
                sensor_and_scenario["ground_range"] = 100

        if w_x is not None:
            sensor_and_scenario["w_x"] = w_x
        if w_y is not None:
            sensor_and_scenario["w_y"] = w_y
        if f is not None:
            sensor_and_scenario["f"] = f

        inst = DetectorPerturber(interp=interp, **sensor_and_scenario)

        out_img = pybsm_perturber_assertions(perturb=inst, image=img, expected=None, **img_md)
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
        inst = DetectorPerturber()
        _, out_boxes = inst.perturb(image=np.ones((256, 256, 3)), boxes=boxes, img_gsd=(3.19 / 160))
        assert boxes == out_boxes
