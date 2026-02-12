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

from nrtk.impls.perturb_image.photometric.enhance import ColorPerturber
from tests.impls import INPUT_VISDRONE_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin
from tests.impls.perturb_image.perturber_utils import perturber_assertions

rng = np.random.default_rng()


@pytest.mark.pillow
class TestColorPerturber(PerturberTestsMixin):
    impl_class = ColorPerturber

    def test_consistency(self, ssim_tiff_snapshot: SnapshotAssertion) -> None:
        """Run on a real image to ensure output matches."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        factor = 0.2
        # Test callable
        out_img = perturber_assertions(
            perturb=ColorPerturber(factor=factor),
            image=image,
        )
        ssim_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        ("image", "factor"),
        [
            (rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), 0.5),
            (np.ones((256, 256, 3), dtype=np.float32), 1.3),
            (np.ones((256, 256, 3), dtype=np.float64), 0.2),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, factor: float) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        inst = ColorPerturber(factor=factor)
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("factor", [3.14, 0.5])
    def test_configuration(self, factor: float) -> None:
        """Test configuration stability."""
        inst = ColorPerturber(factor=factor)
        for i in configuration_test_helper(inst):
            assert i.factor == factor

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"factor": 5}, does_not_raise()),
            ({"factor": 0.0}, does_not_raise()),
            (
                {"factor": -1.2},
                pytest.raises(ValueError, match=r"ColorPerturber invalid factor"),
            ),
        ],
    )
    def test_configuration_bounds(
        self,
        kwargs: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        with expectation:
            ColorPerturber(**kwargs)

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
        inst = ColorPerturber(factor=0.5)
        _, out_boxes = inst.perturb(image=np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes
