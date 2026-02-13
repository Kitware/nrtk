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

from nrtk.impls.perturb_image.photometric.blur import AverageBlurPerturber
from tests.impls import INPUT_VISDRONE_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin
from tests.impls.perturb_image.perturber_utils import perturber_assertions

rng = np.random.default_rng()


@pytest.mark.opencv
class TestAverageBlurPerturber(PerturberTestsMixin):
    impl_class = AverageBlurPerturber

    def test_consistency(self, psnr_tiff_snapshot: SnapshotAssertion, ssim_tiff_snapshot: SnapshotAssertion) -> None:
        """Run on a real to ensure output matches precomputed results."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        ksize = 3

        # Test callable
        out_img = perturber_assertions(
            perturb=AverageBlurPerturber(ksize=ksize),
            image=image,
        )
        psnr_tiff_snapshot.assert_match(out_img)
        ssim_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        ("image", "ksize"),
        [
            (rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), 1),
            (np.ones((256, 256, 3), dtype=np.float32), 3),
            (np.ones((256, 256, 3), dtype=np.float64), 5),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, ksize: int) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        inst = AverageBlurPerturber(ksize=ksize)
        out_image = perturber_assertions(perturb=inst.perturb, image=image, expected=None)
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("ksize", [4, 6])
    def test_configuration(self, ksize: int) -> None:
        """Test configuration stability."""
        inst = AverageBlurPerturber(ksize=ksize)
        for i in configuration_test_helper(inst):
            assert i.ksize == ksize

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"ksize": 2}, does_not_raise()),
            ({"ksize": 1}, does_not_raise()),
            (
                {"ksize": 0},
                pytest.raises(ValueError, match=r"AverageBlurPerturber invalid ksize"),
            ),
        ],
    )
    def test_configuration_bounds(self, kwargs: dict[str, Any], expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        with expectation:
            AverageBlurPerturber(**kwargs)

    @pytest.mark.parametrize(
        ("image", "expectation"),
        [
            (np.ones((256, 256)), does_not_raise()),
            (np.ones((256, 256, 3)), does_not_raise()),
            (np.ones((256, 256, 4)), does_not_raise()),
            (
                np.ones((3, 256, 256)),
                pytest.raises(ValueError, match=r"Image is not in expected format"),
            ),
        ],
    )
    def test_perturb_bounds(self, image: np.ndarray, expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        inst = AverageBlurPerturber()
        with expectation:
            inst.perturb(image=image)

    @pytest.mark.parametrize(
        ("boxes"),
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
        inst = AverageBlurPerturber()
        _, out_boxes = inst.perturb(image=np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes
