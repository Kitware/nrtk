from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.geometric.random_crop_perturber import RandomCropPerturber
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.test_perturber_utils import bbox_perturber_assertions

rng = np.random.default_rng()


class TestRandomCropPerturber:
    @pytest.mark.parametrize(
        ("input_test_box", "expected"),
        [
            (
                [(AxisAlignedBoundingBox(min_vertex=(1, 0), max_vertex=(3, 1)), {"meta": 1})],
                (
                    np.array([[1, 2], [4, 5]], dtype=np.uint8),
                    [(AxisAlignedBoundingBox(min_vertex=(1, 0), max_vertex=(2, 1)), {"meta": 1})],
                ),
            ),
            (
                [(AxisAlignedBoundingBox(min_vertex=(2, 0), max_vertex=(2, 1)), {"meta": 1})],
                (np.array([[1, 2], [4, 5]], dtype=np.uint8), []),
            ),
        ],
    )
    def test_consistency(
        self,
        input_test_box: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
        expected: tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
    ) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        # Test perturb interface directly
        inst = RandomCropPerturber(crop_size=(2, 2))
        out_img_1, out_boxes_1 = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=input_test_box,
            expected=expected,
        )

        # Test callable
        out_img_2, out_boxes_2 = bbox_perturber_assertions(
            perturb=inst,
            image=image,
            boxes=input_test_box,
            expected=expected,
        )
        assert np.array_equal(out_img_1, out_img_2)

        if out_boxes_1 is not None and out_boxes_2 is not None:
            for (box_1, meta_1), (box_2, meta_2) in zip(out_boxes_1, out_boxes_2, strict=False):
                assert box_1 == box_2
                assert meta_1 == meta_2

    def test_default_seed_reproducibility(self) -> None:
        """Ensure results are reproducible with default seed (no seed parameter provided)."""
        image = rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)

        # Test perturb interface directly without providing seed (uses default=1)
        inst = RandomCropPerturber(crop_size=(20, 20))
        out_image, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
        )
        inst = RandomCropPerturber(crop_size=(20, 20))  # Create new instance without seed
        bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=(out_image, []),
        )
        # Test callable
        inst = RandomCropPerturber(crop_size=(20, 20))
        bbox_perturber_assertions(
            perturb=inst,
            image=image,
            boxes=None,
            expected=(out_image, []),
        )

    @pytest.mark.parametrize(
        ("image", "seed"),
        [
            (rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), 2),
            (np.ones((256, 256, 3), dtype=np.float32), 2),
            (np.ones((256, 256, 3), dtype=np.float64), 2),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, seed: int) -> None:
        """Ensure results are reproducible when explicit seed is provided."""
        # Test perturb interface directly
        inst = RandomCropPerturber(crop_size=(20, 20), seed=seed)
        out_image, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
        )
        inst = RandomCropPerturber(crop_size=(20, 20), seed=seed)  # Create new instances with same seed
        bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=(out_image, []),
        )
        inst = RandomCropPerturber(crop_size=(20, 20), seed=seed)
        # Test callable
        bbox_perturber_assertions(
            perturb=inst,
            image=image,
            boxes=None,
            expected=(out_image, []),
        )

    def test_identity_operation(self) -> None:
        """Test that the identity crop returns the original image."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        inst = RandomCropPerturber()  # Full image size as crop size
        out_image, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
        )
        bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=[],
            expected=(out_image, []),
        )

    def test_regression(self, psnr_tiff_snapshot: SnapshotAssertion) -> None:
        """Regression testing results to detect API changes."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        inst = RandomCropPerturber(crop_size=(20, 20))
        out_img, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
        )
        psnr_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        ("crop_size", "seed"),
        [
            ((2, 5), 123),
            ((20, 20), np.random.default_rng(123)),
        ],
    )
    def test_configuration(
        self,
        crop_size: tuple[int, int],
        seed: int | np.random.Generator,
    ) -> None:
        """Test configuration stability."""
        inst = RandomCropPerturber(
            crop_size=crop_size,
            seed=seed,
        )
        for i in configuration_test_helper(inst):
            assert i.crop_size == crop_size
            assert i.seed == seed
