from collections.abc import Hashable, Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from PIL import Image
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb.geometric.random_translation_perturber import RandomTranslationPerturber
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb.test_perturber_utils import bbox_perturber_assertions

rng = np.random.default_rng()


class TestRandomTranslationPerturber:
    @pytest.mark.parametrize(
        ("input_test_box", "expected"),
        [
            (
                [(AxisAlignedBoundingBox(min_vertex=(1, 0), max_vertex=(2, 1)), {"meta": 1})],
                (
                    np.array([[2, 3, 0], [5, 6, 0], [8, 9, 0]], dtype=np.uint8),
                    [(AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"meta": 1})],
                ),
            ),
            (
                [(AxisAlignedBoundingBox(min_vertex=(2, 0), max_vertex=(2, 1)), {"meta": 1})],
                (np.array([[2, 3, 0], [5, 6, 0], [8, 9, 0]], dtype=np.uint8), []),
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
        inst = RandomTranslationPerturber()
        out_img_1, out_boxes_1 = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=input_test_box,
            expected=expected,
        )

        # Test callable
        out_img_2, out_boxes_2 = bbox_perturber_assertions(
            perturb=RandomTranslationPerturber(),
            image=image,
            boxes=input_test_box,
            expected=expected,
        )
        assert np.array_equal(out_img_1, out_img_2)

        if out_boxes_1 is not None and out_boxes_2 is not None:
            for (box_1, meta_1), (box_2, meta_2) in zip(out_boxes_1, out_boxes_2, strict=False):
                assert box_1 == box_2
                assert meta_1 == meta_2

    @pytest.mark.parametrize(
        ("image"),
        [
            (rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)),
            (np.ones((256, 256, 3), dtype=np.float32)),
            (np.ones((256, 256, 3), dtype=np.float64)),
        ],
    )
    def test_default_seed_reproducibility(self, image: np.ndarray) -> None:
        """Ensure results are reproducible with default seed (no seed parameter provided)."""
        # Test perturb interface directly without providing seed (uses default=1)
        inst = RandomTranslationPerturber()
        out_image, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
        )
        inst = RandomTranslationPerturber()  # Create new instance without seed
        bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=(out_image, []),
        )
        # Test callable
        inst = RandomTranslationPerturber()
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
        inst = RandomTranslationPerturber(seed=seed)
        out_image, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
        )
        inst = RandomTranslationPerturber(seed=seed)  # Create new instances with same seed
        bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=(out_image, []),
        )
        # Test callable
        inst = RandomTranslationPerturber(seed=seed)
        bbox_perturber_assertions(
            perturb=inst,
            image=image,
            boxes=None,
            expected=(out_image, []),
        )

    @pytest.mark.parametrize(
        ("image", "max_translation_limit", "boxes", "expectation"),
        [
            (np.ones((256, 256, 3), dtype=np.float32), (100, 200), [], does_not_raise()),
            (
                np.ones((256, 256, 3), dtype=np.float32),
                (100, 200),
                [
                    (AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(50, 50)), {"meta": 1}),
                    (AxisAlignedBoundingBox(min_vertex=(200, 200), max_vertex=(256, 256)), {"meta": 2}),
                ],
                does_not_raise(),
            ),
            (np.ones((256, 256, 3), dtype=np.float32), (0, 0), [], does_not_raise()),
            (
                np.ones((256, 256, 3), dtype=np.float32),
                (0, 0),
                [
                    (AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(50, 50)), {"meta": 1}),
                    (AxisAlignedBoundingBox(min_vertex=(200, 200), max_vertex=(256, 256)), {"meta": 2}),
                ],
                does_not_raise(),
            ),
            (np.ones((256, 256, 3), dtype=np.float32), (256, 256), [], does_not_raise()),
            (
                np.ones((256, 256, 3), dtype=np.float32),
                (256, 256),
                [
                    (AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(50, 50)), {"meta": 1}),
                    (AxisAlignedBoundingBox(min_vertex=(200, 200), max_vertex=(256, 256)), {"meta": 2}),
                ],
                does_not_raise(),
            ),
            (
                np.ones((256, 256, 3), dtype=np.float32),
                (257, 100),
                [],
                pytest.raises(ValueError, match=r"Max translation limit should be less than or equal to \(256, 256\)"),
            ),
            (
                np.ones((512, 512, 3), dtype=np.float32),
                (100, 513),
                [],
                pytest.raises(ValueError, match=r"Max translation limit should be less than or equal to \(512, 512\)"),
            ),
        ],
    )
    def test_additional_params(
        self,
        image: np.ndarray,
        max_translation_limit: tuple[int, int],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
        expectation: AbstractContextManager,
    ) -> None:
        """Ensure the max translation limit image output is consistent."""
        inst = RandomTranslationPerturber()
        with expectation:
            _, _ = bbox_perturber_assertions(
                perturb=inst.perturb,
                image=image,
                boxes=boxes,
                expected=None,
                **{"max_translation_limit": max_translation_limit},
            )

    @pytest.mark.parametrize(
        ("max_translation_limit"),
        [None, (0, 0), (0, 1), (1, 0), (100, 100)],
    )
    def test_regression(self, psnr_tiff_snapshot: SnapshotAssertion, max_translation_limit: tuple[int, int]) -> None:
        """Regression testing results to detect API changes."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        inst = RandomTranslationPerturber()
        additional_params = dict()
        if max_translation_limit is not None:
            additional_params = {"max_translation_limit": max_translation_limit}
        out_img, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
            **additional_params,
        )
        psnr_tiff_snapshot.assert_match(out_img)
