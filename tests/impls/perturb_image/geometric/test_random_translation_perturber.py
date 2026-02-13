from collections.abc import Hashable, Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.geometric.random import RandomTranslationPerturber
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin
from tests.impls.perturb_image.perturber_utils import bbox_perturber_assertions

rng = np.random.default_rng()


@pytest.mark.core
class TestRandomTranslationPerturber(PerturberTestsMixin):
    impl_class = RandomTranslationPerturber

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
                (
                    np.array([[2, 3, 0], [5, 6, 0], [8, 9, 0]], dtype=np.uint8),
                    [(AxisAlignedBoundingBox(min_vertex=(1, 0), max_vertex=(1, 1)), {"meta": 1})],
                ),
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
        # Test perturb interface directly with fixed seed for deterministic test
        inst = RandomTranslationPerturber(seed=1)
        out_img_1, out_boxes_1 = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=input_test_box,
            expected=expected,
        )

        # Test callable
        out_img_2, out_boxes_2 = bbox_perturber_assertions(
            perturb=RandomTranslationPerturber(seed=1),
            image=image,
            boxes=input_test_box,
            expected=expected,
        )
        assert np.array_equal(out_img_1, out_img_2)

        if out_boxes_1 is not None and out_boxes_2 is not None:
            for (box_1, meta_1), (box_2, meta_2) in zip(out_boxes_1, out_boxes_2, strict=False):
                assert box_1 == box_2
                assert meta_1 == meta_2

    def test_non_deterministic_default(self) -> None:
        """Verify different results when seed=None (default)."""
        image = rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)

        # Create two instances with default seed=None
        inst1 = RandomTranslationPerturber()
        inst2 = RandomTranslationPerturber()
        out1, _ = inst1.perturb(image=image)
        out2, _ = inst2.perturb(image=image)
        # Results should (almost certainly) be different with non-deterministic default
        assert not np.array_equal(out1, out2)

    @pytest.mark.parametrize(
        ("image", "seed"),
        [
            (rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), 2),
            (np.ones((256, 256, 3), dtype=np.float32), 2),
            (np.ones((256, 256, 3), dtype=np.float64), 2),
        ],
    )
    def test_seed_reproducibility(self, image: np.ndarray, seed: int) -> None:
        """Verify same results with explicit seed."""
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

    def test_is_static(self) -> None:
        """Verify is_static resets RNG each call."""
        image = rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        # is_static=True ensures identical results on repeated calls with the same input
        inst = RandomTranslationPerturber(seed=42, is_static=True)
        out_image, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
        )
        # Second call should produce identical output due to is_static resetting RNG
        bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=(out_image, []),
        )

    def test_is_static_warning(self) -> None:
        """Verify warning when is_static=True with seed=None."""
        with pytest.warns(UserWarning, match="is_static=True has no effect when seed=None"):
            RandomTranslationPerturber(seed=None, is_static=True)

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
    def test_kwargs(
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
        # Use explicit seed for regression test stability
        inst = RandomTranslationPerturber(seed=1)
        kwargs = {}
        if max_translation_limit is not None:
            kwargs = {"max_translation_limit": max_translation_limit}
        out_img, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
            **kwargs,
        )
        psnr_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        ("seed", "is_static"),
        [
            (123, False),
            (42, True),
            (None, False),
        ],
    )
    def test_configuration(
        self,
        seed: int | None,
        is_static: bool,
    ) -> None:
        """Test configuration stability."""
        inst = RandomTranslationPerturber(
            seed=seed,
            is_static=is_static,
        )
        for i in configuration_test_helper(inst):
            assert i.seed == seed
            assert i.is_static == is_static
