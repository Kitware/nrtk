from collections.abc import Hashable, Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from PIL import Image
from smqtk_image_io import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.generic.translation_perturber import RandomTranslationPerturber
from tests.impls.perturb_image.test_perturber_utils import bbox_perturber_assertions
from tests.impls.test_pybsm_utils import TIFFImageSnapshotExtension

rng = np.random.default_rng()

INPUT_IMG_FILE_PATH = "./examples/pybsm/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"


class TestRandomTranslationPerturber:
    @pytest.mark.parametrize(
        ("input_test_box", "expected"),
        [
            (
                [(AxisAlignedBoundingBox(min_vertex=(1, 0), max_vertex=(2, 1)), {"meta": 1})],
                (
                    np.array([[2, 3, 0], [5, 6, 0], [8, 9, 0]], dtype=np.uint8),
                    [(AxisAlignedBoundingBox(min_vertex=(1, 0), max_vertex=(1, 1)), {"meta": 1})],
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

        for (box_1, meta_1), (box_2, meta_2) in zip(out_boxes_1, out_boxes_2):
            assert box_1 == box_2
            assert meta_1 == meta_2

    @pytest.mark.parametrize(
        ("image"),
        [
            (rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)),
            (np.ones((256, 256, 3), dtype=np.float32)),
            (np.ones((256, 256, 3), dtype=np.float64)),
        ],
    )
    def test_reproducibility(self, image: np.ndarray) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        inst = RandomTranslationPerturber()
        out_image, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
        )
        inst = RandomTranslationPerturber()  # Create new instances to reset random seed
        bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=(out_image, []),
        )
        inst = RandomTranslationPerturber()
        # Test callable
        bbox_perturber_assertions(
            perturb=inst,
            image=image,
            boxes=None,
            expected=(out_image, []),
        )

    @pytest.mark.parametrize(
        ("image", "max_translation_limit", "expectation"),
        [
            (np.ones((256, 256, 3), dtype=np.float32), (100, 200), does_not_raise()),
            (
                np.ones((256, 256, 3), dtype=np.float32),
                (257, 100),
                pytest.raises(ValueError, match=r"Max translation limit should be less than or equal to \(256, 256\)"),
            ),
            (
                np.ones((512, 512, 3), dtype=np.float32),
                (100, 513),
                pytest.raises(ValueError, match=r"Max translation limit should be less than or equal to \(512, 512\)"),
            ),
        ],
    )
    def test_additional_params(
        self,
        image: np.ndarray,
        max_translation_limit: tuple[int, int],
        expectation: AbstractContextManager,
    ) -> None:
        """Ensure the max translation limit image output is consistent"""
        inst = RandomTranslationPerturber()
        with expectation:
            _, _ = bbox_perturber_assertions(
                perturb=inst.perturb,
                image=image,
                boxes=None,
                expected=None,
                additional_params={"max_translation_limit": max_translation_limit},
            )

    def test_regression(self, snapshot: SnapshotAssertion) -> None:
        """Regression testing results to detect API changes."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        inst = RandomTranslationPerturber()
        out_img, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
        )
        assert TIFFImageSnapshotExtension.ndarray2bytes(out_img) == snapshot(extension_class=TIFFImageSnapshotExtension)
