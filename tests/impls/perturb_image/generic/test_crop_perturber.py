from collections.abc import Hashable, Iterable

import numpy as np
import pytest
from smqtk_image_io import AxisAlignedBoundingBox

from nrtk.impls.perturb_image.generic.crop_perturber import CropPerturber
from tests.impls.perturb_image.test_perturber_utils import bbox_perturber_assertions

rng = np.random.default_rng()


class TestCropPerturber:
    @pytest.mark.parametrize(
        ("input_test_box", "expected"),
        [
            (
                [(AxisAlignedBoundingBox(min_vertex=(1, 0), max_vertex=(2, 1)), {"meta": 1})],
                (
                    np.array([[1, 2], [4, 5]], dtype=np.uint8),
                    [(AxisAlignedBoundingBox(min_vertex=(1, 0), max_vertex=(1, 1)), {"meta": 1})],
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
        inst = CropPerturber()
        out_img_1, out_boxes_1 = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=input_test_box,
            expected=expected,
            additional_params={"crop_size": (2, 2)},
        )

        # Test callable
        out_img_2, out_boxes_2 = bbox_perturber_assertions(
            perturb=CropPerturber(),
            image=image,
            boxes=input_test_box,
            expected=expected,
            additional_params={"crop_size": (2, 2)},
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
        inst = CropPerturber()
        out_image, _ = bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=None,
            additional_params={"crop_size": (20, 20)},
        )
        inst = CropPerturber()  # Create new instances to reset random seed
        bbox_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            boxes=None,
            expected=(out_image, []),
            additional_params={"crop_size": (20, 20)},
        )
        inst = CropPerturber()
        # Test callable
        bbox_perturber_assertions(
            perturb=inst,
            image=image,
            boxes=None,
            expected=(out_image, []),
            additional_params={"crop_size": (20, 20)},
        )
