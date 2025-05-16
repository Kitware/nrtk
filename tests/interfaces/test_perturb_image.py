from collections.abc import Hashable
from types import MethodType
from unittest.mock import MagicMock

import numpy as np
import pytest
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image import PerturbImage


@pytest.mark.parametrize(
    ("input_bboxes", "orig_shape", "new_shape", "expected_bboxes"),
    [
        (
            [(AxisAlignedBoundingBox((10, 5), (20, 15)), {"test": 0.53})],
            (3, 4),
            (6, 8),  # double both axes
            [(AxisAlignedBoundingBox((20, 10), (40, 30)), {"test": 0.53})],
        ),
        (
            [
                (AxisAlignedBoundingBox((54, 21), (97, 112)), {"test": 0.23}),
                (AxisAlignedBoundingBox((7, 621), (37, 767)), {"test": 0.97}),
            ],
            (100, 99),
            (25, 33),  # x factor of 1/3 and y factor of 1/4
            [
                (AxisAlignedBoundingBox((18, 5.25), (32.333333, 28)), {"test": 0.23}),
                (AxisAlignedBoundingBox((2.333333, 155.25), (12.333333, 191.75)), {"test": 0.97}),
            ],
        ),
    ],
)
def test_rescale_boxes(
    input_bboxes: list[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
    orig_shape: tuple,
    new_shape: tuple,
    expected_bboxes: list[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
) -> None:
    perturber = MagicMock(spec=PerturbImage)
    # map mock object rescale method to actual one
    perturber._rescale_boxes = MethodType(PerturbImage._rescale_boxes, perturber)
    out_bboxes = perturber._rescale_boxes(
        input_bboxes,
        orig_shape,
        new_shape,
    )
    for (out_box, out_dict), (expeted_box, expected_dict) in zip(out_bboxes, expected_bboxes):
        # score dicts should be unchanged
        assert out_dict == expected_dict
        # use np.allclose() because of float operations
        assert np.allclose(out_box.min_vertex, expeted_box.min_vertex)
        assert np.allclose(out_box.max_vertex, expeted_box.max_vertex)
