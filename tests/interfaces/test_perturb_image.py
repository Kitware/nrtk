from collections.abc import Hashable
from types import MethodType
from unittest.mock import MagicMock

import numpy as np
import pytest
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image import PerturbImage


@pytest.mark.core
@pytest.mark.parametrize(
    ("input_bboxes", "orig_shape", "new_shape", "expected_bboxes"),
    [
        (
            [(AxisAlignedBoundingBox(min_vertex=(10, 5), max_vertex=(20, 15)), {"test": 0.53})],
            (3, 4),
            (6, 8),  # double both axes
            [(AxisAlignedBoundingBox(min_vertex=(20, 10), max_vertex=(40, 30)), {"test": 0.53})],
        ),
        (
            [
                (AxisAlignedBoundingBox(min_vertex=(54, 21), max_vertex=(97, 112)), {"test": 0.23}),
                (AxisAlignedBoundingBox(min_vertex=(7, 621), max_vertex=(37, 767)), {"test": 0.97}),
            ],
            (100, 99),
            (25, 33),  # x factor of 1/3 and y factor of 1/4
            [
                (AxisAlignedBoundingBox(min_vertex=(18, 5.25), max_vertex=(32.333333, 28)), {"test": 0.23}),
                (AxisAlignedBoundingBox(min_vertex=(2.333333, 155.25), max_vertex=(12.333333, 191.75)), {"test": 0.97}),
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
    perturber._rescale_boxes = MethodType(PerturbImage._rescale_boxes, perturber)  # noqa: FKA100, RUF100
    out_bboxes = perturber._rescale_boxes(
        boxes=input_bboxes,
        orig_shape=orig_shape,
        new_shape=new_shape,
    )
    for (out_box, out_dict), (expeted_box, expected_dict) in zip(out_bboxes, expected_bboxes, strict=False):
        # score dicts should be unchanged
        assert out_dict == expected_dict
        # use np.allclose() because of float operations
        assert np.allclose(out_box.min_vertex, expeted_box.min_vertex)
        assert np.allclose(out_box.max_vertex, expeted_box.max_vertex)


@pytest.mark.core
@pytest.mark.parametrize(
    ("vertices", "expected_box"),
    [
        (
            ((1, 2), (5, 6), (-1, 4), (0, 9)),
            AxisAlignedBoundingBox(min_vertex=(-1, 2), max_vertex=(5, 9)),
        ),
        (
            ((-1, 0, 3), (1, -1, -3)),
            AxisAlignedBoundingBox(min_vertex=(-1, -1, -3), max_vertex=(1, 0, 3)),
        ),
    ],
)
def test_align_bboxes(
    vertices: tuple[tuple[int]],
    expected_box: AxisAlignedBoundingBox,
) -> None:
    perturber = MagicMock(spec=PerturbImage)
    # map mock object _align_box method to actual one
    perturber._align_box = MethodType(PerturbImage._align_box, perturber)  # noqa: FKA100, RUF100
    out_box = perturber._align_box(
        vertices,
    )
    assert np.allclose(out_box.min_vertex, expected_box.min_vertex)
    assert np.allclose(out_box.max_vertex, expected_box.max_vertex)
