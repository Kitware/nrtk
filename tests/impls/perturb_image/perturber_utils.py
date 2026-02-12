from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from copy import deepcopy
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from tests.utils import deep_equals


def _create_test_boxes() -> Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]:
    return [(AxisAlignedBoundingBox(min_vertex=(1, 1), max_vertex=(2, 2)), {"test1": 1, "test2": 2.0})]


def _assert_no_box_memory_sharing(
    a: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
    b: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
) -> None:
    if a is None or b is None:
        return
    for box_a, meta_a in a:
        for box_b, meta_b in b:
            assert box_a is not box_b
            assert meta_a is not meta_b


def perturber_assertions(
    perturb: Callable[
        ...,
        tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None],
    ],
    image: np.ndarray,
    expected: None | np.ndarray = None,
    **kwargs: Any,
) -> np.ndarray:
    """Test several blanket assertions for perturbers.

    1) Input should remain unchanged
    2) Output should not share memory with input (e.g no clones, etc)
    3) Output should have the same dtype as input
    Additionally, if ``expected`` is provided
    4) Output should match expected

    :param perturb: Interface with which to generate the perturbation.
    :param image: Input image as numpy array.
    :param expected: (Optional) Expected return value of the perturbation.
    :param kwargs: A dictionary containing perturber implementation-specific input param-values pairs.
    """
    dtype = image.dtype
    copy = np.copy(image)

    boxes = _create_test_boxes()
    boxes_copy = deepcopy(boxes)

    out_image, out_boxes = perturb(image=image, boxes=boxes, **kwargs)
    assert np.array_equal(image, copy)
    assert deep_equals(a=boxes, b=boxes_copy)
    assert not np.shares_memory(image, out_image)
    assert out_image.dtype == dtype
    if expected is not None:
        assert np.array_equal(out_image, expected)
    _assert_no_box_memory_sharing(a=boxes, b=out_boxes)

    return out_image


def bbox_perturber_assertions(
    perturb: Callable[
        ...,
        tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None],
    ],
    image: np.ndarray,
    boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
    expected: tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
    """Test several blanket assertions for perturbers.

    1) Input should remain unchanged
    2) Output should not share memory with input (e.g no clones, etc)
    3) Output should have the same dtype as input
    Additionally, if ``expected`` is provided
    4) Output should match expected

    :param perturb: Interface with which to generate the perturbation.
    :param image: Input image as numpy array.
    :param expected: (Optional) Expected return value of the perturbation.
    :param kwargs: A dictionary containing perturber implementation-specific input param-values pairs.
    """
    dtype = image.dtype
    copy = np.copy(image)

    boxes_copy = deepcopy(boxes)

    out_image, out_boxes = perturb(image=image, boxes=boxes, **kwargs)
    assert np.array_equal(image, copy)
    assert deep_equals(a=boxes, b=boxes_copy)
    assert not np.shares_memory(image, out_image)
    assert out_image.dtype == dtype
    if expected is not None:
        assert np.array_equal(out_image, expected[0])
        assert out_boxes is not None
        for (expected_box, expected_meta), (out_box, out_meta) in zip(expected[1], out_boxes, strict=True):
            assert expected_box == out_box
            assert expected_meta == out_meta
    _assert_no_box_memory_sharing(a=boxes, b=out_boxes)

    return out_image, out_boxes


def pybsm_perturber_assertions(
    perturb: Callable[
        ...,
        tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None],
    ],
    image: np.ndarray,
    expected: None | np.ndarray = None,
    tol: float = 1e-6,
    **kwargs: Any,
) -> np.ndarray:
    """Test some blanket assertions for perturbers.

    1) Input should remain unchanged
    2) Output should not share memory with input (e.g no clones, etc)
    Additionally, if ``expected`` is provided
    3) Output should have the same dtype as expected
    4) Output should match expected

    :param perturb: Interface with which to generate the perturbation.
    :param image: Input image as numpy array.
    :param expected: (Optional) Expected return value of the perturbation.
    """
    copy = np.copy(image)

    boxes = _create_test_boxes()
    boxes_copy = deepcopy(boxes)

    out_image, out_boxes = perturb(image=image, boxes=boxes, **kwargs)

    assert np.array_equal(image, copy)
    assert deep_equals(a=boxes, b=boxes_copy)
    assert not np.shares_memory(image, out_image)
    if expected is not None:
        assert out_image.dtype == expected.dtype
        assert out_image.shape == expected.shape
        assert np.average(np.abs(out_image - expected)) < tol
    _assert_no_box_memory_sharing(a=boxes, b=out_boxes)

    return out_image
