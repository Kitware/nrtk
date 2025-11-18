from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox


def perturber_assertions(
    perturb: Callable[
        ...,
        tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None],
    ],
    image: np.ndarray,
    expected: None | np.ndarray = None,
    **additional_params: Any,
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
    :param additional_oarams: A dictionary containing perturber implementation-specific input param-values pairs.
    """
    dtype = image.dtype
    copy = np.copy(image)

    out_image, _ = perturb(image, None, **additional_params)
    assert np.array_equal(image, copy)
    assert not np.shares_memory(image, out_image)
    assert out_image.dtype == dtype
    if expected is not None:
        assert np.array_equal(out_image, expected)

    return out_image


def bbox_perturber_assertions(
    perturb: Callable[
        ...,
        tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None],
    ],
    image: np.ndarray,
    boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
    expected: tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]] | None = None,
    **additional_params: Any,
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
    :param additional_oarams: A dictionary containing perturber implementation-specific input param-values pairs.
    """
    dtype = image.dtype
    copy = np.copy(image)

    out_image, out_boxes = perturb(image, boxes, **additional_params)
    assert np.array_equal(image, copy)
    assert not np.shares_memory(image, out_image)
    assert out_image.dtype == dtype
    if expected is not None:
        assert np.array_equal(out_image, expected[0])
        assert out_boxes is not None
        for (expected_box, expected_meta), (out_box, out_meta) in zip(expected[1], out_boxes, strict=False):
            assert expected_box == out_box
            assert expected_meta == out_meta

    return out_image, out_boxes


def pybsm_perturber_assertions(
    perturb: Callable[
        ...,
        tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None],
    ],
    image: np.ndarray,
    expected: None | np.ndarray = None,
    tol: float = 1e-6,
    **additional_params: Any,
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

    out_image, _ = perturb(image, None, **additional_params)

    assert np.array_equal(image, copy)
    assert not np.shares_memory(image, out_image)
    if expected is not None:
        assert out_image.dtype == expected.dtype
        assert np.average(np.abs(out_image - expected)) < tol

    return out_image
