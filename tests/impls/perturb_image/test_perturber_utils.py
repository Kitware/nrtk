from __future__ import annotations

from typing import Any, Callable

import numpy as np


def perturber_assertions(
    perturb: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    image: np.ndarray,
    expected: np.ndarray | None = None,
    additional_params: dict[str, Any] | None = None,
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
    if additional_params is None:
        additional_params = dict()
    dtype = image.dtype
    copy = np.copy(image)

    out_image = perturb(image, additional_params)
    assert np.array_equal(image, copy)
    assert not np.shares_memory(image, out_image)
    assert out_image.dtype == dtype
    if expected is not None:
        assert np.array_equal(out_image, expected)

    return out_image


def pybsm_perturber_assertions(
    perturb: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    image: np.ndarray,
    expected: np.ndarray | None = None,
    random_seed: int = 0,
    additional_params: dict[str, Any] | None = None,
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
    if additional_params is None:
        additional_params = dict()
    copy = np.copy(image)
    np.random.seed(random_seed)
    out_image = perturb(image, additional_params)

    assert np.array_equal(image, copy)
    assert not np.shares_memory(image, out_image)
    if expected is not None:
        assert out_image.dtype == expected.dtype
        assert np.array_equal(out_image, expected)

    return out_image
