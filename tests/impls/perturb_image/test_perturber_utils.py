import numpy as np
from typing import Any, Callable, Dict, Optional


def perturber_assertions(
    perturb: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
    image: np.ndarray,
    expected: Optional[np.ndarray] = None,
    additional_params: Dict[str, Any] = {}
) -> np.ndarray:
    """
    Test the blanket assertions for perturbers that
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

    out_image = perturb(image, additional_params)
    assert np.array_equal(image, copy)
    assert not np.shares_memory(image, out_image)
    assert out_image.dtype == dtype
    if expected is not None:
        assert np.array_equal(out_image, expected)

    return out_image


def pybsm_perturber_assertions(
    perturb: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
    image: np.ndarray,
    expected: Optional[np.ndarray] = None,
    random_seed: int = 0,
    additional_params: Dict[str, Any] = {}
) -> np.ndarray:
    """
    Test the blanket assertions for perturbers that
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
    np.random.seed(random_seed)
    out_image = perturb(image, additional_params)

    assert np.array_equal(image, copy)
    assert not np.shares_memory(image, out_image)
    if expected is not None:
        assert out_image.dtype == expected.dtype
        assert np.array_equal(out_image, expected)

    return out_image
