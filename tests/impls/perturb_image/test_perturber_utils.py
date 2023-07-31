import numpy as np
from typing import Callable, Optional


def perturber_assertions(
    perturb: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    expected: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Test the blanket assertions for perturbers that
    1) Input should remain unchanged
    2) Output should not share memory with input (e.g no clones, etc)
    3) Output should have the same shape as input
    4) Output should have the same dtype as input
    Additionally, if ``expected`` is provided
    5) Output should match expected

    :param perturb: Interface with which to generate the perturbation.
    :param image: Input image as numpy array.
    :param expected: (Optional) Expected return value of the perturbation.
    """
    shape = image.shape
    dtype = image.dtype
    copy = np.copy(image)

    out_image = perturb(image)
    assert np.array_equal(image, copy)
    assert not np.shares_memory(image, out_image)
    assert out_image.shape == shape
    assert out_image.dtype == dtype
    if expected is not None:
        assert np.array_equal(out_image, expected)

    return out_image
