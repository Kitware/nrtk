import copy
from typing import Any, Callable, Dict, Optional

import numpy as np


def image_metric_assertions(
    computation: Callable[
        [np.ndarray, Optional[np.ndarray], Optional[Dict[str, Any]]], float
    ],
    img_1: np.ndarray,
    img_2: Optional[np.ndarray] = None,
    additional_params: Optional[Dict[str, Any]] = None,
) -> float:
    """Test that the inputs are not modified while computing an image metric.

    :param computation: Interface to test the compute() function on
    :param img_1: Original input image in the shape (height, width, channels).
    :param img_2: (Optional) Second input image in the shape (height, width, channels).
    :param additional_params: (Optional) A dictionary containing implementation-specific input param-values pairs.
    """
    original_img_1 = copy.deepcopy(img_1)
    original_img_2 = copy.deepcopy(img_2) if img_2 is not None else None
    original_additional_params = (
        copy.deepcopy(additional_params) if additional_params is not None else None
    )

    metric_value = computation(img_1, img_2, additional_params)

    assert np.array_equal(original_img_1, img_1), "img_1 modified, data changed"

    assert (original_img_2 is None) == (
        img_2 is None
    ), "img_2 modified, data became None or no longer None"

    if original_img_2 is not None and img_2 is not None:
        assert np.array_equal(original_img_2, img_2), "img_2 modified, data changed"

    assert (original_additional_params is None) == (
        additional_params is None
    ), "additional_params modified, data became None or no longer None"

    if original_additional_params is not None and additional_params is not None:
        assert (
            original_additional_params == additional_params
        ), "additional_params modified, data changed"

    return metric_value
