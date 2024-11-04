"""
This module provides an implementation of the `ImageMetric` interface for calculating the
Signal-to-Noise Ratio (SNR) of an image. The main class, `SNRImageMetric`, computes the SNR
for a single input image, with options to control the axis and degrees of freedom used in the
calculation.

Classes:
    SNRImageMetric: Calculates the Signal-to-Noise Ratio (SNR) of an image using specified
    parameters for axis and degrees of freedom.

Functions:
    _signal_to_noise: A helper function for computing the SNR of an image based on its mean
    and standard deviation along a given axis and degrees of freedom.

Dependencies:
    - numpy for image processing and statistical calculations.

Usage:
    Instantiate `SNRImageMetric` and call the `compute` method with an image to obtain the
    SNR. Optional parameters for `axis` and `ddof` (degrees of freedom) can be specified
    in `additional_params`.

Example:
    snr_metric = SNRImageMetric()
    snr_value = snr_metric.compute(image_data, additional_params={"axis": 0, "ddof": 0})

Notes:
    - SNR calculations are only supported for single images; attempting to pass a second
      image to `compute` will raise an error.
    - The optional `axis` parameter controls the dimensions over which the mean and standard
      deviation are computed. This affects the SNR calculation but defaults to None, which
      includes all axes and channels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from nrtk.interfaces.image_metric import ImageMetric


class SNRImageMetric(ImageMetric):
    """Implementation of the ``ComputeImageMetrics`` interface to calculate the Signal to Noise Ratio."""

    @override
    def compute(  # noqa: C901
        self,
        img_1: np.ndarray,
        img_2: np.ndarray | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> float:
        """Given one image, compute the Signal to Noise ratio.

        :param img_1: Original input image in the shape (height, width, channels).
        :param img_2: (Optional) Second input image in the shape (height, width, channels). Not allowed for SNR
        :param additional_params: (Optional) A dictionary containing implementation-specific input param-values pairs.

        SNR has the following optional parameters:
        - axis: valid values are None, 0, 1, (0,1). Default is 0. The axis over which to calculate the standard
        deviation during SNR calculation. Keeping this at None will calculate over both axes and the channels.
        - ddof: degrees of freedom for the standard deviation calculation. Default is 0. Max value is
        num_elements, the number of elements used in calculation. Must be non-negative.

        :return: Returns the signal to noise ratio for the input image.
        """
        if img_1.ndim != 3:
            raise ValueError("Incorrect number of dimensions on input image! Expected ndim == 3.")

        img_height, img_width, img_channels = img_1.shape
        if img_height <= 0:
            raise ValueError("Invalid image height! Image height is <= 0.")
        if img_width <= 0:
            raise ValueError("Invalid image width! Image width is <= 0.")
        if img_channels != 1 and img_channels != 3:
            raise ValueError("Invalid number of channels on input image! Expected 1 or 3.")

        if img_2 is not None:
            raise ValueError("Incorrect number of arguments. Computing SNR can only be done on a single image.")

        if not additional_params:
            additional_params = dict()

        # There are two optional params for SNR, axis and ddof, which both have default values of 0
        axis = additional_params.get("axis", 0)
        ddof = additional_params.get("ddof", 0)

        # checking axis
        valid_axis_parameters = [None, 0, 1, (0, 1)]
        if axis not in valid_axis_parameters:
            raise ValueError("Invalid axis parameter! Valid axis parameters are: None, 0, 1, (0,1).")

        # hard coding the default axis parameter to None for now
        # this will cause it to compute over both axis and the channels, if we do not have this then the
        # signal_to_noise call might return multiple values, which does not correspond with our interface
        # because our interface expects a singular float
        default_axis_parameter = None

        # because we hard code the default axis parameter, we also know the num_elements equation
        # this line should be removed when we no longer hard code the axis
        num_elements = img_height * img_width * img_channels

        # checking ddof
        if ddof < 0:
            raise ValueError("Invalid ddof value! ddof must be non-negative.")
        # ddof can be at most num_elements used for calculating the standard deviation
        # num_elements will vary based on the value for axis
        """
        if axis is None:
            num_elements = img_height * img_width * img_channels
        elif axis == 0:
            num_elements = img_width * img_channels
        elif axis == 1:
            num_elements = img_height * img_channels
        elif axis == (0, 1):
            num_elements = img_channels
        """
        if ddof > num_elements:
            raise ValueError(
                f"Invalid ddof value! ddof must be at most {num_elements}, depending on the value of axis.",
            )

        # Check that we are calling the signal_to_noise function correctly
        return _signal_to_noise(img=img_1, axis=default_axis_parameter, ddof=ddof)


def _signal_to_noise(
    img: np.ndarray,
    axis: int | tuple[int, ...] | None = None,
    ddof: int = 0,
) -> float:
    """Computes the signal to noise ratio of an input image.

    :param img: Input image in the shape (height, width, channels).
    :param axis: The axis upon which the mean and standard deviation are computed.
    :param ddof: Delta degrees of freedom
    """
    mean = img.mean(axis)
    standard_deviation = img.std(axis=axis, ddof=ddof)
    return float(np.where(standard_deviation == 0, 0, mean / standard_deviation))
