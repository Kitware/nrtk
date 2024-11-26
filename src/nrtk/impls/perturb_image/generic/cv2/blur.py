"""
This module provides classes for applying different types of blurring to an image,
implementing the `PerturbImage` interface. Blurring methods include average, Gaussian,
and median blurring, each with customizable kernel sizes for controlling the level of
blur effect.

Classes:
    AverageBlurPerturber: Applies average blurring to an image.
    GaussianBlurPerturber: Applies Gaussian blurring to an image, useful for reducing noise
        and detail with a smooth effect.
    MedianBlurPerturber: Applies median blurring to an image, commonly used for removing
        salt-and-pepper noise while preserving edges.

Dependencies:
    - OpenCV (cv2) for image processing.
    - numpy for handling image data.

Usage:
    Each blur perturber class can be instantiated with a specific kernel size, controlling
    the intensity and spread of the blur. The `perturb` method applies the blur effect to
    an input image, returning the processed result.

Example:
    avg_blur = AverageBlurPerturber(ksize=3)
    blurred_image = avg_blur.perturb(input_image)

    gauss_blur = GaussianBlurPerturber(ksize=5)
    blurred_image = gauss_blur.perturb(input_image)

    median_blur = MedianBlurPerturber(ksize=3)
    blurred_image = median_blur.perturb(input_image)

Note:
    Each class requires OpenCV for functionality. An ImportError will be raised if OpenCV is
    not available.
"""

from __future__ import annotations

from typing import Any

try:
    import cv2

    cv2_available = True
except ImportError:
    cv2_available = False
import numpy as np
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage


class AverageBlurPerturber(PerturbImage):
    """Applies average blurring to the image stimulus."""

    def __init__(self, ksize: int = 1) -> None:
        """:param ksize: Blurring kernel size."""
        if not self.is_usable():
            raise ImportError("OpenCV not found. Please install 'nrtk[graphics]' or 'nrtk[headless]'.")
        min_k_size = 1
        if ksize < min_k_size:
            raise ValueError(f"{type(self).__name__} invalid ksize ({ksize}). Must be >= {min_k_size}")

        self.ksize = ksize

    @override
    def perturb(self, image: np.ndarray, additional_params: dict[str, Any] | None = None) -> np.ndarray:
        """Return image stimulus after applying average blurring."""
        if additional_params is None:
            additional_params = dict()

        # Check for channel last format
        if image.ndim == 3 and image.shape[2] > 4:
            raise ValueError("Image is not in expected format (H, W, C)")

        return cv2.blur(image, ksize=(self.ksize, self.ksize))

    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the AverageBlurPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        return {"ksize": self.ksize}

    @classmethod
    def is_usable(cls) -> bool:
        """
        Checks if the required pybsm module is available.

        Returns:
            bool: True if pybsm is installed; False otherwise.
        """
        # Requires opencv to be installed
        return cv2_available


class GaussianBlurPerturber(PerturbImage):
    """Applies Gaussian blurring to the image stimulus."""

    def __init__(self, ksize: int = 1) -> None:
        """:param ksize: Blurring kernel size."""
        if not self.is_usable():
            raise ImportError("OpenCV not found. Please install 'nrtk[graphics]' or 'nrtk[headless]'.")
        min_k_size = 1
        if ksize < min_k_size or ksize % 2 == 0:
            raise ValueError(f"{type(self).__name__} invalid ksize ({ksize}). Must be >= {min_k_size} and odd.")

        self.ksize = ksize

    @override
    def perturb(self, image: np.ndarray, additional_params: dict[str, Any] | None = None) -> np.ndarray:
        """Return image stimulus after applying Gaussian blurring."""
        if additional_params is None:
            additional_params = dict()

        # Check for channel last format
        if image.ndim == 3 and image.shape[2] > 4:
            raise ValueError("Image is not in expected format (H, W, C)")

        return cv2.GaussianBlur(image, ksize=(self.ksize, self.ksize), sigmaX=0)

    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the GaussianBlurPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        return {"ksize": self.ksize}

    @classmethod
    def is_usable(cls) -> bool:
        """
        Checks if the required pybsm module is available.

        Returns:
            bool: True if pybsm is installed; False otherwise.
        """
        # Requires opencv to be installed
        return cv2_available


class MedianBlurPerturber(PerturbImage):
    """Applies median blurring to the image stimulus."""

    def __init__(self, ksize: int = 1) -> None:
        """:param ksize: Blurring kernel size."""
        if not self.is_usable():
            raise ImportError("OpenCV not found. Please install 'nrtk[graphics]' or 'nrtk[headless]'.")
        min_k_size = 3
        if ksize < min_k_size or ksize % 2 == 0:
            raise ValueError(f"{type(self).__name__} invalid ksize ({ksize}). Must be >= {min_k_size} and odd.")

        self.ksize = ksize

    @override
    def perturb(self, image: np.ndarray, additional_params: dict[str, Any] | None = None) -> np.ndarray:
        """Return image stimulus after applying Gaussian blurring."""
        if additional_params is None:
            additional_params = dict()

        # Check for channel last format
        if image.ndim == 3 and image.shape[2] > 4:
            raise ValueError("Image is not in expected format (H, W, C)")

        return cv2.medianBlur(image, ksize=self.ksize)

    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the MedianBlurPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        return {"ksize": self.ksize}

    @classmethod
    def is_usable(cls) -> bool:
        """
        Checks if the required pybsm module is available.

        Returns:
            bool: True if pybsm is installed; False otherwise.
        """
        # Requires opencv to be installed
        return cv2_available
