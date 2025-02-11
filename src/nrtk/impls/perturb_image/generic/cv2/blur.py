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
    blurred_image, boxes = avg_blur.perturb(input_image, boxes)

    gauss_blur = GaussianBlurPerturber(ksize=5)
    blurred_image, boxes = gauss_blur.perturb(input_image, boxes)

    median_blur = MedianBlurPerturber(ksize=3)
    blurred_image, boxes = median_blur.perturb(input_image, boxes)

Note:
    Each class requires OpenCV for functionality. An ImportError will be raised if OpenCV is
    not available.
    The boxes returned from `perturb` are identical to the boxes passed in.
"""

from collections.abc import Hashable, Iterable
from typing import Any, Optional

try:
    # Multiple type ignores added for pyright's handling of guarded imports
    import cv2

    cv2_available = True
except ImportError:  # pragma: no cover
    cv2_available = False
import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import OpenCVImportError


class _PerturbImage(PerturbImage):
    def __init__(self, ksize: int = 1, box_alignment_mode: str = "extent") -> None:
        if not self.is_usable():
            raise OpenCVImportError
        super().__init__(box_alignment_mode=box_alignment_mode)

        self.ksize = ksize

    @override
    def perturb(
        self,
        image: np.ndarray,
        boxes: Optional[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]]:
        """Return image stimulus after applying average blurring."""
        _image, _boxes = super().perturb(image=image, boxes=boxes, additional_params=additional_params)

        # Check for channel last format
        if _image.ndim == 3 and _image.shape[2] > 4:
            raise ValueError("Image is not in expected format (H, W, C)")

        return _image, _boxes

    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the MedianBlurPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = super().get_config()
        cfg["ksize"] = self.ksize
        return cfg

    @classmethod
    def is_usable(cls) -> bool:
        """
        Checks if the required cv2 module is available.

        Returns:
            bool: True if opencv is installed; False otherwise.
        """
        # Requires opencv to be installed
        return cv2_available


class AverageBlurPerturber(_PerturbImage):
    """Applies average blurring to the image stimulus."""

    def __init__(self, ksize: int = 1, box_alignment_mode: str = "extent") -> None:
        """:param ksize: Blurring kernel size."""
        super().__init__(ksize=ksize, box_alignment_mode=box_alignment_mode)
        min_k_size = 1
        if ksize < min_k_size:
            raise ValueError(f"{type(self).__name__} invalid ksize ({ksize}). Must be >= {min_k_size}")

    @override
    def perturb(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        """Return image stimulus after applying average blurring."""
        _image, _boxes = super().perturb(image=image, boxes=boxes, additional_params=additional_params)

        return cv2.blur(_image, ksize=(self.ksize, self.ksize)), _boxes


class GaussianBlurPerturber(_PerturbImage):
    """Applies Gaussian blurring to the image stimulus."""

    def __init__(self, ksize: int = 1, box_alignment_mode: str = "extent") -> None:
        """:param ksize: Blurring kernel size."""
        super().__init__(ksize=ksize, box_alignment_mode=box_alignment_mode)
        min_k_size = 1
        if ksize < min_k_size or ksize % 2 == 0:
            raise ValueError(f"{type(self).__name__} invalid ksize ({ksize}). Must be >= {min_k_size} and odd.")

    @override
    def perturb(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: dict[str, Any] = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        """Return image stimulus after applying Gaussian blurring."""
        _image, _boxes = super().perturb(image=image, boxes=boxes, additional_params=additional_params)

        return cv2.GaussianBlur(_image, ksize=(self.ksize, self.ksize), sigmaX=0), _boxes


class MedianBlurPerturber(_PerturbImage):
    """Applies median blurring to the image stimulus."""

    def __init__(self, ksize: int = 3, box_alignment_mode: str = "extent") -> None:
        """:param ksize: Blurring kernel size."""
        super().__init__(ksize=ksize, box_alignment_mode=box_alignment_mode)
        min_k_size = 3
        if ksize < min_k_size or ksize % 2 == 0:
            raise ValueError(f"{type(self).__name__} invalid ksize ({ksize}). Must be >= {min_k_size} and odd.")

    @override
    def perturb(
        self,
        image: np.ndarray,
        boxes: Optional[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]]:
        """Return image stimulus after applying Gaussian blurring."""
        _image, _boxes = super().perturb(image=image, boxes=boxes, additional_params=additional_params)

        return cv2.medianBlur(_image, ksize=self.ksize), _boxes
