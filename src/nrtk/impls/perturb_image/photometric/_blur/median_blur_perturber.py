"""Implements MedianBlurPerturber for applying median blurring to images.

MedianBlurPerturber applies median blurring to an image, commonly used for removing
salt-and-pepper noise while preserving edges.

Dependencies:
    - OpenCV (cv2) for image processing.
    - numpy for handling image data.

Usage:
    The MedianBlurPerturber can be instantiated with a specific kernel size, controlling
    the intensity and spread of the blur. The `perturb` method applies the blur effect to
    an input image, returning the processed result.

Example:
    >>> image = np.ones((256, 256, 3), dtype=np.float32)
    >>> median_blur = MedianBlurPerturber(ksize=3)
    >>> blurred_image, _ = median_blur(image=image)

Note:
    The boxes returned from `perturb` are identical to the boxes passed in.
"""

from __future__ import annotations

from nrtk.interfaces.perturb_image import PerturbImage

__all__ = ["MedianBlurPerturber"]

from collections.abc import Hashable, Iterable
from typing import Any

import cv2
import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image.photometric._blur.blur_perturber_mixin import BlurPerturberMixin


class MedianBlurPerturber(BlurPerturberMixin, PerturbImage):
    """Applies median blurring to the image stimulus.

    Attributes:
        ksize (int):
            blur kernal size
    """

    def __init__(self, ksize: int = 3) -> None:
        """MedianBlurPerturber applies gaussian blurring to an image.

        Args:
            ksize:
                Blurring kernel size.
        """
        super().__init__(ksize=ksize)
        min_k_size = 3
        if ksize < min_k_size or ksize % 2 == 0:
            raise ValueError(f"{type(self).__name__} invalid ksize ({ksize}). Must be >= {min_k_size} and odd.")

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **additional_params: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Return image stimulus after applying Gaussian blurring."""
        _image, _boxes = super().perturb(image=image, boxes=boxes, **additional_params)

        return cv2.medianBlur(_image, ksize=self.ksize), _boxes
