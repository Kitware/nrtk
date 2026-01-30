"""Implements GaussianBlurPerturber for applying Gaussian blurring to images.

GaussianBlurPerturber applies Gaussian blurring to an image, useful for reducing noise
and detail with a smooth effect.

Dependencies:
    - OpenCV (cv2) for image processing.
    - numpy for handling image data.

Usage:
    The GaussianBlurPerturber can be instantiated with a specific kernel size, controlling
    the intensity and spread of the blur. The `perturb` method applies the blur effect to
    an input image, returning the processed result.

Example:
    >>> image = np.ones((256, 256, 3), dtype=np.float32)
    >>> gauss_blur = GaussianBlurPerturber(ksize=5)
    >>> blurred_image, _ = gauss_blur(image=image)

Note:
    The boxes returned from `perturb` are identical to the boxes passed in.
"""

from __future__ import annotations

from nrtk.interfaces.perturb_image import PerturbImage

__all__ = ["GaussianBlurPerturber"]

from collections.abc import Hashable, Iterable
from typing import Any

import cv2
import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image.photometric._blur.blur_perturber_mixin import BlurPerturberMixin


class GaussianBlurPerturber(BlurPerturberMixin, PerturbImage):
    """Applies Gaussian blurring to the image stimulus.

    Attributes:
        ksize (int):
            blur kernal size
    """

    def __init__(self, ksize: int = 1) -> None:
        """GaussianBlurPerturber applies gaussian blurring to an image.

        Args:
            ksize:
                Blurring kernel size.
        """
        super().__init__(ksize=ksize)
        min_k_size = 1
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

        return cv2.GaussianBlur(_image, ksize=(self.ksize, self.ksize), sigmaX=0), _boxes
