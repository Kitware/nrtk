from typing import Any, Dict, Optional

import cv2
import numpy as np

from nrtk.interfaces.perturb_image import PerturbImage


class AverageBlurPerturber(PerturbImage):
    """Applies average blurring to the image stimulus."""

    def __init__(self, ksize: int = 1):
        """:param ksize: Blurring kernel size."""
        min_k_size = 1
        if ksize < min_k_size:
            raise ValueError(
                f"{type(self).__name__} invalid ksize ({ksize})."
                f" Must be >= {min_k_size}"
            )

        self.ksize = ksize

    def perturb(
        self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Return image stimulus after applying average blurring."""
        if additional_params is None:
            additional_params = dict()
        return cv2.blur(image, ksize=(self.ksize, self.ksize))

    def get_config(self) -> Dict[str, Any]:
        return {"ksize": self.ksize}


class GaussianBlurPerturber(PerturbImage):
    """Applies Gaussian blurring to the image stimulus."""

    def __init__(self, ksize: int = 1):
        """:param ksize: Blurring kernel size."""
        min_k_size = 1
        if ksize < min_k_size or ksize % 2 == 0:
            raise ValueError(
                f"{type(self).__name__} invalid ksize ({ksize})."
                f" Must be >= {min_k_size} and odd."
            )

        self.ksize = ksize

    def perturb(
        self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Return image stimulus after applying Gaussian blurring."""
        if additional_params is None:
            additional_params = dict()
        return cv2.GaussianBlur(image, ksize=(self.ksize, self.ksize), sigmaX=0)

    def get_config(self) -> Dict[str, Any]:
        return {"ksize": self.ksize}


class MedianBlurPerturber(PerturbImage):
    """Applies median blurring to the image stimulus."""

    def __init__(self, ksize: int = 1):
        """:param ksize: Blurring kernel size."""
        min_k_size = 3
        if ksize < min_k_size or ksize % 2 == 0:
            raise ValueError(
                f"{type(self).__name__} invalid ksize ({ksize})."
                f" Must be >= {min_k_size} and odd."
            )

        self.ksize = ksize

    def perturb(
        self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Return image stimulus after applying Gaussian blurring."""
        if additional_params is None:
            additional_params = dict()
        return cv2.medianBlur(image, ksize=self.ksize)

    def get_config(self) -> Dict[str, Any]:
        return {"ksize": self.ksize}
