from typing import Any, Dict

import cv2
import numpy as np

from nrtk.interfaces.perturb_image import PerturbImage


class AverageBlurPerturber(PerturbImage):
    """
    Applies average blurring to the image stimulus.
    """
    def __init__(
        self,
        ksize: int = 1
    ):
        """
        :param ksize: Blurring kernel size.
        """
        min_ksize = 1
        if ksize < min_ksize:
            raise ValueError(f"{type(self).__name__} invalid ksize ({ksize})."
                             f" Must be >= {min_ksize}")

        self.ksize = ksize

    def perturb(
        self,
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Return image stimulus after applying average blurring.
        """

        return cv2.blur(image, ksize=(self.ksize, self.ksize))

    def get_config(self) -> Dict[str, Any]:
        return {
            'ksize': self.ksize
        }


class GaussianBlurPerturber(PerturbImage):
    """
    Applies Gaussian blurring to the image stimulus.
    """
    def __init__(
        self,
        ksize: int = 1
    ):
        """
        :param ksize: Blurring kernel size.
        """
        min_ksize = 1
        if ksize < min_ksize or ksize % 2 == 0:
            raise ValueError(f"{type(self).__name__} invalid ksize ({ksize})."
                             f" Must be >= {min_ksize} and odd.")

        self.ksize = ksize

    def perturb(
        self,
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Return image stimulus after applying Gaussian blurring.
        """

        return cv2.GaussianBlur(image, ksize=(self.ksize, self.ksize), sigmaX=0)

    def get_config(self) -> Dict[str, Any]:
        return {
            'ksize': self.ksize
        }


class MedianBlurPerturber(PerturbImage):
    """
    Applies median blurring to the image stimulus.
    """
    def __init__(
        self,
        ksize: int = 1
    ):
        """
        :param ksize: Blurring kernel size.
        """
        min_ksize = 3
        if ksize < min_ksize or ksize % 2 == 0:
            raise ValueError(f"{type(self).__name__} invalid ksize ({ksize})."
                             f" Must be >= {min_ksize} and odd.")

        self.ksize = ksize

    def perturb(
        self,
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Return image stimulus after applying Gaussian blurring.
        """

        return cv2.medianBlur(image, ksize=self.ksize)

    def get_config(self) -> Dict[str, Any]:
        return {
            'ksize': self.ksize
        }
