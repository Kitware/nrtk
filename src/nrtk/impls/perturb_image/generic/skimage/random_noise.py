"""
This module provides a set of classes for adding different types of noise to images, implementing
the `PerturbImage` interface. The perturbation types include salt, pepper, salt-and-pepper,
Gaussian, and speckle noise, allowing for a wide range of image noise simulations.

Classes:
    _SKImageNoisePerturber: Base class for noise perturbation, using `skimage.util.random_noise`.
    _SPNoisePerturber: Base class for salt-and-pepper type noise.
    SaltNoisePerturber: Adds salt noise to an image.
    PepperNoisePerturber: Adds pepper noise to an image.
    SaltAndPepperNoisePerturber: Adds both salt and pepper noise to an image, with control over
        the ratio of salt to pepper.
    _GSNoisePerturber: Base class for Gaussian-based noise.
    GaussianNoisePerturber: Adds Gaussian-distributed additive noise to an image.
    SpeckleNoisePerturber: Adds Gaussian-based multiplicative noise (speckle) to an image.

Dependencies:
    - numpy for handling image data and random number generation.
    - skimage.util for applying various noise effects to images.

Usage:
    Each noise perturber class can be instantiated with specific parameters, allowing the user
    to customize the type and intensity of noise applied to an image. Use the `perturb` method
    of each class to apply the chosen noise effect.

Example:
    gaussian_perturber = GaussianNoisePerturber(mean=0, var=0.01)
    noisy_image = gaussian_perturber.perturb(image_data)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import skimage.util  # type:ignore
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage


class _SKImageNoisePerturber(PerturbImage):
    def __init__(self, rng: np.random.Generator | int | None = None) -> None:
        """:param rng: Pseudo-random number generator or seed."""
        self.rng = rng

    def _perturb(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Call skimage.util.random_noise with appropriate arguments and convert back to input dtype.

        :param image: Input image as a numpy array.
        :param kwargs: Keyword arguments for random_noise call. ``rng`` will be
            specified separately.

        :return: Peturbed image as numpy array, including matching shape and dtype.
        """
        # Determine if conversion back to original dtype is possible
        dtype_str = str(image.dtype)
        convert_image = {
            str(np.dtype(np.bool_)): skimage.util.img_as_bool,
            str(np.dtype(np.float32)): skimage.util.img_as_float32,
            str(np.dtype(np.float64)): skimage.util.img_as_float64,
            str(np.dtype(np.int16)): skimage.util.img_as_int,
            str(np.dtype(np.uint8)): skimage.util.img_as_ubyte,
            str(np.dtype(np.uint)): skimage.util.img_as_uint,
        }
        if dtype_str not in convert_image:
            if np.issubdtype(image.dtype, np.floating):
                convert = skimage.util.img_as_float
            else:
                raise NotImplementedError(f"Perturb not implemented for {dtype_str}")
        else:
            convert = convert_image[dtype_str]

        # Apply perturbation
        image_noise = skimage.util.random_noise(image, rng=self.rng, **kwargs)

        # Convert image back to original dtype
        return convert(image_noise).astype(image.dtype)

    @override
    def get_config(self) -> dict[str, Any]:
        return {"rng": self.rng}


class _SPNoisePerturber(_SKImageNoisePerturber):
    def __init__(
        self,
        rng: np.random.Generator | int | None = None,
        amount: float = 0.05,
    ) -> None:
        """Initializes the SPNoisePerturber.

        :param rng: Pseudo-random number generator or seed.
        :param amount: Proportion of image pixels to replace with noise on range [0, 1].
        """
        if amount < 0.0 or amount > 1.0:
            raise ValueError(
                f"{type(self).__name__} invalid amount ({amount}). Must be in [0.0, 1.0]",
            )

        super().__init__(rng=rng)

        self.amount = amount

    @override
    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["amount"] = self.amount
        return cfg


class SaltNoisePerturber(_SPNoisePerturber):
    """Adds salt noise to image stimulus."""

    @override
    def perturb(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return image stimulus with salt noise."""
        if additional_params is None:
            additional_params = dict()
        return self._perturb(image, mode="salt", amount=self.amount)


class PepperNoisePerturber(_SPNoisePerturber):
    """Adds pepper noise to image stimulus."""

    @override
    def perturb(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return image stimulus with pepper noise."""
        if additional_params is None:
            additional_params = dict()
        return self._perturb(image, mode="pepper", amount=self.amount)


class SaltAndPepperNoisePerturber(_SPNoisePerturber):
    """Adds salt & pepper noise to image stimulus."""

    def __init__(
        self,
        rng: np.random.Generator | int | None = None,
        amount: float = 0.05,
        salt_vs_pepper: float = 0.5,
    ) -> None:
        """Initializes the SaltAndPepperNoisePerturber.

        :param rng: Pseudo-random number generator or seed.
        :param amount: Proportion of image pixels to replace with noise on range [0, 1].
        :param salt_vs_pepper: Proportion of salt vs. pepper noise on range [0, 1].
            Higher values represent more salt.
        """
        if salt_vs_pepper < 0.0 or salt_vs_pepper > 1.0:
            raise ValueError(
                f"{type(self).__name__} invalid salt_vs_pepper ({salt_vs_pepper}). Must be in [0.0, 1.0]",
            )

        super().__init__(amount=amount, rng=rng)

        self.salt_vs_pepper = salt_vs_pepper

    @override
    def perturb(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return image stimulus with S&P noise."""
        if additional_params is None:
            additional_params = dict()
        return self._perturb(
            image,
            mode="s&p",
            amount=self.amount,
            salt_vs_pepper=self.salt_vs_pepper,
        )

    @override
    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["salt_vs_pepper"] = self.salt_vs_pepper
        return cfg


class _GSNoisePerturber(_SKImageNoisePerturber):
    def __init__(
        self,
        rng: np.random.Generator | int | None = None,
        mean: float = 0.0,
        var: float = 0.05,
    ) -> None:
        """Initializes the GSNoisePerturber.

        :param rng: Pseudo-random number generator or seed.
        :param mean: Mean of random distribution.
        :param var: Variance of random distribution.
        """
        if var < 0:
            raise ValueError(
                f"{type(self).__name__} invalid var ({var}). Must be >= 0.0",
            )

        super().__init__(rng=rng)

        self.mean = mean
        self.var = var

    @override
    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["mean"] = self.mean
        cfg["var"] = self.var
        return cfg


class GaussianNoisePerturber(_GSNoisePerturber):
    """Adds Gaussian-distributed additive noise to image stimulus."""

    @override
    def perturb(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return image stimulus with Gaussian noise."""
        if additional_params is None:
            additional_params = dict()
        return self._perturb(image, mode="gaussian", var=self.var, mean=self.mean)


class SpeckleNoisePerturber(_GSNoisePerturber):
    """Adds multiplicative noise to image stimulus. Noise is Gaussian-based."""

    @override
    def perturb(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return image stimulus with multiplicative noise."""
        if additional_params is None:
            additional_params = dict()
        return self._perturb(image, mode="speckle", var=self.var, mean=self.mean)
