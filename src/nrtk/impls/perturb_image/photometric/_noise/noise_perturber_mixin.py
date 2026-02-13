"""Mixin class providing shared functionality for noise perturbers."""

from __future__ import annotations

from typing import Any

import numpy as np
import skimage.util
from typing_extensions import override

from nrtk.impls.perturb_image._base import NumpyRandomPerturbImage

__all__ = [
    "NoisePerturberMixin",
    "SaltPepperNoisePerturberMixin",
    "GaussianSpeckleNoisePerturberMixin",
]


class NoisePerturberMixin(NumpyRandomPerturbImage):
    def __init__(self, *, seed: int | None = None, is_static: bool = False, clip: bool = True) -> None:
        self.clip = clip
        super().__init__(seed=seed, is_static=is_static)

    def _perturb(self, *, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Call skimage.util.random_noise with appropriate arguments and convert back to input dtype.

        Args:
            image:
                Input image as a numpy array.
            kwargs:
                Keyword arguments for random_noise call. ``rng`` will be specified separately.

        Returns:
            Peturbed image as numpy array, including matching shape and dtype.
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
        image_noise = skimage.util.random_noise(image, rng=self._rng, clip=self.clip, **kwargs)

        # Convert image back to original dtype
        return convert(image_noise).astype(image.dtype)

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the _SKImageNoisePerturber instance."""
        cfg = super().get_config()
        cfg["clip"] = self.clip
        return cfg


class SaltPepperNoisePerturberMixin(NoisePerturberMixin):
    def __init__(
        self,
        *,
        seed: int | None = None,
        is_static: bool = False,
        amount: float = 0.05,
        clip: bool = True,
    ) -> None:
        """Initializes the SPNoisePerturber.

        Args:
            seed:
                Random seed for reproducible results. Defaults to None for
                non-deterministic behavior.
            is_static:
                If True and seed is provided, resets the random state after each
                perturb call for identical results on repeated calls.
            amount:
                Proportion of image pixels to replace with noise on range [0, 1].
            clip:
                Decide if output is clipped between the range of [-1, 1].
        """
        super().__init__(seed=seed, is_static=is_static, clip=clip)

        if amount < 0.0 or amount > 1.0:
            raise ValueError(
                f"{type(self).__name__} invalid amount ({amount}). Must be in [0.0, 1.0]",
            )

        self.amount = amount

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the _SPNoisePerturber instance."""
        cfg = super().get_config()
        cfg["amount"] = self.amount
        return cfg


class GaussianSpeckleNoisePerturberMixin(NoisePerturberMixin):
    def __init__(
        self,
        *,
        seed: int | None = None,
        is_static: bool = False,
        mean: float = 0.0,
        var: float = 0.05,
        clip: bool = True,
    ) -> None:
        """Initializes the GSNoisePerturber.

        Args:
            seed:
                Random seed for reproducible results. Defaults to None for
                non-deterministic behavior.
            is_static:
                If True and seed is provided, resets the random state after each
                perturb call for identical results on repeated calls.
            mean:
                Mean of random distribution
            var:
                Variance of random distribution.
            clip:
                Decide if output is clipped between the range of [-1, 1].
        """
        super().__init__(seed=seed, is_static=is_static, clip=clip)

        if var < 0:
            raise ValueError(
                f"{type(self).__name__} invalid var ({var}). Must be >= 0.0",
            )

        self.mean = mean
        self.var = var

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the _GSNoisePerturber instance."""
        cfg = super().get_config()
        cfg["mean"] = self.mean
        cfg["var"] = self.var
        return cfg
