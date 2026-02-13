"""Implements SaltAndPepperNoisePerturber for applying salt and pepper noise.

Dependencies:
    - skimage (scikit-image) for noise application.
    - numpy for image data handling.

Example:
    >>> import numpy as np
    >>> image = np.ones((256, 256, 3))
    >>> sp_noise_perturber = SaltAndPepperNoisePerturber(amount=0.05)
    >>> noisy_image, _ = sp_noise_perturber(image=image)

Note:
    The boxes returned from `perturb` are identical to the boxes passed in.
"""

from __future__ import annotations

__all__ = ["SaltAndPepperNoisePerturber"]

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image.photometric._noise.noise_perturber_mixin import SaltPepperNoisePerturberMixin


class SaltAndPepperNoisePerturber(SaltPepperNoisePerturberMixin):
    """Adds salt & pepper noise to image stimulus.

    Attributes:
        seed (int | None):
            Random seed for reproducible results. None means non-deterministic.
        is_static (bool):
            If True and seed is set, resets RNG state after each perturb call.
        clip (bool):
            Decide if output is clipped between the range of [-1, 1].
        amount (float):
            Proportion of image pixels to replace with noise on range [0, 1]
        salt_vs_pepper (float):
            Proportion of salt vs. pepper noise. Higher values represent more salt.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        is_static: bool = False,
        amount: float = 0.05,
        salt_vs_pepper: float = 0.5,
        clip: bool = True,
    ) -> None:
        """Initializes the SaltAndPepperNoisePerturber.

        Args:
            seed:
                Random seed for reproducible results. Defaults to None for
                non-deterministic behavior.
            is_static:
                If True and seed is provided, resets the random state after each
                perturb call for identical results on repeated calls.
            amount:
                Proportion of image pixels to replace with noise on range [0, 1].
            salt_vs_pepper:
                Proportion of salt vs. pepper noise on range [0, 1].
                Higher values represent more salt.
            clip:
                Decide if output is clipped between the range of [-1, 1].
        """
        super().__init__(amount=amount, seed=seed, is_static=is_static, clip=clip)

        if salt_vs_pepper < 0.0 or salt_vs_pepper > 1.0:
            raise ValueError(
                f"{type(self).__name__} invalid salt_vs_pepper ({salt_vs_pepper}). Must be in [0.0, 1.0]",
            )

        self.salt_vs_pepper = salt_vs_pepper

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Return image stimulus with S&P noise."""
        perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes, **kwargs)

        return self._perturb(
            image=perturbed_image,
            mode="s&p",
            amount=self.amount,
            salt_vs_pepper=self.salt_vs_pepper,
        ), perturbed_boxes

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the SaltAndPepperNoisePerturber instance."""
        cfg = super().get_config()
        cfg["salt_vs_pepper"] = self.salt_vs_pepper
        return cfg
