"""Defines RandomScalePerturber to apply Albumentations' `Scale` transformation to input images.

Classes:
    RandomScalePerturber: A perturbation class for applying the `Scale` transformation from Albumentations.

Dependencies:
    - albumentations: For the underlying perturbations
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for perturbation algorithms.
    - nrtk.impls.perturb_image.AlbumentationsPerturber: Base implementation for Albumentations perturbers.

Example usage:
    >>> import numpy as np
    >>> limit = 0.5
    >>> perturber = RandomScalePerturber(limit=limit, seed=42)
    >>> image = np.ones((256, 256, 3))
    >>> perturbed_image, _ = perturber(image=image)
"""

from __future__ import annotations

__all__ = ["RandomScalePerturber"]

from typing import Any

import cv2
from typing_extensions import override

from nrtk.impls.perturb_image import AlbumentationsPerturber


class RandomScalePerturber(AlbumentationsPerturber):
    """RandomScalePerturber applies a Scale transformation from Albumentations.

    Attributes:
        limit (float | tuple[float, float]):
            Range of scaling factors. A single float value is equivalent to a tuple of (-value, value).
            One will then be added to each value. Therefore, a given value of (-0.2, 0.1) corresponds
            to scaling the image between 0.8x and 1.1x, and a given value of 0.5 corresponds to scaling
            the image between 0.5x and 1.5x.
        interpolation (int)
            OpenCV flag indicating the interpolation algorithm.
        probability (float):
            Probability of applying the scale transformation.
        seed (int | None):
            Random seed for reproducibility. None for non-deterministic behavior.
        is_static (bool):
            If True, resets seed after each call for consistent results.
    """

    def __init__(  # noqa: C901 - validation branches cannot be reduced further
        self,
        *,
        limit: float | tuple[float, float] = 0.0,
        interpolation: int = cv2.INTER_LINEAR,
        probability: float = 1.0,
        seed: int | None = None,
        is_static: bool = False,
    ) -> None:
        """RandomScalePerturber applies a random scale perturbation to an input image.

        It ensures that bounding boxes are adjusted correctly to reflect the translated
        image coordinates.

        Args:
            limit:
                Range of scaling factors. A single float value is equivalent to a tuple of (-value, value).
                One will then be added to each value. Therefore, a given value of (-0.2, 0.1) corresponds
                to scaling the image between 0.8x and 1.1x, and a given value of 0.5 corresponds to scaling
                the image between 0.5x and 1.5x.
            interpolation:
                OpenCV flag indicating the interpolation algorithm.
            probability:
                Probability of applying the scale transformation.
            seed:
                Random seed for reproducible results. Defaults to None for non-deterministic behavior.
            is_static:
                If True and seed is provided, resets seed after each perturb call for consistent
                results across multiple calls (useful for video frame processing).
        """
        self._allowed_interpolations = {
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_LINEAR_EXACT,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        }

        if isinstance(limit, tuple):
            lower, upper = limit
        else:
            lower = -limit
            upper = limit

        if not 0.0 <= probability <= 1.0:
            raise ValueError("Scale probability must be between 0.0 and 1.0 inclusive.")
        if lower > upper:
            raise ValueError("Lower scale limit must be less than or equal to upper limit.")
        if lower <= -1.0:
            raise ValueError("Lower scale limit must be greater than -1.0.")
        if interpolation not in self._allowed_interpolations:
            raise ValueError("Interpolation value not supported.")

        super().__init__(
            perturber="RandomScale",
            parameters={"scale_limit": limit, "interpolation": interpolation, "p": probability},
            seed=seed,
            is_static=is_static,
        )
        self._limit = limit
        self._interpolation = interpolation
        self._probability = probability

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the RandomScalePerturber instance."""
        cfg = super().get_config()
        cfg["limit"] = self._limit
        cfg["interpolation"] = self._interpolation
        cfg["probability"] = self._probability
        # Remove inherited Albumentations-specific keys not used by this perturber
        cfg.pop("parameters", None)  # noqa: FKA100, RUF100 - suppress flake8-keyword-arguments; RUF100 needed since ruff doesn't recognize FKA100
        cfg.pop("perturber", None)  # noqa: FKA100, RUF100 - suppress flake8-keyword-arguments; RUF100 needed since ruff doesn't recognize FKA100
        return cfg
