"""Defines RandomRotationPerturber to apply Albumentations' `Rotate` transformation to input images.

Classes:
    RandomRotationPerturber: A perturbation class for applying the `Rotate` transformation from Albumentations.

Dependencies:
    - albumentations: For the underlying perturbations
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for perturbation algorithms.
    - nrtk.impls.perturb_image.AlbumentationsPerturber: Base implementation for Albumentations perturbers.

Example usage:
    >>> limit = 90.0
    >>> perturber = RandomRotationPerturber(limit=limit, seed=42)
    >>> image = np.ones((256, 256, 3))
    >>> perturbed_image, _ = perturber(image=image)
"""

from __future__ import annotations

__all__ = ["RandomRotationPerturber"]

from collections.abc import Sequence
from typing import Any

import numpy as np
from typing_extensions import override

from nrtk.impls.perturb_image import AlbumentationsPerturber


class RandomRotationPerturber(AlbumentationsPerturber):
    """RandomRotationPerturber applies a Rotation transformation from Albumentations.

    Attributes:
        limit (float | tuple[float, float]):
            Maximum rotation angle in degrees.
        probability (float):
            Probability of applying the rotation transformation.
        seed (int | None):
            Random seed for reproducibility. None for non-deterministic behavior.
        is_static (bool):
            If True, resets seed after each call for consistent results.
        fill (numpy.array):
            Background color fill for RGB image.
    """

    def __init__(
        self,
        *,
        limit: float | tuple[float, float] = 0.0,
        probability: float = 1.0,
        fill: Sequence[int] | None = None,
        seed: int | None = None,
        is_static: bool = False,
    ) -> None:
        """RandomRotationPerturber applies a Rotation transformation from Albumentations.

        Args:
            limit (float | tuple[float, float]):
                Maximum rotation angle in degrees. For example, 90.0 value means the image can be rotated
                by any angle between [-90, 90] degrees and (0.0, 90.0) means the image can be rotated
                by any angle between [0, 90] degrees. Default value is 0.0, which means no rotation.
            probability (float):
                Probability of applying the rotation transformation. A value of 1.0 means it will always
                apply the rotation, while a value of 0.0 means it will never apply the rotation. Default
                value is 1.0.
            fill (numpy.array | None):
                Background color fill for RGB image. (0 to 255 for each channel).
                Default value is a black background ([0, 0, 0]).
            seed:
                Random seed for reproducible results. Defaults to None for non-deterministic behavior.
            is_static:
                If True and seed is provided, resets seed after each perturb call for consistent
                results across multiple calls (useful for video frame processing).

        Raises:
            ValueError:
                Rotation probability must be between 0.0 and 1.0 inclusive.
            ValueError:
                Color fill values must be integers between 0 and 255 inclusive.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Rotation probability must be between 0.0 and 1.0 inclusive.")

        if fill is None:
            fill = [0, 0, 0]
        fill_arr = np.asarray(fill)
        if not ((fill_arr >= 0) & (fill_arr <= 255)).all():
            raise ValueError("Color fill values must be integers between 0 and 255 inclusive.")

        super().__init__(
            perturber="Rotate",
            parameters={"limit": limit, "p": probability, "fill": fill},
            seed=seed,
            is_static=is_static,
        )
        self._limit = limit
        self._probability = probability
        self._fill = fill

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the RandomRotationPerturber instance."""
        cfg = super().get_config()
        cfg["limit"] = self._limit
        cfg["probability"] = self._probability
        cfg["fill"] = self._fill
        # Remove inherited Albumentations-specific keys not used by this perturber
        cfg.pop("parameters", None)  # noqa: FKA100, RUF100 - suppress flake8-keyword-arguments; RUF100 needed since ruff doesn't recognize FKA100
        cfg.pop("perturber", None)  # noqa: FKA100, RUF100 - suppress flake8-keyword-arguments; RUF100 needed since ruff doesn't recognize FKA100
        return cfg
