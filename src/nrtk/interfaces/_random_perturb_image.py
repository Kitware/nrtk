"""Defines RandomPerturbImage, an interface for image perturbers that use random state.

Classes:
    RandomPerturbImage: An abstract base class for image perturbation algorithms that require
        random number generation, providing standardized seed handling and optional static
        behavior for video frame consistency.

Dependencies:
    - numpy for handling image arrays.
    - nrtk.interfaces.perturb_image for the base PerturbImage interface.

Usage:
    To create a custom random image perturbation class, inherit from `RandomPerturbImage`
    and implement the `_set_seed` and `perturb` methods.

Example:
    class CustomRandomPerturbImage(RandomPerturbImage):
        def _set_seed(self) -> None:
            if self._seed is not None:
                self._rng = np.random.default_rng(self._seed)

        def perturb(self, *, image, boxes=None, **kwargs):
            # Custom perturbation logic using self._rng
            perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes, **kwargs)
            return perturbed_image, perturbed_boxes

    perturber = CustomRandomPerturbImage(seed=42, is_static=True)
    perturbed_image, _ = perturber(image=image_data)
"""

from __future__ import annotations

__all__: list[str] = ["RandomPerturbImage"]

import abc
import warnings
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage


class RandomPerturbImage(PerturbImage):
    """Interface for image perturbers that use random state.

    This abstract class extends PerturbImage to provide standardized handling of
    random number generation for perturbation algorithms. It supports:

    - Configurable seeding for reproducibility
    - Non-deterministic behavior by default (seed=None)
    - Optional "static" mode for consistent results across multiple calls

    The is_static feature is particularly useful for video processing where
    the same perturbation should be applied consistently across all frames.

    Attributes:
        seed: Random seed for reproducibility. None (default) means non-deterministic.
        is_static: If True and seed is set, resets RNG state after each perturb
            call to ensure identical results for repeated calls with the same input.
    """

    def __init__(self, *, seed: int | None = None, is_static: bool = False) -> None:
        """Initialize the RandomPerturbImage with seed and static behavior options.

        Args:
            seed:
                Random seed for reproducible results. Defaults to None for
                non-deterministic behavior.
            is_static:
                If True and seed is provided, resets the random state after each
                perturb call. This ensures that repeated calls with the same input
                produce identical results, useful for video frame consistency.
                Has no effect when seed is None.

        Warns:
            UserWarning: If is_static is True but seed is None, since the
                static behavior only applies when a deterministic seed is set.
        """
        super().__init__()
        self._seed = seed
        self._is_static = is_static
        if self._seed is None and self._is_static:
            warnings.warn(
                "is_static=True has no effect when seed=None",
                UserWarning,
                stacklevel=2,
            )
        self._set_seed()

    @property
    def seed(self) -> int | None:
        """Random seed for reproducibility. None means non-deterministic."""
        return self._seed

    @property
    def is_static(self) -> bool:
        """If True and seed is set, resets RNG state after each perturb call."""
        return self._is_static

    @abc.abstractmethod
    def _set_seed(self) -> None:
        """Seed the random state(s) for this perturber.

        Implementations should initialize their random number generator(s) using
        self._seed. If self._seed is None, the RNG should be initialized without
        a seed for non-deterministic behavior.

        This method is called during __init__ and again after each perturb() call
        when is_static is True and seed is not None.
        """

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Generate a perturbed image, optionally resetting RNG state afterwards.

        This method calls the parent perturb() implementation and then, if
        is_static is True and seed is set, resets the random state to
        ensure subsequent calls produce identical results.

        Args:
            image:
                Input image as a numpy array.
            boxes:
                Input bounding boxes as an Iterable of tuples containing bounding boxes.
            kwargs:
                Implementation-specific keyword arguments.

        Returns:
            Perturbed image as numpy array and optionally modified bounding boxes.
        """
        perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes, **kwargs)
        if self._is_static and self._seed is not None:
            self._set_seed()
        return perturbed_image, perturbed_boxes

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the RandomPerturbImage instance.

        Returns:
            Configuration dictionary containing seed and is_static settings.
        """
        cfg = super().get_config()
        cfg["seed"] = self._seed
        cfg["is_static"] = self._is_static
        return cfg
