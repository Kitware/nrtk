"""
This module provides classes for enhancing image properties, such as brightness, color,
contrast, and sharpness, by implementing the `PerturbImage` interface. These classes use
the PIL library to adjust the enhancement level for a given image, allowing for customized
image modifications. The `_Enhancement` protocol and `_PILEnhancePerturber` base class
facilitate this by defining common behavior for all enhancement-based perturbations.

Classes:
    _PILEnhancePerturber: Base class for applying a specific PIL enhancement to an image.
    BrightnessPerturber: Adjusts the brightness of an image.
    ColorPerturber: Adjusts the color balance of an image.
    ContrastPerturber: Adjusts the contrast of an image.
    SharpnessPerturber: Adjusts the sharpness of an image.

Dependencies:
    - numpy for image data handling.
    - PIL (Pillow) for image enhancements.
    - typing_extensions for type checking and enhancements.

Usage:
    Instantiate any of the enhancement classes with a specific enhancement factor, then call
    `perturb` to apply the desired enhancement to an input image.

Example:
    brightness_perturber = BrightnessPerturber(factor=1.5)
    brighter_image = brightness_perturber.perturb(input_image)

    contrast_perturber = ContrastPerturber(factor=0.8)
    contrasted_image = contrast_perturber.perturb(input_image)

Notes:
    - Each enhancement class has a default factor of 1.0, which applies no change to the image.
    - `SharpnessPerturber` enforces a factor range of [0.0, 2.0].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image, ImageEnhance
from PIL.Image import Image as PILImage
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage


@runtime_checkable
class _Enhancement(Protocol):  # Used for type checking only  # pragma: no cover
    def __init__(self: _Enhancement, image: PILImage) -> None:
        pass

    def enhance(self: _Enhancement, factor: float) -> PILImage:
        pass


class _PILEnhancePerturber(PerturbImage):
    def __init__(self, factor: float = 1.0) -> None:
        """:param factor: Enhancement factor."""
        if factor < 0.0:
            raise ValueError(
                f"{type(self).__name__} invalid factor ({factor}). Must be >= 0.0",
            )

        self.factor = factor

    def _perturb(
        self,
        enhancement: type[_Enhancement],
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Call appropriate enhancement interface and perform any necessary data type conversion.

        :param enhancement: Ehancement to apply.
        :param image: Input image as a numpy array.

        :return: Peturbed image as numpy array, including matching shape and dtype.
        """
        if additional_params is None:
            additional_params = dict()
        dtype = image.dtype
        # PIL does not support RGB floating point images so we must do an
        # intermediary conversion
        if np.issubdtype(dtype, np.floating):
            image = (image * 255).astype(np.uint8)

        image_pil = Image.fromarray(image)
        image_enhanced = enhancement(image_pil).enhance(factor=self.factor)
        image_np = np.array(image_enhanced)

        # Convert back to floating point dtype if needed
        if np.issubdtype(dtype, np.floating):
            image_np = image.astype(dtype) / 255

        return image_np

    @override
    def get_config(self) -> dict[str, Any]:
        return {"factor": self.factor}


class BrightnessPerturber(_PILEnhancePerturber):
    """Adjusts image stimulus brightness."""

    @override
    def perturb(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return image stimulus with adjusted brightness."""
        if additional_params is None:
            additional_params = dict()
        enhancement = ImageEnhance.Brightness
        if TYPE_CHECKING and not isinstance(
            enhancement,
            _Enhancement,
        ):  # pragma: no cover
            raise ValueError("enhancement does not conform to _Enhancement protocol")
        return self._perturb(enhancement=enhancement, image=image)


class ColorPerturber(_PILEnhancePerturber):
    """Adjusts image stimulus color balance."""

    @override
    def perturb(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return image stimulus with adjusted color balance."""
        if additional_params is None:
            additional_params = dict()
        enhancement = ImageEnhance.Color
        if TYPE_CHECKING and not isinstance(
            enhancement,
            _Enhancement,
        ):  # pragma: no cover
            raise ValueError("enhancement does not conform to _Enhancement protocol")
        return self._perturb(enhancement=enhancement, image=image)


class ContrastPerturber(_PILEnhancePerturber):
    """Adjusts image stimulus contrast."""

    @override
    def perturb(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return image stimulus with adjusted contrast."""
        if additional_params is None:
            additional_params = dict()
        enhancement = ImageEnhance.Contrast
        if TYPE_CHECKING and not isinstance(
            enhancement,
            _Enhancement,
        ):  # pragma: no cover
            raise ValueError("enhancement does not conform to _Enhancement protocol")
        return self._perturb(enhancement=enhancement, image=image)


class SharpnessPerturber(_PILEnhancePerturber):
    """Adjusts image stimulus sharpness."""

    def __init__(self, factor: float = 1.0) -> None:
        """:param rng: Enhancement factor."""
        if factor < 0.0 or factor > 2.0:
            raise ValueError(
                f"{type(self).__name__} invalid sharpness factor ({factor}). Must be in [0.0, 2.0]",
            )

        super().__init__(factor=factor)

    @override
    def perturb(
        self,
        image: np.ndarray,
        additional_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return image stimulus with adjusted sharpness."""
        if additional_params is None:
            additional_params = dict()
        enhancement = ImageEnhance.Sharpness
        if TYPE_CHECKING and not isinstance(
            enhancement,
            _Enhancement,
        ):  # pragma: no cover
            raise ValueError("enhancement does not conform to _Enhancement protocol")
        return self._perturb(enhancement=enhancement, image=image)
