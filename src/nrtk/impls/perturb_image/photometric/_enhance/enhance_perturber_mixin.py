"""Mixin class providing shared functionality for enhancement perturbers."""

from __future__ import annotations

import abc
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage


@runtime_checkable
class _Enhancement(Protocol):  # Used for type checking only  # pragma: no cover
    def __init__(self: _Enhancement, image: PILImage) -> None:
        pass

    @abc.abstractmethod
    def enhance(self: _Enhancement, factor: float) -> PILImage:
        pass


class EnhancePerturberMixin(PerturbImage):
    def __init__(self, factor: float = 1.0) -> None:
        """Private class to handle general Enhancement functions.

        Args:
            factor:
                Enhancement factor.
        """
        super().__init__()
        if factor < 0.0:
            raise ValueError(
                f"{type(self).__name__} invalid factor ({factor}). Must be >= 0.0",
            )

        self.factor = factor

    def _perturb(
        self,
        *,
        enhancement: type[_Enhancement],
        image: NDArray[Any],
    ) -> NDArray[Any]:
        """Call appropriate enhancement interface and perform any necessary data type conversion.

        Args:
            enhancement:
                Ehancement to apply.
            image:
                Input image as a numpy array.

        Returns:
            Peturbed image as numpy array, including matching shape and dtype.
        """
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
        """Returns the current configuration of the EnhancePerturberMixin instance."""
        cfg = super().get_config()
        cfg["factor"] = self.factor
        return cfg
