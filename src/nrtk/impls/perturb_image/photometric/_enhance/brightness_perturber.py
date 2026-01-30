"""Implements BrightnessPerturber for adjusting image brightness.

Dependencies:
    - PIL (Pillow) for image enhancements.
    - numpy for image data handling.

Example:
    >>> import numpy as np
    >>> image = np.ones((256, 256, 3))
    >>> brightness_perturber = BrightnessPerturber(factor=1.5)
    >>> brighter_image, _ = brightness_perturber(image=image)

Note:
    The boxes returned from `perturb` are identical to the boxes passed in.
"""

from __future__ import annotations

from nrtk.interfaces.perturb_image import PerturbImage

__all__ = ["BrightnessPerturber"]

from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray
from PIL import ImageEnhance
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image.photometric._enhance.enhance_perturber_mixin import (
    EnhancePerturberMixin,
    _Enhancement,
)


class BrightnessPerturber(EnhancePerturberMixin, PerturbImage):
    """Adjusts image stimulus brightness."""

    @override
    def perturb(
        self,
        *,
        image: NDArray[Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArray[Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Return image stimulus with adjusted brightness."""
        perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes, **kwargs)

        enhancement = ImageEnhance.Brightness
        if TYPE_CHECKING and not isinstance(
            enhancement,
            type(_Enhancement),
        ):  # pragma: no cover
            raise ValueError("enhancement does not conform to _Enhancement protocol")
        return super()._perturb(enhancement=enhancement, image=perturbed_image), perturbed_boxes
