"""Implements SharpnessPerturber for adjusting image sharpness.

Dependencies:
    - PIL (Pillow) for image enhancements.
    - numpy for image data handling.

Example:
    >>> import numpy as np
    >>> image = np.ones((256, 256, 3))
    >>> sharpness_perturber = SharpnessPerturber(factor=1.5)
    >>> sharpened_image, _ = sharpness_perturber(image=image)

Note:
    The boxes returned from `perturb` are identical to the boxes passed in.
    SharpnessPerturber enforces a factor range of [0.0, 2.0].
"""

from __future__ import annotations

from nrtk.interfaces.perturb_image import PerturbImage

__all__ = ["SharpnessPerturber"]

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


class SharpnessPerturber(EnhancePerturberMixin, PerturbImage):
    """Adjusts image stimulus sharpness."""

    def __init__(self, factor: float = 1.0) -> None:
        """Override parent init since factor is capped at 2.0."""
        if factor < 0.0 or factor > 2.0:
            raise ValueError(
                f"{type(self).__name__} invalid sharpness factor ({factor}). Must be in [0.0, 2.0]",
            )

        super().__init__(factor=factor)

    @override
    def perturb(
        self,
        *,
        image: NDArray[Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArray[Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Return image stimulus with adjusted sharpness."""
        perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes, **kwargs)

        enhancement = ImageEnhance.Sharpness
        if TYPE_CHECKING and not isinstance(
            enhancement,
            type(_Enhancement),
        ):  # pragma: no cover
            raise ValueError("enhancement does not conform to _Enhancement protocol")
        return super()._perturb(enhancement=enhancement, image=perturbed_image), perturbed_boxes
