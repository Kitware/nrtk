"""Mixin class providing shared functionality for blur perturbers."""

from __future__ import annotations

__all__ = []

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage


class BlurPerturberMixin(PerturbImage):
    def __init__(self, ksize: int = 1) -> None:
        super().__init__()
        self.ksize = ksize

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **additional_params: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Return image stimulus after applying average blurring."""
        _image, _boxes = super().perturb(image=image, boxes=boxes, **additional_params)

        # Check for channel last format
        if _image.ndim == 3 and _image.shape[2] > 4:
            raise ValueError("Image is not in expected format (H, W, C)")

        return _image, _boxes

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the MedianBlurPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = super().get_config()
        cfg["ksize"] = self.ksize
        return cfg
