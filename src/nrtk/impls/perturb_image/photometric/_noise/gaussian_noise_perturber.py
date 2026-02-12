"""Implements GaussianNoisePerturber for applying Gaussian-distributed noise.

Dependencies:
    - skimage (scikit-image) for noise application.
    - numpy for image data handling.

Example:
    >>> import numpy as np
    >>> image = np.ones((256, 256, 3))
    >>> gauss_noise_perturber = GaussianNoisePerturber(mean=0.0, var=0.05)
    >>> noisy_image, _ = gauss_noise_perturber(image=image)

Note:
    The boxes returned from `perturb` are identical to the boxes passed in.
"""

from __future__ import annotations

__all__ = ["GaussianNoisePerturber"]

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image.photometric._noise.noise_perturber_mixin import GaussianSpeckleNoisePerturberMixin


class GaussianNoisePerturber(GaussianSpeckleNoisePerturberMixin):
    """Adds Gaussian-distributed additive noise to image stimulus."""

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Return image stimulus with Gaussian noise."""
        perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes, **kwargs)
        return self._perturb(image=perturbed_image, mode="gaussian", var=self.var, mean=self.mean), perturbed_boxes
