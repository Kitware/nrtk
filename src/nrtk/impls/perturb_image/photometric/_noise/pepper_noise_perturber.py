"""Implements PepperNoisePerturber for applying pepper noise.

Dependencies:
    - skimage (scikit-image) for noise application.
    - numpy for image data handling.

Example:
    >>> import numpy as np
    >>> image = np.ones((256, 256, 3))
    >>> pepper_noise_perturber = PepperNoisePerturber(amount=0.05)
    >>> noisy_image, _ = pepper_noise_perturber(image=image)

Note:
    The boxes returned from `perturb` are identical to the boxes passed in.
"""

from __future__ import annotations

__all__ = ["PepperNoisePerturber"]

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image.photometric._noise.noise_perturber_mixin import SaltPepperNoisePerturberMixin


class PepperNoisePerturber(SaltPepperNoisePerturberMixin):
    """Adds pepper noise to image stimulus.

    Attributes:
        rng (np.random.Generator | int | None):
            Pseudo-random number generator or seed
        clip (bool):
            Decide if output is clipped between the range of [-1, 1].
        amount (float):
            Proportion of image pixels to replace with pepper noise on range [0, 1]
    """

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Return image stimulus with pepper noise."""
        super().perturb(image=image, boxes=boxes, **kwargs)
        return self._perturb(image=image, mode="pepper", amount=self.amount), boxes
