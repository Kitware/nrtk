"""Blur perturbers using cv2."""

from nrtk.impls.perturb_image.generic._cv2.blur import (
    AverageBlurPerturber,
    GaussianBlurPerturber,
    MedianBlurPerturber,
)

# Override __module__ to reflect the public API path for plugin discovery
AverageBlurPerturber.__module__ = __name__
GaussianBlurPerturber.__module__ = __name__
MedianBlurPerturber.__module__ = __name__

__all__ = [
    "AverageBlurPerturber",
    "GaussianBlurPerturber",
    "MedianBlurPerturber",
]
