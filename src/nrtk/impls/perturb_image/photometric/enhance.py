"""Enhancement perturbers using PIL."""

from nrtk.impls.perturb_image.photometric._impl.enhance import (
    BrightnessPerturber,
    ColorPerturber,
    ContrastPerturber,
    SharpnessPerturber,
)

# Override __module__ to reflect the public API path for plugin discovery
BrightnessPerturber.__module__ = __name__
ColorPerturber.__module__ = __name__
ContrastPerturber.__module__ = __name__
SharpnessPerturber.__module__ = __name__

__all__ = [
    "BrightnessPerturber",
    "ColorPerturber",
    "ContrastPerturber",
    "SharpnessPerturber",
]
