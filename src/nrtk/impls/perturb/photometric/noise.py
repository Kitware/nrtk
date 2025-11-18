"""Random noise perturbers using skimage."""

from nrtk.impls.perturb.photometric._impl.noise import (
    GaussianNoisePerturber,
    PepperNoisePerturber,
    SaltAndPepperNoisePerturber,
    SaltNoisePerturber,
    SpeckleNoisePerturber,
)

# Override __module__ to reflect the public API path for plugin discovery
GaussianNoisePerturber.__module__ = __name__
PepperNoisePerturber.__module__ = __name__
SaltAndPepperNoisePerturber.__module__ = __name__
SaltNoisePerturber.__module__ = __name__
SpeckleNoisePerturber.__module__ = __name__

__all__ = [
    "GaussianNoisePerturber",
    "PepperNoisePerturber",
    "SaltAndPepperNoisePerturber",
    "SaltNoisePerturber",
    "SpeckleNoisePerturber",
]
