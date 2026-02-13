"""Random geometric perturbers."""

from nrtk.impls.perturb_image.geometric._random.random_crop_perturber import RandomCropPerturber
from nrtk.impls.perturb_image.geometric._random.random_translation_perturber import RandomTranslationPerturber

# Override __module__ to reflect the public API path for plugin discovery
RandomCropPerturber.__module__ = __name__
RandomTranslationPerturber.__module__ = __name__

__all__ = ["RandomCropPerturber", "RandomTranslationPerturber"]

# Albumentations-based perturbers (optional)
_ALBUMENTATIONS_CLASSES = ["RandomRotationPerturber", "RandomScalePerturber"]

try:
    from nrtk.impls.perturb_image._albumentations.random_rotation_perturber import (
        RandomRotationPerturber as RandomRotationPerturber,
    )
    from nrtk.impls.perturb_image._albumentations.random_scale_perturber import (
        RandomScalePerturber as RandomScalePerturber,
    )

    RandomRotationPerturber.__module__ = __name__
    RandomScalePerturber.__module__ = __name__

    __all__ += _ALBUMENTATIONS_CLASSES
except ImportError:
    pass


def __getattr__(name: str) -> None:
    if name in _ALBUMENTATIONS_CLASSES:
        raise ImportError(
            f"{name} requires the `albumentations` and (`graphics` or `headless`) extras. "
            f"Install with: `pip install nrtk[albumentations,graphics]` or `pip install nrtk[albumentations,headless]`",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
