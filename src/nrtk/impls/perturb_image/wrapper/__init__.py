"""Module for wrapper implementations of PerturbImage."""

from nrtk.impls.perturb_image.wrapper._compose_perturber import ComposePerturber

# Override __module__ to reflect the public API path for plugin discovery
ComposePerturber.__module__ = __name__

__all__ = ["ComposePerturber"]

# Albumentations-based perturbers (optional)
_ALBUMENTATIONS_CLASSES = ["AlbumentationsPerturber"]

try:
    from nrtk.impls.perturb_image.wrapper._albumentations.albumentations_perturber import (
        AlbumentationsPerturber as AlbumentationsPerturber,
    )

    AlbumentationsPerturber.__module__ = __name__

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
