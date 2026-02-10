"""Module for generative implementations of PerturbImage."""

_DIFFUSION_CLASSES = ["DiffusionPerturber"]

__all__: list[str] = list()

try:
    from nrtk.impls.perturb_image.generative._diffusion_perturber import (
        DiffusionPerturber as DiffusionPerturber,
    )

    # Override __module__ to reflect the public API path for plugin discovery
    DiffusionPerturber.__module__ = __name__

    __all__ += _DIFFUSION_CLASSES
except ImportError:
    pass


def __getattr__(name: str) -> None:
    if name in _DIFFUSION_CLASSES:
        raise ImportError(
            f"{name} requires the `diffusion` extra. Install with: `pip install nrtk[diffusion]`",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
