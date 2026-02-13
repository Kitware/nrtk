"""Module for optical implementations of PerturbImage."""

from nrtk.impls.perturb_image.optical.radial_distortion_perturber import (
    RadialDistortionPerturber as RadialDistortionPerturber,
)

RadialDistortionPerturber.__module__ = __name__

_PYBSM_CLASSES = [
    "PybsmPerturber",
]

__all__: list[str] = ["RadialDistortionPerturber"]

try:
    from nrtk.impls.perturb_image.optical._pybsm_perturber import (
        PybsmPerturber as PybsmPerturber,
    )

    # Override __module__ to reflect the public API path for plugin discovery
    PybsmPerturber.__module__ = __name__

    __all__ += _PYBSM_CLASSES
except ImportError:
    pass


def __getattr__(name: str) -> None:
    if name in _PYBSM_CLASSES:
        raise ImportError(
            f"{name} requires the `pybsm` extra. Install with: `pip install nrtk[pybsm]`",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
