"""Module for optical implementations of PerturbImage."""

import nrtk.impls.perturb_image.optical.otf as otf
from nrtk.impls.perturb_image.optical._radial_distortion_perturber import (
    RadialDistortionPerturber as RadialDistortionPerturber,
)

RadialDistortionPerturber.__module__ = __name__

_PYBSM_CLASSES = [
    "PybsmPerturber",
]

__all__: list[str] = ["RadialDistortionPerturber", "otf"]

_import_error: ImportError | None = None

try:
    from nrtk.impls.perturb_image.optical._pybsm_perturber import (
        PybsmPerturber as PybsmPerturber,
    )

    # Override __module__ to reflect the public API path for plugin discovery
    PybsmPerturber.__module__ = __name__

    __all__ += _PYBSM_CLASSES
except ImportError as _ex:
    _import_error = _ex


def __getattr__(name: str) -> None:
    if name in _PYBSM_CLASSES:
        msg = f"{name} requires the `pybsm` extra. Install with: `pip install nrtk[pybsm]`"
        if _import_error is not None:
            msg += (
                f"\n\nIf the extra is already installed, the following upstream error may be the cause:"
                f"\n  {type(_import_error).__name__}: {_import_error}"
            )
        raise ImportError(msg)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
