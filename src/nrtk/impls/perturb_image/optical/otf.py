"""pyBSM OTF perturber implementations."""

_OTF_CLASSES = [
    "CircularAperturePerturber",
    "DefocusPerturber",
    "DetectorPerturber",
    "JitterPerturber",
    "TurbulenceAperturePerturber",
]

__all__: list[str] = list()

try:
    from nrtk.impls.perturb_image.optical._pybsm.circular_aperture_perturber import (
        CircularAperturePerturber as CircularAperturePerturber,
    )
    from nrtk.impls.perturb_image.optical._pybsm.defocus_perturber import (
        DefocusPerturber as DefocusPerturber,
    )
    from nrtk.impls.perturb_image.optical._pybsm.detector_perturber import (
        DetectorPerturber as DetectorPerturber,
    )
    from nrtk.impls.perturb_image.optical._pybsm.jitter_perturber import (
        JitterPerturber as JitterPerturber,
    )
    from nrtk.impls.perturb_image.optical._pybsm.turbulence_aperture_perturber import (
        TurbulenceAperturePerturber as TurbulenceAperturePerturber,
    )

    # Override __module__ to reflect the public API path for plugin discovery
    CircularAperturePerturber.__module__ = __name__
    DefocusPerturber.__module__ = __name__
    DetectorPerturber.__module__ = __name__
    JitterPerturber.__module__ = __name__
    TurbulenceAperturePerturber.__module__ = __name__

    __all__ += _OTF_CLASSES
except ImportError:
    pass


def __getattr__(name: str) -> None:
    if name in _OTF_CLASSES:
        raise ImportError(
            f"{name} requires the `pybsm` extra. Install with: `pip install nrtk[pybsm]`",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
