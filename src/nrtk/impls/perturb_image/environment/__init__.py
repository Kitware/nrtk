"""Module for environment implementations of PerturbImage."""

from nrtk.impls.perturb_image.environment._haze_perturber import HazePerturber

# Override __module__ to reflect the public API path for plugin discovery
HazePerturber.__module__ = __name__

__all__ = ["HazePerturber"]

# Water droplet perturber (optional - requires scipy and numba)
_WATERDROPLET_CLASSES = ["WaterDropletPerturber"]

try:
    from nrtk.impls.perturb_image.environment._water_droplet_perturber import (
        WaterDropletPerturber as WaterDropletPerturber,
    )

    WaterDropletPerturber.__module__ = __name__

    __all__ += _WATERDROPLET_CLASSES
except ImportError:
    pass


def __getattr__(name: str) -> None:
    if name in _WATERDROPLET_CLASSES:
        raise ImportError(
            f"{name} requires the `waterdroplet` extra. Install with: `pip install nrtk[waterdroplet]`",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
