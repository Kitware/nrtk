"""Module for PyBSM based implementations of PerturbImage."""

from collections.abc import Callable
from typing import Any

import lazy_loader as lazy

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "circular_aperture_otf_perturber",
        "defocus_otf_perturber",
        "detector_otf_perturber",
        "jitter_otf_perturber",
        "pybsm_perturber",
        "scenario",
        "sensor",
        "turbulence_aperture_otf_perturber",
    ],
)
