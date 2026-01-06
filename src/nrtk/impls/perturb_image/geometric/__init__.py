"""Module for geometric implementations of PerturbImage."""

from collections.abc import Callable
from typing import Any

import lazy_loader as lazy

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "random_crop_perturber",
        "random_rotation_perturber",
        "random_scale_perturber",
        "random_translation_perturber",
    ],
)
