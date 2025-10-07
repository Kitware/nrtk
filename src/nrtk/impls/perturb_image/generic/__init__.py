"""Module for all generic implementations of PerturbImage."""

from collections.abc import Callable
from typing import Any

import lazy_loader as lazy

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "albumentations_perturber",
        "compose_perturber",
        "cv2",
        "diffusion_perturber",
        "haze_perturber",
        "nop_perturber",
        "PIL",
        "radial_distortion_perturber",
        "random_crop_perturber",
        "random_rotation_perturber",
        "random_translation_perturber",
        "skimage",
        "water_droplet_perturber",
    ],
)
