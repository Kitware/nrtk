"""Module for all implementations of nrtk interfaces."""

from collections.abc import Callable
from typing import Any

import lazy_loader as lazy

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "gen_object_detector_blackbox_response",
        "image_metric",
        "perturb_image",
        "perturb_image_factory",
        "score_detections",
    ],
)
