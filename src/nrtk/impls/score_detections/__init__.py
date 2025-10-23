"""Module for the scoring implementations."""

from collections.abc import Callable
from typing import Any

import lazy_loader as lazy

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["class_agnostic_pixelwise_iou_scorer", "coco_scorer", "nop_scorer", "random_scorer"],
)
