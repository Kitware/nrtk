"""Object detection utilities for NRTK-XAITK workflow notebooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lazy_loader as lazy

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["dataset", "model"],
    submod_attrs={
        "dataset": ["VisDroneObjectDetectionDataset", "stratified_sample_dataset", "YOLODetectionTarget"],
        "model": ["MaiteYOLODetector"],
    },
)
