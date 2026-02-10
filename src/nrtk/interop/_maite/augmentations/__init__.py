"""Public API for MAITE augmentation wrappers.

This module provides import guards for optional dependencies:
- MAITEImageClassificationAugmentation, MAITEObjectDetectionAugmentation:
  require ``maite`` extra
"""

from __future__ import annotations

_MAITE_CLASSES = [
    "MAITEImageClassificationAugmentation",
    "MAITEObjectDetectionAugmentation",
]

__all__: list[str] = list()

try:
    from nrtk.interop._maite.augmentations._maite_image_classification_augmentation import (
        MAITEImageClassificationAugmentation as MAITEImageClassificationAugmentation,
    )
    from nrtk.interop._maite.augmentations._maite_object_detection_augmentation import (
        MAITEObjectDetectionAugmentation as MAITEObjectDetectionAugmentation,
    )

    __all__ += _MAITE_CLASSES
except ImportError:
    pass


def __getattr__(name: str) -> None:
    if name in _MAITE_CLASSES:
        raise ImportError(
            f"{name} requires the `maite` extra. Install with: `pip install nrtk[maite]`",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
