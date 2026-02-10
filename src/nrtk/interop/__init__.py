"""Define the nrtk.interop package."""

_MAITE_CLASSES = [
    "MAITEImageClassificationAugmentation",
    "MAITEObjectDetectionAugmentation",
]

__all__: list[str] = list()

try:
    from nrtk.interop._maite.augmentations import (
        MAITEImageClassificationAugmentation as MAITEImageClassificationAugmentation,
    )
    from nrtk.interop._maite.augmentations import (
        MAITEObjectDetectionAugmentation as MAITEObjectDetectionAugmentation,
    )

    # Override __module__ to reflect the public API path
    MAITEImageClassificationAugmentation.__module__ = __name__
    MAITEObjectDetectionAugmentation.__module__ = __name__

    __all__ += _MAITE_CLASSES
except ImportError:
    pass


def __getattr__(name: str) -> None:
    if name in _MAITE_CLASSES:
        raise ImportError(
            f"{name} requires the `maite` extra. Install with: `pip install nrtk[maite]`",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
