"""Utilities for interoperability with MAITE OD dataset protocols."""

_GUARDED_NAMES = ["VisDroneObjectDetectionDataset", "stratified_sample_dataset", "YOLODetectionTarget"]

__all__: list[str] = []

_import_error: ImportError | None = None

try:
    from object_detection._dataset import (
        VisDroneObjectDetectionDataset as VisDroneObjectDetectionDataset,
    )
    from object_detection._dataset import (
        YOLODetectionTarget as YOLODetectionTarget,
    )
    from object_detection._dataset import (
        stratified_sample_dataset as stratified_sample_dataset,
    )

    __all__ += _GUARDED_NAMES
except ImportError as _ex:
    _import_error = _ex


def __getattr__(name: str) -> None:
    if name in _GUARDED_NAMES:
        msg = f"{name} requires additional dependencies. Install with: `pip install nrtk[maite,pillow] torch`"
        if _import_error is not None:
            msg += (
                f"\n\nIf the extra is already installed, the following upstream error may be the cause:"
                f"\n  {type(_import_error).__name__}: {_import_error}"
            )
        raise ImportError(msg)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
