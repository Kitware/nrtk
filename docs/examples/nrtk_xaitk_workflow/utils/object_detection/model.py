"""A wrapper class for a YOLO model to simplify its usage with input batches and object detection targets."""

_GUARDED_NAMES = ["MaiteYOLODetector"]

__all__: list[str] = []

_import_error: ImportError | None = None

try:
    from object_detection._model import (
        MaiteYOLODetector as MaiteYOLODetector,
    )

    __all__ += _GUARDED_NAMES
except ImportError as _ex:
    _import_error = _ex


def __getattr__(name: str) -> None:
    if name in _GUARDED_NAMES:
        msg = f"{name} requires additional dependencies. Install with: `pip install nrtk[maite] torch ultralytics`"
        if _import_error is not None:
            msg += (
                f"\n\nIf the extra is already installed, the following upstream error may be the cause:"
                f"\n  {type(_import_error).__name__}: {_import_error}"
            )
        raise ImportError(msg)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
