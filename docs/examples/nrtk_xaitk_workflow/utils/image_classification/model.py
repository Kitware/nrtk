"""Utilities for interoperability with MAITE IC model protocols."""

_GUARDED_NAMES = ["HuggingFaceMaiteModel"]

__all__: list[str] = []

_import_error: ImportError | None = None

try:
    from image_classification._model import (
        HuggingFaceMaiteModel as HuggingFaceMaiteModel,
    )

    __all__ += _GUARDED_NAMES
except ImportError as _ex:
    _import_error = _ex


def __getattr__(name: str) -> None:
    if name in _GUARDED_NAMES:
        msg = (
            f"{name} requires additional dependencies. "
            f"Install with: `pip install nrtk[maite,pillow] torch transformers`"
        )
        if _import_error is not None:
            msg += (
                f"\n\nIf the extra is already installed, the following upstream error may be the cause:"
                f"\n  {type(_import_error).__name__}: {_import_error}"
            )
        raise ImportError(msg)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
