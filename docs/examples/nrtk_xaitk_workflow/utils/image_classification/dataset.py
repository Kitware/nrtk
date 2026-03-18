"""Utilities for interoperability with MAITE IC dataset protocols."""

_GUARDED_NAMES = ["HuggingFaceMaiteDataset", "create_data_subset"]

__all__: list[str] = []

_import_error: ImportError | None = None

try:
    from image_classification._dataset import (
        HuggingFaceMaiteDataset as HuggingFaceMaiteDataset,
    )
    from image_classification._dataset import (
        create_data_subset as create_data_subset,
    )

    __all__ += _GUARDED_NAMES
except ImportError as _ex:
    _import_error = _ex


def __getattr__(name: str) -> None:
    if name in _GUARDED_NAMES:
        msg = f"{name} requires additional dependencies. Install with: `pip install nrtk[maite,pillow] datasets`"
        if _import_error is not None:
            msg += (
                f"\n\nIf the extra is already installed, the following upstream error may be the cause:"
                f"\n  {type(_import_error).__name__}: {_import_error}"
            )
        raise ImportError(msg)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
