"""Public API for MAITE datum-level metadata definitions.

This module provides import guards for optional dependencies:
- NRTKDatumMetadata: requires ``maite`` extra
"""

from __future__ import annotations

_MAITE_CLASSES = [
    "NRTKDatumMetadata",
]

__all__: list[str] = []

try:
    from nrtk.interop._maite.metadata._nrtk_datum_metadata import (
        NRTKDatumMetadata as NRTKDatumMetadata,
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
