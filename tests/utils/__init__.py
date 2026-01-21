"""Shared test utilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _deep_equals_mapping(a: Mapping, b: Mapping, rtol: float, atol: float) -> bool:  # type: ignore[type-arg]
    """Compare two mappings recursively."""
    if a.keys() != b.keys():
        return False
    return all(deep_equals(a=a[k], b=b[k], rtol=rtol, atol=atol) for k in a)


def _deep_equals_sequence(a: list | tuple, b: list | tuple, rtol: float, atol: float) -> bool:  # type: ignore[type-arg]
    """Compare two sequences recursively."""
    if len(a) != len(b):
        return False
    return all(deep_equals(a=x, b=y, rtol=rtol, atol=atol) for x, y in zip(a, b, strict=True))


def _deep_equals_by_type(a: object, b: object, rtol: float, atol: float) -> bool | None:
    """Compare values based on their type. Returns None if type not handled."""
    # Dicts: check keys match, then recursively compare values
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return _deep_equals_mapping(a=a, b=b, rtol=rtol, atol=atol)

    # NumPy arrays: use tolerance-based comparison (|a - b| <= atol + rtol * |b|)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return bool(np.allclose(a, b, rtol=rtol, atol=atol))

    # Lists/tuples: check lengths, then recursively compare elements
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return _deep_equals_sequence(a=a, b=b, rtol=rtol, atol=atol)

    # Floats: use tolerance-based comparison to handle precision issues
    if isinstance(a, float) and isinstance(b, float):
        return bool(np.isclose(a, b, rtol=rtol, atol=atol))

    return None


def deep_equals(*, a: object, b: object, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Recursively compare two structures for equality.

    Handles dicts, sequences, numpy arrays, and floats with appropriate
    comparison methods.

    Args:
        a: First value to compare
        b: Second value to compare
        rtol: Relative tolerance for float/array comparisons
        atol: Absolute tolerance for float/array comparisons

    Returns:
        True if structures are equal within tolerances, False otherwise
    """
    # Types must match exactly (e.g., list vs tuple are not equal)
    if type(a) is not type(b):
        return False

    result = _deep_equals_by_type(a=a, b=b, rtol=rtol, atol=atol)
    if result is not None:
        return result

    # Everything else (int, str, bool, etc.): use standard equality
    return a == b
