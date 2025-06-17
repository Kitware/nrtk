"""This module contains definitions for non-task specific MAITE-compliant metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from typing_extensions import NotRequired, ReadOnly

DatumMetadata: type = object
try:
    # Multiple type ignores added for pyright's handling of guarded imports
    from maite.protocols import DatumMetadata

    maite_available: bool = True

except ImportError:  # pragma: no cover
    maite_available: bool = False


@dataclass
class NRTKDatumMetadata(DatumMetadata):  # pyright:  ignore [reportGeneralTypeIssues]
    """Dataclass for NRTK-perturbed datum-level metdata."""

    # pyright fails when failing to import maite.protocols
    nrtk_perturber_config: NotRequired[ReadOnly[Sequence[dict[str, Any]]]]  # pyright: ignore [reportInvalidTypeForm]
    nrtk_metric: NotRequired[ReadOnly[Sequence[tuple[str, float]]]]  # pyright: ignore [reportInvalidTypeForm]


def _forward_md_keys(
    md: DatumMetadata,  # pyright:  ignore [reportInvalidTypeForm]
    aug_md: NRTKDatumMetadata,
    forwarded_keys: Sequence[str],
) -> NRTKDatumMetadata:
    """Forward input metadata, checking for clobbering."""
    for key, value in md.items():
        if key in forwarded_keys:
            continue
        if key in aug_md:
            raise KeyError(f"'{key}' already present in metadata, erroring out to prevent overwrite")
        aug_md[key] = value

    return aug_md
