"""This module contains definitions for non-task specific MAITE-compliant metadata"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from maite.protocols import DatumMetadata
from typing_extensions import NotRequired, ReadOnly


@dataclass
class NRTKDatumMetadata(DatumMetadata):
    """Dataclass for NRTK-perturbed datum-level metdata"""

    nrtk_perturber_config: NotRequired[ReadOnly[Sequence[dict[str, Any]]]]
    nrtk_metric: NotRequired[ReadOnly[Sequence[tuple[str, float]]]]


def _forward_md_keys(
    md: DatumMetadata,
    aug_md: NRTKDatumMetadata,
    forwarded_keys: Sequence[str],
) -> NRTKDatumMetadata:
    """Forward input metadata, checking for clobbering"""
    for key, value in md.items():
        if key in forwarded_keys:
            continue
        if key in aug_md:
            raise KeyError(f"'{key}' already present in metadata, erroring out to prevent overwrite")
        # TODO: Remove ignore after switch to pyright, mypy doesn't have good typed dict support  # noqa: FIX002
        aug_md[key] = value  # type: ignore

    return aug_md
