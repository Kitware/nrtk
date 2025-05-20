"""This module contains definitions for non-task specific MAITE-compliant metadata"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from nrtk.utils._exceptions import MaiteImportError

try:
    from maite.protocols import DatumMetadata

    maite_available = True

except ImportError:  # pragma: no cover
    maite_available = False

from typing_extensions import NotRequired, ReadOnly

if not maite_available:
    raise MaiteImportError


@dataclass
class NRTKDatumMetadata(DatumMetadata):  # pyright:  ignore [reportPossiblyUnboundVariable]
    """Dataclass for NRTK-perturbed datum-level metdata"""

    # pyright fails when failing to import maite.protocols
    nrtk_perturber_config: NotRequired[ReadOnly[Sequence[dict[str, Any]]]]
    nrtk_metric: NotRequired[ReadOnly[Sequence[tuple[str, float]]]]


def _forward_md_keys(
    md: DatumMetadata,  # pyright:  ignore [reportPossiblyUnboundVariable]
    aug_md: NRTKDatumMetadata,
    forwarded_keys: Sequence[str],
) -> NRTKDatumMetadata:
    """Forward input metadata, checking for clobbering"""
    for key, value in md.items():  # pyright:  ignore [reportPossiblyUnboundVariable]
        if key in forwarded_keys:
            continue
        if key in aug_md:
            raise KeyError(f"'{key}' already present in metadata, erroring out to prevent overwrite")
        aug_md[key] = value

    return aug_md
