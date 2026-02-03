"""Safe plugin discovery that tolerates broken third-party entrypoints.

Overrides smqtk-core's ``Plugfigurable.get_impls`` so that a single
broken entrypoint (e.g. scipy 1.17 ``array_api_compat`` crash) does
not take down the entire discovery pass.
"""

from __future__ import annotations

import logging
import types
from typing import TypeVar, cast

from smqtk_core.plugfigurable import Plugfigurable as _Plugfigurable
from smqtk_core.plugin import (
    _collect_types_in_module,
    discover_via_env_var,
    discover_via_subclasses,
    filter_plugin_types,
    get_ns_entrypoints,
)
from typing_extensions import Self

LOG = logging.getLogger(__name__)

P = TypeVar("P", bound="Plugfigurable")


def _safe_discover_via_entrypoints(entrypoint_ns: str) -> set[type]:
    """Like ``discover_via_entrypoint_extensions`` but tolerates failures.

    Each entrypoint is loaded inside its own try/except so that one
    broken third-party plugin cannot prevent discovery of the others.
    """
    type_set: set[type] = set()
    for ep in get_ns_entrypoints(entrypoint_ns):
        try:
            m = ep.load()
        except Exception:  # noqa: BLE001 - intentionally broad to tolerate any broken entrypoint
            LOG.debug(  # noqa: FKA100, RUF100 - %-style logging format
                "Skipping broken entrypoint %r (%s)",
                ep,
                entrypoint_ns,
                exc_info=True,
            )
            continue
        if isinstance(m, types.ModuleType):
            type_set.update(_collect_types_in_module(m))
    return type_set


class Plugfigurable(_Plugfigurable):
    """Drop-in replacement that swaps in fault-tolerant entrypoint loading."""

    @classmethod
    def get_impls(cls) -> set[type[Self]]:
        """Discover plugins, skipping any entrypoints that fail to load."""
        candidate_types = {
            *discover_via_env_var(cls.PLUGIN_ENV_VAR),
            *_safe_discover_via_entrypoints(cls.PLUGIN_NAMESPACE),
            *discover_via_subclasses(cls),
        }
        return cast(
            set[type[Self]],
            filter_plugin_types(cls, candidate_types),  # noqa: FKA100, RUF100 - upstream smqtk-core signature
        )
