"""Safe plugin discovery that tolerates broken entrypoints and hides private classes.

Overrides smqtk-core's ``Plugfigurable.get_impls`` so that a single
broken entrypoint (e.g. scipy 1.17 ``array_api_compat`` crash) does
not take down the entire discovery pass, and so that nrtk's own private
implementations stay out of the results.
"""

from __future__ import annotations

__all__ = ["Plugfigurable"]

import contextlib
import io
import logging
import types
import warnings
from typing import TypeVar, cast

from smqtk_core.plugfigurable import Plugfigurable as _Plugfigurable
from smqtk_core.plugin import (
    _collect_types_in_module,
    discover_via_env_var,
    discover_via_subclasses,
    filter_plugin_types,
    get_ns_entrypoints,
)
from typing_extensions import Self, override

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
            with (
                warnings.catch_warnings(),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                warnings.simplefilter("ignore")
                m = ep.load()
        except Exception:
            LOG.debug(  # noqa: FKA100 - %-style logging format
                "Skipping broken entrypoint %r (%s)",
                ep,
                entrypoint_ns,
                exc_info=True,
            )
            continue
        if isinstance(m, types.ModuleType):
            type_set.update(_collect_types_in_module(m))
    return type_set


def _is_nrtk_private(impl: type) -> bool:
    """Whether *impl* is an nrtk implementation detail that must stay out of discovery.

    ``discover_via_subclasses`` walks ``__subclasses__()``, which records every class
    at creation time and consults no name, ``__all__``, ``__dir__``, or entrypoint. A
    private base class or helper therefore reaches discovery the moment anything
    imports it, no matter what its module advertises. The import guard cannot prevent
    this, so it is filtered here instead.

    Only nrtk's own classes are judged: another package's naming is not ours to
    interpret.
    """
    module = getattr(impl, "__module__", "")
    parts = module.split(".")
    if parts[0] != "nrtk":
        return False
    return impl.__name__.startswith("_") or any(part.startswith("_") for part in parts)


class Plugfigurable(_Plugfigurable):
    """Drop-in replacement that swaps in fault-tolerant entrypoint loading."""

    @override
    @classmethod
    def get_impls(cls) -> set[type[Self]]:
        """Discover plugins, skipping broken entrypoints and nrtk's private classes."""
        candidate_types = {
            *discover_via_env_var(cls.PLUGIN_ENV_VAR),
            *_safe_discover_via_entrypoints(cls.PLUGIN_NAMESPACE),
            *discover_via_subclasses(cls),
        }
        valid_types = filter_plugin_types(cls, candidate_types)  # noqa: FKA100 - upstream smqtk-core signature
        return cast(
            set[type[Self]],
            {impl for impl in valid_types if not _is_nrtk_private(impl)},
        )
