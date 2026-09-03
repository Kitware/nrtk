"""The leaf module holding a stand-in implementation, pointed at by the guard.

``get_config`` is implemented because ``smqtk_core.Configurable`` declares it
abstract: an implementation without it is rejected by ``is_valid_plugin`` and
discovery returns nothing, which would make the checks that use this pass
vacuously.
"""

from typing import Any

from typing_extensions import override

from tests._utils.guard_plugin._interface import GuardedThing


class GuardedThingImpl(GuardedThing):
    """Stand-in for an experimental implementation with no optional dependency."""

    @override
    def get_config(self) -> dict[str, Any]:
        """Return an empty configuration; nothing here is configurable."""
        return {}
