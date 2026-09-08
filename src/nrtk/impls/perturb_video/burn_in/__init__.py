"""Module for burn-in implementations of :class:`nrtk.interfaces.PerturbVideo`."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from nrtk._guard import Group, guard

if TYPE_CHECKING:
    from nrtk.impls.perturb_video.burn_in._misb_st1909_burn_in_perturber import (
        MISBST1909BurnInPerturber as MISBST1909BurnInPerturber,
    )

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = guard(
    namespace=globals(),
    groups=[
        Group(
            symbols={
                "MISBST1909BurnInPerturber": "nrtk.impls.perturb_video.burn_in._misb_st1909_burn_in_perturber",
            },
            extras=["pillow"],
            experimental=True,
        ),
    ],
)
