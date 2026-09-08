"""Module for all PerturbVideo implementations."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from nrtk._guard import Group, guard

if TYPE_CHECKING:
    from nrtk.impls.perturb_video._codec_macroblock_perturber import (
        CodecMacroblockPerturber as CodecMacroblockPerturber,
    )
    from nrtk.impls.perturb_video._framewise_perturber import (
        FramewisePerturber as FramewisePerturber,
    )

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = guard(
    namespace=globals(),
    submodules=["optical", "burn_in"],
    groups=[
        Group(
            symbols={"FramewisePerturber": "nrtk.impls.perturb_video._framewise_perturber"},
            experimental=True,
        ),
        Group(
            symbols={"CodecMacroblockPerturber": "nrtk.impls.perturb_video._codec_macroblock_perturber"},
            extras=["pyav"],
            experimental=True,
        ),
    ],
)
