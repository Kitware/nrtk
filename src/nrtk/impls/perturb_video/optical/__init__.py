"""Module for optical implementations of :class:`nrtk.interfaces.PerturbVideo`."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from nrtk._guard import Group, guard

if TYPE_CHECKING:
    from nrtk.impls.perturb_video.optical._hcipy.turbulence_video_perturber import (
        TurbulenceVideoPerturber as TurbulenceVideoPerturber,
    )
    from nrtk.impls.perturb_video.optical._jitter.jitter_video_perturber import (
        JitterVideoPerturber as JitterVideoPerturber,
    )

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = guard(
    namespace=globals(),
    groups=[
        Group(
            symbols={
                "TurbulenceVideoPerturber": "nrtk.impls.perturb_video.optical._hcipy.turbulence_video_perturber",
            },
            extras=["hcipy", "scipy"],
            experimental=True,
        ),
        Group(
            symbols={
                "JitterVideoPerturber": "nrtk.impls.perturb_video.optical._jitter.jitter_video_perturber",
            },
            extras=["scipy"],
            experimental=True,
        ),
    ],
)
