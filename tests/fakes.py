"""Fake perturbers for testing perturb image factories."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage


class FakePerturber(PerturbImage):
    """Fake perturber for testing purposes.

    Accepts arbitrary keyword arguments and returns them via get_config().
    This allows it to be used with any theta_key in factory tests.

    Default parameters param1 and param2 are provided for common test cases.
    """

    def __init__(self, *, param1: float = 1, param2: float = 2, **kwargs: Any) -> None:
        self.param1 = param1
        self.param2 = param2
        self._extra_kwargs = kwargs

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **_: Any,
    ) -> tuple[
        np.ndarray[Any, Any],
        Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
    ]:  # pragma: no cover
        return np.copy(image), boxes

    @override
    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"param1": self.param1, "param2": self.param2}
        config.update(self._extra_kwargs)
        return config
