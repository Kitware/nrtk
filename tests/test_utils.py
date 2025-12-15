from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage


class DummyPerturber(PerturbImage):
    """Shared dummy perturber for testing purposes."""

    def __init__(self, *, param1: float = 1, param2: float = 2) -> None:
        self.param1 = param1
        self.param2 = param2

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
        return {"param1": self.param1, "param2": self.param2}
