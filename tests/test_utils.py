from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image import PerturbImage


class DummyPerturber(PerturbImage):
    """Shared dummy perturber for testing purposes."""

    def __init__(self, param1: float = 1, param2: float = 2) -> None:
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def perturb(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        additional_params: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[
        np.ndarray,
        Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
    ]:  # pragma: no cover
        return np.copy(image), boxes

    def get_config(self) -> dict[str, Any]:
        return {"param1": self.param1, "param2": self.param2}
