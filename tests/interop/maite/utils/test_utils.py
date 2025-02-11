from collections.abc import Hashable, Iterable
from typing import Any, Optional

import numpy as np
from PIL import Image
from smqtk_image_io import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image import PerturbImage


class ResizePerturber(PerturbImage):
    def __init__(self, w: int, h: int) -> None:
        self.w = w
        self.h = h

    def perturb(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        """Resize image."""
        if additional_params is None:
            additional_params = {}
        img = Image.fromarray(image)
        img = img.resize((self.w, self.h))
        return np.array(img), boxes

    def get_config(self) -> dict[str, Any]:
        return {"w": self.w, "h": self.h}
