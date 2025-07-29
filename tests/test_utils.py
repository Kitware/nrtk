from __future__ import annotations

import ast
import math
import re
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.extensions.amber import AmberSnapshotExtension

from nrtk.interfaces.perturb_image import PerturbImage


class CustomFloatSnapshotExtension(AmberSnapshotExtension):
    def parse_snapshot_to_numpy_no_eval(self, snapshot: str) -> tuple[np.ndarray]:
        # Remove metadata lines starting with `#`
        snapshot = "\n".join(line for line in snapshot.splitlines() if not line.strip().startswith("#"))

        # Extract array strings using regex
        array_pattern = r"array\((\[.*?\])\)"
        matches = re.findall(array_pattern, snapshot, flags=re.S)

        # Parse each array string into a NumPy array
        arrays = []
        for match in matches:
            # Replace "..." with the repeating last row/column to avoid parsing errors
            cleaned_array = match.replace("...,", "")
            # Convert the array string into a NumPy array using `np.array` and `eval`
            arrays.append(
                np.array(ast.literal_eval(cleaned_array)),
            )  # Use `eval` only for literals, not the whole snapshot
        return tuple(arrays)

    def matches(self, *, serialized_data: str, snapshot_data: str) -> bool:
        try:
            # Convert serialized and snapshot data to floats and compare within tolerance
            a = float(serialized_data)
            b = float(snapshot_data)
            return math.isclose(a, b, abs_tol=1e-4)
        except ValueError:
            # If conversion to float fails, fallback to default comparison
            pass
        try:
            # Convert serialized and snapshot data to np arrays and compare within tolerance
            a = self.parse_snapshot_to_numpy_no_eval(serialized_data)
            b = self.parse_snapshot_to_numpy_no_eval(snapshot_data)
            for array_a, array_b in zip(a, b, strict=False):
                if not all(
                    math.isclose(array_a[index], array_b[index], rel_tol=1e-4) for index in np.ndindex(array_a.shape)
                ):
                    return False
            return True
        except ValueError:
            return serialized_data == snapshot_data


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
