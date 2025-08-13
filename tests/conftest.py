import io
import json
import math
from typing import Any

import numpy as np
import pytest
from PIL import Image
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.json import JSONSnapshotExtension
from syrupy.extensions.single_file import SingleFileSnapshotExtension


def ndarray_isclose(*, computed: np.ndarray, expected: np.ndarray, rtol: float, atol: float, int_tol: float) -> bool:
    if np.issubdtype(computed.dtype, np.integer):
        return bool(np.average(np.abs(expected - computed)) < int_tol)
    return np.allclose(computed, expected, rtol=rtol, atol=atol)


class FuzzyFloatSnapshotExtension(JSONSnapshotExtension):
    def __init__(self, *, rtol: float = 1e-5, atol: float = 1e-8, int_tol: float = 1e-3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rtol = rtol
        self.atol = atol
        self.int_tol = int_tol

    def serialize(self, data: np.ndarray | float, **_: Any) -> str:
        if isinstance(data, np.ndarray):
            return (
                json.dumps(
                    {
                        "__type__": "ndarray",
                        "dtype": str(data.dtype),
                        "shape": data.shape,
                        "data": data.tolist(),
                    },
                )
                + "\n"
            )
        return json.dumps({"__type__": "float", "value": data}) + "\n"

    def deserialize(self, data: str) -> np.ndarray | float:
        parsed = json.loads(data)
        if isinstance(parsed, dict):
            if parsed.get("__type__") == "ndarray":
                return np.array(parsed["data"], dtype=parsed["dtype"]).reshape(parsed["shape"])
            if parsed.get("__type__") == "float":
                return float(parsed["value"])
        raise ValueError("Unknown data type")

    def matches(self, *, serialized_data: str, snapshot_data: str) -> bool:
        try:
            expected = self.deserialize(snapshot_data)
            received = self.deserialize(serialized_data)

            if isinstance(received, float) and isinstance(expected, float):
                return math.isclose(received, expected, rel_tol=self.rtol, abs_tol=self.atol)

            if isinstance(received, np.ndarray) and isinstance(expected, np.ndarray):
                return ndarray_isclose(
                    computed=received,
                    expected=expected,
                    rtol=self.rtol,
                    atol=self.atol,
                    int_tol=self.int_tol,
                )

            return serialized_data == snapshot_data
        except ValueError:
            return False


@pytest.fixture
def fuzzy_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(FuzzyFloatSnapshotExtension)


class TIFFImageSnapshotExtension(SingleFileSnapshotExtension):
    _file_extension = "tiff"

    def __init__(self, *, tol: float = 1e-3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tol = tol

    def serialize(self, data: np.ndarray, **_: Any) -> bytes:
        im = Image.fromarray(data)
        byte_arr = io.BytesIO()
        im.save(byte_arr, format="tiff")
        return byte_arr.getvalue()

    def deserialize(self, data: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(data)) as image:
            # Force load image data to avoid lazy loading
            image.load()
            return np.array(image)

    def matches(self, *, serialized_data: bytes, snapshot_data: bytes) -> bool:
        expected_array = self.deserialize(snapshot_data)
        received_array = self.deserialize(serialized_data)

        return ndarray_isclose(
            computed=received_array,
            expected=expected_array,
            rtol=self.tol,
            atol=self.tol,
            int_tol=self.tol,
        )


@pytest.fixture
def tiff_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(TIFFImageSnapshotExtension)
