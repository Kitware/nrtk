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


@pytest.fixture(scope="session", autouse=True)
def set_numpy_printoptions() -> None:
    """Sets global NumPy print options for the entire test session."""
    np.set_printoptions(
        edgeitems=3,
        threshold=1000,
        floatmode="maxprec",
        precision=8,
        suppress=False,
        linewidth=75,
        nanstr="nan",
        infstr="inf",
        sign="-",
        formatter=None,
        legacy=False,
    )


class FuzzyFloatSnapshotExtension(JSONSnapshotExtension):
    def __init__(self, *, rtol: float = 1e-4, atol: float = 1e-5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rtol = rtol
        self.atol = atol

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
                return math.isclose(expected, received, rel_tol=self.rtol, abs_tol=self.atol)

            if isinstance(received, np.ndarray) and isinstance(expected, np.ndarray):
                return np.allclose(expected, received, rtol=self.rtol, atol=self.atol)

            return serialized_data == snapshot_data
        except ValueError:
            return False


@pytest.fixture
def fuzzy_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(FuzzyFloatSnapshotExtension)


class WeakFuzzySnapshotExtension(FuzzyFloatSnapshotExtension):
    def __init__(self, *, rtol: float = 0.1, atol: float = 1e-5, **kwargs: Any) -> None:
        super().__init__(rtol=rtol, atol=atol, **kwargs)


# Should not be used without maintainer approval
@pytest.fixture
def weak_fuzzy_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(WeakFuzzySnapshotExtension)


class TIFFImageSnapshotExtension(SingleFileSnapshotExtension):
    _file_extension = "tiff"

    def __init__(self, *, rtol: float = 1e-4, atol: float = 1e-5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rtol = rtol
        self.atol = atol

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

        return np.allclose(
            expected_array / 255,
            received_array / 255,
            rtol=self.rtol,
            atol=self.atol,
        )


@pytest.fixture
def tiff_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(TIFFImageSnapshotExtension)


class WeakTIFFImageSnapshotExtension(TIFFImageSnapshotExtension):
    def __init__(self, *, rtol: float = 0.1, atol: float = 1e-5, **kwargs: Any) -> None:
        super().__init__(rtol=rtol, atol=atol, **kwargs)


# Should not be used without maintainer approval
@pytest.fixture
def weak_tiff_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(WeakTIFFImageSnapshotExtension)


class PSNRImageSnapshotExtension(SingleFileSnapshotExtension):
    """Snapshot extension using PSNR metric for image comparison.

    This extension compares images using Peak Signal-to-Noise Ratio (PSNR)
    instead of element-wise numerical comparison. Higher PSNR values indicate
    more similar images. Images pass if their PSNR exceeds a threshold. The
    default threshold of 48.13 corresponds to the psnr for uint8 images
    where each pixel value is off by 1.

    Args:
        min_psnr: Minimum PSNR value in dB required to pass (default: 48.13)
    """

    _file_extension = "tiff"

    def __init__(self, *, min_psnr: float = 48.13, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_psnr = min_psnr

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

        # Ensure images have same shape before computing metric
        if expected_array.shape != received_array.shape:
            return False

        # Compute Mean Squared Error
        mse = np.mean((expected_array.astype(float) - received_array.astype(float)) ** 2)

        # To get MAX value, we assume it is a float normalized between 0-1
        # or a uint8
        max_pixel_value = 1.0 if np.issubdtype(received_array.dtype, np.floating) else 255.0

        # If MSE is zero, the images are identical, so PSNR is infinity
        # otherise, PSNR = 10 * log10(MAX^2 / MSE)
        psnr = float(np.inf) if mse == 0.0 else 10 * np.log10((max_pixel_value**2) / mse)

        # Pass if metric value meets or exceeds minimum threshold
        return psnr >= self.min_psnr


@pytest.fixture
def psnr_tiff_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(PSNRImageSnapshotExtension)
