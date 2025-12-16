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


class SSIMImageSnapshotExtension(SingleFileSnapshotExtension):
    """Snapshot extension using SSIM metric for image comparison.

    This extension compares images using the Structural Similarity Index Measure
    (SSIM) following the Wang et al. (2004) formulation with per-channel windowed
    computation. SSIM is computed over local 11×11 Gaussian windows (σ=1.5) and
    averaged across channels and windows (Mean SSIM). Images pass if their SSIM
    exceeds a threshold. A threshold of 0.99 corresponds to industry
    standards used in video codec testing. To allow for machine level differences,
    a default threshold of 0.985 was chosen.

    Args:
        min_ssim: Minimum SSIM value in [0, 1] required to pass (default: 0.985).
    """

    _file_extension = "tiff"

    def __init__(self, *, min_ssim: float = 0.985, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_ssim = min_ssim

    def serialize(self, data: np.ndarray, **_: Any) -> bytes:
        im = Image.fromarray(data)
        byte_arr = io.BytesIO()
        im.save(byte_arr, format="tiff")
        return byte_arr.getvalue()

    def deserialize(self, data: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return np.array(image)

    @staticmethod
    def _gaussian_kernel(*, size: int = 11, sigma: float = 1.5) -> np.ndarray:
        x = np.arange(size) - (size - 1) / 2.0
        gauss = np.exp(-(x**2) / (2 * sigma**2))
        return gauss / gauss.sum()

    @staticmethod
    def _convolve2d(*, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        h, w = img.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        output = np.zeros_like(img)

        for i in range(h):
            for j in range(w):
                window = padded[i : i + kh, j : j + kw]
                output[i, j] = np.sum(window * kernel)

        return output

    def _compute_ssim_windowed(self, *, img_a: np.ndarray, img_b: np.ndarray) -> float:
        img_a = img_a.astype(np.float64, copy=False)
        img_b = img_b.astype(np.float64, copy=False)

        if np.issubdtype(img_a.dtype, np.floating) or np.issubdtype(
            img_b.dtype,
            np.floating,
        ):
            dynamic_range = 1.0
        else:
            dynamic_range = float(np.iinfo(img_a.dtype).max)

        c1 = (0.01 * dynamic_range) ** 2
        c2 = (0.03 * dynamic_range) ** 2

        kernel = self._gaussian_kernel(size=11, sigma=1.5)
        kernel_2d = np.outer(kernel, kernel)

        if img_a.ndim == 2:
            img_a = img_a[:, :, np.newaxis]
            img_b = img_b[:, :, np.newaxis]

        channels = img_a.shape[2]
        ssim_values = []

        for c in range(channels):
            a_c = img_a[:, :, c]
            b_c = img_b[:, :, c]

            mu_a = self._convolve2d(img=a_c, kernel=kernel_2d)
            mu_b = self._convolve2d(img=b_c, kernel=kernel_2d)
            mu_aa = self._convolve2d(img=a_c * a_c, kernel=kernel_2d)
            mu_bb = self._convolve2d(img=b_c * b_c, kernel=kernel_2d)
            mu_ab = self._convolve2d(img=a_c * b_c, kernel=kernel_2d)

            sigma_a_sq = mu_aa - mu_a**2
            sigma_b_sq = mu_bb - mu_b**2
            sigma_ab = mu_ab - mu_a * mu_b

            numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
            denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a_sq + sigma_b_sq + c2)

            ssim_map = numerator / denominator
            ssim_values.append(np.mean(ssim_map))

        return float(np.mean(ssim_values))

    def matches(self, *, serialized_data: bytes, snapshot_data: bytes) -> bool:
        expected_array = self.deserialize(snapshot_data)
        received_array = self.deserialize(serialized_data)

        if expected_array.shape != received_array.shape:
            return False

        ssim_val = self._compute_ssim_windowed(img_a=expected_array, img_b=received_array)
        return ssim_val >= self.min_ssim


@pytest.fixture
def ssim_tiff_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(SSIMImageSnapshotExtension)
