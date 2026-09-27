"""Mixin providing shared tests for PerturbVideo implementations.

Mirrors tests/impls/perturb_image/perturber_tests_mixin.py for the video domain.

This module provides ``PerturbVideoTestsMixin``, a mixing class that defines
shared contract and discovery tests for PerturbVideo implementations.

Subclasses must define:
    - ``impl_class``: the PerturbVideo implementation class under test
    - ``make_perturber()``: returns a ready-to-use instance of that class
    - ``make_frames()``: returns a fresh list[VideoFrame] valid for that perturber

Shared Test Cases:
    Plugin Discovery
        - Perturber is discoverable via PerturbVideo.get_impls()
    Iterator Contract
        - perturb() requires an Iterator[VideoFrame]; a plain list raises
          TypeError, while a valid iterator can be consumed, yielding VideoFrames.

This module also provides ``ImageConversionTestsMixin``, a mixin class that
defines shared image conversion tests for PerturbVideo implementations.

Usage:
    Subclasses must define:
        - ``make_perturber()``: Returns a ready-to-use perturber instance
        - ``convert_options``: Keyword arguments passed to ``_convert_image``
        - ``restore_options``: Keyword arguments passed to ``_restore_image``
        - ``working_dtype``: Expected dtype after conversion
        - ``expected_conversion``: Expected converted values for input pixels
          ``(0.0, 0.5, 1.0)``
        - ``keeps_rgb``: Whether restoration keeps RGB output instead of
          restoring the input channel layout

Shared Test Cases:
    Conversion Values
        - Conversion follows the implementation's scaling policy
    Restored Layout and Dtype
        - Restoration returns the expected channel layout and dtype
        - Restored arrays share no memory with the original or working arrays
    Channel Layout
        - Conversion returns the expected channel layout and values
        - Converted arrays share no memory with the original array

Note: this mixin deliberately does NOT assert output frame count == input count;
that is not a universal PerturbVideo contract. Keep any equal-length assertion in
the implementation-specific test class.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from nrtk.interfaces import PerturbVideo, VideoFrame
from tests.impls.perturb_video.test_perturber_utils import perturber_assertions


class PerturbVideoTestsMixin:
    """Mixin providing shared tests for PerturbVideo implementations."""

    impl_class: ClassVar[type[PerturbVideo]]

    def make_perturber(self) -> PerturbVideo:
        """Return a ready-to-use instance of ``impl_class``."""
        raise NotImplementedError

    def make_frames(self) -> list[VideoFrame]:
        """Return a fresh list of VideoFrame objects valid for the perturber."""
        raise NotImplementedError

    # ========================== Plugin Discovery ==========================

    def test_plugin_discovery(self) -> None:
        """Perturber is discoverable via PerturbVideo.get_impls()."""
        assert self.impl_class in PerturbVideo.get_impls()

    # ========================== Iterator Contract =========================

    def test_perturb_requires_iterator(self) -> None:
        """``perturb`` requires an Iterator[VideoFrame], not a plain list."""
        perturber = self.make_perturber()
        # A plain list is rejected by the iterator contract.
        with pytest.raises(TypeError, match="not an iterator"):
            list(perturber(frames=self.make_frames()))  # type: ignore[arg-type]
        # A valid iterator can be consumed without error and yields VideoFrames.
        outputs = list(perturber(frames=iter(self.make_frames())))
        assert all(isinstance(frame, VideoFrame) for frame in outputs)

    # ========================== Dtype Contract ============================

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
    def test_output_dtype_matches_input(self, dtype: type) -> None:
        """Frames come back in the dtype they arrived in, for uint8 and normalized floats."""

        def _as_dtype(image: np.ndarray, dtype: type) -> np.ndarray:
            if np.issubdtype(dtype, np.floating):
                return (image.astype(dtype) / np.array(255, dtype=dtype)).astype(dtype)
            return image.astype(dtype)

        frames = [
            VideoFrame(
                image=_as_dtype(image=frame.image, dtype=dtype),
                timestamp=frame.timestamp,
                boxes=frame.boxes,
                additional_params=frame.additional_params,
            )
            for frame in self.make_frames()
        ]
        for _ in perturber_assertions(perturb=self.make_perturber(), frames=iter(frames)):
            pass


class ImageConversionTestsMixin:
    """Mixin providing shared image conversion tests for PerturbVideo implementations.

    See module docstring for full list of shared test cases and usage instructions.

    Attributes:
        convert_options: Keyword arguments passed to ``_convert_image``.
        restore_options: Keyword arguments passed to ``_restore_image``.
        working_dtype: Expected dtype after conversion.
        expected_conversion: Expected converted values for input pixels
            ``(0.0, 0.5, 1.0)``.
        keeps_rgb: Whether restoration keeps RGB output instead of restoring
            the input channel layout.
    """

    convert_options: ClassVar[dict[str, bool]]
    restore_options: ClassVar[dict[str, bool]]
    working_dtype: ClassVar[type]
    expected_conversion: ClassVar[tuple[float, float, float]]
    keeps_rgb: ClassVar[bool]

    def make_perturber(self) -> PerturbVideo:
        """Return the implementation under test."""
        raise NotImplementedError

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_conversion_values(self, dtype: type) -> None:
        """Conversion follows the implementation's scaling policy."""
        perturber = self.make_perturber()
        original = np.array([[0.0, 0.5, 1.0]], dtype=dtype)

        working = perturber._convert_image(
            image=original,
            **self.convert_options,
        )

        values = working[0] if working.ndim == 2 else working[0, :, 0]
        assert working.dtype == self.working_dtype
        assert np.array_equal(values, self.expected_conversion)
        assert not np.shares_memory(working, original)

    @pytest.mark.parametrize("shape", [(7, 9), (7, 9, 1), (7, 9, 3)])
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
    def test_restored_layout_and_dtype(
        self,
        shape: tuple[int, ...],
        dtype: type,
    ) -> None:
        """Restoration follows the channel policy and returns an independent copy."""
        perturber = self.make_perturber()
        original = np.ones(shape=shape, dtype=dtype)

        working = perturber._convert_image(
            image=original,
            **self.convert_options,
        )
        restored = perturber._restore_image(
            image=working,
            original_image=original,
            **self.restore_options,
        )

        expected_shape = (*shape[:2], 3) if self.keeps_rgb else shape
        assert restored.shape == expected_shape
        assert restored.dtype == original.dtype
        expected = original
        if self.keeps_rgb:
            if expected.ndim == 2:
                expected = expected[:, :, None]
            if expected.shape[2] == 1:
                expected = np.repeat(expected, 3, axis=2)

        assert np.array_equal(restored, expected)
        assert not np.shares_memory(restored, original)
        assert not np.shares_memory(restored, working)

    @pytest.mark.parametrize("convert_to_rgb", [True, False])
    @pytest.mark.parametrize(
        ("pixels", "rgb_pixels"),
        [
            ([[10, 20]], [[[10, 10, 10], [20, 20, 20]]]),
            ([[[10], [20]]], [[[10, 10, 10], [20, 20, 20]]]),
            ([[[10, 20, 30]]], [[[10, 20, 30]]]),
        ],
    )
    def test_conversion_channel_layout(
        self,
        pixels: list,
        rgb_pixels: list,
        convert_to_rgb: bool,
    ) -> None:
        original = np.array(pixels, dtype=np.uint8)
        expected = np.array(
            rgb_pixels if convert_to_rgb else pixels,
            dtype=self.working_dtype,
        )

        converted = self.make_perturber()._convert_image(
            image=original,
            convert_to_rgb=convert_to_rgb,
        )

        assert np.array_equal(converted, expected)
        assert not np.shares_memory(converted, expected)
        assert converted.dtype == self.working_dtype
        assert not np.shares_memory(converted, original)
